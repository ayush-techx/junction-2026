"""Anomaly scoring of the held brick: fixed crop, Anomalib OpenVINO model on the NPU."""
import os
import time

import numpy as np


def crop(img, roi):
    if not roi:
        return img
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def _field(pred, name):
    """Anomalib 2.x returns a batch object with attributes; older versions a dict-like result."""
    v = getattr(pred, name, None)
    if v is None and isinstance(pred, dict):
        v = pred.get(name)
    if v is not None and hasattr(v, "detach"):     # torch tensor
        v = v.detach().cpu().numpy()
    return v


class Inspector:
    def __init__(self, cfg):
        ic = cfg["inspection"]
        path = ic.get("model")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"inspection.model is {path!r}. Collect images (`cell collect`), train with "
                "tools/train_anomaly.py, then set the exported model.xml path in config.yaml.")
        self.device, self.roi, self.last_ms = ic["device"], ic["roi"], None
        self.secondary = None
        self.primary_threshold = float(ic.get("threshold", 0.5))
        self.primary_band = float(ic.get("band", 0.0))
        self.secondary_threshold = float(ic.get("secondary_threshold", 0.5))
        self.secondary_band = float(ic.get("secondary_band", 0.03))
        if ic.get("secondary_model"):
            secondary_cfg = {k: v for k, v in ic.items() if not k.startswith("secondary_")}
            secondary_cfg["model"] = ic["secondary_model"]
            self.secondary = Inspector({"inspection": secondary_cfg})
        try:
            from anomalib.deploy import OpenVINOInferencer
            self.inf = OpenVINOInferencer(path=path, device=self.device)
            self.raw = None
        except ImportError:
            # Runtime env without Anomalib (e.g. the Studio backend venv): plain OpenVINO.
            # Anomalib 2.x embeds pre/post-processing in the exported graph.
            import openvino as ov
            core = ov.Core()
            model = core.read_model(path)
            shape = model.input(0).get_partial_shape()
            # Some anomalib exports (PatchCore) bake in a fixed batch > 1; keep it and tile the input.
            self.batch = shape[0].get_length() if shape[0].is_static else 1
            if shape.is_dynamic:
                model.reshape([self.batch, 3, 256, 256])    # the NPU only compiles static shapes
            order = [self.device] + [d for d in ("GPU", "CPU") if d != self.device]
            errors = []
            for dev in order:
                try:
                    self.raw = core.compile_model(model, dev)
                    if dev != self.device:
                        print(f"inspection: {self.device} failed, running on {dev}")
                    self.device = dev
                    break
                except RuntimeError as e:
                    errors.append(f"{dev}: {str(e).splitlines()[-1]}")
            else:
                raise RuntimeError("could not compile the inspection model: " + " | ".join(errors))
            self.inf = None

    def _raw_predict(self, rgb):
        import cv2
        shape = self.raw.input(0).get_partial_shape()
        h, w = (shape[2].get_length(), shape[3].get_length()) if shape[2].is_static else (256, 256)
        x = cv2.resize(rgb, (w, h)).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        n = self.raw.input(0).get_partial_shape()[0].get_length()
        if n > 1:
            x = np.repeat(x, n, axis=0)
        out = {p: np.asarray(v)[:1] for p, v in self.raw(x).items()}   # keep the first (our) image
        score = heat = None
        for port, val in out.items():
            names, arr = port.get_names(), np.asarray(val)
            if "pred_score" in names or (score is None and arr.size == 1):
                score = arr
            elif "anomaly_map" in names or (heat is None and arr.ndim >= 2 and arr.size > 1):
                heat = arr
        return {"pred_score": score, "anomaly_map": heat}

    def score(self, bgr):
        """Returns (score 0..1, heatmap HxW or None, cropped BGR image)."""
        c = crop(bgr, self.roi)
        rgb = np.ascontiguousarray(c[..., ::-1])                       # model expects RGB
        t = time.perf_counter()
        pred = self.inf.predict(rgb) if self.inf is not None else self._raw_predict(rgb)
        self.last_ms = (time.perf_counter() - t) * 1000
        score = float(np.asarray(_field(pred, "pred_score")).reshape(-1)[0])
        heat = _field(pred, "anomaly_map")
        heat = np.asarray(heat).squeeze() if heat is not None else None
        # Preserve primary passes. A second, separately calibrated memory bank can
        # clear pose/colour false alarms only when its score is confidently normal.
        # Uncertain secondary results retain the original rejection decision.
        if self.secondary is not None and score > self.primary_threshold - self.primary_band:
            secondary_score, secondary_heat, _ = self.secondary.score(bgr)
            self.last_ms += self.secondary.last_ms
            cutoff = self.secondary_threshold - self.secondary_band
            if secondary_score <= cutoff:
                original = score
                scale = self.primary_band / self.secondary_band
                score = min(score, max(0.0, min(1.0,
                    self.primary_threshold + (secondary_score - self.secondary_threshold) * scale)))
                heat = secondary_heat
                print(f"inspection: live-photo calibration cleared primary={original:.4f} "
                      f"secondary={secondary_score:.4f} calibrated={score:.4f}", flush=True)
        return score, heat, c
