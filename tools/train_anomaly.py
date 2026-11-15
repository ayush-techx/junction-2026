"""Train the brick anomaly model on good images only, export to OpenVINO (FP16 + INT8).

Written for Anomalib 2.x (check: `uv pip show anomalib` in anomalib-env).
Runs on CPU by default: PaDiM trains in about a minute, and the GPU stays free for ACT.

Folder layout (crops from the inspection pose, same ROI every time):
    data/bricks/good/*.png      ~100-150 good bricks, every colour and shape, both wrist angles
    data/bricks/defect/*.png    ~20-30 marked/taped/chipped bricks (used only to test + set threshold)

Run:
    python tools/train_anomaly.py --root data/bricks --model padim
    python tools/train_anomaly.py --root data/bricks --model patchcore   # if too many false rejects

Output:
    exports/<model>/fp16/weights/openvino/model.xml
    exports/<model>/int8/weights/openvino/model.xml
Exported scores are normalised to 0..1 with the decision threshold at 0.5.
"""
import argparse
import os

from anomalib.data import Folder
from anomalib.deploy import CompressionType, ExportType
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/bricks")
p.add_argument("--model", choices=["padim", "patchcore"], default="padim")
p.add_argument("--accelerator", default="cpu", help="cpu (default) or xpu")
p.add_argument("--no-int8", action="store_true")
a = p.parse_args()

has_defects = os.path.isdir(os.path.join(a.root, "defect")) and os.listdir(os.path.join(a.root, "defect"))
if not has_defects:
    print("WARNING: no defect images -> training works, but no test metrics and no threshold check")
dm = Folder(name="bricks", root=a.root, normal_dir="good", abnormal_dir="defect" if has_defects else None,
            train_batch_size=16, eval_batch_size=16, num_workers=2)
model = Padim() if a.model == "padim" else Patchcore()
engine = Engine(accelerator=a.accelerator, devices=1, max_epochs=1)

engine.fit(model=model, datamodule=dm)
if has_defects:
    engine.test(model=model, datamodule=dm)   # prints image AUROC / F1 on the defect set

out = f"exports/{a.model}"
engine.export(model=model, export_type=ExportType.OPENVINO, export_root=f"{out}/fp16")
if not a.no_int8:
    engine.export(model=model, export_type=ExportType.OPENVINO, export_root=f"{out}/int8",
                  compression_type=CompressionType.INT8_PTQ, datamodule=dm)
print(f"done: {out}/fp16 and {out}/int8")
