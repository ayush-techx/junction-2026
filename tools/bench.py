"""Latency and energy per inference for OpenVINO models on CPU / GPU / NPU.

    python tools/bench.py --models exports/padim/fp16/weights/openvino/model.xml \
        exports/padim/int8/weights/openvino/model.xml --devices CPU GPU NPU --n 200

Prints a markdown table for the README. Energy comes from RAPL (whole package), so it is
reported as joules per inference including idle draw; run with the arm and Studio idle.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def static_input(model, default_hw=(256, 256)):
    import numpy as np
    shape = model.input(0).get_partial_shape()
    dims = [d.get_length() if d.is_static else None for d in shape]
    if len(dims) == 4:
        n, c, h, w = dims
        dims = [n or 1, c or 3, h or default_hw[0], w or default_hw[1]]
    else:
        dims = [d or 1 for d in dims]
    return np.random.default_rng(0).random(dims, dtype=np.float32)


def bench(core, path, device, n, warmup=10):
    from secondlook.power import Power
    t0 = time.perf_counter()
    compiled = core.compile_model(path, device)
    compile_s = time.perf_counter() - t0
    x = static_input(compiled)
    for _ in range(warmup):
        compiled(x)
    power = Power()
    power.sample()
    times = []
    start = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        compiled(x)
        times.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - start
    watts = power.sample()
    times.sort()
    return {"p50": times[n // 2], "p90": times[int(n * 0.9) - 1], "fps": n / total,
            "j_per_inf": watts * total / n if watts else None, "watts": watts, "compile_s": compile_s}


def main():
    import openvino as ov
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()

    core = ov.Core()
    available = core.available_devices
    print("available:", available)
    print("| model | device | p50 ms | p90 ms | FPS | package W | J / inference | compile s |")
    print("|---|---|---|---|---|---|---|---|")
    for path in a.models:
        name = os.sep.join(path.split(os.sep)[-5:-3]) or os.path.basename(path)
        for dev in a.devices:
            if not any(d.startswith(dev) for d in available):
                print(f"| {name} | {dev} | not available | | | | | |")
                continue
            try:
                r = bench(core, path, dev, a.n)
            except Exception as e:
                print(f"| {name} | {dev} | failed: {str(e).splitlines()[0][:60]} | | | | | |")
                continue
            w = f"{r['watts']:.1f}" if r["watts"] else "n/a"
            j = f"{r['j_per_inf']:.3f}" if r["j_per_inf"] else "n/a"
            print(f"| {name} | {dev} | {r['p50']:.2f} | {r['p90']:.2f} | {r['fps']:.0f} | {w} | {j} | {r['compile_s']:.1f} |")


if __name__ == "__main__":
    main()
