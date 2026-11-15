# SecondLook

**A low-cost robot arm that learns to pick bricks from demonstrations, inspects each one with the camera on its
own gripper, and sorts out defects it has never seen. Everything runs on one Intel Core Ultra laptop.**

Built at Junction 2026 for the Ignite Next x Intel Physical AI Challenge.

## The problem

Duplo bricks are made for toddlers, and a damaged one is a safety problem. On many production lines a person
still checks parts by eye for hours, and attention drops over a shift. Automating that job usually means a
specialist programming a robot cell for each product.

## What it does

1. **Pick (learned).** An ACT policy, trained in Intel Physical AI Studio from teleoperated demonstrations,
   picks a brick from the table. The app stops the policy once the brick is held and lifted.
2. **Inspect.** The arm moves to a fixed inspection pose. The wrist camera sees the held brick close up, always
   in the same place, so the camera that helps the policy pick is also the inspection camera.
3. **Judge.** An Anomalib model trained only on clean bricks scores the image and draws a heatmap. It flags
   marks, tape and damage without ever being shown a defect.
4. **Second look.** If the score is unsure, the arm puts the brick back, grasps it again and re-inspects it.
   Still unsure means the reject bowl, where a person checks it.
5. **Sort and tell.** The brick goes to the good or the reject bowl. The result appears on the dashboard and is
   spoken aloud. A missed grasp is detected from the gripper position and retried.

Inference never leaves the laptop: the pick policy runs on the integrated GPU, the anomaly model on the NPU,
both through OpenVINO.

## Accessible by design

Every result reaches the operator on two channels: on screen in large, high-contrast text and spoken aloud
by an offline voice. One button (Space, or a USB button or foot pedal mapped to Space) starts each inspection.
The dashboard announces results to screen readers. The goal is that more people can run the same station on
their own. It has not been tested with users yet.

## Results

Measured on our station. Fill from `./sl results` and `./sl bench`.

| Metric | Normal | Shifted | Clutter |
|---|---|---|---|
| Pick success | | | |
| Sort accuracy (of picked) | | | |
| Defects caught | | | |
| Good bricks wrongly rejected | | | |
| Settled by the second look | | | |
| Cycle time median / p90 | | | |

| Model | Device | Precision | p50 latency | Energy per inference |
|---|---|---|---|---|
| Pick policy (ACT) | iGPU | | | |
| Anomaly (PaDiM) | CPU | FP16 | | |
| Anomaly (PaDiM) | NPU | FP16 | | |
| Anomaly (PaDiM) | NPU | INT8 | | |

## How it is built

```
Physical AI Studio:   teleop recording ─► ACT training ─► OpenVINO export ─────────────┐
Anomalib:             clean-brick photos ─► PaDiM ─► OpenVINO FP16 / INT8 ──────────┐  │
                                                                                     ▼  ▼
SecondLook (this repo): state machine ─ robot driver (LeRobot) ─ inspector (NPU) ─ policy (iGPU)
                        dashboard (browser) ─ voice ─ telemetry (latency, RAPL power, temperature) ─ CSV logs
```

| Path | Purpose |
|---|---|
| `secondlook/cell.py` | the app: `check`, `teach`, `collect`, `run`, `review` |
| `secondlook/machine.py` | one inspection cycle as plain logic (unit-tested) |
| `secondlook/robot.py` | SO-101 + cameras through LeRobot, taught poses, grasp detection |
| `secondlook/policy.py` | learned pick with `physicalai.inference.InferenceModel`; hand-fed mode |
| `secondlook/inspection.py` | crop + Anomalib OpenVINO model (falls back to plain OpenVINO) |
| `secondlook/web.py`, `static/index.html` | local dashboard, no external assets |
| `secondlook/announce.py` | offline speech with espeak-ng |
| `secondlook/power.py` | package power (RAPL) and temperature |
| `secondlook/calibrate.py` | threshold and uncertain band from scored photos |
| `tools/` | dataset preparation, training, evaluation, benchmarks, contact sheets |
| `tests/` | 38 tests; hardware is replaced by stand-ins, library-dependent tests skip when a library is missing |

## Run it

Step-by-step instructions for the station laptop: **[INTEL_LAPTOP.md](INTEL_LAPTOP.md)**.

```bash
./setup.sh
./sl check
./sl teach
./sl collect --label good --n 50
./sl train --root data/bricks
./sl run --handfed
```

## Engineering notes

- Training in Studio first aborted as an orphan job: torchcodec needs the FFmpeg libraries (`apt install ffmpeg`).
- Studio then trained on the CPU only (about 30 s per step) because the Level Zero GPU driver was missing
  (`apt install libze-intel-gpu1`).
- On the GPU, training froze at step 0. `dmesg` showed GPU page faults and a compute-engine reset. A minimal
  repro isolated it to the default fused attention kernel at ACT's sequence length (902 tokens × batch 8); the
  math attention backend runs fine (`torch.nn.attention.sdpa_kernel(SDPBackend.MATH)`).
- Episodes are recorded at 640×480. At 1080p a single first-layer ResNet activation is about 3.2 GB, close to
  the GPU's 4 GB single-allocation limit.
- The recorded demonstrations end in a bowl. The app ends the policy right after the grasp and does inspection
  and sorting itself, so the learned pick and the anomaly decision stay independent.

## What is real

Everything in the demo runs live on the station. Inspection and bowl positions are taught poses. The pick and
the defect judgement are learned.

## License

MIT
