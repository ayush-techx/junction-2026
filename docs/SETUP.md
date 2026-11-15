# Running SecondLook on the Intel laptop

Everything below runs on the station laptop (HP OmniBook Ultra 14, Core Ultra X9, Ubuntu) that has
Physical AI Studio installed in `~/physical-ai-studio`. Run the commands one at a time from a terminal.

Two Python environments are used, and `./sl` picks the right one for you:

| Environment | Used for | Where |
|---|---|---|
| Studio backend venv | running the cell, dashboard, evaluation, benchmarks | `~/physical-ai-studio/application/backend/.venv` |
| `anomalib-env` | training the anomaly model | created by `setup.sh` inside this folder |
| `tts-env` | Piper offline voice | created by `setup.sh` inside this folder |
| `accessinspect/.venv` | optional Ava voice + Gemini helper | created by `setup.sh` |

---

## 1. Clone and install (once, ~10 minutes)

```bash
git clone https://github.com/ayush-techx/junction-2026.git ~/secondlook && cd ~/secondlook
```
```bash
./setup.sh
```
It installs the app libraries into the Studio venv, creates `anomalib-env`, installs the offline voice
(Piper + `espeak-ng`), gives you serial-port access, downloads the trained models
(`./tools/fetch_models.sh`, add `--all` for ACT 20ep and SmolVLA) and runs the test suite. It ends with `Done. Next step: ./sl check`.
If it added you to the `dialout` group, log out and back in once.

Test the speaker:
```bash
./sl voice
```

---

## 2. Take the arm over from Studio

The arm and cameras can only be used by one program at a time.

1. In Physical AI Studio, stop any teleoperation, recording or "Run model" session. A training job can keep running.
2. Check the ports are free (no output means free):
   ```bash
   fuser -v /dev/ttyACM0 /dev/ttyACM1
   ```
3. When you go back to Studio, quit SecondLook first (STOP button or Ctrl+C).

---

## 3. Fill in the settings (`secondlook/config.yaml`)

Run the check. It prints what it finds and saves one snapshot per camera into `runs/`:
```bash
./sl check
```

Fill in these fields, then run `./sl check` again until nothing is reported missing.

| Field | How to find it |
|---|---|
| `robot.port` | `ls /dev/ttyACM*` with only the follower plugged in, or your udev name |
| `robot.calibration_dir`, `robot.id` | **must be Studio's calibration** for the follower: `find ~ -path '*calib*' -name '*.json'`. `calibration_dir` = the folder, `id` = the file name without `.json` |
| `robot.use_degrees` | the same setting Studio uses (ask the Intel mentors if unsure) |
| `cameras.*.serial` (RealSense) | `rs-enumerate-devices -s` |
| `cameras.wrist.path` | `ls /dev/video*`; pick the one whose snapshot in `runs/` shows the gripper view |
| Camera names (`top`, `side`, `wrist`) | must match the names used when the policy's episodes were recorded |
| `robot.gripper.*` | run `./sl check` three times: gripper open, closed on nothing, closed on a brick. `open` = open reading, `close` = a value slightly below the empty-closed reading, `hold_min` = halfway between empty-closed and on-a-brick, `hold_max` = halfway between on-a-brick and open |
| `power.idle_watts` | the watts `./sl check` prints with the arm idle. If it says RAPL is unreadable: `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj` |
| `inspection.roi` | open `runs/check_wrist.jpg` with a brick in the gripper and set `[x, y, width, height]` around the brick |

---

## 4. Teach the positions (~5 minutes)

```bash
./sl teach
```
Torque turns off. Hold the arm, move it by hand to each position, type the name, press Enter.
Leave the line empty to finish (torque comes back on, so hold the arm).

| Name | Position |
|---|---|
| `home` | safe resting pose above the table, clear of the bricks |
| `handoff` | gripper open at a comfortable height where a person can put a brick in it |
| `inspect` | brick lifted, wrist camera looking at it against the white board, same every time |
| `putback` | just above a free spot in the pick area (used for the second look) |
| `bowl_good` | gripper above the good bowl |
| `bowl_defective` | gripper above the reject bowl |

---

## 5. Photos for the anomaly model (~15 minutes)

The model learns what a clean brick looks like from the wrist camera at the `inspect` pose.

```bash
./sl collect --label good --n 50
```
Use every clean brick type. Put each brick in about 10 times with a different grip (turn it, flip it).

Then make the same bricks defective in a removable way (tape, stickers, whiteboard marker) and:
```bash
./sl collect --label defect --n 20
```
Photos are saved cropped in `data/bricks/good` and `data/bricks/defect`, full frames in `*_full`.
Clean the bricks afterwards.

---

## 6. Train, check, and switch on the model (~5 minutes)

The trained PatchCore model is already in `exports/` after `./tools/fetch_models.sh`. To retrain from the photos
in `data/bricks` (a few minutes on the GPU):
```bash
./sl train --root data/bricks --model patchcore
```
It ends with `done: exports/patchcore/fp16 ...`. During runs, photos you label G/D go to `data/bricks_live`;
`./sl recalibrate` retrains with them and writes the new threshold into `config.yaml`. The previous model is
kept in `exports_before_recal_*`; roll back from the Models tab of the control center.

Score every photo, get the threshold, and save heatmap images for the README:
```bash
./sl evaluate --model exports/patchcore/fp16/weights/openvino/model.xml --root data/bricks --device GPU --overlays
```
Copy the printed `threshold` and `band` into `config.yaml` (`inspection.threshold`, `inspection.band`) and set
`inspection.model: exports/patchcore/fp16/weights/openvino/model.xml`.

If clean bricks get rejected, add more good photos (robot-grasped ones matter most), then `./sl recalibrate`.
Use `GPU`: on this laptop the NPU driver fails to compile the model (`ZE_RESULT_ERROR_UNKNOWN`).

Check the model in the dashboard without the robot:
```bash
./sl review data/bricks/defect --precropped
```

---

## 7. Run the demo

The dashboard opens in the browser at http://127.0.0.1:8765. Space or **Start** begins a brick,
S, Esc or **STOP** stops the arm (torque off). Every result is also spoken.

Without the pick model (you hand the brick to the gripper):
```bash
./sl run --handfed
```

With the trained pick model:
1. In Physical AI Studio, export the trained policy to OpenVINO (Models page) and note the folder.
2. Set `policy.path` to that folder in `config.yaml`, and `policy.cameras` to the cameras it was trained with.
3. Check it loads and produces an action:
   ```bash
   ./sl check --policy
   ```
   If the action call fails with a shape or key error, the observation layout differs from the default.
   The only place to adapt is `to_obs()` in `secondlook/policy.py`.
4. Run:
   ```bash
   ./sl run
   ```

How the cycle works: home, pick (the policy runs until the gripper holds a brick and lifts it, then hands
over), inspect at the `inspect` pose, second look if the score is unsure (put back, re-grasp, re-inspect),
drop in the bowl, home. A missed grasp is detected from the gripper position and retried.

---

## 8. Numbers for the judges

Evaluation runs. After each brick press G (it really was good), D (really defective) or X (skip):
```bash
./sl run --eval --cycles 20 --layout normal --name v1_normal
```
```bash
./sl run --eval --cycles 20 --layout shifted --name v1_shifted
```
```bash
./sl run --eval --cycles 10 --layout clutter --name v1_clutter
```
Results table (paste into the README):
```bash
./sl results
```

Speed and energy per device:
```bash
./sl bench --models exports/patchcore/fp16/weights/openvino/model.xml --devices CPU GPU
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied: /dev/ttyACM0` | log out and in after `setup.sh` (dialout group) |
| `device reports readiness to read but returned no data` | Studio still holds the port (section 2), or a loose USB cable |
| Port numbers swap between replugs | use a udev rule by serial number and set `robot.port` to that name |
| `pose 'inspect' not taught` | `./sl teach` |
| `robot.gripper ... not set` | fill the gripper values (section 3) |
| `inspection.model is None` | sections 5 and 6, or use `./sl run --handfed` only after a model exists |
| Policy moves the arm to the wrong places | calibration differs from Studio's; set `robot.calibration_dir` and `robot.id` |
| Clean bricks rejected after dark | collect 20 more good photos in the current light and retrain |
| No sound | `./sl voice`; check volume and output device; `sudo apt-get install espeak-ng` |
| Dashboard does not open | open http://127.0.0.1:8765 yourself; or set `ui.mode: window` |
| GPU training stalls at step 0 (Studio) | see "Engineering notes" in README.md |
