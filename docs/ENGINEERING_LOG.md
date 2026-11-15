# SecondLook / Junction 2026 handoff (2026-11-15 01:25)

## Goal
Intel Physical AI challenge (Junction 2026): SO-101 arm on an Intel Core Ultra laptop picks Duplo bricks,
inspects each with the wrist camera (Anomalib, OpenVINO), sorts **good -> pink bowl, defective -> blue bowl**.
Optional: Cloudinary challenge (300 snapshots already uploaded; dashboard not built).

## Hardware (see memory so101-arm-setup)
- Follower 5AE6082572, leader 5B91043227: always use /dev/serial/by-id, ttyACM numbers swap.
- **Follower servos are 7.4 V type (max 8 V): use the 5 V adapter, NEVER 12 V** (12 V was plugged once, servos survived).
  The 12 V adapter belongs to the leader (12 V servos).
- Cameras: top RealSense 409522061795 (USB serial 310423022922), side RealSense 243622061531 (344223023487),
  wrist Sonix USB2.0_CAM1 (flaky cable, drops often; SecondLook uses /dev/v4l/by-id path).
- Only ONE program may hold the arm and cameras (Studio OR SecondLook).

## Physical AI Studio
- Repo ~/physical-ai-studio (branch junction2026). **Currently STOPPED** (user asked). Restart:
  `cd ~/physical-ai-studio/application/backend && uv run --extra xpu physicalai-studio serve`
  `cd ~/physical-ai-studio/application/ui && npm run start`
- Project 4ad69942…, env `techh`, dataset `NewData` faf53384… : 198 episodes, relabelled
  "pick up the brick and place it in the {pink|blue} bowl" (backups in ~/.local/share/physicalai/datasets_backup/).
- Models: ACT 6ep cb771287 (doesn't pick), **ACT 20ep 918242dc** (picks sometimes, bowl = coin flip, stops after one
  place because demos end above a bowl), **SmolVLA 20k ef34e2e2** (follows pink/blue text 10/10 offline; needs Task text
  in Run model or it crashes with BusRequestError 'task').
- ACT v3 (clean data + augmentation, 12 ep, job 44543851) is **paused at 18%**; it reconnects to Modal when Studio restarts.
- Modal credits ~ $15 used (A10G $1.10/h). HF needs nothing.

## SecondLook (~/Downloads/secondlook) = the main app
- `./sl help`. Control center (persistent, systemd --user `secondlook-center`): **http://127.0.0.1:8770**
  (start/stop runs, live view, results per run, inspection gallery, model/data, hardware, logs). Run UI: :8765.
- Pipeline: start pose -> pick (ACT/SmolVLA, hand-fed, fixed spot, or table scan) -> **squeeze check (held 1/0)** ->
  inspect pose -> **re-check grip (slip -> retry)** -> PatchCore score -> good/unsure/defective -> second look if unsure
  (clean 2nd view can clear it; any defective view rejects; unsure twice -> reject) -> taught bowl pose.
- Config secondlook/config.yaml: policy.path = ACT 20ep, start_pose = demo median, inspection = PatchCore fp16 on GPU,
  **threshold 0.486, band 0.08**. Poses taught (home, handoff, inspect, putback, bowl_good=pink, bowl_defective=blue).
- Defect data: data/bricks (42 good / 15 defect hand-fed) + data/bricks_live (15 good / 11 defect real grasps).
  Every inspection photo saved in runs/inspections/. G/D keys or "It was GOOD/DEFECTIVE" buttons file photos for learning.
  `--learn-every 5` retrains in the BACKGROUND and hot-swaps weights; `./sl recalibrate` does it manually.
  Never use `--truth good` with mixed bricks (it mislabelled 3 defects once; fixed).
- Latest real result (calib2, corrected weights): 4/4 correct, 0 false alarms. Earlier runs had FPs from empty grippers
  and contaminated labels (both fixed).
- Table scan (`--scan`): table_map.yaml has 5 points, **worst fit error 14 deg (too high)**; empty-table photo is stale
  (bowls moved). Needs `./sl table --background` then `./sl table --add` 3-4 consistent points (wrist down, elbow same way).
- NPU crashes on this laptop (driver ZE_RESULT_ERROR_UNKNOWN): use GPU (PatchCore ~12 ms, ACT ~9 ms, SmolVLA ~106 ms).

## NEW 01:55: product UI, sort-the-whole-table, voice, model switcher
- Control center redesigned (http://127.0.0.1:8770): Station (table map, Sort the table, bowls, last brick + score
  meter, plain-language feed) | Insights (KPIs, sorted-per-run + accuracy charts, score strip, history with labels) |
  Models (pick engine switcher: Vision pick / ACT 20ep / SmolVLA 20k; defect model + roll back) | Stage view
  (#stage, fullscreen for the audience). Live run is proxied (/api/live, /live/frame.jpg, /api/live/cmd): no iframe.
- **Sort the table** = `run --scan --auto --until-clear` (vision engine) or `run --auto --until-clear` (learned engine,
  bricks counted by the top camera via TableWatcher). Says what is on the table, per brick "Good brick. Pink bowl.
  Two left.", summary at the end; stops after 3 failed picks in a row. Simulated end to end (3/3 picked first try).
- Voice: secondlook/announce.py; Piper neural TTS offline (tts-env/, voices/en_US-lessac-medium.onnx), espeak-ng
  fallback; on/off in runs/voice.json (header speaker button, works mid-run). `./sl scene --speak` / Describe button.
  No language model: sentences are built from the detections and verdicts.
- Detection: colour now backs up depth (far bricks read only ~7 mm on the D415), and blobs touching the arm are
  split by colour (red 2x4 + green brick next to the parked arm are found; arm parts never become bricks).
- Real-robot check: 01:25 vision pick grasped the blue brick first try (edge of reach). 01:22 one connect failed:
  servo 6 "no status packet" (gripper cable/power) -> now a plain message.
- config.yaml: `station.picker: vision`, `bowls: {good: pink, defective: blue}`.

## NEW 01:25: precise table pick, scene check, history + learning tabs
- **Why scan picks missed:** the old per-joint pixel->joints map does not generalise. Leave-one-out error on the 5
  taught points is 60-160 units on the elbow (one point taught elbow-up). The "14 deg fit error" hid this.
  The empty-table photo was also stale (taken before the bowls were placed) -> bowls and bricks in bowls were counted.
- **Fix = depth + kinematics** (`scan.method: ik`, default once secondlook/handeye.yaml exists):
  top D415 depth (`cameras.top.depth: true`) -> table plane (RANSAC) -> height map -> bricks = 6-60 mm tall,
  brick-sized in mm; bowls found live per colour; box/arm flagged as "not a brick". No background photo needed.
  Brick grasp point -> robot base via hand-eye transform (secondlook/handeye.py) -> IK on the SO-101 URDF
  (secondlook/kinematics.py, copy in secondlook/urdf/). Units = Studio viewer convention (n in -100..100 maps onto
  the URDF limits); verified: FK + one rigid transform fits the 5 taught points to 7-15 mm.
  Wrist roll is turned so the fingers close across the brick (2x4 long side detected).
- Calibration now: `./sl table --from-map` done (10.2 mm mean, 5 hand-held points).
  **Next: `./sl table --auto`** (arm puts ONE brick at 8 spots, camera sees where it landed, re-picks by vision;
  simulated: 24 mm -> 1-3 mm after 4 spots). Then `./sl run --scan --auto --eval --cycles 5`.
  Second looks keep teaching it (`learn_from_putback`).
- `scan.place_by_vision: false` (default): true = drop into the bowl where the camera sees it (pink reachable
  cleanly; blue only with ~54 deg wrist tilt -> falls back to the taught pose if IK can't reach).
- `./sl scene` = camera-only scene check (no arm): runs/scene/latest*.jpg + latest.json; also a button in the center.
- History: SQLite runs/history.db (bricks + pick attempts); old CSVs imported automatically by the center.
- Control center tabs: Live | History (photos, label Good/Defective -> files into data/bricks_live, Recalibrate) |
  Table map (detections / height map / pick heatmap, calibration mm) | Learning (score strip plot, best threshold on
  real photos, what to do next). URL hash selects a tab: http://127.0.0.1:8770/#map
- Bugs fixed: hand-fed first press closed on nothing and logged a failed pick (now asks again; button says
  "Brick is in"); cycle time included operator waits (now wait_s separate, median s/brick); inspection photo names
  collided across days (now YYYYMMDD-HHMMSS-ms); center BrokenPipe tracebacks, leaked file/DB handles;
  2nd look with table scan could re-pick a different brick (now prefers the put-back spot); servo "Input voltage
  error" now gives a plain message (seen once at 00:34 on servo 6 = gripper: check the 5 V adapter/plug).
- Finding: all 4 real-photo false alarms are ACT-grasped GOOD bricks (0.56-0.76); hand-fed good bricks score low.
  -> label more robot-grasped good bricks, then Recalibrate.
- New tests: tests/test_precise.py (62 tests total, all pass). Backups of edited files: *.bak_0101map.

## Open problems / next steps
1. **Scene keeps drifting** (bowls, side camera, box, bricks left in bowls): this is why ACT hovers and the scan miscounts.
   Tape bowls + side camera stand, clear clutter, bricks only in the area between robot and bowls.
2. `./sl table --background`, then `./sl table --add`; target worst error < 5 deg; then test "Table scan pick".
3. Test auto pick (ACT 20ep) in the fixed scene; if weak, restart Studio so ACT v3 finishes, or try SmolVLA in SecondLook
   (set policy.path to ef34e2e2…/exports/openvino; task text already set).
4. More hand-fed eval rounds (G/D) to firm up the defect threshold; then `./sl results` for the README table.
5. Submission: demo video, README results, pitch (docs/pitch.md), Intel form; Cloudinary optional (cld.media/hackathon-survey).

## Running in background now
- secondlook-center (systemd user service, port 8770), station monitor (~/station_monitor, port 8090, logging only,
  NO camera reads), possibly a SecondLook run started from the center.

## Gotchas learned
- Don't kill/restart Studio unless the user explicitly asks (jobs get marked failed; remote jobs reconnect on restart).
- Side processes must not subscribe to Studio's iceoryx2 camera streams (caused EventOpenError DoesNotExist);
  fix: stop the subscriber + `iox2.Node.try_cleanup_dead_nodes(...)`.
- ACT commands the gripper only to ~9, so an empty gripper looks "held": always squeeze fully then read (<5 = empty).
- Studio Run model: arm must start raised/open over the pick area (teleop first), not above a bowl.
- Backups of every edited file are next to it as *.bak_HHMM*.
