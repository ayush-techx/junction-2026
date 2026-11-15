"""SecondLook app: run the inspection cell, teach poses, collect anomaly images, check hardware.

    python -m secondlook.cell check [--policy]     hardware + model check, saves camera snapshots
    python -m secondlook.cell teach                hand-guide the arm, save named poses
    python -m secondlook.cell collect --label good --n 50    hand-fed brick images for Anomalib
    python -m secondlook.cell run --handfed        demo without the policy (operator feeds bricks)
    python -m secondlook.cell run                  learned pick (needs policy.path)
    python -m secondlook.cell run --eval           after each brick press g (truly good) / d (truly defective)
    python -m secondlook.cell review data/clean/defect --precropped   judge saved images, no robot

Common: --config secondlook/config.yaml
"""
import argparse
import os
import time

import yaml

POSES = ["home", "handoff", "inspect", "putback", "bowl_good", "bowl_defective"]


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_check(cfg, args):
    import cv2
    from .robot import SO101
    os.makedirs(cfg["log_dir"], exist_ok=True)
    robot = SO101(cfg)
    robot.connect()
    try:
        obs = robot.observe()
        print("joints:", {k: round(v, 1) for k, v in robot.joints(obs).items()})
        print("gripper reading now:", round(float(obs["gripper.pos"]), 1),
              "-> note it open, closed on nothing, and closed on a brick; fill robot.gripper")
        for cam in cfg["cameras"]:
            img = obs[cam]
            out = os.path.join(cfg["log_dir"], f"check_{cam}.jpg")
            cv2.imwrite(out, img[..., ::-1])
            print(f"camera {cam}: {img.shape} -> {out}")
        missing = [p for p in POSES if p not in robot.poses]
        print("poses missing:", missing or "none")
    finally:
        robot.disconnect()

    try:
        import openvino as ov
        print("OpenVINO devices:", ov.Core().available_devices)
    except ImportError:
        print("OpenVINO not importable in this environment")

    from .power import Power
    p = Power()
    p.sample(); time.sleep(2); w = p.sample()
    print("package watts (2 s):", None if w is None else round(w, 2),
          "| temp C:", p.temp_c(), "| put idle watts into power.idle_watts" if w else
          "| RAPL unreadable: sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj")

    # Report every problem and keep checking the rest.
    try:
        from .inspection import Inspector
        Inspector(cfg)
        print("inspection model: loaded")
    except Exception as e:
        print("inspection model: NOT READY ->", e)
    if args.policy:
        try:
            from .policy import PolicyPicker
            pol = PolicyPicker(cfg)
            robot = SO101(cfg)
            robot.connect()
            try:
                a = pol.act(robot.observe())
                print("policy action:", {k: round(v, 1) for k, v in a.items()}, f"({pol.last_ms:.1f} ms)")
            finally:
                robot.disconnect()
        except Exception as e:
            print("policy model: NOT READY ->", e)


def cmd_teach(cfg, args):
    from .robot import SO101
    robot = SO101(cfg, cameras=[])
    robot.connect()
    try:
        robot.torque(False)
        print("Torque OFF. Hold the arm, move it by hand, then type a pose name and press Enter.")
        print("Needed:", ", ".join(POSES), "| empty line = finish")
        while True:
            name = input("pose name> ").strip()
            if not name:
                break
            robot.save_pose(name)
            print("saved", name, robot.poses[name])
    finally:
        print("Hold the arm: torque turns ON now.")
        time.sleep(1.5)
        robot.torque(True)
        robot.disconnect()


def cmd_table(cfg, args):
    """Calibrate the top-camera table scan. The robot parks itself at `home` for every photo, and
    each point is found as the difference between a photo before and after you place the brick."""
    if getattr(args, "from_map", False):
        return cmd_table_from_map(cfg, args)
    if getattr(args, "auto", False):
        return cmd_table_auto(cfg, args)
    import cv2
    from .robot import SO101
    from .table import BACKGROUND_FILE, MAP_FILE, PoseMap, detect_bricks, fixed_objects, table_region
    if getattr(args, "background", False):
        # New empty-table photo only (bowls in place, no bricks). Calibration points stay valid as long
        # as the top camera has not moved; bowls in this photo (and bricks later lying in them) are ignored.
        robot = SO101(cfg, cameras=["top"])
        robot.connect()
        try:
            robot.hold_here()
            input("Bowls in their places, NO bricks in the pick area, nothing else on the table. Enter = photo ")
            robot.go("home")
            time.sleep(0.6)
            cv2.imwrite(BACKGROUND_FILE, robot.frame("top"))
            print("saved new empty-table photo", BACKGROUND_FILE)
        finally:
            robot.go("home")
            robot.disconnect()
        return
    if args.drop:
        m = PoseMap.load()
        pts = [p for i, p in enumerate(m.data["points"], 1) if i not in set(args.drop)]
        m2 = PoseMap.fit(pts)
        m2.save(MAP_FILE)
        worst = max(max(e.values()) for e in m2.data["fit_max_error_deg"].values())
        print(f"dropped points {args.drop}: {len(pts)} left, worst fit error {worst:.1f} deg")
        return
    points = PoseMap.load().data["points"] if args.add else []
    robot = SO101(cfg, cameras=["top"])
    robot.connect()

    def park_and_photo():
        robot.go("home")
        time.sleep(0.6)
        return robot.frame("top")

    try:
        print("The arm moves to home by itself before every photo. Keep hands clear while it moves.")
        robot.hold_here()
        if args.add:
            print(f"Adding to the saved map ({len(points)} points). Leave bricks where they are.")
            empty = cv2.imread(BACKGROUND_FILE)
            before = park_and_photo()
        else:
            input("1) Clear the pick area completely (no bricks). Enter = park + empty-table photo ")
            before = park_and_photo()
            cv2.imwrite(BACKGROUND_FILE, before)
            empty = before
        table = table_region(before)
        bowls = fixed_objects(before)
        print("   saved", BACKGROUND_FILE, "| table area found" if table else "| WARNING: table area not found")
        while True:
            n = len(points) + 1
            k = input(f"{n + 1}) Put ONE more brick in a NEW spot of the pick area (other bricks may stay), "
                      "then step back: no hands or people in the camera view. "
                      f"Enter = photo   (d = done; {max(0, 6 - len(points))} more recommended) ")
            if k.strip().lower() == "d":
                if len(points) >= 4:
                    break
                print("   need at least 4 points")
                continue
            after = park_and_photo()
            found = detect_bricks(after, empty, table, previous=before, exclude=bowls)
            if len(found) != 1:
                print(f"   {'no new brick' if not found else str(len(found)) + ' changes'} seen. Make sure only ONE "
                      "brick was added, nothing else moved, nobody is over the table. Fix it and press Enter again.")
                before = park_and_photo() if not found else before
                continue
            b = found[0]
            vis = after.copy()
            cv2.circle(vis, (int(b["u"]), int(b["v"])), 12, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(cfg["log_dir"], f"table_point_{n}.jpg"), vis)
            print(f"   new brick at pixel ({b['u']:.0f}, {b['v']:.0f}), area {b['area']}  OK")
            robot.torque(False)
            input("   Motors OFF. Hold the OPEN gripper around that brick, as for grasping it. Enter ")
            pick = {j: v for j, v in robot.joints().items() if j != "gripper"}
            input("   Lift it straight up a few cm (still open, above the brick). Enter ")
            above = {j: v for j, v in robot.joints().items() if j != "gripper"}
            if len(points) >= 4:
                guess = PoseMap.fit(points).poses(b["u"], b["v"])["pick"]
                off = {j: abs(pick[j] - guess[j]) for j in guess}
                worst_j = max(off, key=off.get)
                if off[worst_j] > 25:
                    print(f"   WARNING: this pose is very different from the other points ({worst_j} off by "
                          f"{off[worst_j]:.0f} deg). Hold the arm the SAME way as before: wrist pointing down, "
                          "elbow bent the same way.")
                    if input("   Keep it anyway? (y = keep, Enter = redo this point) ").strip().lower() != "y":
                        input("   Keep holding the arm. Enter = motors on, arm goes home, then place the same brick again ")
                        robot.hold_here()
                        before = park_and_photo()
                        continue
            points.append({"u": b["u"], "v": b["v"], "area": b["area"], "pick": pick, "above": above})
            input("   Keep holding the arm. Enter = motors on, the arm goes home by itself ")
            robot.hold_here()
            before = park_and_photo()       # next point is the difference from this photo
        m = PoseMap.fit(points)
        m.save(MAP_FILE)
        worst = max(max(e.values()) for e in m.data["fit_max_error_deg"].values())
        print(f"saved {MAP_FILE}: {len(points)} points, worst fit error {worst:.1f} deg "
              "(under ~3 deg is good; above ~6, re-check the point that is off)")
        print("Now take ALL bricks off and run:  ./sl run --scan --auto")
    finally:
        try:
            robot.go("home")
        finally:
            robot.disconnect()



def _scene_ref(res, plane):
    return {"plane": [float(x) for x in plane] if plane is not None else None,
            "bowls": {b["color"]: [round(b["u"], 1), round(b["v"], 1)] for b in res["bowls"]},
            "ts": time.strftime("%Y-%m-%d %H:%M:%S")}


def cmd_scene(cfg, args):
    """Scene check with the top camera only (colour + depth). The arm is not touched or moved.
    Saves runs/scene/latest*.jpg + latest.json for the dashboard and prints what to fix."""
    import json
    from . import scene
    from .handeye import HandEye
    bgr, depth, K = scene.capture(cfg)
    he = HandEye.load() if HandEye.exists() else None
    plane, inl = (scene.fit_plane(depth, K) if depth is not None else (he.plane if he else None, None))
    res = scene.detect(bgr, depth, K or (he.K if he else None), plane)
    rep = scene.report(res, plane, ref=he.data.get("scene_ref") if he else None, inlier=inl)
    if he is not None:
        try:
            from .kinematics import Arm
            import yaml
            home = yaml.safe_load(open(cfg["robot"]["poses_file"]))["home"]
            u, v = scene.project(he.to_cam(Arm().tcp(home)), K)
            rep["obstacles"] = [o for o in rep["obstacles"] if ((o["u"] - u) ** 2 + (o["v"] - v) ** 2) ** 0.5 > 120]
            from .precise import IKTablePicker
            rep["warnings"] = [w for w in rep["warnings"] if not IKTablePicker._is_arm_warning(w, (u, v))]
            rep["arm_px"] = [round(u), round(v)]
        except Exception:
            pass
        rep["calibration"] = he.data.get("fit")
    scene.save(bgr, res, rep, ignore=rep.get("arm_px"))
    print(json.dumps({k: rep[k] for k in ("bricks", "bowls", "obstacles", "in_bowls", "warnings")}, indent=1, default=float))
    if getattr(args, "speak", False):
        from .announce import describe_table, say
        text = describe_table(res["bricks"], len(rep["obstacles"]))
        print("saying:", text)
        say(text, force=True)
        time.sleep(1.0 + 0.08 * len(text))          # let the sentence finish before the process ends
    print("pictures: runs/scene/latest.jpg (annotated), latest_height.jpg (height map)")


def cmd_table_from_map(cfg, args):
    """Build the precise-pick calibration (secondlook/handeye.yaml) from the hand-held points in
    table_map.yaml + the table plane seen by the depth camera. Camera only, the arm does not move."""
    import numpy as np
    from . import scene
    from .handeye import HandEye
    from .kinematics import Arm
    from .table import PoseMap
    pts = PoseMap.load().data["points"]
    bgr, depth, K = scene.capture(cfg)
    if depth is None:
        raise SystemExit("no depth from the top camera")
    plane, inl = scene.fit_plane(depth, K)
    arm = Arm()
    he = HandEye.from_table_map(pts, plane, K, arm)
    he.data["ref"] = {j: float(np.median([p["pick"][j] for p in pts])) for j in
                      ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")}
    res = scene.detect(bgr, depth, K, plane)
    he.data["scene_ref"] = _scene_ref(res, plane)
    he.save()
    fit = he.data["fit"]
    print(f"saved {'secondlook/handeye.yaml'}: {fit['pairs_used']} hand-held points, error mean {fit['mean_mm']} mm, "
          f"max {fit['max_mm']} mm (table plane from {inl:.0%} of the depth image)")
    for p in he.data["pairs"]:
        print(f"   point at pixel ({p['u']:.0f}, {p['v']:.0f}): {p['err_mm']} mm")
    print("Next: ./sl table --auto   (the arm calibrates itself with one brick; aim for a mean under 6 mm)")


def _calib_targets(picker, res, plane, K, n=8):
    """Spots for the self-calibration: on the table, reachable, away from bowls and other objects,
    spread out (farthest-point sampling)."""
    import cv2
    import numpy as np
    from . import scene
    if res.get("table") is None:
        raise SystemExit("could not find the table top in the depth image")
    poly = np.asarray(res["table"], np.float32)
    # Keep-out mask from the real outlines: bowls (+25 px), other objects and the parked arm (+30 px),
    # loose bricks (+35 px).
    keep_out = np.zeros((480, 640), np.uint8)
    for b in res["bowls"]:
        cv2.fillPoly(keep_out, [np.asarray(b["hull"], np.int32)], 255)
    keep_out = cv2.dilate(keep_out, np.ones((51, 51), np.uint8))
    for o, m in [(o, 30) for o in res["obstacles"]] + [(b, 35) for b in res["bricks"]]:
        x, y, w, h = o["box"]
        cv2.rectangle(keep_out, (x - m, y - m), (x + w + m, y + h + m), 255, -1)
    up = picker.he.up_base()
    cands = []
    for v in range(40, 470, 24):
        for u in range(30, 630, 24):
            if cv2.pointPolygonTest(poly, (u, v), True) < 25 or keep_out[v, u]:
                continue
            base = picker.he.to_base(scene.ray_plane(u, v, K, plane, 0.012))
            r = float(np.hypot(base[0], base[1]))
            if not 0.12 <= r <= 0.42:
                continue
            q, info = picker.arm.ik(base, up, picker.ref["wrist_roll"], picker.ref, picker.max_tilt)
            if info["ok"]:
                cands.append((u, v, base))
    if not cands:
        raise SystemExit("no reachable free spot on the table for calibration")
    man = np.array([[p["u"], p["v"]] for p in picker.he.data["pairs"] if "u" in p] or [[c[0], c[1]] for c in cands])
    centre = man.mean(0)
    chosen = [min(cands, key=lambda c: (c[0] - centre[0]) ** 2 + (c[1] - centre[1]) ** 2)]
    while len(chosen) < min(n, len(cands)):
        chosen.append(max(cands, key=lambda c: min(np.linalg.norm(c[2] - d[2]) for d in chosen)))
    return chosen


def cmd_table_auto(cfg, args):
    """Self-calibration: the arm puts ONE brick down at spread-out spots, the depth camera finds
    where it landed (a perfect point pair, no hands on the arm), then it picks the brick up again
    by vision (a real test of the calibration) and moves on."""
    import numpy as np
    from .handeye import HandEye
    from .precise import IKTablePicker
    from .robot import SO101
    if not HandEye.exists():
        raise SystemExit("run `./sl table --from-map` first (gives the starting calibration)")
    cfg["cameras"]["top"]["depth"] = True
    picker = IKTablePicker(cfg)
    picker.learn = False
    he, arm = picker.he, picker.arm
    robot = SO101(cfg, cameras=["top"])
    robot.connect()
    picked = tried = 0
    errors = []
    try:
        robot.hold_here()
        print("The arm moves by itself. Keep hands out of the table area while it moves.")
        _, bricks = picker.scan(robot)
        res, plane, K = picker.scene
        if bricks:
            input(f"{len(bricks)} loose brick(s) on the table. Take them off (bricks IN the bowls may stay), Enter ")
            _, bricks = picker.scan(robot)
            res, plane, K = picker.scene
        he.data["scene_ref"] = _scene_ref(res, plane)
        targets = _calib_targets(picker, res, plane, K, args.points)
        print(f"{len(targets)} calibration spots chosen")
        up = he.up_base()

        def hand_brick():
            robot.go("handoff")
            robot.gripper(open=True)
            while True:
                input("Put ONE brick between the fingers, Enter ")
                robot.gripper(open=False)
                time.sleep(0.4)
                if robot.holding():
                    return
                robot.gripper(open=True)
                print("   the gripper closed on nothing; try again")

        hand_brick()
        for i, (u, v, base) in enumerate(targets, 1):
            q, info = arm.ik(base + up * 0.006, up, picker.ref["wrist_roll"], picker.ref, picker.max_tilt)
            qa, _ = arm.ik(base + up * picker.above_m, up, picker.ref["wrist_roll"], picker.ref, picker.max_tilt, starts=[q])
            if not info["ok"]:
                print(f"[{i}] spot ({u}, {v}) not reachable, skipped")
                continue
            print(f"[{i}/{len(targets)}] putting the brick down at pixel ({u}, {v})")
            robot.go("home")                             # travel high: never swing low over the bowls
            robot.move(qa)
            robot.move(q)
            robot.gripper(open=True)
            robot.move(qa)
            put = arm.tcp(q) - up * 0.006                # tool point at grasp height = where the brick is
            _, seen = picker.scan(robot)
            near = [b for b in seen if np.linalg.norm(np.asarray(b["base"]) - put) < 0.07]
            if not near:
                print("   brick not found near that spot (rolled away? fell over?). Hand it to me again.")
                hand_brick()
                continue
            b = min(near, key=lambda b: np.linalg.norm(np.asarray(b["base"]) - put))
            before = float(np.linalg.norm(np.asarray(b["base"]) - put) * 1000)
            fit = he.add_pair(b["grasp_cam"], put, "drop", u=b["u"], v=b["v"])
            errors.append(before)
            he.save()
            print(f"   seen {before:.0f} mm from where it was put -> calibration now {fit['mean_mm']} mm mean "
                  f"({fit['pairs_used']} pairs: {', '.join(fit['sources'])})")
            # Pick it up again BY VISION with the updated calibration: the real test.
            b["base"] = he.to_base(b["grasp_cam"]).tolist()
            plan = picker.plan(b)
            tried += 1
            got = False
            if plan["ok"]:
                robot.gripper(open=True)
                robot.move(plan["above"])
                robot.move(plan["pick"])
                robot.gripper(open=False)
                time.sleep(0.4)
                robot.move(plan["above"])
                got = robot.holding()
            picked += got
            print(f"   pick by vision: {'OK' if got else 'MISSED'}  ({picked}/{tried})")
            try:
                from . import history
                history.add_pick(run="table_auto", method="ik-calib", u=b["u"], v=b["v"], x=plan["target"][0],
                                 y=plan["target"][1], z=plan["target"][2], outcome="grasped" if got else "missed",
                                 roll=plan["roll"], tilt_deg=plan["info"]["tilt_deg"], ik_err_mm=plan["info"]["pos_err_mm"],
                                 joints=plan["pick"])
            except Exception:
                pass
            if not got:
                robot.gripper(open=True)
                robot.go("home")
                hand_brick()
        fit = he.data.get("fit", {})
        print(f"\nDONE: calibration {fit.get('mean_mm')} mm mean / {fit.get('max_mm')} mm max over "
              f"{fit.get('pairs_used')} pairs; picks by vision during calibration {picked}/{tried}")
        if errors:
            print(f"(before each update, the brick was seen {np.mean(errors):.0f} mm from the prediction on average)")
        print("Try it: take the brick, scatter 3-5 bricks, then   ./sl run --scan --auto --eval --cycles 5")
    finally:
        try:
            robot.go("home")
            robot.gripper(open=True)
        finally:
            robot.disconnect()


def cmd_collect(cfg, args):
    import cv2
    from .inspection import crop
    from .robot import SO101
    out = os.path.join("data", "bricks", args.label)
    os.makedirs(out + "_full", exist_ok=True)
    os.makedirs(out, exist_ok=True)
    cam, roi = cfg["inspection"]["camera"], cfg["inspection"]["roi"]
    robot = SO101(cfg, cameras=[cam])
    robot.connect()
    try:
        for i in range(args.n):
            robot.go("handoff")
            robot.gripper(open=True)
            if input(f"[{i + 1}/{args.n}] put a {args.label} brick in the gripper, Enter (q = quit) ") == "q":
                break
            robot.gripper(open=False)
            robot.go("inspect")
            time.sleep(0.4)
            img = robot.frame(cam)
            stamp = time.strftime("%H%M%S") + f"_{i:03d}"
            cv2.imwrite(os.path.join(out + "_full", f"{stamp}.png"), img)
            cv2.imwrite(os.path.join(out, f"{stamp}.png"), crop(img, roi))
        robot.go("handoff")
        robot.gripper(open=True)
    finally:
        robot.disconnect()
    print("saved to", out, "(crops) and", out + "_full", "(full frames, to re-crop later)")


def cmd_run(cfg, args):
    from .eval_log import log
    from .inspection import Inspector
    from .machine import Cell, Stop
    from .policy import FixedSpotPicker, HandFedPicker, PolicyPicker
    from .robot import SO101
    from .ui import UI

    inspector = Inspector(cfg)
    robot = SO101(cfg)
    if args.handfed:
        picker = HandFedPicker()
    elif args.fixed:
        picker = FixedSpotPicker(robot.poses)
    elif args.scan:
        from .handeye import HandEye
        if HandEye.exists() and (cfg.get("scan") or {}).get("method", "ik") == "ik":
            from . import history
            from .precise import IKTablePicker
            picker = IKTablePicker(cfg, history=history.add_pick)       # depth + kinematics (precise)
            print(f"table scan: precise pick (depth + IK), calibration {picker.he.data.get('fit')}", flush=True)
        else:
            from .table import TableScanPicker
            picker = TableScanPicker(cfg)                                # old pixel -> joints map
            print("table scan: old pixel map (run `./sl table --from-map` for the precise pick)", flush=True)
    else:
        picker = PolicyPicker(cfg)
    ui = UI(cfg, inspector, picker)
    robot.on_tick = ui.check_stop
    cell = Cell(robot, picker, inspector, ui, cfg)
    ai = cfg.get("accessinspect") or {}
    if ai.get("gemini", False):
        from . import assist
        if assist.available():
            cell.advisor = assist.check
            cell.advisor_wait = float(ai.get("wait_s", 3.0))
            print("second opinion: Gemini (AccessInspect) names defects for the voice; PatchCore decides", flush=True)
        else:
            print("second opinion: off (accessinspect/.venv not found)", flush=True)

    def file_photos(truth):
        """Copy the last brick's inspection photos into the labelled set (and out of the other one)."""
        import shutil
        dst = os.path.join("data", "bricks_live", "good" if truth == "good" else "defect")
        other = os.path.join("data", "bricks_live", "defect" if truth == "good" else "good")
        os.makedirs(dst, exist_ok=True)
        n = 0
        for f in cell.last_files:
            src = os.path.join(cell.save_dir, f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst, f))
                n += 1
            if os.path.exists(os.path.join(other, f)):
                os.remove(os.path.join(other, f))
        if last_brick["id"] is not None:                 # keep the history in step with the correction
            try:
                from . import history
                with history.connect() as con:
                    con.execute("UPDATE bricks SET truth = ? WHERE id = ?", (truth, last_brick["id"]))
            except Exception as e:
                print("history label not saved:", e, flush=True)
        return f"{n} photo(s) saved for learning."
    ui.on_label = file_photos
    last_brick = {"id": None}
    mode = "handfed" if args.handfed else "fixed" if args.fixed else "scan" if args.scan else "act"
    run_name = args.name or time.strftime('run_%H%M')
    if hasattr(picker, "run"):
        picker.run = run_name

    def record(**row):
        """Run CSV (README table) + the persistent history database (dashboard History tab)."""
        from .eval_log import FIELDS
        log(run_file, **{k: v for k, v in row.items() if k in FIELDS})
        try:
            from . import history
            last_brick["id"] = history.add_brick(run=run_name, trial=n, mode=args.layout, band=cell.band,
                                                 photos=list(cell.last_files),
                                                 **{k: v for k, v in row.items() if k in history.BRICK_COLS and k != "mode"})
        except Exception as e:
            print("history not saved:", e, flush=True)

    def learn_status(since, last=None):
        if args.learn_every:
            ui.state["learn"] = {"since": since, "every": args.learn_every, "last": last or ui.state.get("learn", {}) and ui.state["learn"].get("last")}
    learn_status(0)
    run_file = os.path.join(cfg["log_dir"], f"{run_name}.csv")
    n = 0
    try:
        robot.connect()
    except (RuntimeError, ConnectionError) as e:
        ui.close()
        msg = str(e).lower()
        if "voltage" in msg:
            raise SystemExit("The arm's servos report an INPUT VOLTAGE error. Check the follower's 5 V adapter is "
                             "plugged in and switched on (and that the plug is firm), then try again.") from e
        if "no status packet" in msg:
            sid = __import__("re").search(r"id_=(\d+)", str(e))
            raise SystemExit(f"Servo {sid[1] if sid else '?'} did not answer. Check the arm's power and the cable to that "
                             "servo (6 = gripper), then try again.") from e
        raise
    try:
        robot.go("home")
        n = 0
        learned = 0
        sort = {"watcher": None, "t0": time.time(), "fails": 0}
        if args.until_clear:
            # Sort the whole table: say what is on it, then pick until nothing is left.
            from .announce import describe_table
            ui.state["until_clear"] = True
            if not hasattr(picker, "scan"):
                from .precise import TableWatcher
                sort["watcher"] = TableWatcher(cfg)
            ui.show(stage="LOOKING")
            if sort["watcher"] is None:
                bricks = picker.scan(robot)[1]
            else:
                bricks = sort["watcher"].look(robot)
                picker.found, picker.colors = len(bricks), [b.get("color") for b in bricks]
            text = describe_table(bricks, getattr(sort["watcher"] or picker, "obstacles", 0))
            ui.event(text, "info")
            ui.say_text(text)
            ui.state["table_start"] = len(bricks)
            sort["t0"] = time.time()
        else:
            ui.say("ready")
        learn_job = {"proc": None, "t0": None, "log": None}

        ui.on_idle = lambda: poll_learning_status()

        def poll_learning_status():
            p = learn_job["proc"]
            if p is not None and p.poll() is None:
                ui.state["learn"] = dict(ui.state.get("learn") or {}, training=int(time.time() - learn_job["t0"]))

        def poll_learning():
            """Swap in new weights between bricks once a background retrain has finished."""
            p = learn_job["proc"]
            if p is None:
                return
            if p.poll() is None:
                ui.state["learn"] = dict(ui.state.get("learn") or {}, training=int(time.time() - learn_job["t0"]))
                return
            learn_job["proc"] = None
            ui.state["learn"] = dict(ui.state.get("learn") or {}, training=None)
            if p.returncode != 0:
                ui.event(f"Learning failed (log {os.path.basename(learn_job['log'])}); keeping the current model", "error")
                return
            import yaml
            with open(args.config) as fh:
                new_cfg = yaml.safe_load(fh)
            cfg["inspection"] = new_cfg["inspection"]
            cell.inspector = Inspector(cfg)
            cell.thr, cell.band = cfg["inspection"]["threshold"], cfg["inspection"]["band"]
            ui.inspector = cell.inspector
            ui.state["thr"], ui.state["band"] = cell.thr, cell.band
            took = int(time.time() - learn_job["t0"])
            ui.event(f"Learned in {took} s: new weights loaded, threshold {cell.thr:.2f}", "ok")
            ui.say_text("New weights loaded.")
            learn_status(0, f"{time.strftime('%H:%M')} threshold {cell.thr:.2f}")
            print(f"reloaded defect model: threshold {cell.thr} band {cell.band} ({took} s)", flush=True)
        def finish_sort():
            from .announce import summary_line
            total = sum(cell.counts.get(k, 0) for k in ("good", "defective"))
            expected = ui.state.get("table_start", 0)
            if total == 0 or total < expected:
                text = (f"Sorting paused. {total} bricks sorted. No more bricks detected, "
                        "but the table is not confirmed clear. Check the camera view and remaining bricks.")
                ui.event(text, "warn")
                ui.show(stage="CHECK TABLE  sorting incomplete")
                ui.say_text(text)
                return
            text = summary_line(cell.counts, time.time() - sort["t0"])
            ui.event(text, "ok")
            ui.show(stage="SORTED")
            ui.say_text(text)
            time.sleep(0.5)

        while args.cycles == 0 or n < args.cycles:
            poll_learning()
            if args.until_clear:
                if sort["watcher"] is not None and n > 0:
                    bricks = sort["watcher"].look(robot)
                    picker.found, picker.colors = len(bricks), [b.get("color") for b in bricks]
                if getattr(picker, "found", 1) == 0:
                    finish_sort()
                    break
                if sort["fails"] >= 3:
                    msg = "I could not pick up the bricks that are left. Move them closer to the robot."
                    ui.event(msg, "warn")
                    ui.say_text(msg)
                    break
            if not args.auto and not args.handfed:
                ui.show(stage="READY  press SPACE to inspect")
                ui.wait_key({ord(" ")})
            last_brick["id"] = None             # G/D during this brick's eval is stored by record()
            res = cell.cycle()
            if args.until_clear:
                sort["fails"] = sort["fails"] + 1 if res is None else 0
            if res is None and cell.empty and args.until_clear:
                finish_sort()
                break
            if res is None and cell.empty:       # table scan found nothing: wait for more bricks
                ui.say("table_clear")
                ui.show(stage="TABLE CLEAR  put bricks down, then press SPACE")
                ui.wait_key({ord(" ")})
                continue
            n += 1
            if res is None:
                cell.last_files = []
                record(pick_ok=0, policy=mode, layout=args.layout)
                continue
            truth = ""
            if args.eval:
                ui.show(stage="EVAL  g = brick was good   d = brick was defective   x = skip")
                k = ui.wait_key({ord("g"), ord("d"), ord("x")})
                truth = {ord("g"): "good", ord("d"): "defective"}.get(k, "")
                if truth:                        # learn from real robot grasps (used by `./sl recalibrate`)
                    file_photos(truth)
                    learned += len(cell.last_files)
            if args.truth and not truth:
                truth = args.truth
                file_photos(truth)
                learned += len(cell.last_files)
                ui.event(f"Auto-labelled {truth.upper()} ({learned}/{args.learn_every or '-'} until retrain). "
                         "Wrong? press the correction button.", "info")
            learn_status(learned)
            if args.learn_every and learned >= args.learn_every and learn_job["proc"] is None:
                learned = 0
                # Retrain in the BACKGROUND at low priority: the run keeps going with the current model.
                import subprocess
                lf = os.path.join(cfg["log_dir"], "logs", time.strftime("%Y%m%d_%H%M%S") + "_recalibrate.log")
                os.makedirs(os.path.dirname(lf), exist_ok=True)
                learn_job.update(proc=subprocess.Popen(["nice", "-n", "10", "tools/recalibrate.sh"],
                                                       stdout=open(lf, "w"), stderr=subprocess.STDOUT),
                                 t0=time.time(), log=lf)
                ui.event("Learning in the background: the run continues with the current model", "info")
                ui.say_text("Learning in the background.")
            record(truth=truth, layout=args.layout, policy=mode, **res)
    except Stop:
        robot.torque(False)
        ui.say("stopped")
        print("STOPPED by operator: torque off")
    finally:
        robot.disconnect()
        ui.close()
        print("log:", run_file)


def cmd_review(cfg, args):
    """Judge saved images with the anomaly model in the dashboard. No robot needed."""
    import cv2
    from .decide import final_bowl, verdict
    from .inspection import Inspector
    from .machine import Stop
    from .ui import UI

    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = sorted(os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.lower().endswith(exts))
    if not files:
        raise SystemExit(f"no images in {args.folder}")
    if args.precropped:
        cfg["inspection"]["roi"] = None
    inspector = Inspector(cfg)
    ui = UI(cfg, inspector)
    thr, band = cfg["inspection"]["threshold"], cfg["inspection"]["band"]
    counts = {"good": 0, "defective": 0}
    try:
        for i, path in enumerate(files, 1):
            ui.show(stage=f"REVIEW {i}/{len(files)}  press SPACE for the next image  ({os.path.basename(path)})")
            ui.wait_key({ord(" ")})
            score, heat, crop = inspector.score(cv2.imread(path))
            v = verdict(score, thr, band)
            counts[final_bowl(v)] += 1
            ui.show(stage=f"REVIEW {i}/{len(files)}", verdict=v, score=score, views=[(crop, heat, score)],
                    counts=dict(counts))
            ui.show(stage="DONE")
            ui.say(v if v != "uncertain" else "defective", count=counts[final_bowl(v)])
            print(f"{os.path.basename(path)}: {score:.3f} {v}")
        ui.show(stage="REVIEW finished  press SPACE to close")
        ui.wait_key({ord(" ")})
    except Stop:
        pass
    finally:
        ui.close()


def main():
    p = argparse.ArgumentParser(prog="secondlook.cell")
    p.add_argument("--config", default="secondlook/config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--policy", action="store_true", help="also load the policy and run one step")
    sub.add_parser("teach")
    c = sub.add_parser("table", help="calibrate the top-camera table scan (empty-table photo + 6 points)")
    c.add_argument("--add", action="store_true", help="add points to the saved map (keeps the empty-table photo)")
    c.add_argument("--drop", type=int, nargs="+", help="remove these point numbers from the saved map and refit")
    c.add_argument("--background", action="store_true", help="retake only the empty-table photo (keeps the calibration)")
    c.add_argument("--from-map", action="store_true", help="precise pick: build handeye.yaml from table_map.yaml + depth (no motion)")
    c.add_argument("--auto", action="store_true", help="precise pick: the arm calibrates itself with one brick")
    c.add_argument("--points", type=int, default=8, help="spots for --auto (default 8)")
    c = sub.add_parser("scene", help="scene check with the top camera only (no arm): bricks, bowls, obstacles")
    c.add_argument("--speak", action="store_true", help="say what is on the table")
    c = sub.add_parser("collect")
    c.add_argument("--label", choices=["good", "defect"], required=True)
    c.add_argument("--n", type=int, default=50)
    c = sub.add_parser("run")
    c.add_argument("--handfed", action="store_true", help="operator feeds bricks (no policy needed)")
    c.add_argument("--fixed", action="store_true", help="scripted pick from one marked spot (backup, no policy)")
    c.add_argument("--scan", action="store_true", help="top camera finds bricks on the table and picks them (no policy)")
    c.add_argument("--truth", choices=["good", "defective"], help="every brick on the table is this (auto-label, no keys)")
    c.add_argument("--learn-every", type=int, default=0, help="retrain the defect model after this many new labelled photos")
    c.add_argument("--auto", action="store_true", help="no button press between bricks")
    c.add_argument("--until-clear", action="store_true", help="sort every brick on the table, then stop (says what it sees)")
    c.add_argument("--eval", action="store_true", help="ask for the true label after each brick")
    c.add_argument("--cycles", type=int, default=0, help="0 = until STOP")
    c.add_argument("--layout", default="normal", help="normal / shifted / clutter (for the results table)")
    c.add_argument("--name", help="log file name")
    c = sub.add_parser("review", help="judge saved images in the dashboard, no robot needed")
    c.add_argument("folder")
    c.add_argument("--precropped", action="store_true", help="images are already cropped (data/clean/...)")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    {"check": cmd_check, "teach": cmd_teach, "table": cmd_table, "collect": cmd_collect, "run": cmd_run,
     "review": cmd_review, "scene": cmd_scene}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
