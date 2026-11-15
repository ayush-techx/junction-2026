"""Log eval trials to CSV and turn them into the README results table.

Log from code:
    from eval_log import log
    log("runs/v2_normal.csv", truth="defective", final_bowl="defective", pick_ok=1,
        regrasps=0, cycle_s=13.9, score_v1=0.21, score_v2=0.83, thr=0.5, layout="normal")

Log by hand at the table (one line per brick):
    python -m secondlook.eval_log add runs/v2_normal.csv truth=good final_bowl=good pick_ok=1 cycle_s=14.1

Summarise one or more runs as markdown:
    python -m secondlook.eval_log summarize runs/*.csv
"""
import csv
import os
import statistics
import sys
import time

FIELDS = ["ts", "run", "trial", "layout", "truth", "verdict", "final_bowl", "pick_ok",
          "regrasps", "cycle_s", "score_v1", "score_v2", "thr", "policy", "notes"]


def log(path, **row):
    """Append one trial. Unknown keys go into notes; trial number auto-increments."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    if new:
        trial = 1
    else:
        with open(path) as fh:
            trial = sum(1 for _ in fh)
    extra = {k: v for k, v in row.items() if k not in FIELDS}
    row = {k: v for k, v in row.items() if k in FIELDS}
    if extra:
        row["notes"] = (row.get("notes", "") + " " + " ".join(f"{k}={v}" for k, v in extra.items())).strip()
    row.setdefault("ts", time.strftime("%H:%M:%S"))
    row.setdefault("run", os.path.splitext(os.path.basename(path))[0])
    row.setdefault("trial", trial)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(rows):
    n = len(rows)
    picked = [r for r in rows if r.get("pick_ok") in ("1", "True", "true")]
    labelled = [r for r in picked if r.get("truth")]  # accuracy only where ground truth was recorded
    sorted_ok = [r for r in labelled if r.get("final_bowl") == r["truth"]]
    conf = {(t, p): 0 for t in ("good", "defective") for p in ("good", "defective")}
    for r in picked:
        if (r.get("truth"), r.get("final_bowl")) in conf:
            conf[(r["truth"], r["final_bowl"])] += 1
    defects = conf[("defective", "good")] + conf[("defective", "defective")]
    goods = conf[("good", "good")] + conf[("good", "defective")]
    cyc = sorted(c for c in (_f(r.get("cycle_s")) for r in picked) if c is not None)
    second = 0
    for r in rows:
        s1, s2, thr = _f(r.get("score_v1")), _f(r.get("score_v2")), _f(r.get("thr"))
        if None not in (s1, s2, thr) and s1 < thr <= s2:
            second += 1
    pct = lambda a, b: f"{100 * a / b:.0f}% ({a}/{b})" if b else "n/a"
    out = {
        "trials": n,
        "pick success": pct(len(picked), n),
        "sort accuracy (of picked)": pct(len(sorted_ok), len(labelled)),
        "defects caught (recall)": pct(conf[("defective", "defective")], defects),
        "good bricks wrongly rejected": pct(conf[("good", "defective")], goods),
        "defects missed": conf[("defective", "good")],
        "re-grasps": sum(int(_f(r.get("regrasps")) or 0) for r in rows),
        "caught only by second view": second,
    }
    if cyc:
        out["cycle time median / p90"] = f"{statistics.median(cyc):.1f} s / {cyc[min(len(cyc) - 1, -(-9 * len(cyc) // 10) - 1)]:.1f} s"
    return out


def main(argv):
    if len(argv) >= 2 and argv[0] == "add":
        log(argv[1], **dict(a.split("=", 1) for a in argv[2:]))
        print("logged")
        return
    if len(argv) >= 2 and argv[0] == "summarize":
        groups = {}
        for p in argv[1:]:
            with open(p) as f:
                for r in csv.DictReader(f):
                    key = f"{r.get('run') or os.path.basename(p)} / {r.get('layout') or '-'}"
                    groups.setdefault(key, []).append(r)
        summaries = {k: summarize(v) for k, v in groups.items()}
        metrics = list(next(iter(summaries.values())).keys())
        for s in summaries.values():
            metrics += [m for m in s if m not in metrics]
        print("| metric | " + " | ".join(summaries) + " |")
        print("|---|" + "---|" * len(summaries))
        for m in metrics:
            print(f"| {m} | " + " | ".join(str(s.get(m, "")) for s in summaries.values()) + " |")
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
