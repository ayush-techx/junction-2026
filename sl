#!/usr/bin/env bash
# SecondLook command runner. Picks the right Python environment for each job.
#   ./sl help
set -euo pipefail
cd "$(dirname "$0")"

STUDIO_PY="${STUDIO_PY:-$HOME/physical-ai-studio/application/backend/.venv/bin/python}"
ANOMALIB_PY="${ANOMALIB_PY:-$PWD/anomalib-env/bin/python}"

cmd="${1:-help}"
[ $# -gt 0 ] && shift

case "$cmd" in
  check|teach|table|collect|run|review|scene)
    exec "$STUDIO_PY" -m secondlook.cell "$cmd" "$@" ;;
  prepare)  exec "$STUDIO_PY" tools/prepare_dataset.py "$@" ;;
  train)    exec "$ANOMALIB_PY" tools/train_anomaly.py "$@" ;;
  evaluate) exec "$STUDIO_PY" tools/evaluate_anomaly.py "$@" ;;
  bench)    exec "$STUDIO_PY" tools/bench.py "$@" ;;
  results)
    [ $# -eq 0 ] && set -- runs/*.csv
    exec "$STUDIO_PY" -m secondlook.eval_log summarize "$@" ;;
  recalibrate) exec tools/recalibrate.sh ;;
  voice)    exec "$STUDIO_PY" -m secondlook.announce ;;
  test)     exec "$STUDIO_PY" -m unittest discover tests -v ;;
  *)
    cat <<'EOF'
SecondLook

  ./sl check [--policy]              robot, cameras, devices, models; saves camera snapshots in runs/
  ./sl teach                         move the arm by hand and save the named poses
  ./sl collect --label good --n 50   photos for the anomaly model (also --label defect)
  ./sl train --root data/bricks      train the anomaly model (anomalib-env)
  ./sl evaluate --model <model.xml> --root data/bricks --device NPU --overlays
  ./sl review data/bricks/defect --precropped   judge saved photos in the dashboard, no robot
  ./sl scene                         scene check, top camera only (no arm): bricks, bowls, obstacles, height map
  ./sl table --from-map              precise pick: starting calibration from the old points (no arm motion)
  ./sl table --auto                  precise pick: the arm calibrates itself with ONE brick (~5 min)
  ./sl table                         old pixel-map calibration (only used if scan.method: map)
  ./sl run --scan --auto             demo: top camera + depth find bricks on the table, picks them all
  ./sl run --handfed                 demo: you hand bricks to the gripper
  ./sl run                           demo: learned pick (needs policy.path)
  ./sl run --eval --cycles 20 --layout normal --name v1_normal
  ./sl results                       results table from runs/*.csv
  ./sl recalibrate                   retrain the defect model with photos you labelled (g/d) during runs
  ./sl bench --models <a.xml> <b.xml> --devices CPU GPU NPU
  ./sl prepare --src <frames> --out <folder> --roi X Y W H
  ./sl voice                         test the speaker
  ./sl test                          run the test suite

Full guide: docs/SETUP.md   Models: ./tools/fetch_models.sh
EOF
    ;;
esac
