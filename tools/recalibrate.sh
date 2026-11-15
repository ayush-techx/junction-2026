#!/usr/bin/env bash
# Retrain the defect model on the original photos + photos labelled during real runs (g/d keys),
# re-score everything and write the new model, threshold and band into config.yaml (backup kept).
set -euo pipefail
cd "$(dirname "$0")/.."
STUDIO_PY="${STUDIO_PY:-$HOME/physical-ai-studio/application/backend/.venv/bin/python}"
T=$(date +%H%M%S)
M=data/bricks_merged
rm -rf "$M"; mkdir -p "$M/good" "$M/defect"
cp data/bricks/good/*.png "$M/good/" 2>/dev/null || true
cp data/bricks/defect/*.png "$M/defect/" 2>/dev/null || true
cp data/bricks_live/good/*.png "$M/good/" 2>/dev/null || true
cp data/bricks_live/defect/*.png "$M/defect/" 2>/dev/null || true
echo "training on: good $(ls $M/good | wc -l) (live $(ls data/bricks_live/good 2>/dev/null | wc -l)), defect $(ls $M/defect | wc -l) (live $(ls data/bricks_live/defect 2>/dev/null | wc -l))"
[ -d exports ] && mv exports "exports_before_recal_$T"
anomalib-env/bin/python tools/train_anomaly.py --root "$M" --model patchcore 2>&1 | grep -E "image_AUROC|image_F1|done:" || true
MODEL=exports/patchcore/fp16/weights/openvino/model.xml
[ -f "$MODEL" ] || { echo "training failed; previous model kept in exports_before_recal_$T"; mv "exports_before_recal_$T" exports; exit 1; }
OUT=$("$STUDIO_PY" tools/evaluate_anomaly.py --model "$MODEL" --root "$M" --device GPU --overlays 2>&1 | grep -E "good:|defect:|suggested|caught|latency")
echo "$OUT"
THR=$(echo "$OUT" | sed -n 's/.*suggested threshold \([0-9.]*\).*/\1/p'); BAND=$(echo "$OUT" | sed -n 's/.*band \([0-9.]*\).*/\1/p')
THR=${THR:-0.5}; BAND=$(python3 -c "print(max(0.03, min(0.08, float('${BAND:-0.05}'))))")
cp secondlook/config.yaml "secondlook/config.yaml.bak_recal_$T"
python3 - "$MODEL" "$THR" "$BAND" <<'PY'
import re,sys
p='secondlook/config.yaml'; s=open(p).read()
s=re.sub(r"(\ninspection:(?:\n[^\n]*)*?\n  model: )\S+", r"\g<1>"+sys.argv[1], s, count=1)
s=re.sub(r"(\n  threshold: )[0-9.]+", r"\g<1>"+sys.argv[2], s, count=1)
s=re.sub(r"(\n  band: )[0-9.]+", r"\g<1>"+sys.argv[3], s, count=1)
open(p,'w').write(s)
PY
echo "config updated: threshold $THR band $BAND (backup secondlook/config.yaml.bak_recal_$T)"
