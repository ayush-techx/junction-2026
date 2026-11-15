#!/usr/bin/env bash
# Downloads the trained models from the GitHub release and unpacks them where config.yaml expects them.
#
#   ./tools/fetch_models.sh            defect models + ACT v3 (what the default config uses, ~360 MB)
#   ./tools/fetch_models.sh --all      also ACT 20ep and SmolVLA 20k (~1.1 GB)
#
# Works without a login (public release). Uses the GitHub CLI or GITHUB_TOKEN when available.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${SECONDLOOK_REPO:-ayush-techx/junction-2026}"
TAG="${SECONDLOOK_MODELS_TAG:-models-v1}"
DL="${TMPDIR:-/tmp}/secondlook-models"
mkdir -p "$DL" models exports

# asset name -> folder it unpacks into, and the folder that must exist afterwards
declare -A DEST=(
  [patchcore-fp16-openvino.tar.gz]=exports       [patchcore-live-fp16-openvino.tar.gz]=exports
  [act-v3-openvino.tar.gz]=models                [act-20ep-openvino.tar.gz]=models
  [smolvla-20k-openvino.tar.gz]=models
)
declare -A MADE=(
  [patchcore-fp16-openvino.tar.gz]=exports/patchcore/fp16/weights/openvino
  [patchcore-live-fp16-openvino.tar.gz]=exports/patchcore_live/fp16/weights/openvino
  [act-v3-openvino.tar.gz]=models/act_v3         [act-20ep-openvino.tar.gz]=models/act_20ep
  [smolvla-20k-openvino.tar.gz]=models/smolvla_20k
)
WANT=(patchcore-fp16-openvino.tar.gz patchcore-live-fp16-openvino.tar.gz act-v3-openvino.tar.gz)
[ "${1:-}" = "--all" ] && WANT+=(act-20ep-openvino.tar.gz smolvla-20k-openvino.tar.gz)

fetch() {  # fetch <asset>
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$1" --dir "$DL" --clobber
  elif [ -n "${GITHUB_TOKEN:-}" ]; then
    local id
    id=$(curl -fsSL -H "Authorization: Bearer $GITHUB_TOKEN" \
      "https://api.github.com/repos/$REPO/releases/tags/$TAG" |
      python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='$1'))")
    curl -fL -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/octet-stream" \
      -o "$DL/$1" "https://api.github.com/repos/$REPO/releases/assets/$id"
  else
    curl -fL -o "$DL/$1" "https://github.com/$REPO/releases/download/$TAG/$1"
  fi
}

fetch SHA256SUMS
for a in "${WANT[@]}"; do
  if [ -e "${MADE[$a]}" ]; then
    echo "ok    ${MADE[$a]} (already there)"
    continue
  fi
  echo "get   $a"
  fetch "$a"
  (cd "$DL" && grep " $a\$" SHA256SUMS | sha256sum -c --quiet -)
  tar -xzf "$DL/$a" -C "${DEST[$a]}"
  echo "ok    ${MADE[$a]}"
done
echo
echo "Models ready. Next: ./sl check"
