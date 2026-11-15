#!/usr/bin/env bash
# Installs everything SecondLook needs on the Intel laptop. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

STUDIO_DIR="${STUDIO_DIR:-$HOME/physical-ai-studio/application/backend}"
STUDIO_PY="$STUDIO_DIR/.venv/bin/python"

if [ ! -x "$STUDIO_PY" ]; then
  echo "Physical AI Studio venv not found at $STUDIO_PY"
  echo "Set STUDIO_DIR to the backend folder, for example: STUDIO_DIR=/path/to/backend ./setup.sh"
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh   then open a new terminal."
  exit 1
fi

echo "[1/5] App libraries into the Studio venv"
uv pip install --python "$STUDIO_PY" -r requirements.txt

echo "[2/5] Anomaly training environment (anomalib-env)"
[ -x anomalib-env/bin/python ] || uv venv --python 3.12 anomalib-env
uv pip install --python anomalib-env/bin/python "anomalib[openvino,xpu]"

echo "[3/5] Offline voice (asks for your password)"
command -v espeak-ng >/dev/null || sudo apt-get install -y espeak-ng

echo "[4/5] Serial port access"
if ! id -nG | grep -qw dialout; then
  sudo usermod -aG dialout "$USER"
  echo "      Added you to the dialout group. Log out and back in once before using the arm."
fi

echo "[5/5] Self-test"
chmod +x sl
"$STUDIO_PY" -m unittest discover tests

echo
echo "Done. Next step:  ./sl check"
