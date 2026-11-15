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

echo "[1/7] App libraries into the Studio venv"
uv pip install --python "$STUDIO_PY" -r requirements.txt

echo "[2/7] Anomaly training environment (anomalib-env)"
[ -x anomalib-env/bin/python ] || uv venv --python 3.12 anomalib-env
uv pip install --python anomalib-env/bin/python "anomalib[openvino,xpu]"

echo "[3/7] Offline voice: Piper neural TTS (tts-env) + espeak-ng fallback (asks for your password)"
[ -x tts-env/bin/piper ] || uv venv --python 3.12 tts-env
uv pip install --python tts-env/bin/python "piper-tts==1.8.0"
command -v espeak-ng >/dev/null || sudo apt-get install -y espeak-ng

echo "[4/7] Serial port access"
if ! id -nG | grep -qw dialout; then
  sudo usermod -aG dialout "$USER"
  echo "      Added you to the dialout group. Log out and back in once before using the arm."
fi

echo "[5/7] Trained models from the GitHub release (defect model + ACT v3)"
./tools/fetch_models.sh || echo "      Skipped: run ./tools/fetch_models.sh after 'gh auth login'."

echo "[6/7] Optional: AccessInspect voice + Gemini helper (own venv)"
[ -x accessinspect/.venv/bin/python ] || uv venv --python 3.12 accessinspect/.venv
uv pip install --python accessinspect/.venv/bin/python -r accessinspect/requirements.txt

echo "[7/7] Self-test"
chmod +x sl
"$STUDIO_PY" -m unittest discover tests

echo
echo "Done. Next step:  ./sl check"
