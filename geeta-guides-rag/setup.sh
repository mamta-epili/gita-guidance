#!/usr/bin/env bash
# setup.sh — create .venv and install dependencies.
#   ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> using $($PY --version)"

[ -d .venv ] || { echo "==> creating .venv"; "$PY" -m venv .venv; }

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip --quiet
echo "==> installing torch, fastapi, uvicorn (a few minutes)"
python -m pip install -r requirements.txt

echo
echo "==> locating the trained checkpoint"
python - <<'EOF'
import os
p = os.path.join(os.path.dirname(os.getcwd()), "geeta-guides", "checkpoints", "gita_gpt.pt")
if os.path.exists(p):
    print(f"    found: {p} ({os.path.getsize(p)/1e6:.1f} MB)")
else:
    print(f"    NOT FOUND at {p}")
    print("    Train one first:  cd ../geeta-guides && make train")
    print("    Or set CHARGPT_CKPT to a .pt file.")
EOF

echo
echo "Done.  Start it with:  make run"
