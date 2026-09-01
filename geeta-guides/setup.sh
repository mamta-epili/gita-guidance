#!/usr/bin/env bash
# setup.sh — create the virtualenv and install dependencies.
#
# Run once, from this directory:   ./setup.sh
# Then activate it in every new shell:   source .venv/bin/activate
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> using $($PY --version) at $(command -v "$PY")"

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PY" -m venv .venv
else
  echo "==> .venv already exists, reusing it"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> upgrading pip"
python -m pip install --upgrade pip --quiet

echo "==> installing torch + numpy (this can take a few minutes)"
python -m pip install torch numpy

# For Phase 2 retrieval. Skip with SKIP_RETRIEVAL=1 if you only want the
# char-GPT exercise — this pulls in transformers and is a larger download.
if [ -z "${SKIP_RETRIEVAL:-}" ]; then
  echo "==> installing sentence-transformers (for retrieval)"
  python -m pip install sentence-transformers
fi

echo
echo "==> preparing the dataset"
python prepare_data.py | tail -n 20

echo
echo "==> checking your hardware"
python device_check.py

echo
echo "Done. Activate the env with:   source .venv/bin/activate"
