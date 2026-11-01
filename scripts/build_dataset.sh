#!/usr/bin/env bash
# Bangun dataset dari raw/ ke dataset/. Jalankan ulang setelah ada review baru.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
"$ROOT/.envs/whisper-train/bin/python" src/build_dataset.py --raw-dir raw --out-dir dataset \
  --tokenizer openai/whisper-large-v3 --workers 16 "$@"
