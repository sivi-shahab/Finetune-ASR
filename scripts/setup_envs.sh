#!/usr/bin/env bash
# Membuat env Python 3.11 di /data (partisi /home hanya ±9 GB).
# Pemakaian: bash scripts/setup_envs.sh [train|infer|all]   (jalankan satu per satu, jangan paralel:
# conda memakai cache paket yang sama dan bisa gagal bila dua proses mengunduh bersamaan)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PIP_CACHE_DIR=$ROOT/.cache/pip
what=${1:-all}

if [[ $what == train || $what == all ]]; then
  [[ -x $ROOT/.envs/whisper-train/bin/python ]] || conda create -y -p "$ROOT/.envs/whisper-train" python=3.11
  P=$ROOT/.envs/whisper-train/bin/pip
  $P install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
  $P install -r "$ROOT/requirements-train.txt"
  "$ROOT/.envs/whisper-train/bin/python" -c "import torch,transformers;print('train OK', torch.__version__, transformers.__version__, torch.cuda.device_count(), 'GPU')"
fi

if [[ $what == infer || $what == all ]]; then
  [[ -x $ROOT/.envs/whisperx-infer/bin/python ]] || conda create -y -p "$ROOT/.envs/whisperx-infer" python=3.11
  P=$ROOT/.envs/whisperx-infer/bin/pip
  $P install -r "$ROOT/requirements-infer.txt"
  "$ROOT/.envs/whisperx-infer/bin/python" -c "import whisperx,faster_whisper,ctranslate2,torch;print('infer OK', faster_whisper.__version__, ctranslate2.__version__, torch.__version__, ctranslate2.get_cuda_device_count(), 'GPU')"
fi
