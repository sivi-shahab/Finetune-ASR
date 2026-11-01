#!/usr/bin/env bash
# Konversi model HF hasil training ke CTranslate2 float16 untuk WhisperX.
#   bash scripts/convert_ct2.sh outputs/run-001/final outputs/whisper-telemarketing-ct2
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
src=${1:?direktori model HF (berisi tokenizer.json & preprocessor_config.json)}
dst=${2:?direktori output CT2}
for f in tokenizer.json preprocessor_config.json; do
  [[ -f $src/$f ]] || { echo "$src/$f tidak ada"; exit 1; }
done
"$ROOT/.envs/whisperx-infer/bin/ct2-transformers-converter" --model "$src" --output_dir "$dst" \
  --copy_files tokenizer.json preprocessor_config.json --quantization float16
echo "OK: $dst"
