#!/usr/bin/env bash
# Inferensi dengan konteks di 2 GPU: satu worker per GPU, panggilan dibagi berdasarkan hash.
#   bash scripts/run_infer_2gpu.sh audio    --model outputs/whisper-telemarketing-ct2 --audio-dir raw/audio --out-dir outputs/transcripts
#   bash scripts/run_infer_2gpu.sh manifest --model outputs/whisper-telemarketing-ct2 --manifest dataset/test.jsonl --out outputs/preds/test-C4.jsonl
# Mode manifest: tiap worker menulis <out>.w0 / <out>.w1, lalu digabung ke <out>.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=$ROOT/.envs/whisperx-infer/bin/python
mode=${1:?audio|manifest}; shift
args=("$@")
out=""
for ((i = 0; i < ${#args[@]}; i++)); do [[ ${args[i]} == --out ]] && out=${args[i+1]}; done

pids=()
for w in 0 1; do
  extra=()
  [[ $mode == manifest ]] && extra=(--out "$out.w$w")
  CUDA_VISIBLE_DEVICES=$w $PY src/infer_context.py "$mode" "${args[@]}" "${extra[@]}" \
    --worker-index $w --num-workers 2 > "logs/infer-$mode-w$w.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
if [[ $mode == manifest ]]; then cat "$out.w0" "$out.w1" > "$out" && rm "$out.w0" "$out.w1"; fi
echo "selesai"
