#!/usr/bin/env bash
# Training DDP di 2 GPU. Jalankan di dalam tmux/screen.
#   bash scripts/run_train.sh smoke        # 1 GPU, 200 sampel, 50 step
#   bash scripts/run_train.sh smoke-ddp    # 2 GPU, 200 sampel, 50 step
#   bash scripts/run_train.sh full run-001 # training penuh
#   bash scripts/run_train.sh resume run-001
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export OMP_NUM_THREADS=4
PY=$ROOT/.envs/whisper-train/bin/python
TORCHRUN=$ROOT/.envs/whisper-train/bin/torchrun
mode=${1:?smoke|smoke-ddp|full|resume}; run=${2:-run-001}
mkdir -p outputs logs

case $mode in
  smoke)     CUDA_VISIBLE_DEVICES=0 $PY src/train.py --max_train_samples 200 --max_eval_samples 50 \
               --max_steps 50 --eval_steps 25 --warmup_steps 5 --output_dir outputs/smoke 2>&1 | tee logs/smoke.log ;;
  smoke-ddp) $TORCHRUN --nproc_per_node=2 src/train.py --max_train_samples 200 --max_eval_samples 50 \
               --max_steps 50 --eval_steps 25 --warmup_steps 5 --output_dir outputs/smoke-ddp 2>&1 | tee logs/smoke-ddp.log ;;
  full)      $TORCHRUN --nproc_per_node=2 src/train.py --output_dir "outputs/$run" "${@:3}" 2>&1 | tee "logs/$run.log" ;;
  resume)    $TORCHRUN --nproc_per_node=2 src/train.py --output_dir "outputs/$run" \
               --resume_from_checkpoint true "${@:3}" 2>&1 | tee -a "logs/$run.log" ;;
  *) echo "mode tidak dikenal: $mode"; exit 1 ;;
esac
