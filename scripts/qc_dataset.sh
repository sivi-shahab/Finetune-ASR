#!/usr/bin/env bash
# Alur dataset dari telemarketing-qc-dashboard (docs/SPEC.md bagian 2a).
#
#   bash scripts/qc_dataset.sh export [--scope qc_all|qc_reviewed] [--campaign Cashline] [--since 2026-09-01]
#        → raw/qc_export.jsonl, raw/transcripts/, raw/meta.jsonl, raw/crm.jsonl, raw/audio_list.csv
#          (query READ ONLY di dalam container API; TIDAK mengambil audio)
#   (ambil sendiri audio yang dipilih dari raw/audio_list.csv ke satu folder)
#   bash scripts/qc_dataset.sh import-audio /path/folder_audio      → raw/audio/<ticket_id>.flac
#   bash scripts/qc_dataset.sh dry-run                              → outputs/dryrun/stats.json (tanpa audio)
#   bash scripts/qc_dataset.sh build [--pii-policy review]          → dataset/ + review_queue.jsonl
#   bash scripts/qc_dataset.sh review <nama> [queue|spotcheck]      → web UI 127.0.0.1:8765
#   bash scripts/qc_dataset.sh check                                → READY / NOT READY
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=$ROOT/.envs/whisper-train/bin/python
CONTAINER=${QC_API_CONTAINER:-telemarketing-qc-api-api-1}
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
umask 077   # semua file turunan berisi PII: hanya pemilik
cmd=${1:?export|import-audio|dry-run|build|review|check}; shift || true
mkdir -p raw logs

case $cmd in
  export)
    docker exec -i "$CONTAINER" python - "$@" < src/qc_export_db.py > raw/qc_export.jsonl 2> logs/qc_export_db.log
    tail -1 logs/qc_export_db.log
    $PY src/qc_export.py convert --input raw/qc_export.jsonl --raw-dir raw ;;
  import-audio)
    $PY src/qc_export.py import-audio --src "${1:?folder audio}" --raw-dir raw ;;
  dry-run)
    $PY src/build_dataset.py --raw-dir raw --out-dir outputs/dryrun --dry-run --workers 16 "$@"
    $PY -c "import json;s=json.load(open('outputs/dryrun/stats.json'));print(json.dumps({k:s[k] for k in ('splits','rejected','review_queue','review_queue_hours','review_queue_by_split')},indent=1))" ;;
  build)
    $PY src/build_dataset.py --raw-dir raw --out-dir dataset --workers 16 "$@" ;;
  review)
    $PY src/review_server.py --dataset dataset --reviewer "${1:?nama reviewer}" --mode "${2:-queue}" ;;
  check)
    $PY src/check_ready.py --dataset dataset "$@" ;;
  *) echo "perintah tidak dikenal: $cmd"; exit 1 ;;
esac
