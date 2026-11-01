#!/usr/bin/env bash
# Jalankan semua unit test (env whisper-train).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" && "$ROOT/.envs/whisper-train/bin/python" -m pytest -q -rs "$@"
