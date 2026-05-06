#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --python 3.11 uvicorn asr_router.server:app --host "${ASR_HOST:-0.0.0.0}" --port "${ASR_PORT:-18081}" --reload
