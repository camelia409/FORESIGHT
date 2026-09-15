#!/usr/bin/env bash
# ==============================================================================
# PROJECT FORESIGHT — Launch Production FastAPI Engine (Linux/macOS)
# ==============================================================================
set -euo pipefail
echo "[FORESIGHT] Starting Production FastAPI Engine on http://0.0.0.0:8000 ..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
