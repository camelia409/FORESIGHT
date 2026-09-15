#!/usr/bin/env bash
# ==============================================================================
# PROJECT FORESIGHT — Execute Production Pipeline (Linux/macOS)
# ==============================================================================
set -euo pipefail
echo "[FORESIGHT] Running Production Pipeline Orchestrator DAG ..."
python -m src.production_pipeline "$@"
