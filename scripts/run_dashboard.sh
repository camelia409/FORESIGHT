#!/usr/bin/env bash
# ==============================================================================
# PROJECT FORESIGHT — Launch Production Streamlit Dashboard (Linux/macOS)
# ==============================================================================
set -euo pipefail
echo "[FORESIGHT] Starting Production Executive Dashboard on http://0.0.0.0:8501 ..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
