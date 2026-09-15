@echo off
REM ==============================================================================
REM PROJECT FORESIGHT — Launch Production FastAPI Engine (Windows)
REM ==============================================================================
echo [FORESIGHT] Starting Production FastAPI Engine on http://localhost:8000 ...
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
