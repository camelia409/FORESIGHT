@echo off
REM ==============================================================================
REM PROJECT FORESIGHT — Execute Production Pipeline (Windows)
REM ==============================================================================
echo [FORESIGHT] Running Production Pipeline Orchestrator DAG ...
python -m src.production_pipeline %*
