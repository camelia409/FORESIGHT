@echo off
REM ==============================================================================
REM PROJECT FORESIGHT — Launch Production Streamlit Dashboard (Windows)
REM ==============================================================================
echo [FORESIGHT] Starting Production Executive Dashboard on http://localhost:8501 ...
streamlit run app/streamlit_app.py --server.port 8501
