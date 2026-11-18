@echo off
cd /d "%~dp0"
echo Starting ScriptRAG on http://localhost:8501 ...
python -m streamlit run app.py
pause

