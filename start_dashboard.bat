@echo off
rem 在本機啟動儀表板（永豐即時報價要在本機才連得上）。雙擊本檔即可，瀏覽器會自動打開 http://localhost:8501
chcp 65001 >nul
cd /d "%~dp0"
python -m streamlit run app.py
pause
