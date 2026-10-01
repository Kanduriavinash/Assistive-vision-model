@echo off
echo =======================================================
echo   STARTING ASSISTIVE VISION PIPELINE (IMAGE BATCH)
echo =======================================================
echo Activating virtual environment...
call venv\Scripts\activate.bat
python main.py
pause
