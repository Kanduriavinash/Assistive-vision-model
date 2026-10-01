@echo off
echo =======================================================
echo   STARTING MULTIMODAL ASSISTIVE VISION (LIVE WEBCAM)
echo =======================================================
echo Activating virtual environment...
call venv\Scripts\activate.bat
python live_cam.py
pause
