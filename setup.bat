@echo off
echo ===============================
echo  OCR INSPECTION - FIRST TIME SETUP
echo ===============================
echo.

cd /d D:\ultralytics-main

echo [1/5] Removing old venv (if any)...
if exist venv rmdir /s /q venv

echo [2/5] Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.10 is installed and added to PATH.
    pause
    exit /b 1
)

echo [3/5] Activating virtual environment...
call venv\Scripts\activate

echo [4/5] Upgrading pip...
pip install --upgrade pip

echo [5/5] Installing required packages (this may take a few minutes)...
pip install streamlit ultralytics pillow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo ===============================
echo  SETUP COMPLETE
echo ===============================
echo Next time, just run: dist\run_app.exe
echo (No need to run this setup again on this PC.)
echo.
pause
