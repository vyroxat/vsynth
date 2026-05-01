@echo off
title VSynth — Voice to Klatt Synth
color 0A
echo.
echo  ██╗   ██╗███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗
echo  ██║   ██║██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║
echo  ██║   ██║███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║
echo  ╚██╗ ██╔╝╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║
echo   ╚████╔╝ ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║
echo    ╚═══╝  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝
echo.
echo  Voice ^> Whisper ^> ARPABET ^> klattsch formant synth
echo  --------------------------------------------------------

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

:: Check Node.js / npx  (needed for klattsch CLI)
npx --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js / npx not found. Install from https://nodejs.org
    pause & exit /b 1
)

:: Install Python deps
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)

:: Download NLTK cmudict (idempotent)
echo [2/3] Ensuring NLTK cmudict is downloaded...
python -c "import nltk; nltk.download('cmudict', quiet=True)"

:: Launch app
echo [3/3] Launching VSynth...
echo.
python main.py

pause
