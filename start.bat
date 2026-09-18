@echo off
title FraudGuard AI - Enterprise Fraud Detection Platform
color 0A

echo.
echo  ╔════════════════════════════════════════════════════╗
echo  ║    🛡️  FraudGuard AI — Startup Launcher           ║
echo  ║    Enterprise Real-Time Fraud Detection System     ║
echo  ╚════════════════════════════════════════════════════╝
echo.

:: Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    echo [Setup] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [Setup] No venv found — using system Python
    echo [Setup] Tip: Run setup.bat first for a clean environment
)

:: Kill any existing processes on port 5000
echo [Setup] Clearing port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo  Starting FraudGuard AI Server...
echo  Dashboard will be available at: http://localhost:5000
echo.
echo  ┌─ TIPS ────────────────────────────────────────────┐
echo  │  Open http://localhost:5000 in your browser       │
echo  │  Transactions simulate automatically on start!      │
echo  │  Press CTRL+C to stop the server                  │
echo  └────────────────────────────────────────────────────┘
echo.

:: Launch the server
python app.py

pause
