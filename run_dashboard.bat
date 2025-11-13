@echo off
REM Revnet Dashboard Launcher (Windows)
REM ====================================

echo 🚀 Starting Revnet Agent-Based Model Dashboard...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements_dashboard.txt
)

echo ✅ Dependencies ready
echo.
echo 🌐 Opening dashboard in your browser...
echo    URL: http://localhost:8501
echo.
echo 💡 Tip: Press Ctrl+C to stop the server
echo.

REM Run streamlit
streamlit run streamlit_dashboard.py
