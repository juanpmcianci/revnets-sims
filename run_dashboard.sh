#!/bin/bash

# Revnet Dashboard Launcher
# =========================

echo "🚀 Starting Revnet Agent-Based Model Dashboard..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements_dashboard.txt
fi

echo "✅ Dependencies ready"
echo ""
echo "🌐 Opening dashboard in your browser..."
echo "   URL: http://localhost:8501"
echo ""
echo "💡 Tip: Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run streamlit_dashboard.py
