#!/bin/bash
# ════════════════════════════════════════════════
#  LIFE EXPECTANCY PROJECT — LAUNCHER
#  Run this once to set up and start the dashboard
# ════════════════════════════════════════════════

echo ""
echo "🔬 INDIA LIFE EXPECTANCY — XAI PROJECT"
echo "════════════════════════════════════════"
echo ""

# Step 1: Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages -q
echo "✅ Dependencies installed."
echo ""

# Step 2: Run ML engine to generate all data
echo "🤖 Running ML engine (training model + generating data)..."
python ml_engine.py
echo ""

# Step 3: Launch dashboard
echo "🚀 Launching dashboard..."
echo "   Open your browser at: http://localhost:8501"
echo ""
streamlit run dashboard.py
