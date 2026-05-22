#!/bin/bash
# ── EmotiScan startup script ──────────────────────────────────────────────────
# Run this from ~/Facial/ (where your model .pth file lives)

set -e

VENV="/home/zainaandalhaq/venvs/advance_env"
PORT=5000

echo "========================================"
echo "  EmotiScan — Emotion Prediction Server"
echo "========================================"

# Activate your existing venv
source "$VENV/bin/activate"

# Install Flask if not already installed
pip install flask --quiet

echo ""
echo "  Model : mobilenetv2_emotion_model.pth"
echo "  Device: GPU (CUDA)"
echo "  Port  : $PORT"
echo ""
echo "  Open in browser: http://localhost:$PORT"
echo "  Or from another machine: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "  Press Ctrl+C to stop."
echo "========================================"
echo ""

python app.py
