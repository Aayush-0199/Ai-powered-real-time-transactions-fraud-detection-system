#!/bin/bash
# ──────────────────────────────────────────────────────────────
# FraudGuard AI — Production Startup Script
# Runs the Flask API (via gunicorn) + transaction simulator
# together inside a single Render free-tier web service.
# ──────────────────────────────────────────────────────────────

set -e

# Start gunicorn with threading so the simulator's HTTP calls
# to localhost:5000 never deadlock the single process.
gunicorn \
  --bind "0.0.0.0:${PORT:-5000}" \
  --workers 1 \
  --worker-class gthread \
  --threads 4 \
  --timeout 120 \
  --access-logfile - \
  app:app &

GUNICORN_PID=$!

# Give gunicorn ~8 seconds to fully load the ML models before
# the simulator starts firing transactions at it.
echo "Waiting for Flask API to be ready..."
sleep 8

echo "Starting transaction simulator..."
# FLASK_URL defaults to localhost:5000 in simulate_transactions.py
python simulate_transactions.py &

SIM_PID=$!

# Wait for gunicorn (primary process). If it dies, kill the simulator too.
wait $GUNICORN_PID
kill $SIM_PID 2>/dev/null || true
