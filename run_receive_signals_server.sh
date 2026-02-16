#!/bin/bash

echo "============================================================"
echo "DEPRECATED: run_receive_signals_server.sh is deprecated."
echo "Use the REST server running natively from miner.py instead."
echo "New endpoint: POST /api/submit-order"
echo "============================================================"
echo ""
echo "Continuing in 5 seconds... (Ctrl+C to cancel)"
sleep 5

while true; do
    echo "Starting Python script..."
    . venv/bin/activate
    python -m pip install -e .
    nohup venv/bin/python mining/run_receive_signals_server.py &
    PID=$!
    echo "Python script started with PID: $PID"

    # Wait for the process to finish
    wait $PID

    # Check if the process is still running
    if ps -p $PID > /dev/null; then
        echo "Python script is still running, not restarting."
    else
        echo "Python script stopped, restarting in 5 seconds..."
        sleep 5
    fi
done
