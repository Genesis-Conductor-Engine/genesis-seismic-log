#!/bin/bash
python3 simple_seismic_server.py > /dev/null 2>&1 &
SERVER_PID=$!
sleep 2
python3 bench.py
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
