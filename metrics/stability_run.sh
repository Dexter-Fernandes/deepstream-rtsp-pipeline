#!/usr/bin/env bash
# 30-minute live 3-stream RTSP stability run.
# Measures real-time adherence (25 fps × 3 sources) and checks for RSS/VRAM leaks.
# Results written to metrics/results/stability_live.json.
#
# Prereqs (run on the host before this script):
#   1. mediamtx running:   ps aux | grep mediamtx
#   2. Three ffmpeg streams (capped at 25 fps):
#        ffmpeg -re -stream_loop -1 -i data/mot17_04.mp4 \
#          -c copy -f rtsp rtsp://localhost:8554/stream0
#        (repeat for stream1, stream2)
#   3. Container up with network_mode: host (see docker-compose.yml)
#
# Note: network_mode: host means the container shares the host network stack,
# so rtsp://localhost:8554/... resolves correctly inside the container.
#
# Usage (inside the container):
#   bash metrics/stability_run.sh
#
# Or from the host:
#   docker exec <container> bash /workspace/metrics/stability_run.sh

set -euo pipefail

RESULTS=/workspace/metrics/results
mkdir -p "$RESULTS"

DURATION=${DURATION:-1800}   # 30 min default; override with DURATION=60 for smoke test
INTERVAL=${INTERVAL:-10}     # sample every 10 s

echo "[stability_run] Starting ${DURATION}s live 3-stream run (interval=${INTERVAL}s) ..."
echo "[stability_run] Output: $RESULTS/stability_live.json"

python3 pipelines/multi_stream.py \
    --uri rtsp://localhost:8554/stream0 \
    --uri rtsp://localhost:8554/stream1 \
    --uri rtsp://localhost:8554/stream2 \
    --restream-base-port 8556 \
    --perf-json "$RESULTS/stability_live.json" \
    --perf-interval "$INTERVAL" \
    --duration "$DURATION"

echo "[stability_run] Done. Summary written to $RESULTS/stability_live.json"
