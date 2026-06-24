#!/usr/bin/env bash
# Unthrottled throughput ceiling measurement.
# Feeds three file sources (no real-time cap) through the full graph with sync=false
# sinks to measure the maximum end-to-end FPS the GPU can sustain.
# Results written to metrics/results/throughput_unthrottled.json.
#
# Prereqs:
#   - data/mot17_04.mp4 must exist inside the container at /workspace/data/mot17_04.mp4
#   - No other GPU-heavy processes running (stop compose streams if needed)
#
# Usage (inside the container):
#   bash metrics/throughput_run.sh
#
# Or from the host:
#   docker exec <container> bash /workspace/metrics/throughput_run.sh

set -euo pipefail

RESULTS=/workspace/metrics/results
mkdir -p "$RESULTS"

DURATION=${DURATION:-120}   # 2-minute run
INTERVAL=${INTERVAL:-5}

echo "[throughput_run] Starting ${DURATION}s unthrottled file-source run ..."
echo "[throughput_run] Output: $RESULTS/throughput_unthrottled.json"

# Background nvidia-smi poller (peak VRAM / GPU util) — same pattern as batch_bench.sh
POLLER_OUT="$RESULTS/throughput_smi_poll.tmp"
nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -l 1 > "$POLLER_OUT" 2>/dev/null &
POLLER_PID=$!

python3 pipelines/multi_stream.py \
    --uri data/mot17_04.mp4 \
    --uri data/mot17_04.mp4 \
    --uri data/mot17_04.mp4 \
    --no-sync \
    --perf-json "$RESULTS/throughput_unthrottled.json" \
    --perf-interval "$INTERVAL" \
    --duration "$DURATION"

# Stop the nvidia-smi poller and extract peak values
kill "$POLLER_PID" 2>/dev/null || true
if [[ -f "$POLLER_OUT" ]]; then
    PEAK_VRAM=$(awk -F',' '{print $1+0}' "$POLLER_OUT" | sort -n | tail -1)
    PEAK_UTIL=$(awk -F',' '{print $2+0}' "$POLLER_OUT" | sort -n | tail -1)
    echo "[throughput_run] Peak VRAM: ${PEAK_VRAM} MB   Peak GPU util: ${PEAK_UTIL}%"
    rm -f "$POLLER_OUT"
fi

echo "[throughput_run] Done. Summary written to $RESULTS/throughput_unthrottled.json"
