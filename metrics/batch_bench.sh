#!/usr/bin/env bash
# Profile FP16 engine at batch 1, 2, 3, 15, 20, 25, 33, 100 to model concurrent stream throughput.
# Captures VRAM and GPU utilisation alongside TRT layer timings.
# Run inside the container after building engines.
#
# Usage:
#   docker compose run --rm pipeline bash metrics/batch_bench.sh

set -euo pipefail

ENGINE=/workspace/models/engines/yolo26n_fp16_b100.engine
RESULTS=/workspace/metrics/results

mkdir -p "$RESULTS"

if [ ! -f "$ENGINE" ]; then
    echo "[batch_bench] ERROR: $ENGINE not found. Run docker compose up first to build engines."
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
vram_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

gpu_power_w() {
    # Returns integer watts; 0 if not supported
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' | cut -d'.' -f1 || echo 0
}

# ---------------------------------------------------------------------------
# Profile each batch size with VRAM + GPU util snapshots
# ---------------------------------------------------------------------------
VRAM_IDLE=$(vram_used)
echo "[batch_bench] Idle VRAM: ${VRAM_IDLE} MB"

declare -A VRAM_PEAK GPU_UTIL_PEAK POWER_W

for BATCH in 1 2 3 15 20 25 33 100; do
    echo ""
    echo "[batch_bench] Profiling batch=$BATCH (simulates $BATCH concurrent streams)..."

    # Poll GPU stats in background while trtexec runs
    POLL_OUT=$(mktemp)
    (
        while true; do
            echo "$(vram_used) $(gpu_util) $(gpu_power_w)"
            sleep 0.5
        done
    ) > "$POLL_OUT" 2>/dev/null &
    POLL_PID=$!

    python3 metrics/profile_decode.py \
        --engine "$ENGINE" \
        --batch "$BATCH" \
        --label "FP16 batch=$BATCH (${BATCH}-stream)" \
        --save-json "$RESULTS/batch_${BATCH}.json"

    kill "$POLL_PID" 2>/dev/null || true
    wait "$POLL_PID" 2>/dev/null || true

    # Extract peak values from poll output
    if [ -s "$POLL_OUT" ]; then
        VRAM_PEAK[$BATCH]=$(awk '{print $1}' "$POLL_OUT" | sort -n | tail -1)
        GPU_UTIL_PEAK[$BATCH]=$(awk '{print $2}' "$POLL_OUT" | sort -n | tail -1)
        POWER_W[$BATCH]=$(awk '{print $3}' "$POLL_OUT" | sort -n | tail -1)
    else
        VRAM_PEAK[$BATCH]=0
        GPU_UTIL_PEAK[$BATCH]=0
        POWER_W[$BATCH]=0
    fi
    rm -f "$POLL_OUT"

    echo "[batch_bench] batch=$BATCH — VRAM peak: ${VRAM_PEAK[$BATCH]} MB  GPU util peak: ${GPU_UTIL_PEAK[$BATCH]}%  Power: ${POWER_W[$BATCH]} W"
done

# ---------------------------------------------------------------------------
# Write batch_system_metrics.json
# ---------------------------------------------------------------------------
cat > "$RESULTS/batch_system_metrics.json" <<EOF
{
  "idle_vram_mb": $VRAM_IDLE,
  "batches": {
    "1":   { "vram_mb": ${VRAM_PEAK[1]},   "gpu_util_pct": ${GPU_UTIL_PEAK[1]},   "power_w": ${POWER_W[1]} },
    "2":   { "vram_mb": ${VRAM_PEAK[2]},   "gpu_util_pct": ${GPU_UTIL_PEAK[2]},   "power_w": ${POWER_W[2]} },
    "3":   { "vram_mb": ${VRAM_PEAK[3]},   "gpu_util_pct": ${GPU_UTIL_PEAK[3]},   "power_w": ${POWER_W[3]} },
    "15":  { "vram_mb": ${VRAM_PEAK[15]},  "gpu_util_pct": ${GPU_UTIL_PEAK[15]},  "power_w": ${POWER_W[15]} },
    "20":  { "vram_mb": ${VRAM_PEAK[20]},  "gpu_util_pct": ${GPU_UTIL_PEAK[20]},  "power_w": ${POWER_W[20]} },
    "25":  { "vram_mb": ${VRAM_PEAK[25]},  "gpu_util_pct": ${GPU_UTIL_PEAK[25]},  "power_w": ${POWER_W[25]} },
    "33":  { "vram_mb": ${VRAM_PEAK[33]},  "gpu_util_pct": ${GPU_UTIL_PEAK[33]},  "power_w": ${POWER_W[33]} },
    "100": { "vram_mb": ${VRAM_PEAK[100]}, "gpu_util_pct": ${GPU_UTIL_PEAK[100]}, "power_w": ${POWER_W[100]} }
  },
  "notes": "VRAM/util/power are peak values polled at 0.5s intervals during each trtexec run."
}
EOF

echo ""
echo "[batch_bench] Done. Results written to $RESULTS/"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import json
from pathlib import Path

r = Path("metrics/results")
sys = json.loads((r / "batch_system_metrics.json").read_text())
budget_ms = 40.0  # 25 fps real-time floor

b1_fps = json.loads((r / "batch_1.json").read_text())["fps_per_stream"]

print(
    f"\n  {'Streams':<9} {'mean(ms)':<10} {'p99(ms)':<10} {'max(ms)':<10} "
    f"{'jitter':<10} {'p99 ok?':<8} {'FPS/strm':<10} {'VRAM(MB)':<10} {'GPU%':<6}"
)
print(f"  {'-'*93}")
for b in [1, 2, 3, 15, 20, 25, 33, 100]:
    p = r / f"batch_{b}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    sm = sys["batches"][str(b)]
    p99 = d.get("p99_ms", 0.0)
    max_ms = d.get("max_ms", 0.0)
    min_ms = d.get("min_ms", 0.0)
    jitter = f"{max_ms - min_ms:.2f}ms" if max_ms > 0 else "n/a"
    p99_ok = ("OK" if p99 <= budget_ms else "OVER") if p99 > 0 else "n/a"
    p99_str = f"{p99:.2f}" if p99 > 0 else "n/a"
    eff = d["fps_per_stream"] / b1_fps * 100
    print(
        f"  {b:<9} {d['wall_ms']:<10.2f} {p99_str:<10} {f'{max_ms:.2f}' if max_ms > 0 else 'n/a':<10} "
        f"{jitter:<10} {p99_ok:<8} {d['fps_per_stream']:<10} "
        f"{sm['vram_mb']:<10} {sm['gpu_util_pct']:<6}%  [{eff:.0f}% per-stream eff]"
    )
print(f"\n  Real-time budget: {budget_ms:.0f} ms ({1000/budget_ms:.0f} fps).")
print("  p99 column shows worst-case latency at the 99th percentile.")
PYEOF
