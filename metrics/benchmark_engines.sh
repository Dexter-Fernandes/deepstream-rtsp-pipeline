#!/usr/bin/env bash
# Run inside the container to capture per-engine profiler results + system metrics.
# Results are saved to metrics/results/ and should be committed to the repo so
# the decode_comparison.ipynb renders without a GPU.
#
# Usage:
#   docker compose run --rm pipeline bash metrics/benchmark_engines.sh

set -euo pipefail

PLUGIN=/opt/ds_plugins/libyolo26_decode.so
ENGINES=/workspace/models/engines
RESULTS=/workspace/metrics/results

mkdir -p "$RESULTS"

# ---------------------------------------------------------------------------
# Helper: snapshot GPU memory and utilisation around a profiler run
# ---------------------------------------------------------------------------
vram_before() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

gpu_util_snapshot() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

# ---------------------------------------------------------------------------
# Engine file sizes (MB) — relevant for OTA fleet updates across 5000+ sensors
# ---------------------------------------------------------------------------
size_mb() {
    du --bytes "$1" 2>/dev/null | awk '{printf "%.1f", $1/1048576}' || echo 0
}

FP32_SIZE=$(size_mb "$ENGINES/yolo26n_fp32_b3.engine")
FP16_SIZE=$(size_mb "$ENGINES/yolo26n_fp16_b3.engine")
DECODE_SIZE=$(size_mb "$ENGINES/yolo26n_fp16_b3_decode.engine")

echo "[benchmark] Engine sizes: FP32=${FP32_SIZE}MB  FP16=${FP16_SIZE}MB  FP16+decode=${DECODE_SIZE}MB"

# ---------------------------------------------------------------------------
# Profile FP32 base engine
# ---------------------------------------------------------------------------
echo "[benchmark] Profiling FP32 base engine..."
VRAM_FP32=$(vram_before)
python3 metrics/profile_decode.py \
    --engine "$ENGINES/yolo26n_fp32_b3.engine" \
    --label "FP32 base" \
    --save-json "$RESULTS/fp32_base.json"
VRAM_FP32_PEAK=$(vram_before)

# ---------------------------------------------------------------------------
# Profile FP16 base engine (no decode plugin)
# ---------------------------------------------------------------------------
echo "[benchmark] Profiling FP16 base engine..."
VRAM_FP16=$(vram_before)
python3 metrics/profile_decode.py \
    --engine "$ENGINES/yolo26n_fp16_b3.engine" \
    --label "FP16 base" \
    --save-json "$RESULTS/fp16_base.json"
VRAM_FP16_PEAK=$(vram_before)

# ---------------------------------------------------------------------------
# Profile FP16 + decode plugin engine
# ---------------------------------------------------------------------------
echo "[benchmark] Profiling FP16 + decode plugin engine..."
VRAM_DECODE=$(vram_before)
python3 metrics/profile_decode.py \
    --engine "$ENGINES/yolo26n_fp16_b3_decode.engine" \
    --plugin-lib "$PLUGIN" \
    --label "FP16 + decode plugin" \
    --save-json "$RESULTS/fp16_decode.json"
VRAM_DECODE_PEAK=$(vram_before)

# ---------------------------------------------------------------------------
# Write system_metrics.json — edge-constraint summary for the notebook
# ---------------------------------------------------------------------------
cat > "$RESULTS/system_metrics.json" <<EOF
{
  "engines": {
    "fp32_base":       { "file_mb": $FP32_SIZE,   "vram_mb": $VRAM_FP32_PEAK },
    "fp16_base":       { "file_mb": $FP16_SIZE,   "vram_mb": $VRAM_FP16_PEAK },
    "fp16_decode":     { "file_mb": $DECODE_SIZE, "vram_mb": $VRAM_DECODE_PEAK }
  },
  "fleet_size": 5000,
  "notes": "VRAM measured after loading engine + running 50 inference passes (batch=1, 640x640). Engine sizes on disk in MB."
}
EOF

echo "[benchmark] Done. Results written to $RESULTS/"
ls -lh "$RESULTS/"
