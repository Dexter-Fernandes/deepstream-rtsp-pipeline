"""
TensorRT profiler — measures per-layer latency of any YOLO26n engine.

Drives trtexec (available in the NGC DS 9.0 container) via subprocess so no
Python tensorrt / pycuda package is required.

Usage (inside container):
    # FP32 base engine (no plugin)
    python3 metrics/profile_decode.py \\
        --engine models/engines/yolo26n_fp32_b3.engine \\
        --label "FP32 base" \\
        --save-json metrics/results/fp32_base.json

    # FP16 base engine (no plugin)
    python3 metrics/profile_decode.py \\
        --engine models/engines/yolo26n_fp16_b3.engine \\
        --label "FP16 base" \\
        --save-json metrics/results/fp16_base.json

    # FP16 + decode plugin engine
    python3 metrics/profile_decode.py \\
        --engine models/engines/yolo26n_fp16_b3_decode.engine \\
        --plugin-lib /opt/ds_plugins/libyolo26_decode.so \\
        --label "FP16 + decode plugin" \\
        --save-json metrics/results/fp16_decode.json
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# trtexec helpers
# ---------------------------------------------------------------------------

_TRTEXEC_CANDIDATES = [
    "/usr/bin/trtexec",
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/bin/trtexec",
]


def _find_trtexec() -> str:
    for c in _TRTEXEC_CANDIDATES:
        if Path(c).exists():
            return c
    in_path = shutil.which("trtexec")
    if in_path:
        return in_path
    raise RuntimeError(
        "trtexec not found. Searched: " + ", ".join(_TRTEXEC_CANDIDATES)
    )


def _parse_mean_latency(output: str) -> float:
    """Extract mean inference latency (ms) from trtexec stdout/stderr."""
    # TRT 10: "Latency: min = X ms, max = X ms, mean = X ms, median = X ms"
    m = re.search(r"mean\s*=\s*([\d.]+)\s*ms", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # TRT 10 alternate: "GPU Compute Mean: X ms"
    m = re.search(r"GPU Compute Mean:\s*([\d.]+)\s*ms", output, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Throughput fallback: "Throughput: 123.4 qps"
    m = re.search(r"Throughput:\s*([\d.]+)\s*qps", output, re.IGNORECASE)
    if m:
        return 1000.0 / float(m.group(1))
    # Last resort: sum of all layer averages from the profile itself
    return 0.0


def _parse_layer_profile(profile_path: Path) -> dict[str, float]:
    """Parse trtexec --exportProfile JSON → {layer_name: avg_ms}."""
    data = json.loads(profile_path.read_text())
    layers: dict[str, float] = {}
    for entry in data:
        name = entry.get("name", "unknown")
        # TRT 10 uses "averageMs"; older versions may use "timeMs"
        ms = entry.get("averageMs") or entry.get("timeMs") or 0.0
        if isinstance(ms, list):
            ms = sum(ms) / len(ms) if ms else 0.0
        layers[name] = round(float(ms), 4)
    return layers


# ---------------------------------------------------------------------------
# Profiler class (for programmatic use)
# ---------------------------------------------------------------------------

class _SimpleProfiler:
    """Thin wrapper around trtexec profiling results."""

    def __init__(self, layers: dict[str, float], wall_ms: float):
        self.layers = layers
        self.wall_ms = wall_ms

    def to_dict(self, label: str | None, engine_path: Path) -> dict:
        return {
            "label": label or engine_path.name,
            "engine": str(engine_path),
            "wall_ms": round(self.wall_ms, 4),
            "fps": round(1000.0 / self.wall_ms, 2) if self.wall_ms > 0 else 0,
            "layers": self.layers,
        }

    def print_report(self, title: str = "Layer latency") -> None:
        if not self.layers:
            print("[profiler] No layer data collected.")
            return
        total = sum(self.layers.values())
        rows = sorted(self.layers.items(), key=lambda kv: kv[1], reverse=True)
        col_w = max(len(k) for k in self.layers) + 2
        print(f"\n{'─' * (col_w + 22)}")
        print(f"  {title}")
        print(f"{'─' * (col_w + 22)}")
        print(f"  {'Layer':<{col_w}}  {'Avg (ms)':>8}  {'% total':>7}")
        print(f"{'─' * (col_w + 22)}")
        for name, ms in rows:
            marker = " ◀" if name == "yolo26_decode" else ""
            pct = (100 * ms / total) if total > 0 else 0
            print(f"  {name:<{col_w}}  {ms:8.3f}  {pct:6.1f}%{marker}")
        print(f"{'─' * (col_w + 22)}")
        print(f"  {'TOTAL':<{col_w}}  {total:8.3f}")
        print(f"{'─' * (col_w + 22)}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def profile_engine(
    engine_path: Path,
    plugin_lib: Path | None = None,
    label: str | None = None,
    n_warmup: int = 5,
    n_runs: int = 50,
    save_json: Path | None = None,
) -> dict:
    trtexec = _find_trtexec()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        profile_json = Path(tmp.name)

    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--iterations={n_runs}",
        "--warmUp=2000",           # 2 s warmup (ms, not iterations)
        "--profilingVerbosity=detailed",
        f"--exportProfile={profile_json}",
    ]
    if plugin_lib is not None:
        cmd.append(f"--plugins={plugin_lib}")

    print(f"[profiler] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"trtexec failed (exit {proc.returncode})")

    wall_ms = _parse_mean_latency(output)
    layers  = _parse_layer_profile(profile_json) if profile_json.exists() else {}
    profile_json.unlink(missing_ok=True)
    # If trtexec output format didn't match any regex, sum layer times as proxy
    if wall_ms == 0.0 and layers:
        wall_ms = round(sum(layers.values()), 4)

    profiler = _SimpleProfiler(layers, wall_ms)
    profiler.print_report(title=f"{label or engine_path.name} ({n_runs} runs)")
    if wall_ms > 0:
        print(f"  Wall time per inference: {wall_ms:.2f} ms  ({1000/wall_ms:.1f} FPS)\n")

    result = profiler.to_dict(label, engine_path)

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(result, indent=2))
        print(f"  Results saved → {save_json}\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile TRT engine per-layer latency via trtexec")
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("models/engines/yolo26n_fp16_b3_decode.engine"),
        help="Path to TRT engine file",
    )
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=None,
        dest="plugin_lib",
        help="Path to libyolo26_decode.so (omit for engines without the decode plugin)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Human-readable label stored in JSON output",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        dest="save_json",
        help="Write profiler results to this JSON file",
    )
    parser.add_argument("--n-warmup", type=int, default=5,  dest="n_warmup",
                        help="Ignored (warmup is time-based in trtexec; kept for API compat)")
    parser.add_argument("--n-runs",   type=int, default=50, dest="n_runs")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    profile_engine(
        engine_path=args.engine,
        plugin_lib=args.plugin_lib,
        label=args.label,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        save_json=args.save_json,
    )
