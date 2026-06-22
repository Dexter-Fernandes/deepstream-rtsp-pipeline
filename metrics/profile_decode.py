"""
TensorRT IProfiler wrapper — measures per-layer latency of the decode engine.

Usage (inside container):
    python3 metrics/profile_decode.py \\
        --engine models/engines/yolo26n_fp16_b3_decode.engine \\
        --plugin-lib /workspace/models/plugins/libyolo26_decode.so \\
        [--n-warmup 5] [--n-runs 50]

Output: table of per-layer latency (ms) sorted by time descending.
The yolo26_decode row isolates the CUDA kernel decode step.
"""

import argparse
import ctypes
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile TRT decode engine layer latency")
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("models/engines/yolo26n_fp16_b3_decode.engine"),
        help="Path to decode engine (default: yolo26n_fp16_b3_decode.engine)",
    )
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=Path("/opt/ds_plugins/libyolo26_decode.so"),
        dest="plugin_lib",
        help="Path to libyolo26_decode.so",
    )
    parser.add_argument("--n-warmup", type=int, default=5, dest="n_warmup")
    parser.add_argument("--n-runs",   type=int, default=50, dest="n_runs")
    return parser.parse_args(argv)


class _SimpleProfiler:
    """Accumulates reportLayerTime calls between clear() and print_report()."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}

    def reportLayerTime(self, layer_name: str, ms: float) -> None:
        self.records.setdefault(layer_name, []).append(ms)

    def clear(self) -> None:
        self.records.clear()

    def print_report(self, title: str = "Layer latency") -> None:
        if not self.records:
            print("[profiler] No records collected.")
            return
        avgs = {k: sum(v) / len(v) for k, v in self.records.items()}
        total = sum(avgs.values())
        rows = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)

        col_w = max(len(k) for k in avgs) + 2
        print(f"\n{'─' * (col_w + 22)}")
        print(f"  {title}")
        print(f"{'─' * (col_w + 22)}")
        print(f"  {'Layer':<{col_w}}  {'Avg (ms)':>8}  {'% total':>7}")
        print(f"{'─' * (col_w + 22)}")
        for name, ms in rows:
            marker = " ◀" if name == "yolo26_decode" else ""
            print(f"  {name:<{col_w}}  {ms:8.3f}  {100*ms/total:6.1f}%{marker}")
        print(f"{'─' * (col_w + 22)}")
        print(f"  {'TOTAL':<{col_w}}  {total:8.3f}")
        print(f"{'─' * (col_w + 22)}\n")


def profile_engine(
    engine_path: Path,
    plugin_lib: Path,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> None:
    import numpy as np
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context

    ctypes.CDLL(str(plugin_lib))

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate dummy input (batch=1, 3×640×640) and output buffers
    input_shape  = (1, 3, 640, 640)
    output_shape = (1, 300, 6)
    h_input  = np.zeros(input_shape,  dtype=np.float32)
    h_output = np.zeros(output_shape, dtype=np.float32)
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    cuda.memcpy_htod(d_input, h_input)

    # Set dynamic shapes
    context.set_input_shape("images", input_shape)
    bindings = [int(d_input), int(d_output)]

    profiler = _SimpleProfiler()
    context.profiler = profiler

    # Warmup
    for _ in range(n_warmup):
        context.execute_v2(bindings)

    # Timed runs
    profiler.clear()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        context.execute_v2(bindings)
    wall_ms = (time.perf_counter() - t0) * 1000 / n_runs

    profiler.print_report(title=f"Decode engine — {engine_path.name} ({n_runs} runs)")
    print(f"  Wall time per inference: {wall_ms:.2f} ms  ({1000/wall_ms:.1f} FPS)\n")


if __name__ == "__main__":
    args = parse_args()
    profile_engine(
        engine_path=args.engine,
        plugin_lib=args.plugin_lib,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
    )
