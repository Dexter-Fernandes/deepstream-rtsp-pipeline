"""
Sequential container init: export YOLO26n → ONNX, build TRT engines, then
exec the pipeline command. Steps are skipped if output files already exist
so repeated container starts are instant.
"""
import os
import sys
from pathlib import Path

from models.export_yolo26 import export as _export
from models.convert import convert as _convert


def ensure_models(
    weights: Path,
    engines_dir: Path,
    export_fn=_export,
    convert_fn=_convert,
) -> None:
    onnx = engines_dir / "yolo26n.onnx"
    fp32 = engines_dir / "yolo26n_fp32.engine"
    fp16 = engines_dir / "yolo26n_fp16.engine"

    if onnx.exists():
        print(f"[init] ONNX model found at {onnx} — skipping export", flush=True)
    else:
        print(f"[init] Exporting YOLO26n → ONNX (weights: {weights})...", flush=True)
        export_fn(weights, engines_dir)
        print(f"[init] ONNX export complete → {onnx}", flush=True)

    if fp32.exists():
        print(f"[init] FP32 engine found at {fp32} — skipping build", flush=True)
    else:
        print(f"[init] Building FP32 TensorRT engine from {onnx}...", flush=True)
        convert_fn(onnx, fp16=False, output_dir=engines_dir)
        print(f"[init] FP32 engine ready → {fp32}", flush=True)

    if fp16.exists():
        print(f"[init] FP16 engine found at {fp16} — skipping build", flush=True)
    else:
        print(f"[init] Building FP16 TensorRT engine from {onnx}...", flush=True)
        convert_fn(onnx, fp16=True, output_dir=engines_dir)
        print(f"[init] FP16 engine ready → {fp16}", flush=True)


if __name__ == "__main__":
    print("[init] Starting model initialisation...", flush=True)
    ensure_models(
        weights=Path("models/yolo26n.pt"),
        engines_dir=Path("models/engines"),
    )
    print("[init] All models ready. Launching pipeline...", flush=True)
    if sys.argv[1:]:
        os.execvp(sys.argv[1], sys.argv[1:])
