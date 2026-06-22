"""
Convert YOLO26n ONNX → TensorRT engine via trtexec.

INT8 is excluded: GTX 1660Ti has no INT8 Tensor Cores — calibration would not
yield a throughput gain and requires a representative calibration dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO26n ONNX to TensorRT engine")
    parser.add_argument("onnx", type=Path, help="Input ONNX model path")
    parser.add_argument("--fp16", action="store_true", help="Build FP16 engine (default: FP32)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/engines"),
        dest="output_dir",
        help="Directory for engine output",
    )
    return parser.parse_args(argv)


def engine_path(onnx_path: Path, fp16: bool, output_dir: Path) -> Path:
    precision = "fp16" if fp16 else "fp32"
    return output_dir / f"{onnx_path.stem}_{precision}.engine"


def build_trtexec_cmd(onnx_path: Path, engine_out: Path, fp16: bool) -> list[str]:
    # Shape flags are omitted: YOLO26n is exported with dynamic=False so the
    # input shape is fixed in the ONNX graph; trtexec reads it directly.
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
    ]
    if fp16:
        cmd.append("--fp16")
    return cmd


def convert(
    onnx_path: Path,
    fp16: bool = False,
    output_dir: Path = Path("models/engines"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = engine_path(onnx_path, fp16, output_dir)
    cmd = build_trtexec_cmd(onnx_path, out, fp16)
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    args = parse_args()
    result = convert(args.onnx, fp16=args.fp16, output_dir=args.output_dir)
    print(f"Engine saved to {result}")
