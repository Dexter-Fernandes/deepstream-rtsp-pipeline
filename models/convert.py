"""
Convert YOLO26n ONNX → TensorRT engine via trtexec.

INT8 is excluded: GTX 1660Ti has no INT8 Tensor Cores — calibration would not
yield a throughput gain and requires a representative calibration dataset.
"""

import argparse
import subprocess
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
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        dest="max_batch",
        help="Maximum batch size for dynamic-batch ONNX (default: 1, no shape flags)",
    )
    parser.add_argument(
        "--input-name",
        default="images",
        dest="input_name",
        help="ONNX input tensor name used in shape flags (default: images)",
    )
    return parser.parse_args(argv)


def engine_path(onnx_path: Path, fp16: bool, output_dir: Path, max_batch: int = 1) -> Path:
    precision = "fp16" if fp16 else "fp32"
    suffix = f"_{precision}_b{max_batch}" if max_batch > 1 else f"_{precision}"
    return output_dir / f"{onnx_path.stem}{suffix}.engine"


def build_trtexec_cmd(
    onnx_path: Path,
    engine_out: Path,
    fp16: bool,
    max_batch: int = 1,
    input_name: str = "images",
) -> list[str]:
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_out}",
    ]
    if fp16:
        cmd.append("--fp16")
    if max_batch > 1:
        # Dynamic-batch ONNX: shape profile so TRT builds a plan covering
        # batch 1..max_batch at the fixed 640×640 network resolution.
        cmd += [
            f"--minShapes={input_name}:1x3x640x640",
            f"--optShapes={input_name}:{max_batch}x3x640x640",
            f"--maxShapes={input_name}:{max_batch}x3x640x640",
        ]
    return cmd


def convert(
    onnx_path: Path,
    fp16: bool = False,
    output_dir: Path = Path("models/engines"),
    max_batch: int = 1,
    input_name: str = "images",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = engine_path(onnx_path, fp16, output_dir, max_batch)
    cmd = build_trtexec_cmd(onnx_path, out, fp16, max_batch, input_name)
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    args = parse_args()
    result = convert(
        args.onnx,
        fp16=args.fp16,
        output_dir=args.output_dir,
        max_batch=args.max_batch,
        input_name=args.input_name,
    )
    print(f"Engine saved to {result}")
