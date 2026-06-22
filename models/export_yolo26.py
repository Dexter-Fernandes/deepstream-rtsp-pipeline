"""
Export YOLO26n PyTorch weights → ONNX for TensorRT conversion.

YOLO26 is NMS-free by default (end-to-end inference via one-to-one head),
so no nms=False flag is required on export unlike YOLOv8.

Run inside the DeepStream container after `pip install ultralytics`:
    python3 models/export_yolo26.py --weights yolo26n.pt
"""

import argparse
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26n .pt → ONNX")
    parser.add_argument("weights", type=Path, help="Path to yolo26n.pt weights file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/engines"),
        dest="output_dir",
        help="Directory to place the exported ONNX file",
    )
    return parser.parse_args(argv)


def export(weights: Path, output_dir: Path) -> Path:
    from ultralytics import YOLO  # imported here so the module loads without ultralytics installed

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    # YOLO26 is NMS-free by default; dynamic=True makes the batch dimension
    # flexible so trtexec can build engines for batch 1..N with --minShapes etc.
    model.export(format="onnx", dynamic=True, imgsz=640)

    # ultralytics writes <stem>.onnx alongside the .pt file
    onnx_src = weights.with_suffix(".onnx")
    onnx_dst = output_dir / onnx_src.name
    shutil.move(str(onnx_src), str(onnx_dst))
    return onnx_dst


if __name__ == "__main__":
    args = parse_args()
    result = export(args.weights, args.output_dir)
    print(f"ONNX model saved to {result}")
