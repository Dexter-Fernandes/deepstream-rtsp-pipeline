"""
Build a TensorRT engine that appends the yolo26_decode plugin to the YOLO26n
ONNX network.  The plugin converts xyxy pixel-space output → xywh so the
Python probe can read coordinates directly without a Python for-loop transform.

Usage (inside container, after building libyolo26_decode.so):
    python3 models/decode_engine.py models/engines/yolo26n.onnx \\
        --plugin-lib /workspace/models/plugins/libyolo26_decode.so \\
        --fp16 --max-batch 3
"""

import argparse
from pathlib import Path


def decode_engine_path(output_dir: Path, fp16: bool, max_batch: int) -> Path:
    precision = "fp16" if fp16 else "fp32"
    return output_dir / f"yolo26n_{precision}_b{max_batch}_decode.engine"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TRT engine with yolo26_decode plugin appended"
    )
    parser.add_argument("onnx", type=Path, help="YOLO26n ONNX model path")
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=Path("/opt/ds_plugins/libyolo26_decode.so"),
        dest="plugin_lib",
        help="Path to libyolo26_decode.so (default: /workspace/models/plugins/libyolo26_decode.so)",
    )
    parser.add_argument("--fp16", action="store_true", help="Build FP16 engine")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=3,
        dest="max_batch",
        help="Maximum batch size for shape profile (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/engines"),
        dest="output_dir",
        help="Output directory for the decode engine",
    )
    return parser.parse_args(argv)


def build_decode_engine(
    onnx_path: Path,
    plugin_lib: Path,
    fp16: bool,
    max_batch: int,
    output_dir: Path,
) -> Path:
    """Append yolo26_decode plugin to the YOLO26n TRT network and serialise.

    Requires libyolo26_decode.so built from plugins/yolo26_decode/ and the
    tensorrt Python package (available in the NGC DeepStream 9.0 container).
    The plugin registers Yolo26DecodePluginCreator on import via ctypes.CDLL.
    """
    import ctypes

    import tensorrt as trt

    ctypes.CDLL(str(plugin_lib))

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errors))

    # Unmark YOLO26n's output so we can feed it into the decode plugin
    yolo_out = network.get_output(0)
    network.unmark_output(yolo_out)

    # Append yolo26_decode plugin layer
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("Yolo26DecodePlugin", "1", "")
    if creator is None:
        raise RuntimeError(
            "Yolo26DecodePlugin not found in registry — "
            f"check that {plugin_lib} loaded correctly"
        )
    plugin_obj = creator.create_plugin(
        "yolo26_decode", trt.PluginFieldCollection([])
    )
    decode_layer = network.add_plugin_v2([yolo_out], plugin_obj)
    decode_layer.name = "yolo26_decode"
    network.mark_output(decode_layer.get_output(0))

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        (1, 3, 640, 640),
        (max_batch, 3, 640, 640),
        (max_batch, 3, 640, 640),
    )
    cfg.add_optimization_profile(profile)

    print(f"[decode_engine] Building engine (fp16={fp16}, max_batch={max_batch})…")
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError("Engine build failed — check TRT logs above")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = decode_engine_path(output_dir, fp16, max_batch)
    out.write_bytes(serialized)
    print(f"[decode_engine] Engine saved → {out}")
    return out


if __name__ == "__main__":
    args = parse_args()
    build_decode_engine(
        onnx_path=args.onnx,
        plugin_lib=args.plugin_lib,
        fp16=args.fp16,
        max_batch=args.max_batch,
        output_dir=args.output_dir,
    )
