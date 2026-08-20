"""
Sequential container init: export YOLO26n → ONNX, build TRT engines, then
exec the pipeline command. Steps are skipped if output files already exist
so repeated container starts are instant.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from models.convert import convert as _convert
from models.convert import engine_path as _engine_path
from models.decode_engine import decode_engine_path as _decode_path
from models.export_yolo26 import export as _export

_DEFAULT_PLUGIN_LIB = Path("/opt/ds_plugins/libyolo26_decode.so")
_BUILD_ENGINE_BIN   = Path("/opt/ds_plugins/build_yolo26_engine")
_DEFAULT_REID_CONFIG = Path("configs/nvinfer_reid.txt")
_REID_MODEL_NAME = "resnet50_market1501_aicity156.onnx"
_REID_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/"
    "reidentificationnet/versions/deployable_v1.2/files/"
    f"{_REID_MODEL_NAME}"
)


def _build_decode_default(onnx, plugin_lib, fp16, max_batch, output_dir):
    out = _decode_path(output_dir, fp16=fp16, max_batch=max_batch)
    cmd = [str(_BUILD_ENGINE_BIN), str(onnx), str(plugin_lib), str(out)]
    if fp16:
        cmd.append("--fp16")
    cmd += ["--max-batch", str(max_batch)]
    subprocess.run(cmd, check=True)


def _download_file(url: str, destination: Path) -> None:
    """Download one model artifact atomically so an interrupted fetch is never reused."""
    partial = destination.with_suffix(f"{destination.suffix}.part")
    with urlopen(url, timeout=30) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)


def ensure_reid_model(
    engines_dir: Path,
    *,
    max_batch: int = 32,
    model_url: str = _REID_MODEL_URL,
    download_fn=_download_file,
    convert_fn=_convert,
) -> Path | None:
    """Download NVIDIA's supported ReID ONNX and build its FP16 SGIE engine."""
    engines_dir.mkdir(parents=True, exist_ok=True)
    onnx = engines_dir / _REID_MODEL_NAME
    engine = _engine_path(onnx, fp16=True, output_dir=engines_dir, max_batch=max_batch)

    if onnx.exists():
        print(f"[init] ReID ONNX found at {onnx} — skipping download", flush=True)
    else:
        print(f"[init] Downloading NVIDIA ReIdentificationNet → {onnx}...", flush=True)
        try:
            download_fn(model_url, onnx)
        except OSError as exc:
            print(
                "[init] ReID download unavailable — skipping optional appearance model: "
                f"{exc}",
                flush=True,
            )
            return None

    if engine.exists():
        print(f"[init] ReID engine found at {engine} — skipping build", flush=True)
        return engine

    print(f"[init] Building ReID FP16 engine (max_batch={max_batch}) from {onnx}...", flush=True)
    return convert_fn(
        onnx,
        fp16=True,
        output_dir=engines_dir,
        max_batch=max_batch,
        input_name="input",
        input_shape=(3, 256, 128),
    )


def should_prepare_reid(argv: list[str]) -> bool:
    """Keep the optional model off the baseline container startup path."""
    return "--mtmc" in argv and "--mtmc-appearance" in argv


def _reid_config_path(argv: list[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument == "--reid-config" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--reid-config="):
            return Path(argument.split("=", 1)[1])
    return _DEFAULT_REID_CONFIG


def prepare_launch_args(
    argv: list[str],
    *,
    engines_dir: Path,
    ensure_reid_fn=ensure_reid_model,
) -> list[str]:
    """Prepare optional appearance assets or make the geometry fallback explicit."""
    launch_args = list(argv)
    if not should_prepare_reid(launch_args):
        return launch_args
    if _reid_config_path(launch_args).resolve() != _DEFAULT_REID_CONFIG.resolve():
        return launch_args
    if ensure_reid_fn(engines_dir) is not None:
        return launch_args

    print(
        "[init] ReID unavailable — continuing with geometry-only MTMC",
        flush=True,
    )
    launch_args.remove("--mtmc-appearance")
    return launch_args


def ensure_models(
    weights: Path,
    engines_dir: Path,
    max_batch: int = 3,
    plugin_lib: Path = _DEFAULT_PLUGIN_LIB,
    export_fn=_export,
    convert_fn=_convert,
    decode_fn=_build_decode_default,
) -> None:
    onnx = engines_dir / "yolo26n.onnx"
    fp32 = _engine_path(onnx, fp16=False, output_dir=engines_dir, max_batch=max_batch)
    fp16 = _engine_path(onnx, fp16=True, output_dir=engines_dir, max_batch=max_batch)
    decode = _decode_path(engines_dir, fp16=True, max_batch=max_batch)

    if onnx.exists():
        print(f"[init] ONNX model found at {onnx} — skipping export", flush=True)
    else:
        print(f"[init] Exporting YOLO26n → ONNX (weights: {weights})...", flush=True)
        export_fn(weights, engines_dir)
        print(f"[init] ONNX export complete → {onnx}", flush=True)

    if fp32.exists():
        print(f"[init] FP32 engine found at {fp32} — skipping build", flush=True)
    else:
        print(f"[init] Building FP32 TensorRT engine (max_batch={max_batch}) from {onnx}...", flush=True)
        convert_fn(onnx, fp16=False, output_dir=engines_dir, max_batch=max_batch)
        print(f"[init] FP32 engine ready → {fp32}", flush=True)

    if fp16.exists():
        print(f"[init] FP16 engine found at {fp16} — skipping build", flush=True)
    else:
        print(f"[init] Building FP16 TensorRT engine (max_batch={max_batch}) from {onnx}...", flush=True)
        convert_fn(onnx, fp16=True, output_dir=engines_dir, max_batch=max_batch)
        print(f"[init] FP16 engine ready → {fp16}", flush=True)

    if decode.exists():
        print(f"[init] Decode engine found at {decode} — skipping build", flush=True)
    elif not plugin_lib.exists():
        print(
            f"[init] Plugin lib not found at {plugin_lib} — skipping decode engine build.\n"
            f"[init]   Build first: cd plugins/yolo26_decode && cmake -B build && cmake --build build",
            flush=True,
        )
    else:
        print(
            f"[init] Building decode engine (fp16, max_batch={max_batch}) from {onnx}...",
            flush=True,
        )
        decode_fn(onnx, plugin_lib, fp16=True, max_batch=max_batch, output_dir=engines_dir)
        print(f"[init] Decode engine ready → {decode}", flush=True)


if __name__ == "__main__":
    print("[init] Starting model initialisation...", flush=True)
    ensure_models(
        weights=Path("models/yolo26n.pt"),
        engines_dir=Path("models/engines"),
        max_batch=3,
    )
    launch_args = prepare_launch_args(
        sys.argv[1:],
        engines_dir=Path("models/engines"),
    )
    print("[init] All models ready. Launching pipeline...", flush=True)
    if launch_args:
        os.execvp(launch_args[0], launch_args)
