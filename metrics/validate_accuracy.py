"""
M2.7.1 accuracy validation harness.

Runs FP32, FP16, and FP16+decode TRT engines on a fixed set of MOT17 frames and
computes numerical box agreement metrics, closing the M2.4 "visual check only" gap.

Usage (inside container):
    python3 metrics/validate_accuracy.py \\
        --seq-dir /media/dexter/PortableSSD/Datasets/MOT17/train/MOT17-04-SDP/img1 \\
        --fp32-engine models/engines/yolo26n_fp32_b3.engine \\
        --fp16-engine models/engines/yolo26n_fp16_b3.engine \\
        --decode-engine models/engines/yolo26n_fp16_b3_decode.engine \\
        --plugin-lib /opt/ds_plugins/libyolo26_decode.so \\
        --n-frames 50 \\
        --save-json metrics/results/accuracy.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from models.output_parser import parse_yolo26_output


# ---------------------------------------------------------------------------
# Pure-Python / NumPy functions (CPU-safe, unit-tested)
# ---------------------------------------------------------------------------

def box_iou(a: dict, b: dict) -> float:
    """IoU between two boxes in {left, top, width, height} format."""
    ax1, ay1 = a["left"], a["top"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["left"], b["top"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    dets_a: list[dict],
    dets_b: list[dict],
    iou_thresh: float = 0.5,
) -> tuple[list, list, list]:
    """Greedy IoU matching, highest-IoU pairs first.

    Returns:
        matches:     list of (det_a, det_b, iou)
        unmatched_a: dets in dets_a with no match in dets_b
        unmatched_b: dets in dets_b with no match in dets_a
    """
    if not dets_a or not dets_b:
        return [], list(dets_a), list(dets_b)

    candidates = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou = box_iou(da, db)
            if iou >= iou_thresh:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    matches = []
    for iou, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        matches.append((dets_a[i], dets_b[j], iou))
        matched_a.add(i)
        matched_b.add(j)

    unmatched_a = [d for i, d in enumerate(dets_a) if i not in matched_a]
    unmatched_b = [d for j, d in enumerate(dets_b) if j not in matched_b]
    return matches, unmatched_a, unmatched_b


def compare_engines(
    frames_dets_a: list[list[dict]],
    frames_dets_b: list[list[dict]],
    iou_thresh: float = 0.5,
) -> dict:
    """Aggregate IoU-based detection agreement across all frames.

    Returns mean_iou, n_matched, n_dropped (unmatched in A), n_added (unmatched in B),
    and max_conf_delta for matched pairs.
    """
    all_ious: list[float] = []
    n_matched = n_dropped = n_added = 0
    max_conf_delta = 0.0

    for dets_a, dets_b in zip(frames_dets_a, frames_dets_b):
        matches, unmatched_a, unmatched_b = match_detections(dets_a, dets_b, iou_thresh)
        n_matched += len(matches)
        n_dropped += len(unmatched_a)
        n_added += len(unmatched_b)
        for da, db, iou in matches:
            all_ious.append(iou)
            delta = abs(da["confidence"] - db["confidence"])
            if delta > max_conf_delta:
                max_conf_delta = delta

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    return {
        "mean_iou": round(mean_iou, 6),
        "n_matched": n_matched,
        "n_dropped": n_dropped,
        "n_added": n_added,
        "max_conf_delta": round(max_conf_delta, 6),
    }


def compare_decode_plugin(
    frames_decode: list[list[dict]],
    frames_python: list[list[dict]],
    iou_thresh: float = 0.5,
    coord_epsilon: float = 1.0,
) -> dict:
    """Check that decode plugin xywh output matches Python-decode within coord_epsilon pixels.

    Both inputs use {left, top, width, height} format. The CUDA kernel outputs
    [x1, y1, x2-x1, y2-y1, conf, cls] which maps directly to {left, top, width, height},
    identical to parse_yolo26_output(). Any delta reflects FP16 rounding only.
    """
    all_deltas: list[float] = []
    per_coord_max: dict[str, float] = {"left": 0.0, "top": 0.0, "width": 0.0, "height": 0.0}
    n_matched = 0

    for dets_dec, dets_py in zip(frames_decode, frames_python):
        matches, _, _ = match_detections(dets_dec, dets_py, iou_thresh)
        n_matched += len(matches)
        for da, db, _ in matches:
            for key in per_coord_max:
                delta = abs(da[key] - db[key])
                all_deltas.append(delta)
                if delta > per_coord_max[key]:
                    per_coord_max[key] = delta

    deltas_arr = np.array(all_deltas) if all_deltas else np.array([0.0])
    max_delta  = float(deltas_arr.max())
    mean_delta = float(deltas_arr.mean())
    p99_delta  = float(np.percentile(deltas_arr, 99))

    return {
        "n_matched": n_matched,
        "per_coord_max_delta_px": {k: round(v, 6) for k, v in per_coord_max.items()},
        "max_coord_delta_px": round(max_delta, 6),
        "mean_coord_delta_px": round(mean_delta, 6),
        "p99_coord_delta_px": round(p99_delta, 6),
        "coord_epsilon_px": coord_epsilon,
        "within_epsilon": p99_delta < coord_epsilon,
    }


def preprocess_frame(bgr: np.ndarray) -> np.ndarray:
    """Resize BGR frame to 640×640, normalize to [0, 1]; return [1, 3, 640, 640] float32."""
    import cv2
    resized = cv2.resize(bgr, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis]


# ---------------------------------------------------------------------------
# Frame / inference functions (require OpenCV or TRT; not unit-tested)
# ---------------------------------------------------------------------------

def load_mot17_frames(seq_dir: Path, n: int = 50) -> list[np.ndarray]:
    """Load n evenly-spaced JPEG frames from a MOT17 img1/ directory."""
    import cv2
    frame_paths = sorted(seq_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPEG files found in {seq_dir}")
    indices = np.linspace(0, len(frame_paths) - 1, n, dtype=int)
    frames = []
    for idx in indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            raise RuntimeError(f"Failed to read {frame_paths[idx]}")
        frames.append(img)
    return frames


def run_inference(
    engine_path: Path,
    frames: list[np.ndarray],
    plugin_lib: Path | None = None,
) -> list[np.ndarray]:
    """Run TRT inference on BGR frames; return list of raw output tensors per frame."""
    import ctypes

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    if plugin_lib is not None:
        ctypes.CDLL(str(plugin_lib))

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    inp_shape = (1, 3, 640, 640)
    context.set_input_shape(input_name, inp_shape)
    out_shape = tuple(context.get_tensor_shape(output_name))

    inp_nbytes = int(np.prod(inp_shape)) * 4  # float32
    out_size = int(np.prod(np.abs(out_shape)))
    out_nbytes = out_size * 4

    d_input = cuda.mem_alloc(inp_nbytes)
    d_output = cuda.mem_alloc(out_nbytes)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    results = []
    for bgr in frames:
        inp = preprocess_frame(bgr)
        cuda.memcpy_htod_async(d_input, inp.ravel(), stream)
        context.execute_async_v3(stream.handle)
        out = np.empty(out_size, dtype=np.float32)
        cuda.memcpy_dtoh_async(out, d_output, stream)
        stream.synchronize()
        results.append(out.reshape(out_shape))

    return results


def detect_all(
    engine_path: Path,
    frames: list[np.ndarray],
    *,
    is_decode_engine: bool = False,
    plugin_lib: Path | None = None,
    conf_threshold: float = 0.25,
) -> list[list[dict]]:
    """Run engine on BGR frames; return per-frame detection lists in {left,top,width,height} format."""
    raw_outputs = run_inference(engine_path, frames, plugin_lib=plugin_lib)
    all_dets = []
    for raw in raw_outputs:
        if is_decode_engine:
            # Decode plugin output: [1, 300, 6] with [left, top, width, height, conf, cls]
            # (CUDA kernel converts xyxy → xywh as left/top/width/height in pixel space)
            t = raw[0] if raw.ndim == 3 else raw
            dets = [
                {
                    "left": float(row[0]),
                    "top": float(row[1]),
                    "width": float(row[2]),
                    "height": float(row[3]),
                    "confidence": float(row[4]),
                    "class_id": int(row[5]),
                }
                for row in t
                if float(row[4]) >= conf_threshold
            ]
        else:
            dets = parse_yolo26_output(raw, conf_threshold=conf_threshold)
        all_dets.append(dets)
    return all_dets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M2.7.1 accuracy validation harness")
    p.add_argument("--seq-dir", type=Path, required=True, dest="seq_dir",
                   help="MOT17 img1/ directory (e.g. .../MOT17-04-SDP/img1)")
    p.add_argument("--fp32-engine", type=Path, required=True, dest="fp32_engine")
    p.add_argument("--fp16-engine", type=Path, required=True, dest="fp16_engine")
    p.add_argument("--decode-engine", type=Path, default=None, dest="decode_engine",
                   help="FP16+decode engine (optional; skipped if absent)")
    p.add_argument("--plugin-lib", type=Path, default=None, dest="plugin_lib",
                   help="libyolo26_decode.so (required with --decode-engine)")
    p.add_argument("--n-frames", type=int, default=50, dest="n_frames")
    p.add_argument("--conf-threshold", type=float, default=0.25, dest="conf_threshold")
    p.add_argument("--save-json", type=Path, default=None, dest="save_json")
    return p.parse_args(argv)


def _engine_stats(dets: list[list[dict]]) -> dict:
    total = sum(len(f) for f in dets)
    n = len(dets)
    return {
        "total_detections": total,
        "mean_per_frame": round(total / n, 2) if n else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"[validate] Loading {args.n_frames} frames from {args.seq_dir}")
    frames = load_mot17_frames(args.seq_dir, n=args.n_frames)

    print("[validate] Running FP32 engine...")
    dets_fp32 = detect_all(args.fp32_engine, frames, conf_threshold=args.conf_threshold)

    print("[validate] Running FP16 engine...")
    dets_fp16 = detect_all(args.fp16_engine, frames, conf_threshold=args.conf_threshold)

    fp16_vs_fp32 = compare_engines(dets_fp16, dets_fp32)
    stats_fp32 = _engine_stats(dets_fp32)
    stats_fp16 = _engine_stats(dets_fp16)

    total_fp32 = stats_fp32["total_detections"]
    n_matched  = fp16_vs_fp32["n_matched"]
    match_rate = round(n_matched / total_fp32, 4) if total_fp32 else 0.0

    print(
        f"[validate] FP16 vs FP32: mean_iou={fp16_vs_fp32['mean_iou']:.4f}, "
        f"match_rate={match_rate:.4f}, "
        f"n_matched={n_matched}, "
        f"n_dropped={fp16_vs_fp32['n_dropped']}, "
        f"n_added={fp16_vs_fp32['n_added']}, "
        f"max_conf_delta={fp16_vs_fp32['max_conf_delta']:.4f}"
    )

    result: dict = {
        "n_frames": args.n_frames,
        "seq_dir": str(args.seq_dir),
        "engines": {
            "fp32": {"engine": str(args.fp32_engine), **stats_fp32},
            "fp16": {"engine": str(args.fp16_engine), **stats_fp16},
        },
        "fp16_vs_fp32": {**fp16_vs_fp32, "match_rate": match_rate},
    }

    summary: dict = {
        "fp16_vs_fp32": (
            f"mean IoU {fp16_vs_fp32['mean_iou']:.4f} across {n_matched} matched boxes; "
            f"{fp16_vs_fp32['n_dropped']} dropped, {fp16_vs_fp32['n_added']} added "
            f"({total_fp32} FP32 detections, {stats_fp16['total_detections']} FP16); "
            f"max confidence delta {fp16_vs_fp32['max_conf_delta']:.4f}"
        ),
    }

    if args.decode_engine is not None and args.plugin_lib is not None:
        print("[validate] Running FP16+decode engine...")
        dets_decode = detect_all(
            args.decode_engine,
            frames,
            is_decode_engine=True,
            plugin_lib=args.plugin_lib,
            conf_threshold=args.conf_threshold,
        )
        stats_decode = _engine_stats(dets_decode)
        dec = compare_decode_plugin(dets_decode, dets_fp16)

        result["engines"]["decode"] = {"engine": str(args.decode_engine), **stats_decode}
        result["decode_vs_python"] = dec

        max_delta = dec["max_coord_delta_px"]
        pct = round(max_delta / 640 * 100, 3)
        summary["decode_vs_python"] = (
            f"max coordinate delta {max_delta:.4f}px ({pct}% of 640px input) "
            f"over {dec['n_matched']} matched boxes; "
            f"per-coord: left={dec['per_coord_max_delta_px']['left']:.4f}, "
            f"top={dec['per_coord_max_delta_px']['top']:.4f}, "
            f"width={dec['per_coord_max_delta_px']['width']:.4f}, "
            f"height={dec['per_coord_max_delta_px']['height']:.4f}; "
            f"within {dec['coord_epsilon_px']}px tolerance: {dec['within_epsilon']}"
        )
        print(
            f"[validate] Decode vs Python: max_coord_delta={max_delta:.4f}px ({pct}%), "
            f"within_{dec['coord_epsilon_px']}px={dec['within_epsilon']}"
        )
    else:
        print("[validate] Skipping decode plugin comparison (--decode-engine / --plugin-lib not provided)")

    result["summary"] = summary

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2))
        print(f"[validate] Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
