"""
Sweep conf_threshold against WildTrack ground truth to maximize F1.

This is inference-only — no training. Runs against the raw PyTorch .pt
weights (models/yolo26n.pt) via ultralytics, NOT the deployed FP16 TRT
engine + custom decode plugin — TensorRT/pycuda aren't set up on this host,
so this trades exact production fidelity for a zero-extra-installs path
(torch/ultralytics/opencv are already in .venv). Numbers here approximate
the deployed pipeline's behavior; they are not identical to it.

YOLO26's one-to-one head is NMS-free, and ultralytics skips its own NMS for
end-to-end models, so conf_threshold is the only free parameter in the
decode path either way. Raw mAP is threshold-invariant by definition (it's
the area under the PR curve over all confidences), so this script reports
mAP as a fixed reference number and sweeps conf_threshold against F1 —
the metric that actually reflects a deployed operating point.

Each image is run through the model exactly once at conf=0.001 to get all
ranked boxes; every threshold in the sweep is then just a filter over that
cached, already-computed output, so the sweep is O(1) inference passes, not
O(n_thresholds).

Usage:
    python3 -m metrics.sweep_conf_threshold \\
        --wildtrack-root /media/dexter/PortableSSD/Datasets/WildTrack/labelled_ds \\
        --weights models/yolo26n.pt \\
        --wandb-project deepstream-rtsp-pipeline \\
        --save-json metrics/results/conf_threshold_sweep.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from metrics.validate_accuracy import match_detections
from metrics.wildtrack_gt import find_wildtrack_pairs, load_wildtrack_gt

_PERSON_CLASS_ID = 0

# ---------------------------------------------------------------------------
# Pure / CPU-safe functions (unit-tested)
# ---------------------------------------------------------------------------


def precision_recall_f1(gt_frames: list[list[dict]], det_frames: list[list[dict]], iou_thresh: float = 0.5) -> dict:
    """Micro-averaged precision/recall/F1 across all frames via greedy IoU matching."""
    tp = fp = fn = 0
    for gt, det in zip(gt_frames, det_frames):
        matches, unmatched_gt, unmatched_det = match_detections(gt, det, iou_thresh)
        tp += len(matches)
        fp += len(unmatched_det)
        fn += len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def sweep_thresholds(
    gt_frames: list[list[dict]],
    raw_det_frames: list[list[dict]],
    thresholds: list[float],
    iou_thresh: float = 0.5,
) -> list[dict]:
    """Filter cached raw (conf_threshold=0) detections at each threshold and score.

    raw_det_frames are already rescaled to native image space and restricted
    to the target class.
    """
    results = []
    for t in thresholds:
        filtered = [[d for d in frame if d["confidence"] >= t] for frame in raw_det_frames]
        metrics = precision_recall_f1(gt_frames, filtered, iou_thresh)
        results.append({"conf_threshold": t, **metrics})
    return results


def best_threshold(sweep_results: list[dict], metric: str = "f1") -> dict:
    return max(sweep_results, key=lambda r: r[metric])


def average_precision(gt_frames: list[list[dict]], raw_det_frames: list[list[dict]], iou_thresh: float = 0.5) -> float:
    """AP via the 101-point interpolated PR curve, ranking all detections by confidence.

    Threshold-invariant by construction — this is the reference mAP number,
    independent of the conf_threshold sweep below.
    """
    scored: list[tuple[float, int, dict]] = [
        (det["confidence"], frame_idx, det) for frame_idx, frame in enumerate(raw_det_frames) for det in frame
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    n_gt = sum(len(g) for g in gt_frames)
    if n_gt == 0 or not scored:
        return 0.0

    matched_gt: dict[int, set[int]] = {i: set() for i in range(len(gt_frames))}
    precisions, recalls = [], []
    tp = fp = 0

    for _, frame_idx, det in tqdm(scored, desc="Scoring detections"):
        gt = gt_frames[frame_idx]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if j in matched_gt[frame_idx]:
                continue
            iou = _iou(det, g)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thresh:
            matched_gt[frame_idx].add(best_j)
            tp += 1
        else:
            fp += 1

        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)

    ap = 0.0
    for r_thresh in np.linspace(0, 1, 101):
        p_at_r = max((p for p, r in zip(precisions, recalls) if r >= r_thresh), default=0.0)
        ap += p_at_r / 101
    return ap


def _iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["left"], a["top"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["left"], b["top"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Torch inference (requires ultralytics/torch; not unit-tested)
# ---------------------------------------------------------------------------


def run_wildtrack_detections(
    weights: Path,
    image_paths: list[Path],
    device: str = "0",
) -> list[list[dict]]:
    """Run the .pt model once per image at conf=0.001, restrict to the person class.

    ultralytics rescales predicted boxes back to each image's native
    resolution internally, so no manual coordinate rescaling is needed here.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))

    # Predicting on a full list of paths with stream=True hits ultralytics'
    # multi-stream code path (sized for concurrent video streams, not a list
    # of still images): warmup allocates a buffer sized off len(sources)
    # rather than batch size, OOMing the 1660Ti's 6GB regardless of an
    # explicit batch=1. Predicting one path at a time avoids it — confirmed
    # by direct reproduction; the model itself stays loaded/warmed across
    # calls, so this doesn't reintroduce per-image warmup cost.
    all_dets = []
    for path in tqdm(image_paths, desc="YOLO inference"):
        result = model.predict(
            str(path),
            conf=0.001,
            imgsz=640,  # matches the deployed engine's fixed 640x640 input
            device=device,
            verbose=False,
        )[0]
        boxes = result.boxes
        dets = [
            {
                "left": float(x1),
                "top": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "confidence": float(conf),
                "class_id": int(cls),
            }
            for (x1, y1, x2, y2), conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
            if int(cls) == _PERSON_CLASS_ID
        ]
        all_dets.append(dets)
    return all_dets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep conf_threshold against WildTrack GT to maximize F1")
    p.add_argument("--wildtrack-root", type=Path, required=True, dest="wildtrack_root")
    p.add_argument("--cameras", nargs="+", default=None, help="Restrict to these cameras (default: all)")
    p.add_argument("--weights", type=Path, default=Path("models/yolo26n.pt"))
    p.add_argument("--device", type=str, default="0", help="ultralytics device string, e.g. '0' or 'cpu'")
    p.add_argument("--iou-thresh", type=float, default=0.5, dest="iou_thresh")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[round(t, 2) for t in np.arange(0.003, 1.0, 0.003)],
    )
    p.add_argument("--wandb-project", type=str, default=None, dest="wandb_project")
    p.add_argument("--save-json", type=Path, default=None, dest="save_json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    pairs = find_wildtrack_pairs(args.wildtrack_root, cameras=args.cameras)
    print(f"[sweep] {len(pairs)} labelled WildTrack images")

    gt_frames = [load_wildtrack_gt(json_path) for _, json_path in tqdm(pairs, desc="Loading GT")]
    image_paths = [image_path for image_path, _ in pairs]

    print(f"[sweep] Running {args.weights.name} once per image (conf=0.001)...")
    raw_det_frames = run_wildtrack_detections(args.weights, image_paths, device=args.device)

    ap = average_precision(gt_frames, raw_det_frames, iou_thresh=args.iou_thresh)
    print(f"[sweep] AP@{args.iou_thresh} = {ap:.4f} (reference mAP, threshold-invariant)")

    results = sweep_thresholds(gt_frames, raw_det_frames, args.thresholds, iou_thresh=args.iou_thresh)
    best = best_threshold(results, metric="f1")
    print(
        f"[sweep] best conf_threshold={best['conf_threshold']:.2f} → "
        f"F1={best['f1']:.4f} (P={best['precision']:.4f}, R={best['recall']:.4f})"
    )

    if args.wandb_project is not None:
        import wandb

        run = wandb.init(project=args.wandb_project, job_type="conf_threshold_sweep", config={
            "n_images": len(pairs),
            "cameras": args.cameras or "all",
            "iou_thresh": args.iou_thresh,
            "weights": str(args.weights),
        })
        table = wandb.Table(columns=["conf_threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        for r in results:
            table.add_data(r["conf_threshold"], r["precision"], r["recall"], r["f1"], r["tp"], r["fp"], r["fn"])
        wandb.log({
            "pr_sweep": table,
            "reference_ap": ap,
            "best_conf_threshold": best["conf_threshold"],
            "best_f1": best["f1"],
            "best_precision": best["precision"],
            "best_recall": best["recall"],
        })
        run.finish()

    output = {
        "n_images": len(pairs),
        "cameras": args.cameras or "all",
        "iou_thresh": args.iou_thresh,
        "reference_ap": ap,
        "sweep": results,
        "best": best,
    }
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(output, indent=2))
        print(f"[sweep] Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
