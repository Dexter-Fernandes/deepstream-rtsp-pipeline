"""
M3.2 tracker evaluation harness.

Computes MOTA, MOTP, IDF1, ID-switches, and fragmentations by comparing
pipeline CSV detections against MOT17 ground-truth annotations.

Usage (on host or inside container):
    python3 metrics/evaluate_tracker.py \\
        --gt /media/dexter/PortableSSD/Datasets/MOT17/train/MOT17-04-SDP/gt/gt.txt \\
        --pred metrics/tracker_results/iou/output_stream0.csv \\
        --output-json metrics/results/tracker_metrics_iou.json

GT format (MOT17): frame,id,x,y,w,h,conf,class,visibility  — 1-indexed frames
Pred format:       frame_num,object_id,...,left,top,width,height — 0-indexed frames
"""

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict

import motmetrics as mm
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt(path: str, min_visibility: float = 0.0, class_ids: tuple = (1,)) -> list[dict]:
    """Load MOT17 GT annotations, returning active pedestrian rows only.

    Filters out conf=0 (crowd/ignore regions) and rows below min_visibility.
    Returns list of {frame, obj_id, left, top, width, height}.
    """
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame = int(parts[0])
            obj_id = int(parts[1])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            conf = int(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            if conf == 0:
                continue
            if class_id not in class_ids:
                continue
            if visibility < min_visibility:
                continue

            rows.append({"frame": frame, "obj_id": obj_id,
                         "left": left, "top": top, "width": width, "height": height})
    return rows


def load_predictions(path: str) -> list[dict]:
    """Load pipeline CSV, converting 0-indexed frame_num to 1-indexed.

    Drops rows with object_id=0 (untracked; should not appear after the
    UNTRACKED_OBJECT_ID fix but kept as a guard). Warns if >50% dropped.
    Returns list of {frame, obj_id, left, top, width, height}.
    """
    rows = []
    dropped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            obj_id = int(row["object_id"])
            if obj_id == 0:
                dropped += 1
                continue
            rows.append({
                "frame": int(row["frame_num"]) + 1,  # 0-indexed → 1-indexed
                "obj_id": obj_id,
                "left": float(row["left"]),
                "top": float(row["top"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
            })
    total = len(rows) + dropped
    if total > 0 and dropped / total > 0.5:
        warnings.warn(
            f"load_predictions: {dropped}/{total} rows had object_id=0 and were dropped. "
            "Check that the UNTRACKED_OBJECT_ID fix is in the pipeline.",
            stacklevel=2,
        )
    return rows


# ---------------------------------------------------------------------------
# MOT accumulator
# ---------------------------------------------------------------------------

def _box_iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["left"] + a["width"], a["top"] + a["height"]
    bx2, by2 = b["left"] + b["width"], b["top"] + b["height"]
    iw = max(0.0, min(ax2, bx2) - max(a["left"], b["left"]))
    ih = max(0.0, min(ay2, by2) - max(a["top"], b["top"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def build_accumulator(gt_rows: list[dict], pred_rows: list[dict],
                      iou_threshold: float = 0.5) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator from GT and prediction row lists.

    Uses 1 - IoU as the distance metric; threshold maps to max_d = 1 - iou_threshold.
    """
    acc = mm.MOTAccumulator(auto_id=True)

    gt_by_frame: dict[int, list[dict]] = defaultdict(list)
    pred_by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in gt_rows:
        gt_by_frame[r["frame"]].append(r)
    for r in pred_rows:
        pred_by_frame[r["frame"]].append(r)

    all_frames = sorted(gt_by_frame.keys())
    max_d = 1.0 - iou_threshold

    for frame in all_frames:
        gts = gt_by_frame[frame]
        preds = pred_by_frame.get(frame, [])

        gt_ids = [g["obj_id"] for g in gts]
        pred_ids = [p["obj_id"] for p in preds]

        if gts and preds:
            distances = np.full((len(gts), len(preds)), np.nan)
            for i, g in enumerate(gts):
                for j, p in enumerate(preds):
                    iou = _box_iou(g, p)
                    d = 1.0 - iou
                    if d <= max_d:
                        distances[i, j] = d
        else:
            distances = np.full((len(gts), len(preds)), np.nan)

        acc.update(gt_ids, pred_ids, distances)

    return acc


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def compute_mot_metrics(acc: mm.MOTAccumulator) -> dict:
    """Compute MOTA, MOTP, IDF1, num_switches, num_fragmentations from accumulator."""
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "num_switches", "num_fragmentations"],
        name="summary",
    )
    row = summary.iloc[0]

    def _scalar(v):
        v = float(v)
        return round(v, 6) if not np.isnan(v) else None

    return {
        "MOTA": _scalar(row["mota"]),
        "MOTP": _scalar(row["motp"]),
        "IDF1": _scalar(row["idf1"]),
        "num_switches": int(row["num_switches"]),
        "num_fragmentations": int(row["num_fragmentations"]),
    }


# ---------------------------------------------------------------------------
# No-GT metrics
# ---------------------------------------------------------------------------

def count_unique_tracks(pred_rows: list[dict]) -> int:
    """Number of distinct object_ids — a fragmentation proxy without GT."""
    return len({r["obj_id"] for r in pred_rows})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate tracker CSV against MOT17 GT")
    parser.add_argument("--gt", required=True, help="Path to MOT17 gt.txt")
    parser.add_argument("--pred", required=True, help="Path to pipeline output CSV")
    parser.add_argument("--iou-threshold", type=float, default=0.5, dest="iou_threshold")
    parser.add_argument("--min-visibility", type=float, default=0.0, dest="min_visibility")
    parser.add_argument("--output-json", default=None, dest="output_json",
                        help="Save metrics dict as JSON (prints to stdout if omitted)")
    args = parser.parse_args(argv)

    gt_rows = load_gt(args.gt, min_visibility=args.min_visibility)
    pred_rows = load_predictions(args.pred)

    acc = build_accumulator(gt_rows, pred_rows, iou_threshold=args.iou_threshold)
    metrics = compute_mot_metrics(acc)
    metrics["unique_tracks"] = count_unique_tracks(pred_rows)
    metrics["total_predictions"] = len(pred_rows)
    metrics["total_gt"] = len(gt_rows)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved → {args.output_json}")
    else:
        json.dump(metrics, sys.stdout, indent=2)
        print()

    return metrics


if __name__ == "__main__":
    main()
