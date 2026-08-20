"""Offline MTMC fusion and WildTrack evaluation."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import motmetrics as mm
import numpy as np

from metrics.evaluate_tracker import build_accumulator, compute_mot_metrics
from metrics.validate_accuracy import match_detections
from metrics.wildtrack_gt import load_wildtrack_mtmc_gt
from pipelines.ground_plane import (
    Homography,
    foot_pixel_sigma,
    foot_point,
    is_foot_reliable,
    load_homography,
    project_point,
    projection_sigma,
)
from pipelines.mtmc import GroundObservation, MtmcConfig, fuse_offline_qualified


@dataclass(frozen=True)
class CameraInput:
    """Explicit source/camera binding for one prediction and homography pair."""

    source_id: int
    camera: str
    prediction_path: Path
    homography_path: Path


def apply_final_id_map(rows: list[dict], path: Path) -> list[dict]:
    """Backfill rows from an online run's authoritative final assignment map."""
    payload = json.loads(path.read_text())
    assignments = payload.get("id_map", payload)
    if not isinstance(assignments, list):
        raise TypeError("final assignment JSON must contain an id_map list")
    id_map = {
        (
            int(item["source_id"]),
            int(item.get("generation", 0)),
            int(item["object_id"]),
        ): int(item["global_id"])
        for item in assignments
    }
    result = []
    for row in rows:
        key = (
            int(row["source_id"]),
            int(row.get("generation", 0)),
            int(row["object_id"]),
        )
        result.append({**row, "global_id": id_map.get(key, int(row.get("global_id", -1)))})
    return result


def assignment_agreement(
    offline_rows: list[dict],
    online_rows: list[dict],
    *,
    warmup_frames: int = 0,
) -> dict[str, int | float]:
    """Compare row assignments after a one-to-one global-ID permutation.

    The two inputs may use different row ordering, but each qualified detection
    key must be unique and present on both sides. Warm-up is measured from each
    tracklet's own first frame. Unassigned rows are ignored only when both sides
    decline an assignment; a missing assignment on one side counts as
    disagreement. The optimal one-to-one label mapping deliberately penalises
    cluster splits and merges.
    """
    if warmup_frames < 0:
        raise ValueError("warmup_frames cannot be negative")

    def index_rows(rows: list[dict], label: str) -> dict[tuple[int, int, int, int], dict]:
        indexed = {}
        for row in rows:
            key = (
                int(row["source_id"]),
                int(row.get("generation", 0)),
                int(row["frame_num"]),
                int(row["object_id"]),
            )
            if key in indexed:
                raise ValueError(f"duplicate {label} assignment row for key {key}")
            indexed[key] = row
        return indexed

    offline_by_key = index_rows(offline_rows, "offline")
    online_by_key = index_rows(online_rows, "online")
    if offline_by_key.keys() != online_by_key.keys():
        raise ValueError("offline and online rows must contain the same detections")

    first_frame: dict[tuple[int, int, int], int] = {}
    for source_id, generation, frame_num, object_id in offline_by_key:
        tracklet = source_id, generation, object_id
        first_frame[tracklet] = min(frame_num, first_frame.get(tracklet, frame_num))

    assigned_pairs: list[tuple[int, int]] = []
    evaluated_rows = 0
    excluded_warmup_rows = 0
    excluded_unassigned_rows = 0
    for key in sorted(offline_by_key):
        source_id, generation, frame_num, object_id = key
        tracklet = source_id, generation, object_id
        if frame_num - first_frame[tracklet] < warmup_frames:
            excluded_warmup_rows += 1
            continue

        offline_id = int(offline_by_key[key].get("global_id", -1))
        online_id = int(online_by_key[key].get("global_id", -1))
        if offline_id < 0 and online_id < 0:
            excluded_unassigned_rows += 1
            continue
        evaluated_rows += 1
        if offline_id >= 0 and online_id >= 0:
            assigned_pairs.append((offline_id, online_id))

    matched_rows = 0
    if assigned_pairs:
        offline_ids = sorted({offline_id for offline_id, _ in assigned_pairs})
        online_ids = sorted({online_id for _, online_id in assigned_pairs})
        offline_index = {global_id: index for index, global_id in enumerate(offline_ids)}
        online_index = {global_id: index for index, global_id in enumerate(online_ids)}
        contingency = np.zeros((len(offline_ids), len(online_ids)), dtype=np.int64)
        for offline_id, online_id in assigned_pairs:
            contingency[offline_index[offline_id], online_index[online_id]] += 1
        offline_matches, online_matches = mm.lap.linear_sum_assignment(-contingency)
        matched_rows = int(contingency[offline_matches, online_matches].sum())

    return {
        "warmup_frames_per_tracklet": int(warmup_frames),
        "input_rows": len(offline_by_key),
        "excluded_warmup_rows": excluded_warmup_rows,
        "excluded_unassigned_rows": excluded_unassigned_rows,
        "evaluated_rows": evaluated_rows,
        "assigned_on_both_rows": len(assigned_pairs),
        "matched_rows": matched_rows,
        "disagreement_rows": evaluated_rows - matched_rows,
        "agreement": matched_rows / evaluated_rows if evaluated_rows else 0.0,
    }


def load_prediction_csvs(inputs: list[CameraInput]) -> list[dict]:
    """Load legacy or identity-extended pipeline CSVs using explicit source bindings."""
    rows = []
    for camera_input in inputs:
        with camera_input.prediction_path.open(newline="") as file:
            for csv_row in csv.DictReader(file):
                object_id = int(csv_row["object_id"])
                class_id = int(csv_row.get("class_id", "0") or 0)
                if class_id != 0:
                    continue
                csv_source = csv_row.get("source_id", "")
                if csv_source not in ("", "-1") and int(csv_source) != camera_input.source_id:
                    raise ValueError(
                        f"{camera_input.prediction_path}: CSV source_id {csv_source} does not "
                        f"match explicit binding {camera_input.source_id}"
                    )
                global_value = csv_row.get("global_id", "-1")
                frame_num = int(csv_row["frame_num"])
                bucket_value = csv_row.get("association_bucket", "")
                accepted_value = csv_row.get("association_accepted", "")
                rows.append(
                    {
                        "frame_num": frame_num,
                        "timestamp_ns": frame_num * 500_000_000,
                        "camera": camera_input.camera,
                        "source_id": camera_input.source_id,
                        "generation": int(csv_row.get("generation", "0") or 0),
                        "association_bucket": (
                            frame_num if bucket_value in (None, "") else int(bucket_value)
                        ),
                        "association_accepted": _parse_csv_bool(
                            accepted_value,
                            default=True,
                        ),
                        "object_id": object_id,
                        "class_id": class_id,
                        "global_id": int(global_value or -1),
                        "left": float(csv_row["left"]),
                        "top": float(csv_row["top"]),
                        "width": float(csv_row["width"]),
                        "height": float(csv_row["height"]),
                    }
                )
    return rows


def _parse_csv_bool(value: object, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes"}:
        return True
    if normalised in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid CSV boolean {value!r}")


def prepare_prediction_rows(inputs: list[CameraInput]) -> list[dict]:
    """Load CSVs, validate calibration bindings, and project reliable foot points."""
    if len({item.source_id for item in inputs}) != len(inputs):
        raise ValueError("camera inputs must use unique source IDs")
    homographies: dict[int, Homography] = {}
    for camera_input in inputs:
        homography = load_homography(camera_input.homography_path)
        if homography.source_id != camera_input.source_id:
            raise ValueError(
                f"{camera_input.homography_path}: source_id {homography.source_id} does not "
                f"match explicit binding {camera_input.source_id}"
            )
        if homography.camera != camera_input.camera:
            raise ValueError(
                f"{camera_input.homography_path}: camera {homography.camera!r} does not "
                f"match explicit binding {camera_input.camera!r}"
            )
        homographies[camera_input.source_id] = homography

    projected_rows = []
    for row in load_prediction_csvs(inputs):
        homography = homographies[int(row["source_id"])]
        reliable = is_foot_reliable(
            row["left"],
            row["top"],
            row["width"],
            row["height"],
            image_width=homography.image_width,
            image_height=homography.image_height,
        )
        projected = {**row, "ground_reliable": reliable}
        if reliable:
            u, v = foot_point(row["left"], row["top"], row["width"], row["height"])
            pixel_sigma = foot_pixel_sigma(
                row["left"],
                row["top"],
                row["width"],
                row["height"],
                image_height=homography.image_height,
            )
            ground_x, ground_y = project_point(homography.matrix, u, v)
            projected.update(
                {
                    "ground_x": ground_x,
                    "ground_y": ground_y,
                    "sigma": projection_sigma(homography.matrix, u, v, pixel_sigma),
                }
            )
        projected_rows.append(projected)
    return projected_rows


def fuse_prediction_rows(
    rows: list[dict],
    config: MtmcConfig | None = None,
) -> list[dict]:
    """Fuse projected rows, deliberately ignoring any CSV global IDs.

    Receipt-backed CSV rows carry the exact accepted mux ``association_bucket``.
    Rejected rows are retained for output labelling but excluded from association
    evidence. Legacy aligned files fall back to ``frame_num`` and accepted=True.
    """
    observations = [
        GroundObservation(
            timestamp_ns=int(row["timestamp_ns"]),
            source_id=int(row["source_id"]),
            object_id=int(row["object_id"]),
            x=float(row["ground_x"]),
            y=float(row["ground_y"]),
            sigma=float(row["sigma"]),
            generation=int(row.get("generation", 0)),
            association_bucket=int(row.get("association_bucket", row["frame_num"])),
        )
        for row in rows
        if row.get("ground_reliable", True)
        and _parse_csv_bool(row.get("association_accepted", True), default=True)
        and "ground_x" in row
        and "ground_y" in row
        and "sigma" in row
    ]
    id_map = fuse_offline_qualified(observations, config or MtmcConfig())
    return [
        {
            **row,
            "global_id": id_map.get(
                (
                    int(row["source_id"]),
                    int(row.get("generation", 0)),
                    int(row["object_id"]),
                ),
                -1,
            ),
        }
        for row in rows
    ]


def sweep_fusion_parameters(
    gt_rows: list[dict],
    pred_rows: list[dict],
    config: MtmcConfig,
    *,
    min_affinities: list[float],
    z_gates: list[float],
    iou_threshold: float = 0.5,
    ground_gate_m: float = 0.5,
) -> list[dict]:
    """Evaluate the Cartesian min-affinity/z-gate fusion grid."""
    results = []
    for min_affinity in min_affinities:
        for z_gate in z_gates:
            sweep_config = replace(
                config,
                min_affinity=float(min_affinity),
                z_gate=float(z_gate),
            )
            fused_rows = fuse_prediction_rows(pred_rows, sweep_config)
            pooled = pooled_image_plane_metrics(
                gt_rows,
                fused_rows,
                iou_threshold=iou_threshold,
            )
            xcam = cross_camera_identity_metrics(
                gt_rows,
                fused_rows,
                iou_threshold=iou_threshold,
            )
            ground = ground_plane_metrics(gt_rows, fused_rows, gate_m=ground_gate_m)
            results.append(
                {
                    "min_affinity": float(min_affinity),
                    "z_gate": float(z_gate),
                    "pooled_IDF1": pooled["IDF1"],
                    "xcam_precision": xcam["xcam_precision"],
                    "xcam_recall": xcam["xcam_recall"],
                    "xcam_f1": xcam["xcam_f1"],
                    "ground_MODA": ground["MODA"],
                    "ground_MODP": ground["MODP"],
                }
            )
    return results


def _pooled_frame(frame_num: int, source_id: int) -> int:
    """Interleave camera frames while keeping each image a separate timestep."""
    if not 0 <= source_id < 10:
        raise ValueError("pooled image-plane metrics require source IDs in [0, 9]")
    return frame_num * 10 + source_id + 1


def _global_identity(row: dict) -> int:
    global_id = int(row.get("global_id", -1))
    if global_id != -1:
        return global_id
    return -(int(row["source_id"]) * 10**7 + int(row["object_id"]))


def pooled_image_plane_metrics(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """Score pooled camera images in a shared cross-camera identity space."""
    pooled_gt = [
        {
            **row,
            "frame": _pooled_frame(int(row["frame_num"]), int(row["source_id"])),
            "obj_id": int(row["person_id"]),
        }
        for row in gt_rows
    ]
    pooled_pred = [
        {
            **row,
            "frame": _pooled_frame(int(row["frame_num"]), int(row["source_id"])),
            "obj_id": _global_identity(row),
        }
        for row in pred_rows
    ]
    accumulator = build_accumulator(pooled_gt, pooled_pred, iou_threshold=iou_threshold)
    return compute_mot_metrics(accumulator)


def image_plane_report(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """Return the local-ID baseline beside pooled shared-ID metrics."""
    per_camera = {}
    for source_id in sorted({int(row["source_id"]) for row in gt_rows}):
        camera_gt = [
            {
                **row,
                "frame": int(row["frame_num"]) + 1,
                "obj_id": int(row["person_id"]),
            }
            for row in gt_rows
            if int(row["source_id"]) == source_id
        ]
        camera_pred = [
            {
                **row,
                "frame": int(row["frame_num"]) + 1,
                "obj_id": int(row["object_id"]),
            }
            for row in pred_rows
            if int(row["source_id"]) == source_id
        ]
        accumulator = build_accumulator(
            camera_gt,
            camera_pred,
            iou_threshold=iou_threshold,
        )
        per_camera[str(source_id)] = compute_mot_metrics(accumulator)

    return {
        "per_camera": per_camera,
        "pooled": pooled_image_plane_metrics(
            gt_rows,
            pred_rows,
            iou_threshold=iou_threshold,
        ),
        "pooled_mota_motp_note": (
            "Pooled MOTA/MOTP remain detection-weighted camera aggregates; only identity "
            "metrics, switches, and fragmentations gain cross-camera meaning."
        ),
    }


def cross_camera_identity_metrics(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> dict:
    """Measure identity agreement for IoU-matched detections across cameras."""
    gt_by_image: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pred_by_image: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in gt_rows:
        gt_by_image[(int(row["frame_num"]), int(row["source_id"]))].append(row)
    for row in pred_rows:
        pred_by_image[(int(row["frame_num"]), int(row["source_id"]))].append(row)

    matched_by_frame: dict[int, list[dict]] = defaultdict(list)
    for (frame_num, source_id), image_gt in gt_by_image.items():
        matches, _, _ = match_detections(
            image_gt,
            pred_by_image.get((frame_num, source_id), []),
            iou_thresh=iou_threshold,
        )
        for gt_row, pred_row, _ in matches:
            matched_by_frame[frame_num].append(
                {
                    "source_id": source_id,
                    "person_id": int(gt_row["person_id"]),
                    "global_id": _global_identity(pred_row),
                }
            )

    tp = fp = fn = 0
    for detections in matched_by_frame.values():
        for first, second in combinations(detections, 2):
            if first["source_id"] == second["source_id"]:
                continue
            same_person = first["person_id"] == second["person_id"]
            same_global = first["global_id"] == second["global_id"]
            if same_person and same_global:
                tp += 1
            elif not same_person and same_global:
                fp += 1
            elif same_person and not same_global:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    cameras_by_global: dict[int, set[int]] = defaultdict(set)
    for row in pred_rows:
        cameras_by_global[_global_identity(row)].add(int(row["source_id"]))
    mean_cameras = (
        sum(len(sources) for sources in cameras_by_global.values()) / len(cameras_by_global)
        if cameras_by_global
        else 0.0
    )

    return {
        "xcam_tp": tp,
        "xcam_fp": fp,
        "xcam_fn": fn,
        "xcam_precision": precision,
        "xcam_recall": recall,
        "xcam_f1": f1,
        "mean_cameras_per_global_id": mean_cameras,
        "n_global_ids": len(cameras_by_global),
        "gt_reference": {
            "mean_cameras_per_global_id": 2.78,
            "n_global_ids": 313,
        },
    }


def ground_plane_metrics(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    gate_m: float = 0.5,
) -> dict:
    """Compute WildTrack MODA/MODP after sigma-weighted multi-view fusion."""
    if gate_m <= 0:
        raise ValueError("gate_m must be positive")

    gt_by_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in gt_rows:
        gt_by_frame[int(row["frame_num"])].setdefault(
            int(row["person_id"]),
            (float(row["world_x"]), float(row["world_y"])),
        )

    views_by_prediction: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in pred_rows:
        if (
            not row.get("ground_reliable", True)
            or "ground_x" not in row
            or "ground_y" not in row
            or "sigma" not in row
        ):
            continue
        views_by_prediction[(int(row["frame_num"]), _global_identity(row))].append(row)

    pred_by_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for (frame_num, global_id), views in views_by_prediction.items():
        sigmas = np.asarray([max(float(view["sigma"]), 1e-9) for view in views])
        weights = 1.0 / np.square(sigmas)
        xs = np.asarray([float(view["ground_x"]) for view in views])
        ys = np.asarray([float(view["ground_y"]) for view in views])
        pred_by_frame[frame_num][global_id] = (
            float(np.average(xs, weights=weights)),
            float(np.average(ys, weights=weights)),
        )

    accumulator = mm.MOTAccumulator(auto_id=True)
    tp = fp = fn = 0
    matched_distances: list[float] = []
    total_gt = sum(len(people) for people in gt_by_frame.values())
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    for frame_num in frames:
        gt_points = gt_by_frame.get(frame_num, {})
        pred_points = pred_by_frame.get(frame_num, {})
        gt_ids = list(gt_points)
        pred_ids = list(pred_points)
        distances = np.full((len(gt_ids), len(pred_ids)), np.nan)
        for gt_idx, gt_id in enumerate(gt_ids):
            for pred_idx, pred_id in enumerate(pred_ids):
                distance = float(
                    np.linalg.norm(
                        np.asarray(gt_points[gt_id]) - np.asarray(pred_points[pred_id])
                    )
                )
                if distance <= gate_m:
                    distances[gt_idx, pred_idx] = distance

        matched_gt, matched_pred = mm.lap.linear_sum_assignment(distances)
        frame_distances = [
            float(distances[gt_idx, pred_idx])
            for gt_idx, pred_idx in zip(matched_gt, matched_pred)
        ]
        tp += len(frame_distances)
        fn += len(gt_ids) - len(frame_distances)
        fp += len(pred_ids) - len(frame_distances)
        matched_distances.extend(frame_distances)
        accumulator.update(gt_ids, pred_ids, distances)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mean_distance = float(np.mean(matched_distances)) if matched_distances else None
    identity_metrics = compute_mot_metrics(accumulator)
    return {
        "gate_m": gate_m,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "MODA": 1.0 - (fp + fn) / total_gt if total_gt else 0.0,
        "MODP": (
            float(np.mean([1.0 - distance / gate_m for distance in matched_distances]))
            if matched_distances
            else 0.0
        ),
        "mean_distance_m": mean_distance,
        "ground_IDF1": identity_metrics["IDF1"],
        "ground_MOTA": identity_metrics["MOTA"],
    }


def evaluate_rows(
    gt_rows: list[dict],
    pred_rows: list[dict],
    *,
    iou_threshold: float = 0.5,
    ground_gates: list[float] | tuple[float, ...] = (0.5, 1.0),
) -> dict:
    """Evaluate one prepared assignment through all MTMC metric families."""
    return {
        "image_plane": image_plane_report(
            gt_rows,
            pred_rows,
            iou_threshold=iou_threshold,
        ),
        "cross_camera": cross_camera_identity_metrics(
            gt_rows,
            pred_rows,
            iou_threshold=iou_threshold,
        ),
        "ground_plane": {
            f"{gate:g}m": ground_plane_metrics(gt_rows, pred_rows, gate_m=gate)
            for gate in ground_gates
        },
    }


def _source_binding(value: str) -> tuple[int, str]:
    try:
        source_text, bound_value = value.split("=", 1)
        source_id = int(source_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected SOURCE_ID=VALUE, got {value!r}"
        ) from exc
    if source_id < 0 or not bound_value:
        raise argparse.ArgumentTypeError(f"invalid source binding {value!r}")
    return source_id, bound_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the positional-parallel camera inputs."""
    parser = argparse.ArgumentParser(description="Evaluate MTMC identities on WildTrack")
    parser.add_argument("--wildtrack-root", type=Path, required=True, dest="wildtrack_root")
    parser.add_argument("--pred", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--camera",
        nargs="+",
        required=True,
        type=_source_binding,
        metavar="SOURCE_ID=CAMERA",
    )
    parser.add_argument(
        "--homography",
        nargs="+",
        required=True,
        type=_source_binding,
        metavar="SOURCE_ID=PATH",
    )
    parser.add_argument("--fuse-offline", action="store_true", dest="fuse_offline")
    parser.add_argument("--final-map", type=Path, default=None, dest="final_map")
    parser.add_argument(
        "--agreement-warmup-frames",
        type=int,
        default=10,
        dest="agreement_warmup_frames",
        help="exclude this many frames from the start of each tracklet",
    )
    parser.add_argument("--z-gate", type=float, default=3.0, dest="z_gate")
    parser.add_argument("--min-affinity", type=float, default=0.5, dest="min_affinity")
    parser.add_argument("--min-cooccurrence", type=int, default=5, dest="min_cooccurrence")
    parser.add_argument("--min-support", type=int, default=3, dest="min_support")
    parser.add_argument("--max-distance-m", type=float, default=2.0, dest="max_distance_m")
    parser.add_argument("--max-sigma-m", type=float, default=1.5, dest="max_sigma_m")
    parser.add_argument("--iou-threshold", type=float, default=0.5, dest="iou_threshold")
    parser.add_argument("--ground-gates", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--sweep-min-affinity",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        dest="sweep_min_affinity",
    )
    parser.add_argument(
        "--sweep-z-gate",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.0],
        dest="sweep_z_gate",
    )
    parser.add_argument("--output-json", type=Path, default=None, dest="output_json")
    parser.add_argument("--wandb-project", default=None, dest="wandb_project")

    args = parser.parse_args(argv)
    counts = (len(args.pred), len(args.camera), len(args.homography))
    if len(set(counts)) != 1:
        parser.error(
            "--pred, --camera, and --homography must contain the same number of values"
        )
    camera_sources = [source_id for source_id, _ in args.camera]
    cameras = [camera for _, camera in args.camera]
    homography_sources = [source_id for source_id, _ in args.homography]
    if len(set(camera_sources)) != len(camera_sources):
        parser.error("--camera source IDs must be unique")
    if len(set(cameras)) != len(cameras):
        parser.error("--camera names must be unique")
    if camera_sources != homography_sources:
        parser.error("--camera and --homography source IDs must match in positional order")
    if args.agreement_warmup_frames < 0:
        parser.error("--agreement-warmup-frames cannot be negative")
    return args


def _log_wandb(report: dict, project: str, output_json: Path | None) -> None:
    import wandb

    run = wandb.init(
        project=project,
        job_type="mtmc_evaluation",
        config=report["association"]["config"],
    )
    pooled = report["metrics"]["image_plane"]["pooled"]
    cross_camera = report["metrics"]["cross_camera"]
    wandb.log(
        {
            "pooled_idf1": pooled["IDF1"],
            "xcam_precision": cross_camera["xcam_precision"],
            "xcam_recall": cross_camera["xcam_recall"],
            "xcam_f1": cross_camera["xcam_f1"],
        }
    )
    if "sweep" in report:
        columns = [
            "min_affinity",
            "z_gate",
            "pooled_IDF1",
            "xcam_precision",
            "xcam_recall",
            "xcam_f1",
            "ground_MODA",
            "ground_MODP",
        ]
        table = wandb.Table(columns=columns)
        for row in report["sweep"]:
            table.add_data(*(row[column] for column in columns))
        wandb.log({"mtmc_parameter_sweep": table})
    if output_json is not None:
        artifact = wandb.Artifact("mtmc-evaluation", type="metrics")
        artifact.add_file(str(output_json))
        run.log_artifact(artifact)
    run.finish()


def main(argv: list[str] | None = None) -> dict:
    """Run offline MTMC evaluation and optionally persist/log its artifact."""
    args = parse_args(argv)
    inputs = [
        CameraInput(
            source_id=source_id,
            camera=camera,
            prediction_path=prediction_path,
            homography_path=Path(homography_path),
        )
        for prediction_path, (source_id, camera), (_, homography_path) in zip(
            args.pred,
            args.camera,
            args.homography,
        )
    ]
    camera_to_source = {item.camera: item.source_id for item in inputs}
    gt_rows = load_wildtrack_mtmc_gt(
        args.wildtrack_root,
        [item.camera for item in inputs],
        camera_to_source,
    )
    projected_rows = prepare_prediction_rows(inputs)
    config = MtmcConfig(
        z_gate=args.z_gate,
        max_distance_m=args.max_distance_m,
        max_sigma_m=args.max_sigma_m,
        min_cooccurrence=args.min_cooccurrence,
        min_support=args.min_support,
        min_affinity=args.min_affinity,
    )

    evaluated_rows = projected_rows
    offline_rows = None
    agreement = None
    mode = "csv_global_id"
    if args.fuse_offline:
        offline_rows = fuse_prediction_rows(projected_rows, config)
        evaluated_rows = offline_rows
        mode = "offline_fusion"
    if args.final_map is not None:
        online_rows = apply_final_id_map(projected_rows, args.final_map)
        if offline_rows is not None:
            agreement = assignment_agreement(
                offline_rows,
                online_rows,
                warmup_frames=args.agreement_warmup_frames,
            )
        evaluated_rows = online_rows
        mode = "online_final_map"

    report = {
        "inputs": [
            {
                "source_id": item.source_id,
                "camera": item.camera,
                "prediction": str(item.prediction_path),
                "homography": str(item.homography_path),
            }
            for item in inputs
        ],
        "association": {
            "mode": mode,
            "final_map_override": str(args.final_map) if args.final_map is not None else None,
            "config": asdict(config),
        },
        "metrics": evaluate_rows(
            gt_rows,
            evaluated_rows,
            iou_threshold=args.iou_threshold,
            ground_gates=args.ground_gates,
        ),
    }
    if agreement is not None:
        report["assignment_agreement"] = agreement
    if args.sweep:
        report["sweep"] = sweep_fusion_parameters(
            gt_rows,
            projected_rows,
            config,
            min_affinities=args.sweep_min_affinity,
            z_gates=args.sweep_z_gate,
            iou_threshold=args.iou_threshold,
            ground_gate_m=args.ground_gates[0],
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"Saved → {args.output_json}")
    else:
        json.dump(report, sys.stdout, indent=2)
        print()

    if args.wandb_project is not None:
        _log_wandb(report, args.wandb_project, args.output_json)
    return report


if __name__ == "__main__":
    main()
