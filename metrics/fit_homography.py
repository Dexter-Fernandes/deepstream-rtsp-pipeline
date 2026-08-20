"""Fit WildTrack image-pixel to world-metre homographies with frame holdout gates."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from metrics.wildtrack_gt import load_wildtrack_mtmc_gt
from pipelines.ground_plane import Homography, foot_point, is_foot_reliable, project_point

DEFAULT_CAMERA_TO_SOURCE = {f"C{camera_number}": camera_number - 1 for camera_number in range(1, 8)}


def split_rows_by_frame(
    rows: list[dict],
    *,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split observations without leaking boxes from one frame across the split."""
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be between 0 and 1")
    frames = np.asarray(sorted({int(row["frame_num"]) for row in rows}), dtype=int)
    if len(frames) < 2:
        raise ValueError("at least two distinct frames are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(frames)
    n_holdout = min(len(frames) - 1, max(1, round(len(frames) * holdout_frac)))
    holdout_frames = set(frames[:n_holdout].tolist())
    train_rows = [row for row in rows if int(row["frame_num"]) not in holdout_frames]
    holdout_rows = [row for row in rows if int(row["frame_num"]) in holdout_frames]
    return train_rows, holdout_rows


def _projection_errors(matrix: np.ndarray, rows: list[dict]) -> np.ndarray:
    errors = []
    for row in rows:
        u, v = foot_point(row["left"], row["top"], row["width"], row["height"])
        predicted = np.asarray(project_point(matrix, u, v))
        expected = np.asarray((row["world_x"], row["world_y"]))
        errors.append(np.linalg.norm(predicted - expected))
    return np.asarray(errors)


def fit_camera_homography(
    rows: list[dict],
    *,
    camera: str,
    source_id: int,
    dataset_root: Path,
    holdout_frac: float = 0.2,
    seed: int = 0,
    ransac_thresh_m: float = 0.10,
) -> Homography:
    """Fit one camera calibration and measure it only on unseen frames."""
    usable_rows = [
        row
        for row in rows
        if is_foot_reliable(
            row["left"],
            row["top"],
            row["width"],
            row["height"],
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
        )
    ]
    train_rows, holdout_rows = split_rows_by_frame(
        usable_rows,
        holdout_frac=holdout_frac,
        seed=seed,
    )
    if len(train_rows) < 4:
        raise ValueError(f"{camera} needs at least four reliable training points")

    image_sizes = {
        (int(row["image_width"]), int(row["image_height"])) for row in usable_rows
    }
    if len(image_sizes) != 1:
        raise ValueError(f"{camera} annotations contain inconsistent image sizes")
    image_width, image_height = image_sizes.pop()

    image_points = np.asarray(
        [foot_point(row["left"], row["top"], row["width"], row["height"]) for row in train_rows],
        dtype=np.float64,
    )
    world_points = np.asarray(
        [(row["world_x"], row["world_y"]) for row in train_rows],
        dtype=np.float64,
    )
    cv2.setRNGSeed(seed)
    matrix, inlier_mask = cv2.findHomography(
        image_points,
        world_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_m,
    )
    if matrix is None:
        raise ValueError(f"OpenCV could not fit a homography for {camera}")
    if not np.isfinite(matrix).all() or matrix[2, 2] == 0.0:
        raise ValueError(f"OpenCV returned a degenerate homography for {camera}")
    matrix = matrix / matrix[2, 2]

    holdout_errors = _projection_errors(matrix, holdout_rows)
    train_frames = sorted({int(row["frame_num"]) for row in train_rows})
    holdout_frames = sorted({int(row["frame_num"]) for row in holdout_rows})
    fit = {
        "n_points": len(usable_rows),
        "n_frames": len(train_frames) + len(holdout_frames),
        "n_rejected_points": len(rows) - len(usable_rows),
        "n_train_points": len(train_rows),
        "n_holdout_points": len(holdout_rows),
        "n_train_frames": len(train_frames),
        "n_holdout_frames": len(holdout_frames),
        "train_frames": train_frames,
        "holdout_frames": holdout_frames,
        "inlier_ratio": float(np.mean(inlier_mask)),
        "holdout_median_m": float(np.median(holdout_errors)),
        "holdout_p90_m": float(np.percentile(holdout_errors, 90)),
        "holdout_frac": holdout_frac,
        "seed": seed,
        "ransac_thresh_m": ransac_thresh_m,
        "dataset_root": str(Path(dataset_root).resolve()),
        "model": "single_plane_homography",
    }
    if camera == "C2":
        fit["note"] = (
            "C2 is expected to have a lower RANSAC inlier ratio than C1/C3 because its "
            "viewpoint contains more truncated pedestrians."
        )
    return Homography(
        source_id=source_id,
        camera=camera,
        matrix=matrix,
        image_width=image_width,
        image_height=image_height,
        fit=fit,
    )


def passes_quality_gate(
    calibration: Homography,
    *,
    median_limit_m: float = 0.10,
    p90_limit_m: float = 0.30,
) -> bool:
    """Return whether held-out projection error passes both strict limits."""
    if calibration.fit is None:
        return False
    return (
        float(calibration.fit["holdout_median_m"]) < median_limit_m
        and float(calibration.fit["holdout_p90_m"]) < p90_limit_m
    )


def homography_to_dict(calibration: Homography) -> dict:
    """Convert a calibration to its portable JSON artifact representation."""
    return {
        "source_id": calibration.source_id,
        "camera": calibration.camera,
        "matrix": calibration.matrix.tolist(),
        "image_width": calibration.image_width,
        "image_height": calibration.image_height,
        "model": calibration.model,
        "fit": calibration.fit,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit WildTrack image-to-ground homographies")
    parser.add_argument("--wildtrack-root", type=Path, required=True, dest="wildtrack_root")
    parser.add_argument("--cameras", nargs="+", default=["C1", "C2", "C3"])
    parser.add_argument("--output-dir", type=Path, default=Path("configs"), dest="output_dir")
    parser.add_argument("--holdout-frac", type=float, default=0.2, dest="holdout_frac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ransac-thresh-m",
        type=float,
        default=0.10,
        dest="ransac_thresh_m",
    )
    parser.add_argument("--median-limit-m", type=float, default=0.10, dest="median_limit_m")
    parser.add_argument("--p90-limit-m", type=float, default=0.30, dest="p90_limit_m")
    parser.add_argument("--save-json", type=Path, default=None, dest="save_json")
    parser.add_argument("--wandb-project", type=str, default=None, dest="wandb_project")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    unknown_cameras = set(args.cameras) - DEFAULT_CAMERA_TO_SOURCE.keys()
    if unknown_cameras:
        unknown = ", ".join(sorted(unknown_cameras))
        raise ValueError(f"no source ID binding configured for camera(s): {unknown}")
    if len(set(args.cameras)) != len(args.cameras):
        raise ValueError("--cameras must not contain duplicates")

    camera_to_source = {camera: DEFAULT_CAMERA_TO_SOURCE[camera] for camera in args.cameras}
    rows = load_wildtrack_mtmc_gt(args.wildtrack_root, args.cameras, camera_to_source)
    calibrations = []
    for camera in tqdm(args.cameras, desc="Fitting homographies"):
        camera_rows = [row for row in rows if row["camera"] == camera]
        calibrations.append(
            fit_camera_homography(
                camera_rows,
                camera=camera,
                source_id=camera_to_source[camera],
                dataset_root=args.wildtrack_root,
                holdout_frac=args.holdout_frac,
                seed=args.seed,
                ransac_thresh_m=args.ransac_thresh_m,
            )
        )

    gate_results = {
        calibration.camera: passes_quality_gate(
            calibration,
            median_limit_m=args.median_limit_m,
            p90_limit_m=args.p90_limit_m,
        )
        for calibration in calibrations
    }
    result = {
        "model": "single_plane_homography",
        "dataset_root": str(args.wildtrack_root.resolve()),
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "ransac_thresh_m": args.ransac_thresh_m,
        "quality_gate": {
            "holdout_median_lt_m": args.median_limit_m,
            "holdout_p90_lt_m": args.p90_limit_m,
        },
        "cameras": {
            calibration.camera: {
                **calibration.fit,
                "source_id": calibration.source_id,
                "quality_gate_passed": gate_results[calibration.camera],
            }
            for calibration in calibrations
        },
    }

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2) + "\n")

    if args.wandb_project is not None:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            job_type="homography_fit",
            config={
                "dataset_root": str(args.wildtrack_root),
                "cameras": args.cameras,
                "holdout_frac": args.holdout_frac,
                "seed": args.seed,
                "ransac_thresh_m": args.ransac_thresh_m,
            },
        )
        table = wandb.Table(
            columns=["camera", "inlier_ratio", "holdout_median_m", "holdout_p90_m", "passed"]
        )
        for calibration in calibrations:
            table.add_data(
                calibration.camera,
                calibration.fit["inlier_ratio"],
                calibration.fit["holdout_median_m"],
                calibration.fit["holdout_p90_m"],
                gate_results[calibration.camera],
            )
        wandb.log({"homography_quality": table})
        run.finish()

    failures = [camera for camera, passed in gate_results.items() if not passed]
    if failures:
        failed = ", ".join(failures)
        raise SystemExit(f"homography quality gate failed for: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for calibration in calibrations:
        output_path = args.output_dir / f"homography_{calibration.camera}.json"
        output_path.write_text(json.dumps(homography_to_dict(calibration), indent=2) + "\n")
        print(
            f"[homography] {calibration.camera}: "
            f"median={calibration.fit['holdout_median_m']:.4f} m, "
            f"p90={calibration.fit['holdout_p90_m']:.4f} m -> {output_path}"
        )
    return result


if __name__ == "__main__":
    main()
