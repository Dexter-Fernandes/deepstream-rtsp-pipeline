import json
from pathlib import Path

import pytest

from metrics.fit_homography import (
    fit_camera_homography,
    main,
    passes_quality_gate,
    split_rows_by_frame,
)
from pipelines.ground_plane import Homography, project_point


def _synthetic_rows() -> list[dict]:
    rows = []
    for frame_num in range(10):
        frame_shift = frame_num * 3
        for point_index, (base_u, base_v) in enumerate(
            [(100, 200), (300, 210), (120, 500), (330, 520)]
        ):
            u = base_u + frame_shift
            v = base_v + frame_num * 2
            rows.append(
                {
                    "frame_num": frame_num,
                    "person_id": point_index,
                    "left": u - 10.0,
                    "top": v - 50.0,
                    "width": 20.0,
                    "height": 50.0,
                    "world_x": 0.01 * u - 3.0,
                    "world_y": 0.02 * v - 9.0,
                    "image_width": 640,
                    "image_height": 720,
                }
            )
    return rows


def _write_synthetic_labelled_ds(tmp_path: Path, camera: str = "C1") -> Path:
    root = tmp_path / "labelled_ds"
    image_dir = root / "images" / camera
    annotation_dir = root / "gt_labels" / camera
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    for frame_num in range(10):
        objects = []
        for person_id, (base_grid_x, base_grid_y) in enumerate(
            [(100, 100), (200, 110), (120, 200), (220, 210)]
        ):
            grid_x = base_grid_x + frame_num
            grid_y = base_grid_y + frame_num
            u = 2 * grid_x + 50
            v = 3 * grid_y + 20
            objects.append(
                {
                    "classTitle": "pedestrian",
                    "tags": [
                        {"name": "person id", "value": person_id},
                        {"name": "position id", "value": grid_y * 480 + grid_x},
                    ],
                    "points": {
                        "exterior": [[u - 10, v - 60], [u + 10, v]],
                        "interior": [],
                    },
                }
            )
        image_name = f"{camera}_{frame_num * 5:08d}.png"
        (image_dir / image_name).touch()
        (annotation_dir / f"{image_name}.json").write_text(
            json.dumps(
                {
                    "size": {"width": 1000, "height": 1000},
                    "objects": objects,
                }
            )
        )
    return root


def test_split_rows_by_frame_keeps_every_box_from_a_frame_on_one_side():
    rows = [
        {"frame_num": frame_num, "person_id": person_id}
        for frame_num in range(5)
        for person_id in range(2)
    ]

    train_rows, holdout_rows = split_rows_by_frame(rows, holdout_frac=0.4, seed=7)

    train_frames = {row["frame_num"] for row in train_rows}
    holdout_frames = {row["frame_num"] for row in holdout_rows}
    assert train_frames.isdisjoint(holdout_frames)
    assert train_frames | holdout_frames == set(range(5))
    assert len(holdout_frames) == 2


def test_fit_camera_homography_recovers_transform_and_scores_held_out_frames():
    calibration = fit_camera_homography(
        _synthetic_rows(),
        camera="C1",
        source_id=0,
        dataset_root=Path("/dataset/labelled_ds"),
        holdout_frac=0.2,
        seed=3,
        ransac_thresh_m=0.05,
    )

    assert project_point(calibration.matrix, 250.0, 400.0) == pytest.approx((-0.5, -1.0))
    assert calibration.fit["holdout_median_m"] < 1e-5
    assert calibration.fit["holdout_p90_m"] < 1e-5
    assert set(calibration.fit["train_frames"]).isdisjoint(calibration.fit["holdout_frames"])


def test_quality_gate_uses_strict_median_and_p90_limits():
    def calibration(median: float, p90: float) -> Homography:
        return Homography(
            source_id=0,
            camera="C1",
            matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            image_width=1920,
            image_height=1080,
            fit={"holdout_median_m": median, "holdout_p90_m": p90},
        )

    assert passes_quality_gate(calibration(0.099, 0.299))
    assert not passes_quality_gate(calibration(0.100, 0.299))
    assert not passes_quality_gate(calibration(0.099, 0.300))


def test_cli_writes_loadable_camera_config_and_aggregate_provenance(tmp_path: Path):
    root = _write_synthetic_labelled_ds(tmp_path)
    output_dir = tmp_path / "configs"
    summary_path = tmp_path / "fit-summary.json"

    result = main(
        [
            "--wildtrack-root",
            str(root),
            "--cameras",
            "C1",
            "--output-dir",
            str(output_dir),
            "--holdout-frac",
            "0.2",
            "--seed",
            "3",
            "--save-json",
            str(summary_path),
        ]
    )

    config = json.loads((output_dir / "homography_C1.json").read_text())
    summary = json.loads(summary_path.read_text())
    assert result["cameras"]["C1"]["quality_gate_passed"] is True
    assert config["model"] == "single_plane_homography"
    assert config["source_id"] == 0
    assert config["fit"]["dataset_root"] == str(root.resolve())
    assert config["fit"]["n_frames"] == 10
    assert summary["cameras"]["C1"]["quality_gate_passed"] is True


def test_cli_does_not_publish_a_calibration_that_fails_the_quality_gate(tmp_path: Path):
    root = _write_synthetic_labelled_ds(tmp_path)
    output_dir = tmp_path / "configs"

    with pytest.raises(SystemExit, match="quality gate failed for: C1"):
        main(
            [
                "--wildtrack-root",
                str(root),
                "--cameras",
                "C1",
                "--output-dir",
                str(output_dir),
                "--median-limit-m",
                "0",
            ]
        )

    assert not (output_dir / "homography_C1.json").exists()


def test_c2_fit_provenance_documents_its_expected_lower_inlier_ratio():
    calibration = fit_camera_homography(
        _synthetic_rows(),
        camera="C2",
        source_id=1,
        dataset_root=Path("/dataset/labelled_ds"),
        seed=3,
    )

    assert "lower RANSAC inlier ratio" in calibration.fit["note"]
