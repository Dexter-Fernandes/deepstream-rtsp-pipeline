import json

import pytest

from metrics.csv_sink import CsvSink
from metrics.evaluate_mtmc import (
    CameraInput,
    apply_final_id_map,
    assignment_agreement,
    cross_camera_identity_metrics,
    evaluate_rows,
    fuse_prediction_rows,
    ground_plane_metrics,
    image_plane_report,
    load_prediction_csvs,
    main,
    parse_args,
    pooled_image_plane_metrics,
    prepare_prediction_rows,
    sweep_fusion_parameters,
)
from pipelines.metadata_parser import Detection
from pipelines.mtmc import GroundObservation, MtmcConfig, MtmcFrame
from pipelines.mtmc_runtime import MtmcRuntime


def _write_csv(path, header, rows):
    path.write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(str(value) for value in row) for row in rows)
        + "\n"
    )


def test_cli_rejects_misaligned_parallel_camera_inputs(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--wildtrack-root",
                str(tmp_path),
                "--pred",
                "camera-0.csv",
                "camera-1.csv",
                "--camera",
                "0=C1",
                "1=C2",
                "--homography",
                "0=homography-C1.json",
            ]
        )


def test_cli_requires_explicit_matching_source_bindings(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--wildtrack-root",
                str(tmp_path),
                "--pred",
                "camera.csv",
                "--camera",
                "C1",
                "--homography",
                "0=homography.json",
            ]
        )


def test_cli_accepts_offline_fusion_sweep_and_artifact_options(tmp_path):
    args = parse_args(
        [
            "--wildtrack-root",
            str(tmp_path),
            "--pred",
            "camera.csv",
            "--camera",
            "0=C1",
            "--homography",
            "0=homography.json",
            "--fuse-offline",
            "--final-map",
            "final.json",
            "--agreement-warmup-frames",
            "12",
            "--sweep-min-affinity",
            "0.5",
            "0.8",
            "--sweep-z-gate",
            "2.0",
            "3.0",
            "--ground-gates",
            "0.5",
            "1.0",
            "--output-json",
            "result.json",
            "--wandb-project",
            "mtmc",
        ]
    )

    assert args.fuse_offline is True
    assert args.agreement_warmup_frames == 12
    assert args.sweep_min_affinity == [0.5, 0.8]
    assert args.sweep_z_gate == [2.0, 3.0]
    assert args.ground_gates == [0.5, 1.0]
    assert args.output_json.name == "result.json"


def test_cli_rejects_duplicate_camera_bindings(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--wildtrack-root",
                str(tmp_path),
                "--pred",
                "camera-0.csv",
                "camera-1.csv",
                "--camera",
                "0=C1",
                "1=C1",
                "--homography",
                "0=h0.json",
                "1=h1.json",
            ]
        )


def test_pooled_metrics_treat_unfused_tracks_as_camera_local_singletons():
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "person_id": 7,
            "left": 0.0,
            "top": 0.0,
            "width": 10.0,
            "height": 20.0,
        }
        for source_id in (0, 1)
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "object_id": 100 + source_id,
            "global_id": -1,
            "left": 0.0,
            "top": 0.0,
            "width": 10.0,
            "height": 20.0,
        }
        for source_id in (0, 1)
    ]

    metrics = pooled_image_plane_metrics(gt_rows, pred_rows)

    assert metrics["IDF1"] == pytest.approx(0.5)


def test_image_plane_report_compares_local_tracker_ids_with_shared_global_ids():
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "person_id": 7,
            "left": 0.0,
            "top": 0.0,
            "width": 10.0,
            "height": 20.0,
        }
        for source_id in (0, 1)
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "object_id": 100 + source_id,
            "global_id": 55,
            "left": 0.0,
            "top": 0.0,
            "width": 10.0,
            "height": 20.0,
        }
        for source_id in (0, 1)
    ]

    report = image_plane_report(gt_rows, pred_rows)

    assert report["per_camera"]["0"]["IDF1"] == pytest.approx(1.0)
    assert report["per_camera"]["1"]["IDF1"] == pytest.approx(1.0)
    assert report["pooled"]["IDF1"] == pytest.approx(1.0)
    assert "detection-weighted" in report["pooled_mota_motp_note"]


def test_cross_camera_identity_metrics_count_correct_split_and_false_merges():
    gt_rows = []
    pred_rows = []
    cases = [
        (0, (1, 1), (10, 10)),  # correct fusion
        (1, (1, 1), (10, 20)),  # false split
        (2, (1, 2), (30, 30)),  # false merge
    ]
    for frame_num, person_ids, global_ids in cases:
        for source_id in (0, 1):
            box = {"left": 0.0, "top": 0.0, "width": 10.0, "height": 20.0}
            gt_rows.append(
                {
                    "frame_num": frame_num,
                    "source_id": source_id,
                    "person_id": person_ids[source_id],
                    **box,
                }
            )
            pred_rows.append(
                {
                    "frame_num": frame_num,
                    "source_id": source_id,
                    "object_id": frame_num * 10 + source_id + 1,
                    "global_id": global_ids[source_id],
                    **box,
                }
            )

    result = cross_camera_identity_metrics(gt_rows, pred_rows)

    assert result["xcam_tp"] == 1
    assert result["xcam_fp"] == 1
    assert result["xcam_fn"] == 1
    assert result["xcam_precision"] == pytest.approx(0.5)
    assert result["xcam_recall"] == pytest.approx(0.5)
    assert result["xcam_f1"] == pytest.approx(0.5)
    assert result["n_global_ids"] == 3
    assert result["mean_cameras_per_global_id"] == pytest.approx(5 / 3)
    assert result["gt_reference"] == {
        "mean_cameras_per_global_id": 2.78,
        "n_global_ids": 313,
    }


def test_ground_plane_metrics_sigma_weight_multiview_positions():
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "person_id": 7,
            "world_x": 0.0,
            "world_y": 0.0,
        }
        for source_id in (0, 1)
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "source_id": 0,
            "object_id": 10,
            "global_id": 99,
            "ground_x": 0.2,
            "ground_y": 0.0,
            "sigma": 1.0,
        },
        {
            "frame_num": 0,
            "source_id": 1,
            "object_id": 20,
            "global_id": 99,
            "ground_x": 0.0,
            "ground_y": 0.0,
            "sigma": 0.5,
        },
    ]

    result = ground_plane_metrics(gt_rows, pred_rows, gate_m=0.5)

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["mean_distance_m"] == pytest.approx(0.04)
    assert result["MODA"] == pytest.approx(1.0)
    assert result["MODP"] == pytest.approx(0.92)


def test_ground_plane_metrics_exclude_unreliable_unprojected_detections():
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": 0,
            "person_id": 7,
            "world_x": 0.0,
            "world_y": 0.0,
        }
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "source_id": 0,
            "object_id": 10,
            "global_id": -1,
            "ground_reliable": False,
        }
    ]

    result = ground_plane_metrics(gt_rows, pred_rows)

    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_prediction_loader_accepts_legacy_and_global_id_csvs(tmp_path):
    legacy_path = tmp_path / "legacy.csv"
    current_path = tmp_path / "current.csv"
    base_header = ["frame_num", "object_id", "left", "top", "width", "height"]
    _write_csv(legacy_path, base_header, [[0, 10, 1, 2, 3, 4]])
    _write_csv(
        current_path,
        [
            *base_header,
            "source_id",
            "global_id",
            "generation",
            "association_bucket",
            "association_accepted",
        ],
        [[0, 20, 5, 6, 7, 8, 1, 88, 3, 17, 0]],
    )
    inputs = [
        CameraInput(0, "C1", legacy_path, tmp_path / "h0.json"),
        CameraInput(1, "C2", current_path, tmp_path / "h1.json"),
    ]

    rows = load_prediction_csvs(inputs)

    assert [
        (
            row["source_id"],
            row["object_id"],
            row["global_id"],
            row["generation"],
            row["association_bucket"],
            row["association_accepted"],
        )
        for row in rows
    ] == [
        (0, 10, -1, 0, 0, True),
        (1, 20, 88, 3, 17, False),
    ]


def test_prediction_loader_uses_the_same_person_only_scope_as_online_fusion(tmp_path):
    prediction_path = tmp_path / "predictions.csv"
    _write_csv(
        prediction_path,
        ["frame_num", "object_id", "class_id", "left", "top", "width", "height"],
        [
            [0, 0, 0, 1, 2, 3, 4],
            [0, 20, 26, 5, 6, 7, 8],
        ],
    )
    inputs = [CameraInput(0, "C1", prediction_path, tmp_path / "h0.json")]

    rows = load_prediction_csvs(inputs)

    assert [(row["object_id"], row["class_id"]) for row in rows] == [(0, 0)]


def test_final_assignment_map_overrides_online_warmup_ids(tmp_path):
    final_map_path = tmp_path / "final-map.json"
    final_map_path.write_text(
        '{"metadata": {"run": "online"}, '
        '"id_map": [{"source_id": 0, "generation": 0, '
        '"object_id": 10, "global_id": 77}]}'
    )
    rows = [
        {
            "source_id": 0,
            "generation": 0,
            "object_id": 10,
            "global_id": -1,
        }
    ]

    overridden = apply_final_id_map(rows, final_map_path)

    assert overridden[0]["global_id"] == 77
    assert rows[0]["global_id"] == -1


def test_assignment_agreement_is_id_permutation_invariant_after_warmup():
    offline_rows = [
        {"frame_num": 0, "source_id": 0, "object_id": 1, "global_id": 10},
        {"frame_num": 0, "source_id": 1, "object_id": 2, "global_id": 20},
        {"frame_num": 1, "source_id": 0, "object_id": 1, "global_id": 10},
        {"frame_num": 1, "source_id": 1, "object_id": 2, "global_id": 20},
    ]
    online_rows = [
        {"frame_num": 0, "source_id": 0, "object_id": 1, "global_id": -1},
        {"frame_num": 0, "source_id": 1, "object_id": 2, "global_id": -1},
        {"frame_num": 1, "source_id": 0, "object_id": 1, "global_id": 900},
        {"frame_num": 1, "source_id": 1, "object_id": 2, "global_id": 800},
    ]

    report = assignment_agreement(offline_rows, online_rows, warmup_frames=1)

    assert report == {
        "warmup_frames_per_tracklet": 1,
        "input_rows": 4,
        "excluded_warmup_rows": 2,
        "excluded_unassigned_rows": 0,
        "evaluated_rows": 2,
        "assigned_on_both_rows": 2,
        "matched_rows": 2,
        "disagreement_rows": 0,
        "agreement": 1.0,
    }


def test_assignment_warmup_starts_at_each_tracklets_first_frame():
    offline_rows = [
        {"frame_num": 0, "source_id": 0, "object_id": 1, "global_id": 10},
        {"frame_num": 1, "source_id": 0, "object_id": 1, "global_id": 10},
        {"frame_num": 100, "source_id": 1, "object_id": 2, "global_id": 20},
        {"frame_num": 101, "source_id": 1, "object_id": 2, "global_id": 20},
    ]
    online_rows = [
        {"frame_num": 101, "source_id": 1, "object_id": 2, "global_id": 800},
        {"frame_num": 100, "source_id": 1, "object_id": 2, "global_id": -1},
        {"frame_num": 1, "source_id": 0, "object_id": 1, "global_id": 900},
        {"frame_num": 0, "source_id": 0, "object_id": 1, "global_id": -1},
    ]

    report = assignment_agreement(offline_rows, online_rows, warmup_frames=1)

    assert report["excluded_warmup_rows"] == 2
    assert report["evaluated_rows"] == 2
    assert report["agreement"] == 1.0


def test_assignment_agreement_penalises_cluster_fragmentation_one_to_one():
    offline_rows = [
        {"frame_num": frame, "source_id": 0, "object_id": frame, "global_id": global_id}
        for frame, global_id in enumerate((10, 10, 20, 20))
    ]
    online_rows = [
        {"frame_num": frame, "source_id": 0, "object_id": frame, "global_id": global_id}
        for frame, global_id in enumerate((900, 900, 900, 800))
    ]

    report = assignment_agreement(offline_rows, online_rows)

    assert report["matched_rows"] == 3
    assert report["disagreement_rows"] == 1
    assert report["agreement"] == pytest.approx(0.75)


def test_assignment_agreement_rejects_duplicate_detection_keys():
    row = {"frame_num": 0, "source_id": 0, "object_id": 1, "global_id": 10}

    with pytest.raises(ValueError, match="duplicate offline assignment row"):
        assignment_agreement([row, row], [row])


def test_offline_fusion_ignores_csv_global_ids_and_uses_tracker_observations():
    rows = []
    for frame_num in range(5):
        for source_id, object_id, stale_global_id in ((0, 10, 501), (1, 20, 602)):
            rows.append(
                {
                    "frame_num": frame_num,
                    "timestamp_ns": frame_num * 500_000_000,
                    "source_id": source_id,
                    "object_id": object_id,
                    "global_id": stale_global_id,
                    "ground_x": frame_num * 0.1,
                    "ground_y": 0.0,
                    "sigma": 0.1,
                }
            )

    fused = fuse_prediction_rows(rows, MtmcConfig())

    assert {row["global_id"] for row in fused} == {1}
    assert {row["global_id"] for row in rows} == {501, 602}


def test_recycled_tracker_id_stays_generation_qualified_from_csv_to_final_agreement(
    tmp_path,
):
    prediction_paths = [tmp_path / f"predictions-{source_id}.csv" for source_id in range(3)]
    detections_by_source = [
        [
            Detection(0, 7, 0, "person", 0.9, 1, 2, 3, 4, source_id=0, generation=0),
            Detection(1, 7, 0, "person", 0.9, 1, 2, 3, 4, source_id=0, generation=1),
        ],
        [Detection(0, 8, 0, "person", 0.9, 1, 2, 3, 4, source_id=1)],
        [Detection(1, 9, 0, "person", 0.9, 1, 2, 3, 4, source_id=2)],
    ]
    for prediction_path, detections in zip(prediction_paths, detections_by_source):
        sink = CsvSink(prediction_path)
        sink.write(detections)
        sink.close()
    inputs = [
        CameraInput(source_id, f"C{source_id + 1}", path, tmp_path / f"h{source_id}.json")
        for source_id, path in enumerate(prediction_paths)
    ]
    projected_rows = [
        {
            **row,
            "ground_x": float(row["frame_num"] * 10),
            "ground_y": 0.0,
            "sigma": 0.1,
            "ground_reliable": True,
        }
        for row in load_prediction_csvs(inputs)
    ]
    config = MtmcConfig(
        max_distance_m=1.0,
        min_cooccurrence=1,
        min_support=1,
        min_affinity=1.0,
    )
    offline_rows = fuse_prediction_rows(projected_rows, config)
    final_map_path = tmp_path / "mtmc.json"
    final_map_path.write_text(
        json.dumps(
            {
                "id_map": [
                    {"source_id": 0, "generation": 0, "object_id": 7, "global_id": 101},
                    {"source_id": 1, "generation": 0, "object_id": 8, "global_id": 101},
                    {"source_id": 0, "generation": 1, "object_id": 7, "global_id": 202},
                    {"source_id": 2, "generation": 0, "object_id": 9, "global_id": 202},
                ]
            }
        )
    )
    online_rows = apply_final_id_map(projected_rows, final_map_path)

    source_zero_ids = {
        row["generation"]: row["global_id"]
        for row in offline_rows
        if row["source_id"] == 0
    }
    agreement = assignment_agreement(offline_rows, online_rows)

    assert source_zero_ids[0] != source_zero_ids[1]
    assert agreement["matched_rows"] == 4
    assert agreement["agreement"] == 1.0


def test_offline_frame_batches_match_atomic_online_across_timestamp_bucket_boundaries(
    tmp_path,
):
    config = MtmcConfig(
        sync_bucket_ns=33_000_000,
        max_skew_ns=100_000_000,
        min_cooccurrence=6,
        min_support=6,
        min_affinity=1.0,
        reassign_interval=10,
    )
    prediction_paths = [tmp_path / f"offset-{source_id}.csv" for source_id in (0, 1)]
    sinks = [CsvSink(path) for path in prediction_paths]
    runtime = MtmcRuntime(config, expected_sources={0, 1})
    runtime.start()
    output = tmp_path / "online.json"
    try:
        for frame_num in range(6):
            source_0_timestamp = 32_000_000 + frame_num * 16_683_350
            source_1_timestamp = source_0_timestamp + 9_744_450
            frames = []
            for source_id, object_id, timestamp_ns, source_frame_num in (
                (0, 10, source_0_timestamp, frame_num),
                (1, 20, source_1_timestamp, frame_num + 100),
            ):
                observation = GroundObservation(
                    timestamp_ns, source_id, object_id, 1.0, 2.0, 0.1
                )
                frames.append(MtmcFrame(timestamp_ns, source_id, (observation,)))
            receipt = runtime.submit_batch(frames)
            assert receipt.attempt_epoch == frame_num + 1
            assert receipt.association_bucket == frame_num + 1
            assert receipt.association_accepted is True
            for source_id, object_id, source_frame_num in (
                (0, 10, frame_num),
                (1, 20, frame_num + 100),
            ):
                sinks[source_id].write(
                    [
                        Detection(
                            source_frame_num,
                            object_id,
                            0,
                            "person",
                            0.9,
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            source_id=source_id,
                            generation=receipt.identity_snapshot.generations[source_id],
                            association_bucket=receipt.association_bucket,
                            association_accepted=receipt.association_accepted,
                        )
                    ]
                )
    finally:
        for sink in sinks:
            sink.close()
        runtime.shutdown(json_path=output)

    inputs = [
        CameraInput(source_id, f"C{source_id + 1}", path, tmp_path / f"h{source_id}.json")
        for source_id, path in enumerate(prediction_paths)
    ]
    projected_rows = [
        {
            **row,
            "ground_x": 1.0,
            "ground_y": 2.0,
            "sigma": 0.1,
            "ground_reliable": True,
        }
        for row in load_prediction_csvs(inputs)
    ]
    offline_rows = fuse_prediction_rows(projected_rows, config)
    online_rows = apply_final_id_map(projected_rows, output)

    agreement = assignment_agreement(offline_rows, online_rows)

    assert agreement["matched_rows"] == 12
    assert agreement["agreement"] == 1.0


def test_rejected_receipt_rows_are_labelled_but_do_not_add_offline_evidence():
    runtime = MtmcRuntime(
        MtmcConfig(min_cooccurrence=1, min_support=1, min_affinity=1.0),
        expected_sources={0, 1, 2},
    )
    runtime.start()
    try:
        accepted_receipt = runtime.submit_batch(
            (
                MtmcFrame(0, 0, (GroundObservation(0, 0, 10, 1.0, 2.0, 0.1),)),
                MtmcFrame(0, 1, (GroundObservation(0, 1, 20, 1.0, 2.0, 0.1),)),
                MtmcFrame(0, 2, ()),
            )
        )
        rejected_receipt = runtime.advance_liveness()
    finally:
        runtime.shutdown()

    assert accepted_receipt.association_accepted is True
    assert accepted_receipt.association_bucket == 1
    assert rejected_receipt.association_accepted is False
    assert rejected_receipt.association_bucket is None
    assert rejected_receipt.attempt_epoch == 2
    rows = [
        {
            "frame_num": frame_num,
            "timestamp_ns": frame_num * 500_000_000,
            "source_id": source_id,
            "generation": 0,
            "object_id": object_id,
            "ground_x": 1.0,
            "ground_y": 2.0,
            "sigma": 0.1,
            "ground_reliable": True,
            "association_bucket": association_bucket,
            "association_accepted": accepted,
        }
        for frame_num, source_id, object_id, association_bucket, accepted in (
            (
                0,
                0,
                10,
                accepted_receipt.association_bucket,
                accepted_receipt.association_accepted,
            ),
            (
                0,
                1,
                20,
                accepted_receipt.association_bucket,
                accepted_receipt.association_accepted,
            ),
            (
                1,
                1,
                20,
                rejected_receipt.association_bucket,
                rejected_receipt.association_accepted,
            ),
            (
                101,
                2,
                30,
                rejected_receipt.association_bucket,
                rejected_receipt.association_accepted,
            ),
        )
    ]

    fused = fuse_prediction_rows(
        rows,
        MtmcConfig(min_cooccurrence=1, min_support=1, min_affinity=1.0),
    )

    ids = {(row["source_id"], row["object_id"]): row["global_id"] for row in fused}
    assert len(fused) == len(rows)
    assert ids[(0, 10)] == ids[(1, 20)]
    assert ids[(2, 30)] == -1


def test_parameter_sweep_reports_each_min_affinity_and_z_gate_combination():
    box = {"left": 0.0, "top": 0.0, "width": 10.0, "height": 20.0}
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "person_id": 7,
            "world_x": 0.0,
            "world_y": 0.0,
            **box,
        }
        for source_id in (0, 1)
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "timestamp_ns": 0,
            "source_id": source_id,
            "object_id": source_id + 10,
            "global_id": -1,
            "ground_x": x,
            "ground_y": 0.0,
            "sigma": 0.1,
            **box,
        }
        for source_id, x in ((0, 0.0), (1, 0.4))
    ]
    config = MtmcConfig(min_cooccurrence=1, min_support=1, min_affinity=1.0)

    sweep = sweep_fusion_parameters(
        gt_rows,
        pred_rows,
        config,
        min_affinities=[1.0],
        z_gates=[2.0, 3.0],
    )

    assert [(row["min_affinity"], row["z_gate"]) for row in sweep] == [
        (1.0, 2.0),
        (1.0, 3.0),
    ]
    assert sweep[0]["xcam_f1"] == 0.0
    assert sweep[1]["xcam_f1"] == pytest.approx(1.0)


def test_prepare_prediction_rows_projects_reliable_feet_with_bound_homography(tmp_path):
    pred_path = tmp_path / "pred.csv"
    homography_path = tmp_path / "homography.json"
    _write_csv(
        pred_path,
        ["frame_num", "object_id", "left", "top", "width", "height"],
        [[0, 10, 10, 20, 10, 20]],
    )
    homography_path.write_text(
        json.dumps(
            {
                "source_id": 0,
                "camera": "C1",
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "image_width": 100,
                "image_height": 100,
                "model": "single_plane_homography",
                "fit": {},
            }
        )
    )
    inputs = [CameraInput(0, "C1", pred_path, homography_path)]

    rows = prepare_prediction_rows(inputs)

    assert rows[0]["ground_reliable"] is True
    assert rows[0]["ground_x"] == pytest.approx(15.0)
    assert rows[0]["ground_y"] == pytest.approx(40.0)
    assert rows[0]["sigma"] == pytest.approx(2.0)


def test_evaluate_rows_reports_primary_and_ground_plane_sensitivity_gates():
    box = {"left": 0.0, "top": 0.0, "width": 10.0, "height": 20.0}
    gt_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "person_id": 7,
            "world_x": 0.0,
            "world_y": 0.0,
            **box,
        }
        for source_id in (0, 1)
    ]
    pred_rows = [
        {
            "frame_num": 0,
            "source_id": source_id,
            "object_id": source_id + 10,
            "global_id": 99,
            "ground_x": 0.0,
            "ground_y": 0.0,
            "sigma": 0.1,
            **box,
        }
        for source_id in (0, 1)
    ]

    report = evaluate_rows(gt_rows, pred_rows, ground_gates=[0.5, 1.0])

    assert report["image_plane"]["pooled"]["IDF1"] == pytest.approx(1.0)
    assert report["cross_camera"]["xcam_f1"] == pytest.approx(1.0)
    assert set(report["ground_plane"]) == {"0.5m", "1m"}


def test_cli_runs_legacy_csv_offline_fusion_and_persists_json(tmp_path):
    dataset_root = tmp_path / "labelled_ds"
    pred_paths = []
    homography_paths = []
    position_id_at_origin = 360 * 480 + 120
    for source_id, camera in enumerate(("C1", "C2")):
        image_dir = dataset_root / "images" / camera
        label_dir = dataset_root / "gt_labels" / camera
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image_name = f"{camera}_00000000.png"
        (image_dir / image_name).write_bytes(b"")
        annotation = {
            "size": {"width": 100, "height": 100},
            "objects": [
                {
                    "classTitle": "pedestrian",
                    "points": {"exterior": [[10, 20], [20, 40]]},
                    "tags": [
                        {"name": "person id", "value": 7},
                        {"name": "position id", "value": position_id_at_origin},
                    ],
                }
            ],
        }
        (label_dir / f"{image_name}.json").write_text(json.dumps(annotation))

        pred_path = tmp_path / f"pred-{source_id}.csv"
        _write_csv(
            pred_path,
            ["frame_num", "object_id", "left", "top", "width", "height"],
            [[0, 10 + source_id, 10, 20, 10, 20]],
        )
        pred_paths.append(pred_path)

        homography_path = tmp_path / f"homography-{camera}.json"
        homography_path.write_text(
            json.dumps(
                {
                    "source_id": source_id,
                    "camera": camera,
                    "matrix": [[0.001, 0, -0.015], [0, 0.001, -0.04], [0, 0, 1]],
                    "image_width": 100,
                    "image_height": 100,
                    "model": "single_plane_homography",
                    "fit": {},
                }
            )
        )
        homography_paths.append(homography_path)

    output_path = tmp_path / "metrics.json"
    report = main(
        [
            "--wildtrack-root",
            str(dataset_root),
            "--pred",
            *(str(path) for path in pred_paths),
            "--camera",
            "0=C1",
            "1=C2",
            "--homography",
            *(f"{source_id}={path}" for source_id, path in enumerate(homography_paths)),
            "--fuse-offline",
            "--min-cooccurrence",
            "1",
            "--min-support",
            "1",
            "--output-json",
            str(output_path),
        ]
    )

    assert report["association"]["mode"] == "offline_fusion"
    assert report["metrics"]["image_plane"]["pooled"]["IDF1"] == pytest.approx(1.0)
    assert json.loads(output_path.read_text()) == report
