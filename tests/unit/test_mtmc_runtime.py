import json
from threading import get_ident

import numpy as np
import pytest

from pipelines.ground_plane import Homography
from pipelines.metadata_parser import Detection
from pipelines.mtmc import GroundObservation, MtmcConfig
from pipelines.mtmc_runtime import (
    MtmcRuntime,
    frame_timestamp_ns,
    load_homography_bindings,
    project_person_detections,
    with_global_ids,
)


def _write_homography(path, source_id: int, *, width: int = 1920, height: int = 1080):
    path.write_text(
        json.dumps(
            {
                "source_id": source_id,
                "camera": f"C{source_id + 1}",
                "matrix": np.diag([0.01, 0.01, 1.0]).tolist(),
                "image_width": width,
                "image_height": height,
                "model": "single_plane_homography",
            }
        )
    )


def test_homography_bindings_are_explicitly_bound_to_every_source(tmp_path):
    paths = [tmp_path / "c1.json", tmp_path / "c2.json"]
    for source_id, path in enumerate(paths):
        _write_homography(path, source_id)

    homographies = load_homography_bindings(
        [f"0={paths[0]}", f"1={paths[1]}"],
        expected_sources={0, 1},
        image_width=1920,
        image_height=1080,
    )

    assert set(homographies) == {0, 1}
    assert homographies[0].source_id == 0
    assert homographies[1].source_id == 1


def test_frame_timestamp_uses_file_cadence_and_rtsp_ntp_clock():
    assert frame_timestamp_ns(frame_num=3, ntp_timestamp=99, is_rtsp=False) == 1_500_000_000
    assert frame_timestamp_ns(frame_num=3, ntp_timestamp=1_704_000_000_123, is_rtsp=True) == (
        1_704_000_000_123
    )


def test_reliable_person_bbox_projects_to_ground_position_with_uncertainty():
    homography = Homography(
        source_id=0,
        camera="C1",
        matrix=np.diag([0.01, 0.01, 1.0]),
        image_width=1920,
        image_height=1080,
    )
    detection = Detection(
        frame_num=3,
        object_id=17,
        class_id=0,
        class_label="person",
        confidence=0.9,
        left=100.0,
        top=200.0,
        width=20.0,
        height=100.0,
    )

    observations = project_person_detections(
        timestamp_ns=1_500_000_000,
        source_id=0,
        detections=[detection],
        homography=homography,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.object_id == 17
    assert observation.timestamp_ns == 1_500_000_000
    assert observation.source_id == 0
    assert (observation.x, observation.y) == (1.1, 3.0)
    assert observation.sigma == 0.1


def test_unprojectable_bbox_is_skipped_without_losing_other_people():
    homography = Homography(
        source_id=0,
        camera="C1",
        matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, -300.0]]),
        image_width=1920,
        image_height=1080,
    )
    at_horizon = Detection(0, 17, 0, "person", 0.9, 100.0, 200.0, 20.0, 100.0)
    projectable = Detection(0, 18, 0, "person", 0.9, 200.0, 300.0, 20.0, 100.0)

    observations = project_person_detections(
        timestamp_ns=0,
        source_id=0,
        detections=[at_horizon, projectable],
        homography=homography,
    )

    assert [observation.object_id for observation in observations] == [18]


def test_global_ids_are_read_from_snapshot_without_replacing_tracker_ids():
    detection = Detection(3, 17, 0, "person", 0.9, 1.0, 2.0, 3.0, 4.0)

    annotated = with_global_ids([detection], source_id=2, id_map={(2, 17): 41})

    assert annotated[0].source_id == 2
    assert annotated[0].global_id == 41
    assert annotated[0].object_id == 17
    assert detection.source_id == -1
    assert detection.global_id == -1


def test_runtime_publishes_an_immutable_assignment_snapshot_from_its_worker():
    runtime = MtmcRuntime(
        MtmcConfig(
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    runtime.start()
    try:
        runtime.submit(
            0,
            0,
            (GroundObservation(0, 0, 10, 1.0, 2.0, 0.1),),
        )
        runtime.submit(
            0,
            1,
            (GroundObservation(0, 1, 20, 1.0, 2.0, 0.1),),
        )
        runtime.wait_until_idle()

        snapshot = runtime.id_map_snapshot()
        assert snapshot[(0, 10)] == snapshot[(1, 20)]
        with pytest.raises(TypeError):
            snapshot[(0, 10)] = 99
    finally:
        runtime.shutdown()


def test_runtime_reports_desync_from_the_worker_without_blocking_streaming_thread():
    caller_thread = get_ident()
    events = []
    runtime = MtmcRuntime(
        MtmcConfig(sync_bucket_ns=100, max_skew_ns=150),
        expected_sources={0, 1},
        on_desync=lambda event: events.append((get_ident(), event)),
    )
    runtime.start()
    try:
        runtime.submit(0, 0, (GroundObservation(0, 0, 10, 0.0, 0.0, 0.1),))
        runtime.submit(201, 0, ())
        runtime.wait_until_idle()

        assert len(events) == 1
        worker_thread, event = events[0]
        assert worker_thread != caller_thread
        assert event["reason"] == "missing_sources"
        assert event["desync_buckets"] == 1
    finally:
        runtime.shutdown()


def test_runtime_refuses_and_reports_pairwise_timestamp_skew():
    events = []
    runtime = MtmcRuntime(
        MtmcConfig(
            sync_bucket_ns=1_000,
            max_skew_ns=100,
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
        on_desync=events.append,
    )
    runtime.start()
    try:
        runtime.submit(0, 0, (GroundObservation(0, 0, 10, 0.0, 0.0, 0.1),))
        runtime.submit(200, 1, (GroundObservation(200, 1, 20, 0.0, 0.0, 0.1),))
        runtime.wait_until_idle()

        snapshot = runtime.id_map_snapshot()
        assert snapshot[(0, 10)] != snapshot[(1, 20)]
        assert [event["reason"] for event in events] == ["timestamp_skew"]
    finally:
        runtime.shutdown()


def test_reconnect_bump_invalidates_live_map_and_shutdown_writes_qualified_json(tmp_path):
    output = tmp_path / "mtmc.json"
    runtime = MtmcRuntime(
        MtmcConfig(
            min_cooccurrence=1,
            min_support=1,
            min_affinity=1.0,
            reassign_interval=1,
        ),
        expected_sources={0, 1},
    )
    runtime.start()
    runtime.submit(0, 0, (GroundObservation(0, 0, 7, 0.0, 0.0, 0.1),))
    runtime.submit(0, 1, (GroundObservation(0, 1, 8, 0.0, 0.0, 0.1),))
    runtime.wait_until_idle()
    assert (0, 7) in runtime.id_map_snapshot()

    assert runtime.bump_generation(0) == 1
    assert (0, 7) not in runtime.id_map_snapshot()
    runtime.shutdown(json_path=output)

    payload = json.loads(output.read_text())
    assert set(payload) == {"id_map", "stats"}
    assert {
        "source_id": 0,
        "generation": 0,
        "object_id": 7,
        "global_id": payload["id_map"][0]["global_id"],
    } in payload["id_map"]
    assert all(set(row) == {"source_id", "generation", "object_id", "global_id"}
               for row in payload["id_map"])


def test_shutdown_json_retains_assignments_evicted_from_the_live_ttl_map(tmp_path):
    output = tmp_path / "mtmc-history.json"
    runtime = MtmcRuntime(
        MtmcConfig(
            reassign_interval=1,
            tracklet_ttl=1,
            sync_bucket_ns=100,
        ),
        expected_sources={0},
    )
    runtime.start()
    runtime.submit(0, 0, (GroundObservation(0, 0, 7, 0.0, 0.0, 0.1),))
    runtime.wait_until_idle()
    global_id = runtime.id_map_snapshot()[(0, 7)]

    runtime.submit(100, 0, ())
    runtime.wait_until_idle()
    assert (0, 7) not in runtime.id_map_snapshot()

    runtime.shutdown(json_path=output)

    assert {
        "source_id": 0,
        "generation": 0,
        "object_id": 7,
        "global_id": global_id,
    } in json.loads(output.read_text())["id_map"]
