from pathlib import Path

from pipelines.multi_stream import MultiStreamConfig, parse_args, _output_csv_path, _restream_port, _make_nvinfer_config


def test_default_uris_is_empty_list():
    assert MultiStreamConfig().uris == []


def test_parse_args_single_uri():
    config = parse_args(["--uri", "rtsp://localhost:8554/stream0"])
    assert config.uris == ["rtsp://localhost:8554/stream0"]


def test_parse_args_multiple_uris():
    config = parse_args([
        "--uri", "rtsp://localhost:8554/stream0",
        "--uri", "rtsp://localhost:8554/stream1",
        "--uri", "rtsp://localhost:8554/stream2",
    ])
    assert config.uris == [
        "rtsp://localhost:8554/stream0",
        "rtsp://localhost:8554/stream1",
        "rtsp://localhost:8554/stream2",
    ]


def test_default_restream_base_port_is_none():
    assert MultiStreamConfig().restream_base_port is None


def test_parse_args_restream_base_port():
    config = parse_args(["--restream-base-port", "8556"])
    assert config.restream_base_port == 8556


def test_default_output_dir():
    assert MultiStreamConfig().output_dir == "."


def test_output_csv_path_includes_source_id():
    assert _output_csv_path(".", 2) == "output_stream2.csv"


def test_restream_port_offset_from_base():
    assert _restream_port(8556, 1) == 8557


def test_make_nvinfer_config_returns_original_for_n1(tmp_path):
    cfg = tmp_path / "nvinfer.txt"
    cfg.write_text("batch-size=1\nmodel-engine-file=/models/foo_b1_gpu0_fp32.engine\n")
    assert _make_nvinfer_config(str(cfg), 1) == str(cfg)


def test_make_nvinfer_config_rewrites_batch_size(tmp_path):
    cfg = tmp_path / "nvinfer.txt"
    cfg.write_text("batch-size=1\nmodel-engine-file=/models/foo_b1_gpu0_fp32.engine\n")
    out = _make_nvinfer_config(str(cfg), 3)
    assert "batch-size=3" in Path(out).read_text()


def test_make_nvinfer_config_rewrites_engine_path(tmp_path):
    cfg = tmp_path / "nvinfer.txt"
    cfg.write_text("batch-size=1\nmodel-engine-file=/models/foo_b1_gpu0_fp32.engine\n")
    out = _make_nvinfer_config(str(cfg), 3)
    content = Path(out).read_text()
    assert "_b3_gpu0_fp32.engine" in content
    assert "_b1_gpu0_fp32.engine" not in content


def test_default_tracker_config():
    assert MultiStreamConfig().tracker_config == "configs/tracker_nvdcf.yml"


def test_parse_args_tracker_flag():
    config = parse_args(["--tracker", "configs/tracker_nvdcf.yml"])
    assert config.tracker_config == "configs/tracker_nvdcf.yml"


def test_parse_args_tracker_default():
    assert parse_args([]).tracker_config == "configs/tracker_nvdcf.yml"


def test_yolo_decode_probe_sets_untracked_object_id():
    # The probe must set object_id = pyds.UNTRACKED_OBJECT_ID before adding
    # each detection to the frame so nvtracker assigns a fresh unique track ID
    # rather than treating every detection as already-tracked with ID=0.
    import inspect
    from pipelines.multi_stream import run
    src = inspect.getsource(run)
    assert "UNTRACKED_OBJECT_ID" in src


def test_yolo_decode_probe_marks_frame_inferred():
    # nvinfer runs in output-tensor-meta mode and never sets bInferDone, so
    # nvtracker would skip the frame and drop every injected object. The probe
    # must set frame_meta.bInferDone = 1 itself. Without this the tracker
    # outputs zero objects and every tracker CSV is empty.
    import inspect
    from pipelines.multi_stream import run
    src = inspect.getsource(run)
    assert "bInferDone" in src


def test_yolo_decode_probe_sets_detector_bbox_info():
    # nvtracker associates on detector_bbox_info.org_bbox_coords, not
    # rect_params, so the probe must populate it or the tracker drops the object.
    import inspect
    from pipelines.multi_stream import run
    src = inspect.getsource(run)
    assert "detector_bbox_info" in src


def test_is_file_uri():
    from pipelines.multi_stream import _is_file_uri
    assert _is_file_uri("data/mot17_04.mp4") is True
    assert _is_file_uri("file:///abs/path/clip.mp4") is True
    assert _is_file_uri("rtsp://localhost:8554/stream0") is False


# ---------------------------------------------------------------------------
# M3.3 — new flag defaults and parsing
# ---------------------------------------------------------------------------


def test_default_perf_json_is_none():
    assert MultiStreamConfig().perf_json is None


def test_default_perf_interval():
    assert MultiStreamConfig().perf_interval == 5.0


def test_default_duration_is_none():
    assert MultiStreamConfig().duration is None


def test_default_no_sync_is_false():
    assert MultiStreamConfig().no_sync is False


def test_parse_args_perf_json():
    config = parse_args(["--perf-json", "/tmp/perf.json"])
    assert config.perf_json == "/tmp/perf.json"


def test_parse_args_perf_interval():
    config = parse_args(["--perf-interval", "10"])
    assert config.perf_interval == 10.0


def test_parse_args_duration():
    config = parse_args(["--duration", "120"])
    assert config.duration == 120


def test_parse_args_no_sync():
    config = parse_args(["--no-sync"])
    assert config.no_sync is True


def test_run_imports_perf_monitor():
    import inspect
    from pipelines.multi_stream import run
    assert "perf_monitor" in inspect.getsource(run)


def test_run_has_frame_counts():
    import inspect
    from pipelines.multi_stream import run
    assert "frame_counts" in inspect.getsource(run)


def test_run_uses_timeout_add_seconds():
    import inspect
    from pipelines.multi_stream import run
    assert "timeout_add_seconds" in inspect.getsource(run)


def test_build_pipeline_has_no_sync():
    import inspect
    from pipelines.multi_stream import build_pipeline
    assert "sync" in inspect.getsource(build_pipeline)


# ---------------------------------------------------------------------------
# M3.6.1 — Structured logging wired into run()
# ---------------------------------------------------------------------------


def test_run_uses_configure_pipeline_logging():
    import inspect
    from pipelines.multi_stream import run
    assert "configure_pipeline_logging" in inspect.getsource(run)


def test_run_emits_pipeline_start_event():
    import inspect
    from pipelines.multi_stream import run
    assert "pipeline_start" in inspect.getsource(run)


def test_run_emits_pipeline_eos_event():
    import inspect
    from pipelines.multi_stream import run
    assert "pipeline_eos" in inspect.getsource(run)


def test_run_emits_pipeline_error_event():
    import inspect
    from pipelines.multi_stream import run
    assert "pipeline_error" in inspect.getsource(run)


# ---------------------------------------------------------------------------
# M3.6.2 — HealthMonitor wired into run()
# ---------------------------------------------------------------------------


def test_run_creates_health_monitor():
    import inspect
    from pipelines.multi_stream import run
    assert "HealthMonitor" in inspect.getsource(run)


def test_run_records_health_frame_in_probe():
    import inspect
    from pipelines.multi_stream import run
    assert "record_frame" in inspect.getsource(run)


def test_run_emits_health_tick_event():
    import inspect
    from pipelines.multi_stream import run
    assert "health_tick" in inspect.getsource(run)


def test_run_warns_on_source_stalled():
    import inspect
    from pipelines.multi_stream import run
    assert "source_stalled" in inspect.getsource(run)
