import json
from pathlib import Path
from types import MappingProxyType

from pipelines.mtmc_runtime import MtmcBatchReceipt, MtmcIdentitySnapshot
from pipelines.multi_stream import (
    FrameReceiptLedger,
    MultiStreamConfig,
    StreamReconnectDetector,
    _make_nvinfer_config,
    _output_csv_path,
    _restream_port,
    attach_mtmc_probe,
    format_mtmc_osd_label,
    mux_timing_properties,
    parse_args,
    prepare_mtmc,
    rtsp_clock_properties,
)


def test_default_uris_is_empty_list():
    assert MultiStreamConfig().uris == []


def test_frame_receipt_ledger_binds_offset_source_frames_and_fails_closed():
    receipt = MtmcBatchReceipt(
        identity_snapshot=MtmcIdentitySnapshot(
            id_map=MappingProxyType({(0, 7): 41, (1, 8): 41}),
            generations=MappingProxyType({0: 0, 1: 0}),
        ),
        attempt_epoch=1,
        association_bucket=1,
        association_accepted=True,
    )
    ledger = FrameReceiptLedger(expected_sources={0, 1}, max_pending_per_source=2)

    ledger.record(((0, 0), (1, 100)), receipt)

    assert ledger.consume(0, 0) is receipt
    assert ledger.consume(1, 100) is receipt

    ledger.record(((0, 1),), receipt)
    assert ledger.consume(0, 2) is None
    assert ledger.consume(0, 1) is None

    bounded = FrameReceiptLedger(expected_sources={0}, max_pending_per_source=1)
    bounded.record(((0, 3),), receipt)
    assert bounded.record(((0, 4),), receipt) == ((0, 3),)
    assert bounded.consume(0, 4) is None


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


def test_parse_args_mtmc_online_options():
    config = parse_args([
        "--mtmc",
        "--homography", "0=configs/homography_C1.json",
        "--homography", "1=configs/homography_C2.json",
        "--mtmc-z-gate", "2.5",
        "--mtmc-min-affinity", "0.75",
        "--mtmc-reassign-interval", "7",
        "--mtmc-json", "results/mtmc.json",
        "--mtmc-osd-labels",
        "--mtmc-sync-bucket-ms", "40",
        "--mtmc-max-skew-ms", "90",
    ])

    assert config.mtmc is True
    assert config.homographies == [
        "0=configs/homography_C1.json",
        "1=configs/homography_C2.json",
    ]
    assert config.mtmc_z_gate == 2.5
    assert config.mtmc_min_affinity == 0.75
    assert config.mtmc_reassign_interval == 7
    assert config.mtmc_json == "results/mtmc.json"
    assert config.mtmc_osd_labels is True
    assert config.mtmc_sync_bucket_ms == 40.0
    assert config.mtmc_max_skew_ms == 90.0


def test_parse_args_enables_optional_reid_appearance():
    config = parse_args(
        [
            "--mtmc",
            "--mtmc-appearance",
            "--reid-config",
            "configs/custom_reid.txt",
            "--mtmc-w-app",
            "0.75",
        ]
    )

    assert config.mtmc_appearance is True
    assert config.reid_config == "configs/custom_reid.txt"
    assert config.mtmc_w_app == 0.75


def test_rtsp_clock_and_mux_properties_use_ntp_and_one_frame_timeout():
    assert rtsp_clock_properties() == {
        "ntp-sync": True,
        "buffer-mode": 4,
    }
    assert mux_timing_properties(["rtsp://camera/stream0", "rtsp://camera/stream1"]) == {
        "live-source": 1,
        "attach-sys-ts": 0,
        "batched-push-timeout": 40_000,
    }


def test_rtsp_clock_configuration_enables_deepstream_rtcp_timestamps():
    from pipelines.multi_stream import configure_rtsp_clock

    class Source:
        def __init__(self):
            self.properties = {}

        def __hash__(self):
            return 41

        def set_property(self, name, value):
            self.properties[name] = value

    class Pyds:
        def __init__(self):
            self.configured_sources = []

        def configure_source_for_ntp_sync(self, source_pointer):
            self.configured_sources.append(source_pointer)

    source = Source()
    pyds = Pyds()

    configure_rtsp_clock(source, pyds_module=pyds)

    assert source.properties == {"ntp-sync": True, "buffer-mode": 4}
    assert pyds.configured_sources == [41]


def test_rtsp_clock_gate_waits_for_staggered_sources_and_skips_transition_batch():
    from pipelines.multi_stream import RtspClockGate

    gate = RtspClockGate(expected_sources={0, 1})

    waiting = gate.observe_batch({0: 0, 1: 0})
    repeated_wait = gate.observe_batch({0: 0, 1: 0})
    first_ready = gate.observe_batch({0: 100, 1: 0})
    transition = gate.observe_batch({0: 101, 1: 102})
    healthy = gate.observe_batch({0: 103, 1: 104})

    assert waiting.fuse_batch is False
    assert waiting.waiting_sources == (0, 1)
    assert repeated_wait.waiting_sources == ()
    assert first_ready.ready_sources == (0,)
    assert first_ready.fuse_batch is False
    assert transition.ready_sources == (1,)
    assert transition.fuse_batch is False
    assert transition.advance_liveness is False
    assert healthy.fuse_batch is True
    assert healthy.advance_liveness is True


def test_rtsp_clock_gate_fails_closed_if_a_ready_source_loses_ntp():
    from pipelines.multi_stream import RtspClockGate

    gate = RtspClockGate(expected_sources={0, 1})
    gate.observe_batch({0: 100, 1: 100})
    assert gate.observe_batch({0: 101, 1: 101}).fuse_batch is True

    clock_loss = gate.observe_batch({0: 0, 1: 102})
    repeated_loss = gate.observe_batch({0: 0, 1: 103})
    recovered = gate.observe_batch({0: 104, 1: 104})

    assert clock_loss.fuse_batch is False
    assert clock_loss.advance_liveness is True
    assert clock_loss.lost_sources == (0,)
    assert clock_loss.invalid_sources == (0,)
    assert repeated_loss.lost_sources == ()
    assert repeated_loss.invalid_sources == (0,)
    assert recovered.fuse_batch is True
    assert recovered.recovered_sources == (0,)


def test_reconnect_detector_ignores_initial_connect_and_requires_a_disconnect():
    detector = StreamReconnectDetector()

    assert detector.connected() is False
    assert detector.connected() is False
    detector.disconnected()
    assert detector.connected() is True
    assert detector.connected() is False


def test_prepare_mtmc_validates_bindings_and_converts_cli_units(tmp_path):
    bindings = []
    for source_id in range(2):
        path = tmp_path / f"h{source_id}.json"
        path.write_text(json.dumps({
            "source_id": source_id,
            "camera": f"C{source_id + 1}",
            "matrix": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 1.0]],
            "image_width": 1920,
            "image_height": 1080,
        }))
        bindings.append(f"{source_id}={path}")
    config = MultiStreamConfig(
        uris=["c1.mp4", "c2.mp4"],
        mtmc=True,
        homographies=bindings,
        mtmc_z_gate=2.5,
        mtmc_min_affinity=0.75,
        mtmc_reassign_interval=7,
        mtmc_sync_bucket_ms=40.0,
        mtmc_max_skew_ms=90.0,
        mtmc_appearance=True,
        mtmc_w_app=0.75,
    )

    setup = prepare_mtmc(config)

    assert set(setup.homographies) == {0, 1}
    assert setup.config.z_gate == 2.5
    assert setup.config.min_affinity == 0.75
    assert setup.config.reassign_interval == 7
    assert setup.config.sync_bucket_ns == 40_000_000
    assert setup.config.max_skew_ns == 90_000_000
    assert setup.config.w_app == 0.75


def test_mtmc_osd_label_keeps_local_and_global_identity_visible():
    assert format_mtmc_osd_label("person", object_id=17, global_id=41) == (
        "person local=17 global=41"
    )


def test_mtmc_probe_attaches_to_tracker_source_before_demux():
    calls = []

    class Pad:
        def add_probe(self, probe_type, callback, user_data):
            calls.append((probe_type, callback, user_data))

    class Element:
        def get_static_pad(self, name):
            assert name == "src"
            return Pad()

    class Pipeline:
        def get_by_name(self, name):
            assert name == "tracker"
            return Element()

    callback = object()
    attach_mtmc_probe(Pipeline(), probe_type="BUFFER", callback=callback)

    assert calls == [("BUFFER", callback, 0)]


def test_mtmc_probe_attaches_to_reid_source_when_appearance_is_enabled():
    requested_elements = []

    class Pad:
        def add_probe(self, probe_type, callback, user_data):
            assert (probe_type, callback, user_data) == ("BUFFER", "callback", 0)

    class Element:
        def get_static_pad(self, name):
            assert name == "src"
            return Pad()

    class Pipeline:
        def get_by_name(self, name):
            requested_elements.append(name)
            return Element()

    attach_mtmc_probe(
        Pipeline(),
        probe_type="BUFFER",
        callback="callback",
        appearance_enabled=True,
    )

    assert requested_elements == ["reid"]


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


def test_module_calls_configure_pipeline_logging():
    # Called once at import time (not per-run) so logging is configured
    # before the module-level plugin-load log line fires.
    import inspect

    from pipelines import multi_stream
    assert "configure_pipeline_logging()" in inspect.getsource(multi_stream)


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
