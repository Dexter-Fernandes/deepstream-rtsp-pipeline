from pipelines.rtsp import PipelineConfig, parse_args, _source_props, _restream_sink_props


def test_default_uri():
    assert PipelineConfig().uri == "rtsp://localhost:8554/stream0"


def test_default_nvinfer_config():
    assert PipelineConfig().nvinfer_config == "configs/nvinfer_primary.txt"


def test_default_retry():
    assert PipelineConfig().retry == 3


def test_default_timeout_us():
    assert PipelineConfig().timeout_us == 5_000_000


def test_parse_args_defaults():
    config = parse_args([])
    assert config.uri == "rtsp://localhost:8554/stream0"


def test_parse_args_custom_uri():
    config = parse_args(["--uri", "rtsp://localhost:8554/stream1"])
    assert config.uri == "rtsp://localhost:8554/stream1"


def test_source_props_location():
    config = PipelineConfig(uri="rtsp://localhost:8554/stream0")
    assert _source_props(config)["location"] == "rtsp://localhost:8554/stream0"


def test_source_props_retry():
    config = PipelineConfig(retry=5)
    assert _source_props(config)["retry"] == 5


def test_source_props_timeout():
    config = PipelineConfig(timeout_us=10_000_000)
    assert _source_props(config)["timeout"] == 10_000_000


def test_default_restream_uri_is_none():
    assert PipelineConfig().restream_uri is None


def test_parse_args_restream_uri_set():
    config = parse_args(["--restream-uri", "rtsp://localhost:8554/stream0_out"])
    assert config.restream_uri == "rtsp://localhost:8554/stream0_out"


def test_parse_args_no_restream_uri_stays_none():
    config = parse_args([])
    assert config.restream_uri is None


def test_restream_sink_props_port():
    props = _restream_sink_props("rtsp://localhost:8554/stream0_out")
    assert props["rtsp-port"] == 8554


def test_restream_sink_props_mount_point():
    props = _restream_sink_props("rtsp://localhost:8554/stream0_out")
    assert props["rtsp-mount-point"] == "/stream0_out"
