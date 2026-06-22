from pipelines.multi_stream import MultiStreamConfig, parse_args, _output_csv_path, _restream_port


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
