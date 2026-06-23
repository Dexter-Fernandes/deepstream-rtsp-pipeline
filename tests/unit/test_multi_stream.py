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
    assert MultiStreamConfig().tracker_config == "configs/tracker_iou.yml"


def test_parse_args_tracker_flag():
    config = parse_args(["--tracker", "configs/tracker_nvdcf.yml"])
    assert config.tracker_config == "configs/tracker_nvdcf.yml"


def test_parse_args_tracker_default():
    assert parse_args([]).tracker_config == "configs/tracker_iou.yml"
