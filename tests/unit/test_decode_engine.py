from pathlib import Path

from models.decode_engine import decode_engine_path, parse_args


# ---------------------------------------------------------------------------
# decode_engine_path
# ---------------------------------------------------------------------------


def test_decode_engine_path_fp16(tmp_path):
    result = decode_engine_path(tmp_path, fp16=True, max_batch=3)
    assert result == tmp_path / "yolo26n_fp16_b3_decode.engine"


def test_decode_engine_path_fp32(tmp_path):
    result = decode_engine_path(tmp_path, fp16=False, max_batch=3)
    assert result == tmp_path / "yolo26n_fp32_b3_decode.engine"


def test_decode_engine_path_single_batch(tmp_path):
    result = decode_engine_path(tmp_path, fp16=True, max_batch=1)
    assert result == tmp_path / "yolo26n_fp16_b1_decode.engine"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    args = parse_args([str(onnx)])
    assert args.onnx == onnx
    assert args.fp16 is False
    assert args.max_batch == 3
    assert args.output_dir == Path("models/engines")


def test_parse_args_fp16_flag(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    args = parse_args([str(onnx), "--fp16"])
    assert args.fp16 is True
