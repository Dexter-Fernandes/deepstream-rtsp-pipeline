from pathlib import Path

from models.convert import build_trtexec_cmd, engine_path, parse_args


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    args = parse_args([str(onnx)])
    assert args.onnx == onnx
    assert args.fp16 is False
    assert args.output_dir == Path("models/engines")


def test_parse_args_fp16_flag(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    args = parse_args([str(onnx), "--fp16"])
    assert args.fp16 is True


def test_parse_args_custom_output_dir(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    args = parse_args([str(onnx), "--output-dir", str(tmp_path)])
    assert args.output_dir == tmp_path


# ---------------------------------------------------------------------------
# engine_path
# ---------------------------------------------------------------------------


def test_engine_path_fp32(tmp_path):
    onnx = Path("models/engines/yolo26n.onnx")
    result = engine_path(onnx, fp16=False, output_dir=tmp_path)
    assert result == tmp_path / "yolo26n_fp32.engine"


def test_engine_path_fp16(tmp_path):
    onnx = Path("models/engines/yolo26n.onnx")
    result = engine_path(onnx, fp16=True, output_dir=tmp_path)
    assert result == tmp_path / "yolo26n_fp16.engine"


def test_engine_path_preserves_stem(tmp_path):
    onnx = Path("some/deep/path/my_model_v2.onnx")
    result = engine_path(onnx, fp16=False, output_dir=tmp_path)
    assert result.stem == "my_model_v2_fp32"


# ---------------------------------------------------------------------------
# build_trtexec_cmd
# ---------------------------------------------------------------------------


def test_trtexec_cmd_includes_onnx_and_engine(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False)
    assert any(f"--onnx={onnx}" == part for part in cmd)
    assert any(f"--saveEngine={out}" == part for part in cmd)


def test_trtexec_cmd_has_no_shape_flags(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False)
    joined = " ".join(cmd)
    assert "--minShapes" not in joined
    assert "--optShapes" not in joined
    assert "--maxShapes" not in joined


def test_trtexec_cmd_no_fp16_flag_when_fp32(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False)
    assert "--fp16" not in cmd


def test_trtexec_cmd_includes_fp16_flag(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp16.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=True)
    assert "--fp16" in cmd


def test_trtexec_cmd_starts_with_trtexec(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False)
    assert cmd[0] == "trtexec"


# ---------------------------------------------------------------------------
# dynamic-batch shape flags (max_batch > 1)
# ---------------------------------------------------------------------------


def test_engine_path_includes_batch_suffix_when_dynamic(tmp_path):
    onnx = Path("models/engines/yolo26n.onnx")
    result = engine_path(onnx, fp16=False, output_dir=tmp_path, max_batch=3)
    assert result == tmp_path / "yolo26n_fp32_b3.engine"


def test_trtexec_cmd_has_shape_flags_for_dynamic_batch(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32_b3.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False, max_batch=3)
    joined = " ".join(cmd)
    assert "--minShapes" in joined
    assert "--optShapes" in joined
    assert "--maxShapes" in joined


def test_trtexec_cmd_shape_flags_use_input_name_and_batch(tmp_path):
    onnx = tmp_path / "yolo26n.onnx"
    out = tmp_path / "yolo26n_fp32_b3.engine"
    cmd = build_trtexec_cmd(onnx, out, fp16=False, max_batch=3, input_name="images")
    joined = " ".join(cmd)
    assert "--minShapes=images:1x3x640x640" in joined
    assert "--optShapes=images:3x3x640x640" in joined
    assert "--maxShapes=images:3x3x640x640" in joined
