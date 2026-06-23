from pathlib import Path

from docker.init_models import ensure_models
from models.convert import engine_path
from models.decode_engine import decode_engine_path


def _make_engines_dir(tmp_path: Path, max_batch: int = 3) -> tuple[Path, Path, Path, Path]:
    engines = tmp_path / "engines"
    engines.mkdir()
    onnx = engines / "yolo26n.onnx"
    fp32 = engine_path(onnx, fp16=False, output_dir=engines, max_batch=max_batch)
    fp16 = engine_path(onnx, fp16=True, output_dir=engines, max_batch=max_batch)
    return engines, onnx, fp32, fp16


def test_skips_all_when_files_exist(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()
    fp16.touch()

    calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=lambda *a, **kw: calls.append("export"),
        convert_fn=lambda *a, **kw: calls.append("convert"),
    )
    assert calls == []


def test_exports_when_onnx_missing(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    fp32.touch()
    fp16.touch()

    exported = []

    def fake_export(weights, engines_dir):
        exported.append((weights, engines_dir))
        onnx.touch()  # simulate the file being created

    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=fake_export,
        convert_fn=lambda *a, **kw: None,
    )
    assert len(exported) == 1


def test_skips_export_when_onnx_exists(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()
    fp16.touch()

    calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=lambda *a, **kw: calls.append("export"),
        convert_fn=lambda *a, **kw: None,
    )
    assert "export" not in calls


def test_builds_fp32_when_missing(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp16.touch()

    convert_calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=lambda *a, **kw: None,
        convert_fn=lambda onnx_path, fp16, output_dir, max_batch=1: convert_calls.append(fp16),
    )
    assert False in convert_calls


def test_skips_fp32_when_exists(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()
    fp16.touch()

    convert_calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=lambda *a, **kw: None,
        convert_fn=lambda onnx_path, fp16, output_dir, max_batch=1: convert_calls.append(fp16),
    )
    assert convert_calls == []


def test_builds_fp16_when_missing(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()

    convert_calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=lambda *a, **kw: None,
        convert_fn=lambda onnx_path, fp16, output_dir, max_batch=1: convert_calls.append(fp16),
    )
    assert True in convert_calls


def test_cold_start_calls_export_and_both_conversions(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)

    calls = []

    def fake_export(weights, engines_dir):
        calls.append("export")
        onnx.touch()

    def fake_convert(onnx_path, fp16, output_dir, max_batch=1):
        calls.append(f"convert_fp{'16' if fp16 else '32'}")

    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        export_fn=fake_export,
        convert_fn=fake_convert,
    )
    assert calls == ["export", "convert_fp32", "convert_fp16"]


def test_skips_decode_when_plugin_lib_missing(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()
    fp16.touch()

    decode_calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        plugin_lib=tmp_path / "libyolo26_decode.so",  # does not exist
        export_fn=lambda *a, **kw: None,
        convert_fn=lambda *a, **kw: None,
        decode_fn=lambda *a, **kw: decode_calls.append("decode"),
    )
    assert decode_calls == []


def test_builds_decode_when_plugin_lib_exists(tmp_path):
    engines, onnx, fp32, fp16 = _make_engines_dir(tmp_path)
    onnx.touch()
    fp32.touch()
    fp16.touch()

    plugin_lib = tmp_path / "libyolo26_decode.so"
    plugin_lib.touch()  # simulate .so present

    decode_calls = []
    ensure_models(
        weights=tmp_path / "yolo26n.pt",
        engines_dir=engines,
        plugin_lib=plugin_lib,
        export_fn=lambda *a, **kw: None,
        convert_fn=lambda *a, **kw: None,
        decode_fn=lambda onnx_path, pl, fp16, max_batch, output_dir: decode_calls.append("decode"),
    )
    assert decode_calls == ["decode"]
