from pathlib import Path

from docker.init_models import (
    ensure_models,
    ensure_reid_model,
    prepare_launch_args,
    should_prepare_reid,
)
from models.convert import engine_path


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
    engines, onnx, _fp32, fp16 = _make_engines_dir(tmp_path)
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
    engines, onnx, fp32, _fp16 = _make_engines_dir(tmp_path)
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
    engines, onnx, _fp32, _fp16 = _make_engines_dir(tmp_path)

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


def test_reid_model_downloads_official_onnx_and_builds_sparse_sgie_engine(tmp_path):
    engines = tmp_path / "engines"
    engines.mkdir()
    calls = []

    def fake_download(url, destination):
        calls.append(("download", url, destination.name))
        destination.touch()

    def fake_convert(onnx_path, **kwargs):
        calls.append(("convert", onnx_path.name, kwargs))
        return engines / "reid.engine"

    engine = ensure_reid_model(
        engines,
        download_fn=fake_download,
        convert_fn=fake_convert,
    )

    assert engine == engines / "reid.engine"
    assert calls[0][0] == "download"
    assert calls[0][1].startswith("https://api.ngc.nvidia.com/")
    assert calls[1] == (
        "convert",
        "resnet50_market1501_aicity156.onnx",
        {
            "fp16": True,
            "output_dir": engines,
            "max_batch": 32,
            "input_name": "input",
            "input_shape": (3, 256, 128),
        },
    )


def test_reid_model_is_optional_when_download_is_unavailable(tmp_path, capsys):
    def unavailable(_url, _destination):
        raise OSError("network unavailable")

    result = ensure_reid_model(
        tmp_path / "engines",
        download_fn=unavailable,
        convert_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert result is None
    assert "skipping optional appearance model" in capsys.readouterr().out


def test_reid_initialisation_only_runs_for_explicit_appearance_mode():
    assert should_prepare_reid(["python3", "pipelines/multi_stream.py"]) is False
    assert should_prepare_reid(
        ["python3", "pipelines/multi_stream.py", "--mtmc", "--mtmc-appearance"]
    ) is True
    assert should_prepare_reid(
        ["python3", "pipelines/multi_stream.py", "--mtmc-appearance"]
    ) is False


def test_missing_optional_reid_model_explicitly_falls_back_to_geometry(tmp_path, capsys):
    argv = [
        "python3",
        "pipelines/multi_stream.py",
        "--mtmc",
        "--mtmc-appearance",
        "--mtmc-w-app",
        "0.5",
    ]

    launch_args = prepare_launch_args(
        argv,
        engines_dir=tmp_path,
        ensure_reid_fn=lambda _engines_dir: None,
    )

    assert launch_args == [
        "python3",
        "pipelines/multi_stream.py",
        "--mtmc",
        "--mtmc-w-app",
        "0.5",
    ]
    assert "continuing with geometry-only MTMC" in capsys.readouterr().out


def test_available_optional_reid_model_preserves_appearance_flag(tmp_path):
    argv = ["python3", "pipelines/multi_stream.py", "--mtmc", "--mtmc-appearance"]

    launch_args = prepare_launch_args(
        argv,
        engines_dir=tmp_path,
        ensure_reid_fn=lambda engines_dir: engines_dir / "reid.engine",
    )

    assert launch_args == argv


def test_custom_reid_config_does_not_provision_or_disable_the_default_model(tmp_path):
    argv = [
        "python3",
        "pipelines/multi_stream.py",
        "--mtmc",
        "--mtmc-appearance",
        "--reid-config",
        "configs/custom_reid.txt",
    ]

    launch_args = prepare_launch_args(
        argv,
        engines_dir=tmp_path,
        ensure_reid_fn=lambda _engines_dir: (_ for _ in ()).throw(
            AssertionError("custom config owns its model lifecycle")
        ),
    )

    assert launch_args == argv
