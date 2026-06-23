from pathlib import Path

from models.export_yolo26 import parse_args


def test_parse_args_defaults():
    args = parse_args(["yolo26n.pt"])
    assert args.weights == Path("yolo26n.pt")
    assert args.output_dir == Path("models/engines")


def test_parse_args_custom_output_dir(tmp_path):
    args = parse_args(["yolo26n.pt", "--output-dir", str(tmp_path)])
    assert args.output_dir == tmp_path


def test_parse_args_custom_weights(tmp_path):
    weights = tmp_path / "custom.pt"
    args = parse_args([str(weights)])
    assert args.weights == weights
