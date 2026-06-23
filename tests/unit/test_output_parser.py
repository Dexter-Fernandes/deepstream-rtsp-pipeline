import numpy as np
import pytest

from models.output_parser import parse_yolo26_output


def _make_tensor(rows: list[list[float]], pad_to: int = 300) -> np.ndarray:
    """Build a [pad_to, 6] array with given rows; remaining rows are zeroed."""
    t = np.zeros((pad_to, 6), dtype=np.float32)
    for i, row in enumerate(rows):
        t[i] = row
    return t


def test_empty_below_threshold():
    tensor = _make_tensor([])
    assert parse_yolo26_output(tensor, conf_threshold=0.25) == []


def test_single_detection_above_threshold():
    tensor = _make_tensor([[100, 200, 300, 400, 0.9, 0]])
    results = parse_yolo26_output(tensor, conf_threshold=0.25)
    assert len(results) == 1


def test_threshold_filtering():
    tensor = _make_tensor([
        [0, 0, 10, 10, 0.9, 0],
        [0, 0, 10, 10, 0.1, 1],
        [0, 0, 10, 10, 0.8, 2],
    ])
    results = parse_yolo26_output(tensor, conf_threshold=0.5)
    assert len(results) == 2


def test_coord_conversion():
    # [x1=10, y1=20, x2=50, y2=80, conf=0.9, cls=3]
    tensor = _make_tensor([[10, 20, 50, 80, 0.9, 3]])
    det = parse_yolo26_output(tensor, conf_threshold=0.5)[0]
    assert det["left"] == pytest.approx(10.0)
    assert det["top"] == pytest.approx(20.0)
    assert det["width"] == pytest.approx(40.0)
    assert det["height"] == pytest.approx(60.0)


def test_class_id_extraction():
    tensor = _make_tensor([[0, 0, 10, 10, 0.9, 7.0]])
    det = parse_yolo26_output(tensor, conf_threshold=0.5)[0]
    assert det["class_id"] == 7
    assert isinstance(det["class_id"], int)


def test_batch_dim_squeezed():
    # Shape [1, 300, 6] should be handled identically to [300, 6]
    flat = _make_tensor([[10, 20, 50, 80, 0.9, 0]])
    batched = flat[np.newaxis, ...]  # [1, 300, 6]
    assert parse_yolo26_output(batched) == parse_yolo26_output(flat)
