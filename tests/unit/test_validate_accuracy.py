import numpy as np
import pytest

from metrics.validate_accuracy import (
    box_iou,
    compare_decode_plugin,
    compare_engines,
    match_detections,
    preprocess_frame,
)


def _det(left, top, width, height, confidence=0.9, class_id=0):
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "confidence": confidence,
        "class_id": class_id,
    }


# ---------------------------------------------------------------------------
# box_iou
# ---------------------------------------------------------------------------

def test_box_iou_perfect_overlap():
    a = _det(10, 20, 50, 60)
    assert box_iou(a, a) == pytest.approx(1.0)


def test_box_iou_no_overlap():
    a = _det(0, 0, 10, 10)
    b = _det(20, 20, 10, 10)
    assert box_iou(a, b) == pytest.approx(0.0)


def test_box_iou_partial():
    # a: [0,0]→[10,10], b: [5,5]→[15,15]
    # intersection: [5,5]→[10,10] = 25, union = 100 + 100 - 25 = 175
    a = _det(0, 0, 10, 10)
    b = _det(5, 5, 10, 10)
    assert box_iou(a, b) == pytest.approx(25 / 175)


def test_box_iou_contained():
    # b fully inside a; intersection = area_b = 2500, union = area_a = 10000
    a = _det(0, 0, 100, 100)
    b = _det(25, 25, 50, 50)
    assert box_iou(a, b) == pytest.approx(2500 / 10000)


def test_box_iou_zero_area():
    a = _det(0, 0, 0, 0)
    b = _det(0, 0, 10, 10)
    assert box_iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# match_detections
# ---------------------------------------------------------------------------

def test_match_detections_identical():
    dets = [_det(0, 0, 100, 100), _det(200, 200, 50, 50)]
    matches, unmatched_a, unmatched_b = match_detections(dets, dets)
    assert len(matches) == 2
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0


def test_match_detections_dropped():
    # dets_b has one fewer detection than dets_a
    dets_a = [_det(0, 0, 100, 100), _det(200, 200, 50, 50)]
    dets_b = [_det(0, 0, 100, 100)]
    matches, unmatched_a, unmatched_b = match_detections(dets_a, dets_b)
    assert len(matches) == 1
    assert len(unmatched_a) == 1
    assert len(unmatched_b) == 0


def test_match_detections_added():
    # dets_b has one extra detection
    dets_a = [_det(0, 0, 100, 100)]
    dets_b = [_det(0, 0, 100, 100), _det(200, 200, 50, 50)]
    matches, unmatched_a, unmatched_b = match_detections(dets_a, dets_b)
    assert len(matches) == 1
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 1


def test_match_detections_below_thresh():
    a = _det(0, 0, 10, 10)
    b = _det(20, 20, 10, 10)  # no overlap → IoU = 0
    matches, unmatched_a, unmatched_b = match_detections([a], [b], iou_thresh=0.5)
    assert len(matches) == 0
    assert len(unmatched_a) == 1
    assert len(unmatched_b) == 1


def test_match_detections_empty_inputs():
    matches, unmatched_a, unmatched_b = match_detections([], [])
    assert matches == []
    assert unmatched_a == []
    assert unmatched_b == []


# ---------------------------------------------------------------------------
# compare_engines
# ---------------------------------------------------------------------------

def test_compare_engines_identical():
    dets = [_det(0, 0, 100, 100, confidence=0.9), _det(200, 200, 50, 50, confidence=0.7)]
    frames = [dets, dets, dets]
    result = compare_engines(frames, frames)
    assert result["mean_iou"] == pytest.approx(1.0)
    assert result["n_matched"] == 6
    assert result["n_dropped"] == 0
    assert result["n_added"] == 0
    assert result["max_conf_delta"] == pytest.approx(0.0)


def test_compare_engines_mismatch():
    frames_a = [[_det(0, 0, 100, 100, confidence=0.9)]]
    frames_b = [[_det(0, 0, 100, 100, confidence=0.8), _det(200, 200, 50, 50, confidence=0.7)]]
    result = compare_engines(frames_a, frames_b)
    assert result["n_matched"] == 1
    assert result["n_dropped"] == 0
    assert result["n_added"] == 1
    assert result["max_conf_delta"] == pytest.approx(0.1, abs=1e-5)


def test_compare_engines_no_matches():
    frames_a = [[_det(0, 0, 10, 10)]]
    frames_b = [[_det(300, 300, 10, 10)]]
    result = compare_engines(frames_a, frames_b)
    assert result["n_matched"] == 0
    assert result["n_dropped"] == 1
    assert result["n_added"] == 1
    assert result["mean_iou"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_decode_plugin
# ---------------------------------------------------------------------------

def test_compare_decode_plugin_exact():
    det = _det(10, 20, 50, 60)
    frames = [[det]] * 3
    result = compare_decode_plugin(frames, frames)
    assert result["max_coord_delta_px"] == pytest.approx(0.0)
    assert result["mean_coord_delta_px"] == pytest.approx(0.0)
    assert result["p99_coord_delta_px"] == pytest.approx(0.0)
    assert result["within_epsilon"] is True
    assert result["per_coord_max_delta_px"] == {"left": 0.0, "top": 0.0, "width": 0.0, "height": 0.0}


def test_compare_decode_plugin_small_delta():
    det_decode = _det(10.005, 20.002, 50.0, 60.0)
    det_python = _det(10.0, 20.0, 50.0, 60.0)
    result = compare_decode_plugin([[det_decode]], [[det_python]])
    assert result["max_coord_delta_px"] == pytest.approx(0.005, abs=1e-4)
    assert result["within_epsilon"] is True  # p99 < default 1.0px epsilon
    assert result["per_coord_max_delta_px"]["left"] == pytest.approx(0.005, abs=1e-4)
    assert result["per_coord_max_delta_px"]["top"] == pytest.approx(0.002, abs=1e-4)


def test_compare_decode_plugin_exceeds_epsilon():
    # Every matched pair has a 1.5px left delta — p99 exceeds 1.0px epsilon
    det_decode = _det(11.5, 20.0, 50.0, 60.0)
    det_python = _det(10.0, 20.0, 50.0, 60.0)
    result = compare_decode_plugin([[det_decode]], [[det_python]], coord_epsilon=1.0)
    assert result["max_coord_delta_px"] == pytest.approx(1.5, abs=1e-4)
    assert result["within_epsilon"] is False
    assert result["per_coord_max_delta_px"]["left"] == pytest.approx(1.5, abs=1e-4)
    assert result["per_coord_max_delta_px"]["top"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# preprocess_frame
# ---------------------------------------------------------------------------

def test_preprocess_frame_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.shape == (1, 3, 640, 640)


def test_preprocess_frame_dtype():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.dtype == np.float32


def test_preprocess_frame_range():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(1.0)


def test_preprocess_frame_non_square_input():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = preprocess_frame(frame)
    assert out.shape == (1, 3, 640, 640)
