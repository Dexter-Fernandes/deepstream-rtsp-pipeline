import numpy as np
import pytest

from pipelines.metadata_parser import Detection
from pipelines.anonymisation import blur_bboxes


def _det(**kw):
    d = dict(
        frame_num=0, object_id=1, class_id=0, class_label="person",
        confidence=0.9, left=10.0, top=10.0, width=50.0, height=50.0,
    )
    d.update(kw)
    return Detection(**d)


def _white(h=100, w=100):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _striped(h=100, w=100):
    """Alternating black/white columns — high spatial variance for blur detection."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_detections_returns_frame_unchanged():
    frame = _white()
    original = frame.copy()
    result = blur_bboxes(frame, [])
    assert np.array_equal(result, original)


def test_returns_ndarray():
    frame = _white()
    result = blur_bboxes(frame, [_det()])
    assert isinstance(result, np.ndarray)


def test_bbox_region_is_blurred():
    frame = _striped()
    roi_before = frame[10:60, 10:60].copy()
    result = blur_bboxes(frame, [_det(left=10, top=10, width=50, height=50)])
    roi_after = result[10:60, 10:60]
    assert not np.array_equal(roi_before, roi_after)


def test_pixels_outside_bbox_unchanged():
    frame = _striped()
    # det covers rows 10-60, cols 10-60; check a corner well outside that
    corner_before = frame[80:100, 80:100].copy()
    blur_bboxes(frame, [_det(left=10, top=10, width=50, height=50)])
    assert np.array_equal(frame[80:100, 80:100], corner_before)


def test_multiple_bboxes_all_blurred():
    frame = _striped()
    det_a = _det(left=5, top=5, width=20, height=20, object_id=1)
    det_b = _det(left=70, top=70, width=20, height=20, object_id=2)
    roi_a_before = frame[5:25, 5:25].copy()
    roi_b_before = frame[70:90, 70:90].copy()
    blur_bboxes(frame, [det_a, det_b])
    assert not np.array_equal(frame[5:25, 5:25], roi_a_before)
    assert not np.array_equal(frame[70:90, 70:90], roi_b_before)


def test_bbox_clipped_to_frame_bounds_no_crash():
    frame = _white(h=100, w=100)
    # bbox extends 50px outside the frame on every side
    det = _det(left=-50.0, top=-50.0, width=200.0, height=200.0)
    blur_bboxes(frame, [det])  # must not raise
