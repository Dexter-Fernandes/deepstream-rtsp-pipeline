import pytest

from metrics.sweep_conf_threshold import (
    average_precision,
    best_threshold,
    precision_recall_f1,
    sweep_thresholds,
)


def _box(left, top, width, height, confidence=0.9, class_id=0):
    return {"left": left, "top": top, "width": width, "height": height, "confidence": confidence, "class_id": class_id}


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------


def test_precision_recall_f1_perfect_match():
    gt = [[_box(0, 0, 10, 10)]]
    det = [[_box(0, 0, 10, 10)]]
    r = precision_recall_f1(gt, det)
    assert r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)


def test_precision_recall_f1_false_positive():
    gt = [[]]
    det = [[_box(0, 0, 10, 10)]]
    r = precision_recall_f1(gt, det)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 0
    assert r["precision"] == pytest.approx(0.0)
    assert r["recall"] == pytest.approx(0.0)


def test_precision_recall_f1_false_negative():
    gt = [[_box(0, 0, 10, 10)]]
    det = [[]]
    r = precision_recall_f1(gt, det)
    assert r["tp"] == 0 and r["fp"] == 0 and r["fn"] == 1
    assert r["recall"] == pytest.approx(0.0)


def test_precision_recall_f1_no_detections_no_gt():
    r = precision_recall_f1([[]], [[]])
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["f1"] == 0.0


def test_precision_recall_f1_multi_frame():
    gt = [[_box(0, 0, 10, 10)], [_box(20, 20, 10, 10)]]
    det = [[_box(0, 0, 10, 10)], [_box(999, 999, 10, 10)]]
    r = precision_recall_f1(gt, det)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1


# ---------------------------------------------------------------------------
# sweep_thresholds / best_threshold
# ---------------------------------------------------------------------------


def test_sweep_thresholds_filters_by_confidence():
    gt = [[_box(0, 0, 10, 10), _box(50, 50, 10, 10)]]
    raw = [[_box(0, 0, 10, 10, confidence=0.9), _box(50, 50, 10, 10, confidence=0.2)]]

    results = sweep_thresholds(gt, raw, thresholds=[0.1, 0.5])
    low, high = results[0], results[1]
    assert low["tp"] == 2
    assert high["tp"] == 1 and high["fn"] == 1


def test_best_threshold_picks_max_f1():
    results = [
        {"conf_threshold": 0.1, "f1": 0.5},
        {"conf_threshold": 0.5, "f1": 0.9},
        {"conf_threshold": 0.9, "f1": 0.3},
    ]
    assert best_threshold(results)["conf_threshold"] == 0.5


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


def test_average_precision_perfect_detector():
    gt = [[_box(0, 0, 10, 10)], [_box(20, 20, 10, 10)]]
    raw = [[_box(0, 0, 10, 10, confidence=0.9)], [_box(20, 20, 10, 10, confidence=0.8)]]
    assert average_precision(gt, raw) == pytest.approx(1.0)


def test_average_precision_no_detections():
    gt = [[_box(0, 0, 10, 10)]]
    assert average_precision(gt, [[]]) == pytest.approx(0.0)


def test_average_precision_no_ground_truth():
    raw = [[_box(0, 0, 10, 10, confidence=0.9)]]
    assert average_precision([[]], raw) == 0.0


def test_average_precision_ranks_by_confidence():
    # a false positive ranked above the true positive should still let AP reach
    # 1.0 once recall catches up, since AP takes the max precision at each recall level
    gt = [[_box(0, 0, 10, 10)]]
    raw = [[_box(999, 999, 10, 10, confidence=0.99), _box(0, 0, 10, 10, confidence=0.5)]]
    ap = average_precision(gt, raw)
    assert 0.0 < ap < 1.0
