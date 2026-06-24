import json
import pytest

from metrics.evaluate_tracker import (
    load_gt,
    load_predictions,
    build_accumulator,
    compute_mot_metrics,
    count_unique_tracks,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gt(tmp_path, lines):
    p = tmp_path / "gt.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _write_pred(tmp_path, rows):
    p = tmp_path / "pred.csv"
    header = "frame_num,object_id,class_id,class_label,confidence,left,top,width,height"
    lines = [header] + [",".join(str(v) for v in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _perfect_acc():
    gt = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0}]
    pred = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0}]
    return build_accumulator(gt, pred)


# ---------------------------------------------------------------------------
# Slice 1 — load_gt
# ---------------------------------------------------------------------------

def test_load_gt_basic(tmp_path):
    path = _write_gt(tmp_path, [
        "1,1,100,200,50,100,1,1,0.9",
        "1,2,300,400,60,110,1,1,0.5",
        "2,1,105,205,50,100,1,1,0.9",
    ])
    rows = load_gt(path)
    assert len(rows) == 3
    assert rows[0] == {"frame": 1, "obj_id": 1, "left": 100.0, "top": 200.0, "width": 50.0, "height": 100.0}


def test_load_gt_filters_inactive(tmp_path):
    path = _write_gt(tmp_path, [
        "1,1,100,200,50,100,0,1,0.9",   # conf=0 → excluded
        "1,2,300,400,60,110,1,1,0.5",
    ])
    rows = load_gt(path)
    assert len(rows) == 1
    assert rows[0]["obj_id"] == 2


def test_load_gt_min_visibility(tmp_path):
    path = _write_gt(tmp_path, [
        "1,1,100,200,50,100,1,1,0.2",
        "1,2,300,400,60,110,1,1,0.9",
    ])
    rows = load_gt(path, min_visibility=0.5)
    assert len(rows) == 1
    assert rows[0]["obj_id"] == 2


def test_load_gt_filters_non_pedestrian(tmp_path):
    path = _write_gt(tmp_path, [
        "1,1,100,200,50,100,1,2,0.9",   # class_id=2 → excluded
        "1,2,300,400,60,110,1,1,0.9",
    ])
    rows = load_gt(path, class_ids=(1,))
    assert len(rows) == 1
    assert rows[0]["obj_id"] == 2


# ---------------------------------------------------------------------------
# Slice 2 — load_predictions
# ---------------------------------------------------------------------------

def test_load_predictions_frame_conversion(tmp_path):
    path = _write_pred(tmp_path, [(0, 1, 0, "", 0.8, 100.0, 200.0, 50.0, 100.0)])
    rows = load_predictions(path)
    assert rows[0]["frame"] == 1   # 0-indexed → 1-indexed


def test_load_predictions_schema(tmp_path):
    path = _write_pred(tmp_path, [(0, 3, 0, "", 0.9, 10.0, 20.0, 30.0, 40.0)])
    rows = load_predictions(path)
    assert rows[0] == {"frame": 1, "obj_id": 3, "left": 10.0, "top": 20.0, "width": 30.0, "height": 40.0}


def test_load_predictions_excludes_zero_object_id(tmp_path):
    path = _write_pred(tmp_path, [
        (0, 0, 0, "", 0.8, 100.0, 200.0, 50.0, 100.0),   # object_id=0 → excluded
        (0, 5, 0, "", 0.9, 200.0, 300.0, 60.0, 110.0),
    ])
    rows = load_predictions(path)
    assert len(rows) == 1
    assert rows[0]["obj_id"] == 5


def test_load_predictions_multiple_frames(tmp_path):
    path = _write_pred(tmp_path, [
        (0, 1, 0, "", 0.8, 0.0, 0.0, 50.0, 50.0),
        (1, 1, 0, "", 0.8, 5.0, 5.0, 50.0, 50.0),
        (1, 2, 0, "", 0.9, 100.0, 100.0, 50.0, 50.0),
    ])
    rows = load_predictions(path)
    assert len(rows) == 3
    frames = {r["frame"] for r in rows}
    assert frames == {1, 2}


# ---------------------------------------------------------------------------
# Slice 3 — build_accumulator
# ---------------------------------------------------------------------------

def test_build_accumulator_perfect_match():
    import motmetrics as mm
    gt = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0}]
    pred = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0}]
    acc = build_accumulator(gt, pred)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_matches", "num_misses", "num_false_positives"])
    assert int(summary["num_matches"].iloc[0]) == 1
    assert int(summary["num_misses"].iloc[0]) == 0
    assert int(summary["num_false_positives"].iloc[0]) == 0


def test_build_accumulator_no_match_below_iou_threshold():
    import motmetrics as mm
    gt = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 10.0, "height": 10.0}]
    pred = [{"frame": 1, "obj_id": 1, "left": 500.0, "top": 500.0, "width": 10.0, "height": 10.0}]
    acc = build_accumulator(gt, pred, iou_threshold=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_matches", "num_misses"])
    assert int(summary["num_matches"].iloc[0]) == 0
    assert int(summary["num_misses"].iloc[0]) == 1


def test_build_accumulator_empty_pred():
    import motmetrics as mm
    gt = [{"frame": 1, "obj_id": 1, "left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0}]
    acc = build_accumulator(gt, [], iou_threshold=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_misses"])
    assert int(summary["num_misses"].iloc[0]) == 1


# ---------------------------------------------------------------------------
# Slice 4 — compute_mot_metrics
# ---------------------------------------------------------------------------

def test_compute_mot_metrics_keys():
    acc = _perfect_acc()
    result = compute_mot_metrics(acc)
    assert set(result.keys()) >= {"MOTA", "MOTP", "IDF1", "num_switches", "num_fragmentations"}


def test_compute_mot_metrics_perfect():
    acc = _perfect_acc()
    result = compute_mot_metrics(acc)
    assert result["MOTA"] == pytest.approx(1.0)
    assert result["num_switches"] == 0


# ---------------------------------------------------------------------------
# Slice 5 — count_unique_tracks
# ---------------------------------------------------------------------------

def test_count_unique_tracks():
    pred = [
        {"frame": 1, "obj_id": 1},
        {"frame": 1, "obj_id": 2},
        {"frame": 2, "obj_id": 1},
    ]
    assert count_unique_tracks(pred) == 2


def test_count_unique_tracks_single():
    pred = [{"frame": 1, "obj_id": 7}]
    assert count_unique_tracks(pred) == 1


def test_count_unique_tracks_empty():
    assert count_unique_tracks([]) == 0


# ---------------------------------------------------------------------------
# Slice 6 — CLI
# ---------------------------------------------------------------------------

def test_cli_writes_json(tmp_path):
    gt_path = _write_gt(tmp_path, [
        "1,1,0,0,100,100,1,1,1.0",
        "2,1,2,2,100,100,1,1,1.0",
    ])
    pred_path = _write_pred(tmp_path, [
        (0, 1, 0, "", 0.9, 0.0, 0.0, 100.0, 100.0),
        (1, 1, 0, "", 0.9, 2.0, 2.0, 100.0, 100.0),
    ])
    out_path = str(tmp_path / "out.json")
    main(["--gt", gt_path, "--pred", pred_path, "--output-json", out_path])
    result = json.loads((tmp_path / "out.json").read_text())
    assert "MOTA" in result
    assert "IDF1" in result
    assert "unique_tracks" in result
