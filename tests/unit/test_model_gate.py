import json
import pytest
from pathlib import Path

from metrics.model_gate import (
    compute_gate_result,
    build_manifest,
    write_manifest,
    parse_args,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PASSING_REPORT = {
    "fp16_vs_fp32": {
        "match_rate": 0.9997,
        "mean_iou": 0.9983,
        "n_matched": 120,
        "n_dropped": 0,
        "n_added": 1,
        "max_conf_delta": 0.0012,
    }
}

FAILING_REPORT_LOW_MATCH = {
    "fp16_vs_fp32": {
        "match_rate": 0.93,
        "mean_iou": 0.999,
        "n_matched": 97,
        "n_dropped": 3,
        "n_added": 0,
        "max_conf_delta": 0.002,
    }
}

FAILING_REPORT_LOW_IOU = {
    "fp16_vs_fp32": {
        "match_rate": 0.999,
        "mean_iou": 0.93,
        "n_matched": 120,
        "n_dropped": 0,
        "n_added": 0,
        "max_conf_delta": 0.001,
    }
}

FAILING_REPORT_BOTH = {
    "fp16_vs_fp32": {
        "match_rate": 0.90,
        "mean_iou": 0.88,
        "n_matched": 90,
        "n_dropped": 10,
        "n_added": 0,
        "max_conf_delta": 0.05,
    }
}

# ---------------------------------------------------------------------------
# Slice 1 — compute_gate_result
# ---------------------------------------------------------------------------


def test_gate_passes_when_all_metrics_above_threshold():
    result = compute_gate_result(PASSING_REPORT)
    assert result["passed"] is True


def test_gate_fails_when_matched_rate_below_threshold():
    result = compute_gate_result(FAILING_REPORT_LOW_MATCH)
    assert result["passed"] is False


def test_gate_fails_when_mean_iou_below_threshold():
    result = compute_gate_result(FAILING_REPORT_LOW_IOU)
    assert result["passed"] is False


def test_gate_fails_when_both_metrics_below_threshold():
    result = compute_gate_result(FAILING_REPORT_BOTH)
    assert result["passed"] is False


def test_gate_result_includes_reasons_on_failure():
    result = compute_gate_result(FAILING_REPORT_BOTH)
    assert isinstance(result["reasons"], list)
    assert len(result["reasons"]) >= 2
    reasons_text = " ".join(result["reasons"])
    assert "match_rate" in reasons_text
    assert "mean_iou" in reasons_text


def test_gate_result_pass_has_empty_reasons():
    result = compute_gate_result(PASSING_REPORT)
    assert result["reasons"] == []


def test_gate_result_includes_measured_metrics():
    result = compute_gate_result(PASSING_REPORT)
    assert "metrics" in result
    assert "match_rate" in result["metrics"]
    assert "mean_iou" in result["metrics"]


def test_gate_respects_custom_thresholds():
    strict = {"match_rate": 0.9999, "mean_iou": 0.9999}
    result = compute_gate_result(PASSING_REPORT, thresholds=strict)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Slice 2 — build_manifest
# ---------------------------------------------------------------------------


def test_manifest_includes_engine_path(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"fake engine bytes")
    gate_result = {"passed": True, "reasons": [], "metrics": {}}
    manifest = build_manifest(str(engine), gate_result)
    assert manifest["engine_path"] == str(engine)


def test_manifest_includes_sha256_of_engine_file(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"fake engine bytes")
    gate_result = {"passed": True, "reasons": [], "metrics": {}}
    manifest = build_manifest(str(engine), gate_result)
    assert "sha256" in manifest
    assert len(manifest["sha256"]) == 64  # hex SHA-256


def test_manifest_includes_gate_passed_flag(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"x")
    gate_result = {"passed": False, "reasons": ["low iou"], "metrics": {}}
    manifest = build_manifest(str(engine), gate_result)
    assert manifest["gate_passed"] is False


def test_manifest_includes_timestamp(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"x")
    gate_result = {"passed": True, "reasons": [], "metrics": {}}
    fixed_ts = "2026-06-24T00:00:00Z"
    manifest = build_manifest(str(engine), gate_result, _now=lambda: fixed_ts)
    assert manifest["timestamp"] == fixed_ts


def test_manifest_includes_gate_result_block(tmp_path):
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"x")
    gate_result = {"passed": True, "reasons": [], "metrics": {"match_rate": 0.999}}
    manifest = build_manifest(str(engine), gate_result)
    assert manifest["gate_result"] == gate_result


# ---------------------------------------------------------------------------
# Slice 3 — write_manifest / round-trip
# ---------------------------------------------------------------------------


def test_write_manifest_creates_json_file(tmp_path):
    out = tmp_path / "manifest.json"
    manifest = {"gate_passed": True, "sha256": "abc"}
    write_manifest(manifest, str(out))
    assert out.exists()


def test_write_manifest_roundtrip(tmp_path):
    out = tmp_path / "manifest.json"
    manifest = {"gate_passed": True, "sha256": "abc", "engine_path": "/some/path"}
    write_manifest(manifest, str(out))
    loaded = json.loads(out.read_text())
    assert loaded == manifest


# ---------------------------------------------------------------------------
# Slice 4 — parse_args
# ---------------------------------------------------------------------------


def test_parse_args_accuracy_json():
    args = parse_args(["--accuracy-json", "metrics/results/accuracy.json",
                       "--engine", "models/engines/yolo26n_fp16_b3.engine"])
    assert args.accuracy_json == "metrics/results/accuracy.json"


def test_parse_args_engine_path():
    args = parse_args(["--accuracy-json", "acc.json",
                       "--engine", "model.engine"])
    assert args.engine == "model.engine"


def test_parse_args_threshold_overrides():
    args = parse_args(["--accuracy-json", "acc.json", "--engine", "m.engine",
                       "--min-match-rate", "0.995", "--min-mean-iou", "0.995"])
    assert args.min_match_rate == pytest.approx(0.995)
    assert args.min_mean_iou == pytest.approx(0.995)


def test_parse_args_output_manifest_default():
    args = parse_args(["--accuracy-json", "acc.json", "--engine", "m.engine"])
    assert args.output_manifest is None
