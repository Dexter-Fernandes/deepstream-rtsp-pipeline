from pathlib import Path

import pytest

from metrics.profile_decode import _SimpleProfiler, _parse_tail_latencies, budget_check

_LATENCY_LINE = (
    "[I] Latency: min = 3.12 ms, max = 4.56 ms, mean = 3.26 ms, "
    "median = 3.25 ms, percentile(90%) = 3.44 ms, percentile(95%) = 3.57 ms, "
    "percentile(99%) = 3.89 ms"
)


# ---------------------------------------------------------------------------
# _parse_tail_latencies
# ---------------------------------------------------------------------------


def test_parse_tail_latencies_all_fields():
    result = _parse_tail_latencies(_LATENCY_LINE)
    assert result["min_ms"] == pytest.approx(3.12)
    assert result["max_ms"] == pytest.approx(4.56)
    assert result["median_ms"] == pytest.approx(3.25)
    assert result["p99_ms"] == pytest.approx(3.89)


def test_parse_tail_latencies_no_match():
    result = _parse_tail_latencies("no latency data here")
    assert result == {"min_ms": 0.0, "median_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}


def test_parse_tail_latencies_missing_p99():
    line = "[I] Latency: min = 3.12 ms, max = 4.56 ms, mean = 3.26 ms, median = 3.25 ms"
    result = _parse_tail_latencies(line)
    assert result["min_ms"] == pytest.approx(3.12)
    assert result["median_ms"] == pytest.approx(3.25)
    assert result["p99_ms"] == pytest.approx(0.0)


def test_parse_tail_latencies_multiline_output():
    output = "\n".join([
        "[I] Starting inference...",
        _LATENCY_LINE,
        "[I] Throughput: 306.7 qps",
    ])
    result = _parse_tail_latencies(output)
    assert result["p99_ms"] == pytest.approx(3.89)
    assert result["min_ms"] == pytest.approx(3.12)


def test_parse_tail_latencies_integer_values():
    line = "[I] Latency: min = 3 ms, max = 5 ms, mean = 4 ms, median = 4 ms, percentile(99%) = 5 ms"
    result = _parse_tail_latencies(line)
    assert result["min_ms"] == pytest.approx(3.0)
    assert result["max_ms"] == pytest.approx(5.0)
    assert result["p99_ms"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# budget_check
# ---------------------------------------------------------------------------


def test_budget_check_within_budget():
    result = {"wall_ms": 36.2, "p99_ms": 38.5}
    check = budget_check(result, budget_ms=40.0)
    assert check["mean_ok"] is True
    assert check["p99_ok"] is True
    assert check["budget_ms"] == pytest.approx(40.0)


def test_budget_check_p99_exceeds_budget():
    result = {"wall_ms": 36.2, "p99_ms": 42.1}
    check = budget_check(result, budget_ms=40.0)
    assert check["mean_ok"] is True
    assert check["p99_ok"] is False


def test_budget_check_no_p99_falls_back_to_mean():
    result = {"wall_ms": 36.2}
    check = budget_check(result, budget_ms=40.0)
    assert check["p99_ms"] == pytest.approx(36.2)
    assert check["p99_ok"] is True


# ---------------------------------------------------------------------------
# _SimpleProfiler.to_dict with tail data
# ---------------------------------------------------------------------------


def test_to_dict_includes_tail_fields():
    tail = {"min_ms": 3.12, "median_ms": 3.25, "p99_ms": 3.89, "max_ms": 4.56}
    profiler = _SimpleProfiler(layers={}, wall_ms=3.26, tail=tail)
    d = profiler.to_dict(label="test", engine_path=Path("dummy.engine"))
    assert d["min_ms"] == pytest.approx(3.12)
    assert d["median_ms"] == pytest.approx(3.25)
    assert d["p99_ms"] == pytest.approx(3.89)
    assert d["max_ms"] == pytest.approx(4.56)


def test_to_dict_tail_defaults_to_zero_when_absent():
    profiler = _SimpleProfiler(layers={}, wall_ms=3.26)
    d = profiler.to_dict(label="test", engine_path=Path("dummy.engine"))
    assert d["min_ms"] == pytest.approx(0.0)
    assert d["median_ms"] == pytest.approx(0.0)
    assert d["p99_ms"] == pytest.approx(0.0)
    assert d["max_ms"] == pytest.approx(0.0)
