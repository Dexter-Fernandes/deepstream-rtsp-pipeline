"""CPU-safe unit tests for metrics/perf_monitor.py — no GPU or GStreamer required."""
import json
import math
from pathlib import Path

import pytest

from metrics.perf_monitor import (
    PerfMonitor,
    compute_interval_fps,
    sample_rss_mb,
    sample_vram_mb,
)


# ---------------------------------------------------------------------------
# Slice 1 — compute_interval_fps
# ---------------------------------------------------------------------------


def test_compute_interval_fps_basic():
    assert compute_interval_fps(curr=50, prev=0, dt_s=2.0) == pytest.approx(25.0)


def test_compute_interval_fps_zero_dt_returns_zero():
    assert compute_interval_fps(curr=100, prev=0, dt_s=0.0) == 0.0


def test_compute_interval_fps_negative_dt_returns_zero():
    assert compute_interval_fps(curr=100, prev=0, dt_s=-1.0) == 0.0


def test_compute_interval_fps_no_new_frames_returns_zero():
    assert compute_interval_fps(curr=50, prev=50, dt_s=5.0) == 0.0


# ---------------------------------------------------------------------------
# Slice 2 — PerfMonitor.record
# ---------------------------------------------------------------------------


def test_perf_monitor_record_stores_sample():
    mon = PerfMonitor(num_sources=2)
    mon.record(t=1.0, frame_counts={0: 25, 1: 25}, vram_mb=1000.0, rss_mb=512.0)
    assert len(mon._samples) == 1


def test_perf_monitor_record_multiple_samples():
    mon = PerfMonitor(num_sources=1)
    mon.record(t=1.0, frame_counts={0: 10}, vram_mb=500.0, rss_mb=300.0)
    mon.record(t=6.0, frame_counts={0: 135}, vram_mb=520.0, rss_mb=305.0)
    assert len(mon._samples) == 2


# ---------------------------------------------------------------------------
# Slice 3 — PerfMonitor.summary
# ---------------------------------------------------------------------------


def _make_monitor_with_two_ticks():
    mon = PerfMonitor(num_sources=2)
    # t=0 is start_time implicitly; first record at t=5 with 125 frames each source
    mon.record(t=5.0, frame_counts={0: 125, 1: 125}, vram_mb=1000.0, rss_mb=512.0)
    mon.record(t=10.0, frame_counts={0: 250, 1: 250}, vram_mb=1050.0, rss_mb=514.0)
    return mon


def test_summary_mean_fps_per_source():
    mon = _make_monitor_with_two_ticks()
    s = mon.summary()
    # 250 frames / 10 s = 25.0 fps/source
    assert s["mean_fps_per_source"] == pytest.approx(25.0)


def test_summary_fps_total():
    mon = _make_monitor_with_two_ticks()
    s = mon.summary()
    # 2 sources × 25 fps = 50 fps total
    assert s["fps_total"] == pytest.approx(50.0)


def test_summary_peak_vram():
    mon = _make_monitor_with_two_ticks()
    assert mon.summary()["peak_vram_mb"] == pytest.approx(1050.0)


def test_summary_mean_vram():
    mon = _make_monitor_with_two_ticks()
    assert mon.summary()["mean_vram_mb"] == pytest.approx(1025.0)


def test_summary_rss_fields():
    mon = _make_monitor_with_two_ticks()
    s = mon.summary()
    assert s["rss_start_mb"] == pytest.approx(512.0)
    assert s["rss_end_mb"] == pytest.approx(514.0)
    assert s["rss_delta_mb"] == pytest.approx(2.0)


def test_summary_duration_and_total_frames():
    mon = _make_monitor_with_two_ticks()
    s = mon.summary()
    assert s["duration_s"] == pytest.approx(10.0)
    assert s["total_frames"] == 500  # 250 × 2 sources


def test_summary_leak_suspected_false_for_flat_rss():
    mon = PerfMonitor(num_sources=1)
    for i in range(6):
        mon.record(t=float(i * 60), frame_counts={0: i * 1500}, vram_mb=1000.0, rss_mb=512.0)
    assert mon.summary()["leak_suspected"] is False


def test_summary_leak_suspected_true_for_growing_rss():
    mon = PerfMonitor(num_sources=1)
    for i in range(6):
        # RSS grows 50 MB/min — well above threshold
        mon.record(t=float(i * 60), frame_counts={0: i * 1500}, vram_mb=1000.0, rss_mb=512.0 + i * 50)
    assert mon.summary()["leak_suspected"] is True


# ---------------------------------------------------------------------------
# Slice 4 — to_dict / write_json round-trip
# ---------------------------------------------------------------------------


def test_to_dict_contains_summary_and_samples():
    mon = _make_monitor_with_two_ticks()
    d = mon.to_dict()
    assert "summary" in d
    assert "samples" in d
    assert len(d["samples"]) == 2


def test_write_json_round_trips(tmp_path):
    mon = _make_monitor_with_two_ticks()
    path = tmp_path / "perf.json"
    mon.write_json(str(path))
    loaded = json.loads(path.read_text())
    assert "summary" in loaded
    assert loaded["summary"]["mean_fps_per_source"] == pytest.approx(25.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Slice 5 — sample_rss_mb
# ---------------------------------------------------------------------------


def test_sample_rss_mb_parses_vmrss_line():
    fake_status = "VmPeak:\t 1024 kB\nVmRSS:\t  524288 kB\nVmData:\t 256 kB\n"
    result = sample_rss_mb(reader=lambda: fake_status)
    assert result == pytest.approx(512.0)


def test_sample_rss_mb_returns_zero_on_missing_line():
    result = sample_rss_mb(reader=lambda: "VmPeak:\t1024 kB\n")
    assert result == 0.0


# ---------------------------------------------------------------------------
# Slice 6 — sample_vram_mb
# ---------------------------------------------------------------------------


def test_sample_vram_mb_parses_nvidia_smi_output():
    result = sample_vram_mb(runner=lambda _cmd: "3456\n")
    assert result == pytest.approx(3456.0)


def test_sample_vram_mb_returns_zero_on_failure():
    def bad_runner(_cmd):
        raise OSError("nvidia-smi not found")
    assert sample_vram_mb(runner=bad_runner) == 0.0


def test_sample_vram_mb_returns_zero_on_empty_output():
    assert sample_vram_mb(runner=lambda _cmd: "") == 0.0
