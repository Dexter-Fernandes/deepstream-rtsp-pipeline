"""CPU-safe unit tests for metrics/health_monitor.py — no GStreamer required."""
import pytest

from metrics.health_monitor import HealthMonitor


# ---------------------------------------------------------------------------
# Slice 1 — initialisation
# ---------------------------------------------------------------------------


def test_health_monitor_init_num_sources():
    hm = HealthMonitor(num_sources=2)
    snap = hm.snapshot(t_now=0.0)
    assert len(snap["sources"]) == 2


def test_health_monitor_init_sources_have_expected_keys():
    hm = HealthMonitor(num_sources=1)
    src = hm.snapshot(t_now=0.0)["sources"][0]
    assert src["source_id"] == 0
    assert "is_live" in src
    assert "time_since_last_frame_s" in src
    assert "time_since_last_detection_s" in src
    assert "current_fps" in src
    assert "fps_vs_expected" in src


# ---------------------------------------------------------------------------
# Slice 2 — record_frame updates last_frame_t
# ---------------------------------------------------------------------------


def test_record_frame_updates_time_since_last_frame():
    hm = HealthMonitor(num_sources=1)
    hm.record_frame(source_id=0, t=10.0)
    src = hm.snapshot(t_now=11.0)["sources"][0]
    assert src["time_since_last_frame_s"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Slice 3 — is_live False when > liveness_window_s since last frame
# ---------------------------------------------------------------------------


def test_snapshot_source_is_dead_after_liveness_window():
    hm = HealthMonitor(num_sources=1, liveness_window_s=5.0)
    hm.record_frame(source_id=0, t=1.0)
    src = hm.snapshot(t_now=6.1)["sources"][0]
    assert src["is_live"] is False


# ---------------------------------------------------------------------------
# Slice 4 — is_live True when within liveness_window_s
# ---------------------------------------------------------------------------


def test_snapshot_source_is_live_within_window():
    hm = HealthMonitor(num_sources=1, liveness_window_s=5.0)
    hm.record_frame(source_id=0, t=1.0)
    src = hm.snapshot(t_now=2.0)["sources"][0]
    assert src["is_live"] is True


# ---------------------------------------------------------------------------
# Slice 5 — has_detection=True updates last_detection_t
# ---------------------------------------------------------------------------


def test_record_frame_with_detection_updates_detection_time():
    hm = HealthMonitor(num_sources=1)
    hm.record_frame(source_id=0, t=5.0, has_detection=True)
    src = hm.snapshot(t_now=6.0)["sources"][0]
    assert src["time_since_last_detection_s"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Slice 6 — has_detection=False does NOT update last_detection_t
# ---------------------------------------------------------------------------


def test_record_frame_without_detection_does_not_update_detection_time():
    hm = HealthMonitor(num_sources=1)
    hm.record_frame(source_id=0, t=1.0, has_detection=True)
    hm.record_frame(source_id=0, t=2.0, has_detection=False)
    src = hm.snapshot(t_now=4.0)["sources"][0]
    # last detection was at t=1.0, so 3 s ago
    assert src["time_since_last_detection_s"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Slice 7 — current_fps tracks rolling frame rate
# ---------------------------------------------------------------------------


def test_snapshot_current_fps_matches_uniform_rate():
    hm = HealthMonitor(num_sources=1, expected_fps=25.0)
    # Feed 50 frames at 25 fps (one every 0.04 s) over 2 seconds
    for i in range(50):
        hm.record_frame(source_id=0, t=float(i) * 0.04)
    src = hm.snapshot(t_now=50 * 0.04)["sources"][0]
    assert src["current_fps"] == pytest.approx(25.0, rel=0.1)


# ---------------------------------------------------------------------------
# Slice 8 — fps_vs_expected ratio
# ---------------------------------------------------------------------------


def test_snapshot_fps_vs_expected_ratio():
    hm = HealthMonitor(num_sources=1, expected_fps=25.0)
    for i in range(50):
        hm.record_frame(source_id=0, t=float(i) * 0.04)
    src = hm.snapshot(t_now=50 * 0.04)["sources"][0]
    assert src["fps_vs_expected"] == pytest.approx(1.0, rel=0.1)


# ---------------------------------------------------------------------------
# Slice 9 — system dict when vram_mb / rss_mb supplied
# ---------------------------------------------------------------------------


def test_snapshot_system_dict_present_when_supplied():
    hm = HealthMonitor(num_sources=1)
    snap = hm.snapshot(t_now=0.0, vram_mb=781.0, rss_mb=420.5)
    assert snap["system"]["vram_mb"] == pytest.approx(781.0)
    assert snap["system"]["rss_mb"] == pytest.approx(420.5)


def test_snapshot_system_dict_absent_when_not_supplied():
    hm = HealthMonitor(num_sources=1)
    snap = hm.snapshot(t_now=0.0)
    assert "system" not in snap


# ---------------------------------------------------------------------------
# Slice 10 — never-seen source: is_live=False, time_since_last_detection_s=None
# ---------------------------------------------------------------------------


def test_snapshot_never_seen_source_is_dead():
    hm = HealthMonitor(num_sources=2, liveness_window_s=5.0)
    # Only record frames for source 1; source 0 never seen
    hm.record_frame(source_id=1, t=1.0)
    src0 = hm.snapshot(t_now=3.0)["sources"][0]
    assert src0["is_live"] is False
    assert src0["time_since_last_detection_s"] is None
    assert src0["time_since_last_frame_s"] is None
