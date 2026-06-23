"""Per-sensor health tracking for the DeepStream pipeline.

Pure Python — no GStreamer or pyds imports. Fully CPU-testable.

HealthMonitor tracks per-source liveness, rolling FPS, and detection
staleness. Call record_frame() from the GStreamer probe on every frame;
call snapshot() from the periodic GLib callback to emit a health log line.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional


class HealthMonitor:
    """Tracks liveness, FPS, and detection staleness per source.

    Thread safety: not required — record_frame() and snapshot() both run
    on the GLib main loop thread.
    """

    # Rolling window size for FPS estimation (frames)
    _FPS_WINDOW = 150

    def __init__(
        self,
        num_sources: int,
        expected_fps: float = 25.0,
        liveness_window_s: float = 5.0,
    ) -> None:
        self._num_sources = num_sources
        self._expected_fps = expected_fps
        self._liveness_window_s = liveness_window_s

        # Per-source state — None means "never seen"
        self._last_frame_t: Dict[int, Optional[float]] = {i: None for i in range(num_sources)}
        self._last_detection_t: Dict[int, Optional[float]] = {i: None for i in range(num_sources)}
        # Deque of recent frame timestamps for rolling FPS calculation
        self._frame_times: Dict[int, Deque[float]] = {
            i: deque(maxlen=self._FPS_WINDOW) for i in range(num_sources)
        }

    def record_frame(
        self,
        source_id: int,
        t: float,
        has_detection: bool = False,
    ) -> None:
        """Record that source_id produced a frame at monotonic time t."""
        self._last_frame_t[source_id] = t
        self._frame_times[source_id].append(t)
        if has_detection:
            self._last_detection_t[source_id] = t

    def snapshot(
        self,
        t_now: float,
        vram_mb: Optional[float] = None,
        rss_mb: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return current health for all sources at monotonic time t_now."""
        sources = []
        for i in range(self._num_sources):
            sources.append(self._source_snapshot(i, t_now))

        result: Dict[str, Any] = {"sources": sources}
        if vram_mb is not None or rss_mb is not None:
            result["system"] = {"vram_mb": vram_mb, "rss_mb": rss_mb}
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _source_snapshot(self, source_id: int, t_now: float) -> Dict[str, Any]:
        last_frame = self._last_frame_t[source_id]
        last_det = self._last_detection_t[source_id]

        if last_frame is None:
            return {
                "source_id": source_id,
                "is_live": False,
                "time_since_last_frame_s": None,
                "time_since_last_detection_s": None,
                "current_fps": 0.0,
                "fps_vs_expected": 0.0,
            }

        time_since_frame = t_now - last_frame
        is_live = time_since_frame <= self._liveness_window_s
        time_since_det = (t_now - last_det) if last_det is not None else None

        current_fps = self._rolling_fps(source_id, t_now)
        fps_vs_expected = (
            current_fps / self._expected_fps if self._expected_fps > 0 else 0.0
        )

        return {
            "source_id": source_id,
            "is_live": is_live,
            "time_since_last_frame_s": time_since_frame,
            "time_since_last_detection_s": time_since_det,
            "current_fps": current_fps,
            "fps_vs_expected": fps_vs_expected,
        }

    def _rolling_fps(self, source_id: int, t_now: float) -> float:
        """Estimate FPS from the rolling frame-timestamp window."""
        times = self._frame_times[source_id]
        n = len(times)
        if n < 2:
            return 0.0
        dt = t_now - times[0]
        if dt <= 0:
            return 0.0
        # n frames were recorded over dt seconds; the rate is (n-1)/dt
        # but using n/dt is the conventional "frames in window / window length"
        return (n - 1) / dt
