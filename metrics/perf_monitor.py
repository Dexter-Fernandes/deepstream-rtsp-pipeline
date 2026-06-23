"""FPS / RSS / VRAM monitoring for the DeepStream pipeline.

Pure Python — no GStreamer or pyds imports. Fully CPU-testable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


def compute_interval_fps(curr: int, prev: int, dt_s: float) -> float:
    """Frames per second over [prev, curr] frames in dt_s seconds."""
    if dt_s <= 0:
        return 0.0
    delta = curr - prev
    if delta <= 0:
        return 0.0
    return delta / dt_s


def sample_rss_mb(reader: Optional[Callable[[], str]] = None) -> float:
    """Return current RSS in MB from /proc/self/status."""
    if reader is None:
        def reader():  # noqa: E306
            with open("/proc/self/status") as f:
                return f.read()
    text = reader()
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            # "VmRSS:  524288 kB"
            return int(parts[1]) / 1024.0
    return 0.0


def sample_vram_mb(runner: Optional[Callable[[str], str]] = None) -> float:
    """Return used VRAM in MB via nvidia-smi."""
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
    if runner is None:
        import subprocess
        def runner(c):  # noqa: E306
            return subprocess.check_output(c.split(), text=True)
    try:
        out = runner(cmd).strip()
        if not out:
            return 0.0
        return float(out.splitlines()[0].strip())
    except Exception:
        return 0.0


@dataclass
class _Sample:
    t: float
    frame_counts: Dict[int, int]
    vram_mb: float
    rss_mb: float


class PerfMonitor:
    """Accumulates per-interval samples and computes a summary at shutdown."""

    # RSS slope above this (MB/min) + delta above _LEAK_DELTA_MB → leak_suspected
    _LEAK_SLOPE_MB_PER_MIN = 5.0
    _LEAK_DELTA_MB = 20.0

    def __init__(self, num_sources: int, start_t: float = 0.0):
        self._num_sources = num_sources
        self._samples: List[_Sample] = []
        self._start_t: float = start_t

    def record(
        self,
        t: float,
        frame_counts: Dict[int, int],
        vram_mb: float,
        rss_mb: float,
    ) -> None:
        self._samples.append(_Sample(t=t, frame_counts=dict(frame_counts), vram_mb=vram_mb, rss_mb=rss_mb))

    def summary(self) -> dict:
        if not self._samples:
            return {}

        start_t = self._start_t
        end_t = self._samples[-1].t
        duration_s = end_t - start_t

        # Total frames per source (last sample's cumulative counts)
        last_counts = self._samples[-1].frame_counts
        total_frames = sum(last_counts.values())

        mean_fps_per_source = (
            (total_frames / self._num_sources / duration_s) if duration_s > 0 and self._num_sources > 0 else 0.0
        )
        fps_total = total_frames / duration_s if duration_s > 0 else 0.0

        vrams = [s.vram_mb for s in self._samples]
        peak_vram_mb = max(vrams)
        mean_vram_mb = sum(vrams) / len(vrams)

        rss_start = self._samples[0].rss_mb
        rss_end = self._samples[-1].rss_mb
        rss_delta = rss_end - rss_start

        rss_slope = self._rss_slope_mb_per_min()

        leak_suspected = (
            rss_slope > self._LEAK_SLOPE_MB_PER_MIN and rss_delta > self._LEAK_DELTA_MB
        )

        return {
            "mean_fps_per_source": mean_fps_per_source,
            "fps_total": fps_total,
            "peak_vram_mb": peak_vram_mb,
            "mean_vram_mb": mean_vram_mb,
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_end,
            "rss_delta_mb": rss_delta,
            "rss_slope_mb_per_min": rss_slope,
            "leak_suspected": leak_suspected,
            "duration_s": duration_s,
            "total_frames": total_frames,
        }

    def _rss_slope_mb_per_min(self) -> float:
        """Linear regression slope of RSS vs time (MB/min)."""
        n = len(self._samples)
        if n < 2:
            return 0.0
        ts = [s.t / 60.0 for s in self._samples]  # minutes
        rs = [s.rss_mb for s in self._samples]
        t_mean = sum(ts) / n
        r_mean = sum(rs) / n
        num = sum((ts[i] - t_mean) * (rs[i] - r_mean) for i in range(n))
        den = sum((ts[i] - t_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "samples": [
                {
                    "t": s.t,
                    "frame_counts": s.frame_counts,
                    "vram_mb": s.vram_mb,
                    "rss_mb": s.rss_mb,
                }
                for s in self._samples
            ],
        }

    def write_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        s = self.summary()
        if not s:
            return
        print(
            f"[perf] duration={s['duration_s']:.1f}s  "
            f"fps/stream={s['mean_fps_per_source']:.1f}  "
            f"fps_total={s['fps_total']:.1f}  "
            f"vram_peak={s['peak_vram_mb']:.0f}MB  "
            f"rss_delta={s['rss_delta_mb']:+.1f}MB  "
            f"leak={s['leak_suspected']}"
        )
