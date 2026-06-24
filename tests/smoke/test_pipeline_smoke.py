"""Smoke tests for the multi-stream pipeline.

Requires a physical NVIDIA GPU and the NGC DeepStream container.
Run with: pytest tests/ --gpu -v -k smoke
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipelines" / "multi_stream.py"
VIDEO = ROOT / "data" / "mot17_04.mp4"

_CMD_BASE = [
    sys.executable, str(PIPELINE),
    "--uri", str(VIDEO),
    "--duration", "10",
    "--no-sync",
]


@pytest.mark.gpu
def test_pipeline_exits_cleanly(tmp_path):
    result = subprocess.run(
        _CMD_BASE + ["--output-dir", str(tmp_path),
                     "--perf-json", str(tmp_path / "perf.json")],
        cwd=ROOT,
        timeout=60,
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"Pipeline exited with {result.returncode}\n"
        f"STDERR: {result.stderr.decode()[-2000:]}"
    )


@pytest.mark.gpu
def test_pipeline_writes_frames(tmp_path):
    perf_json = tmp_path / "perf.json"
    subprocess.run(
        _CMD_BASE + ["--output-dir", str(tmp_path),
                     "--perf-json", str(perf_json)],
        cwd=ROOT,
        timeout=60,
        check=True,
        capture_output=True,
    )
    assert perf_json.exists(), "perf JSON was not written"
    data = json.loads(perf_json.read_text())
    summary = data["summary"]
    assert summary["total_frames"] > 0, "no frames processed"
    assert summary["mean_fps_per_source"] > 0, "zero FPS reported"


@pytest.mark.gpu
def test_pipeline_writes_csv(tmp_path):
    subprocess.run(
        _CMD_BASE + ["--output-dir", str(tmp_path),
                     "--perf-json", str(tmp_path / "perf.json")],
        cwd=ROOT,
        timeout=60,
        check=True,
        capture_output=True,
    )
    csv_path = tmp_path / "output_stream0.csv"
    assert csv_path.exists(), f"CSV not found at {csv_path}"
    lines = csv_path.read_text().splitlines()
    assert lines, "CSV is empty"
    assert lines[0].startswith("frame_num"), f"unexpected CSV header: {lines[0]}"
    assert len(lines) > 1, "CSV has header but no data rows"
