"""Integration tests: pipeline output → MOT metrics evaluation.

Stage 1 (GPU): run multi_stream.py on mot17_04.mp4 to produce a tracker CSV.
Stage 2 (CPU): evaluate the CSV against MOT17-04 ground truth via evaluate_tracker.

Requires a physical NVIDIA GPU and the NGC DeepStream container.
Run with: pytest tests/ --gpu -v -k integration
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipelines" / "multi_stream.py"
VIDEO = ROOT / "data" / "mot17_04.mp4"
GT = ROOT / "data" / "MOT17-04-FRCNN" / "gt" / "gt.txt"

_RUN_SECONDS = 30


def _run_pipeline(output_dir: Path) -> Path:
    """Run the pipeline for _RUN_SECONDS seconds and return the CSV path."""
    subprocess.run(
        [sys.executable, str(PIPELINE),
         "--uri", str(VIDEO),
         "--duration", str(_RUN_SECONDS),
         "--no-sync",
         "--output-dir", str(output_dir),
         "--perf-json", str(output_dir / "perf.json")],
        cwd=ROOT,
        timeout=_RUN_SECONDS + 30,
        check=True,
        capture_output=True,
    )
    return output_dir / "output_stream0.csv"


@pytest.mark.gpu
def test_mota_is_not_catastrophic(tmp_path):
    """MOTA > -0.5 means the tracker isn't creating more phantom tracks than real ones."""
    if not GT.exists():
        pytest.skip(f"GT file not found: {GT}")

    csv_path = _run_pipeline(tmp_path)

    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from metrics.evaluate_tracker import main as eval_main

    metrics = eval_main([
        "--gt", str(GT),
        "--pred", str(csv_path),
        "--min-visibility", "0.0",
    ])
    assert metrics["MOTA"] is not None, "MOTA could not be computed"
    assert metrics["MOTA"] > -0.5, f"MOTA={metrics['MOTA']:.3f} is catastrophically low"


@pytest.mark.gpu
def test_idf1_is_positive(tmp_path):
    """IDF1 > 0 means the tracker maintains at least some identity continuity."""
    if not GT.exists():
        pytest.skip(f"GT file not found: {GT}")

    csv_path = _run_pipeline(tmp_path)

    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from metrics.evaluate_tracker import main as eval_main

    metrics = eval_main([
        "--gt", str(GT),
        "--pred", str(csv_path),
        "--min-visibility", "0.0",
    ])
    assert metrics["IDF1"] is not None, "IDF1 could not be computed"
    assert metrics["IDF1"] > 0, f"IDF1={metrics['IDF1']:.3f} — no identity continuity"
