"""Model-promotion gate.

Reads the JSON output of validate_accuracy.py, checks measured metrics against
configurable thresholds, and writes a signed manifest.  Exits 0 (pass) or 1
(fail) so it can be used as a shell/CI gate.

Usage:
    python3 metrics/model_gate.py \\
        --accuracy-json metrics/results/accuracy.json \\
        --engine models/engines/yolo26n_fp16_b3.engine \\
        [--output-manifest metrics/results/gate_manifest.json]
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_THRESHOLDS = {
    "match_rate": 0.95,
    "mean_iou": 0.95,
}


def compute_gate_result(
    report: dict,
    thresholds: dict | None = None,
) -> dict:
    """Evaluate an accuracy report against thresholds.

    Args:
        report: dict as written by validate_accuracy.py (must contain
                report["fp16_vs_fp32"]["match_rate"] and ["mean_iou"]).
        thresholds: optional overrides for "match_rate" and "mean_iou".
                    Defaults to 0.99 for both.

    Returns:
        {"passed": bool, "reasons": list[str], "metrics": dict}
    """
    t = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    fp16 = report["fp16_vs_fp32"]
    match_rate = fp16["match_rate"]
    mean_iou = fp16["mean_iou"]

    reasons: list[str] = []
    if match_rate < t["match_rate"]:
        reasons.append(
            f"match_rate {match_rate:.4f} < threshold {t['match_rate']:.4f}"
        )
    if mean_iou < t["mean_iou"]:
        reasons.append(
            f"mean_iou {mean_iou:.4f} < threshold {t['mean_iou']:.4f}"
        )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {"match_rate": match_rate, "mean_iou": mean_iou},
    }


def build_manifest(
    engine_path: str,
    gate_result: dict,
    _hasher=None,
    _now=None,
) -> dict:
    """Build a manifest dict for the given engine and gate result.

    Args:
        engine_path: path to the TRT engine file (used for SHA-256 hash).
        gate_result: output of compute_gate_result().
        _hasher: injectable callable(path) -> hex_str (for tests without a real engine).
        _now: injectable callable() -> ISO-8601 timestamp string (for tests).
    """
    if _hasher is None:
        def _hasher(p: str) -> str:
            h = hashlib.sha256()
            h.update(Path(p).read_bytes())
            return h.hexdigest()

    if _now is None:
        def _now() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "engine_path": engine_path,
        "sha256": _hasher(engine_path),
        "timestamp": _now(),
        "gate_passed": gate_result["passed"],
        "gate_result": gate_result,
    }


def write_manifest(manifest: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-promotion gate")
    p.add_argument("--accuracy-json", required=True, dest="accuracy_json",
                   metavar="PATH", help="JSON output from validate_accuracy.py")
    p.add_argument("--engine", required=True, dest="engine",
                   metavar="PATH", help="TRT engine file to hash and gate")
    p.add_argument("--min-match-rate", type=float, default=_DEFAULT_THRESHOLDS["match_rate"],
                   dest="min_match_rate", metavar="FLOAT",
                   help=f"min match_rate threshold (default {_DEFAULT_THRESHOLDS['match_rate']})")
    p.add_argument("--min-mean-iou", type=float, default=_DEFAULT_THRESHOLDS["mean_iou"],
                   dest="min_mean_iou", metavar="FLOAT",
                   help=f"min mean_iou threshold (default {_DEFAULT_THRESHOLDS['mean_iou']})")
    p.add_argument("--output-manifest", default=None, dest="output_manifest",
                   metavar="PATH", help="write manifest JSON to PATH (optional)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = json.loads(Path(args.accuracy_json).read_text())
    thresholds = {"match_rate": args.min_match_rate, "mean_iou": args.min_mean_iou}

    gate_result = compute_gate_result(report, thresholds)
    manifest = build_manifest(args.engine, gate_result)

    status = "PASS" if gate_result["passed"] else "FAIL"
    print(f"[gate] {status}: match_rate={gate_result['metrics']['match_rate']:.4f}  "
          f"mean_iou={gate_result['metrics']['mean_iou']:.4f}")
    if gate_result["reasons"]:
        for r in gate_result["reasons"]:
            print(f"[gate]   ✗ {r}")

    if args.output_manifest:
        write_manifest(manifest, args.output_manifest)
        print(f"[gate] Manifest written → {args.output_manifest}")

    return 0 if gate_result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
