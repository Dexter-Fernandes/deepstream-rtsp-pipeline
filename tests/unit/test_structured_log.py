"""CPU-safe unit tests for pipelines/structured_log.py — no GStreamer required."""
import io
import json
import logging

import pytest

from pipelines.structured_log import (
    configure_pipeline_logging,
    get_pipeline_logger,
    log_event,
)


def _fresh_namespace(suffix: str) -> str:
    """Return a unique logger namespace so tests don't share handler state."""
    return f"pipeline._test_{suffix}"


# ---------------------------------------------------------------------------
# Slice 1 — get_pipeline_logger returns a Logger
# ---------------------------------------------------------------------------


def test_get_pipeline_logger_returns_logger():
    logger = get_pipeline_logger("test_basic")
    assert isinstance(logger, logging.Logger)


def test_get_pipeline_logger_name_is_namespaced():
    logger = get_pipeline_logger("mymodule")
    assert logger.name == "pipeline.mymodule"


# ---------------------------------------------------------------------------
# Slice 2 — log_event writes valid JSON to the configured stream
# ---------------------------------------------------------------------------


def test_log_event_writes_json_line():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_json")
    log_event(logger, logging.INFO, event="pipeline_start")
    output = buf.getvalue().strip()
    assert output != ""
    parsed = json.loads(output)
    assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Slice 3 — JSON contains required fields: ts, logger, level, event
# ---------------------------------------------------------------------------


def test_log_event_contains_required_fields():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_fields")
    log_event(logger, logging.INFO, event="pipeline_start")
    parsed = json.loads(buf.getvalue().strip())
    assert "ts" in parsed
    assert "logger" in parsed
    assert "level" in parsed
    assert "event" in parsed
    assert parsed["event"] == "pipeline_start"
    assert parsed["level"] == "INFO"
    assert isinstance(parsed["ts"], float)


# ---------------------------------------------------------------------------
# Slice 4 — source_id present when supplied, absent when omitted
# ---------------------------------------------------------------------------


def test_log_event_source_id_present_when_supplied():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_src_id")
    log_event(logger, logging.INFO, source_id=2, event="frame_received")
    parsed = json.loads(buf.getvalue().strip())
    assert parsed["source_id"] == 2


def test_log_event_source_id_absent_when_not_supplied():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_no_src_id")
    log_event(logger, logging.INFO, event="pipeline_start")
    parsed = json.loads(buf.getvalue().strip())
    assert "source_id" not in parsed


# ---------------------------------------------------------------------------
# Slice 5 — arbitrary **fields appear in JSON
# ---------------------------------------------------------------------------


def test_log_event_extra_fields_appear_in_json():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_extra")
    log_event(logger, logging.INFO, event="perf_tick", fps_per_stream=24.9, vram_mb=781.0)
    parsed = json.loads(buf.getvalue().strip())
    assert parsed["fps_per_stream"] == pytest.approx(24.9)
    assert parsed["vram_mb"] == pytest.approx(781.0)


# ---------------------------------------------------------------------------
# Slice 6 — configure_pipeline_logging is idempotent (no double handler)
# ---------------------------------------------------------------------------


def test_configure_pipeline_logging_idempotent():
    buf = io.StringIO()
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    configure_pipeline_logging(level=logging.DEBUG, stream=buf)
    logger = get_pipeline_logger("test_idempotent")
    log_event(logger, logging.INFO, event="pipeline_start")
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    # Should be exactly one JSON line, not two (handler not doubled)
    assert len(lines) == 1
