"""Structured JSON logging for the DeepStream pipeline.

One JSON object per line, written to stderr by default so stdout stays
clean for downstream tooling. Use configure_pipeline_logging() once at
process startup; then get_pipeline_logger() per module.
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Optional


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "logger": record.name,
            "level": record.levelname,
        }
        # log_event stores the structured fields as record.msg (a dict)
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            payload["msg"] = record.getMessage()
        return json.dumps(payload)


def configure_pipeline_logging(
    level: int = logging.INFO,
    stream: Optional[io.IOBase] = None,
) -> None:
    """Install a JSON handler on the 'pipeline' logger namespace.

    Idempotent — repeated calls do not add duplicate handlers.
    """
    import sys

    root = logging.getLogger("pipeline")
    root.setLevel(level)

    # Remove any existing _JsonFormatter handlers to avoid duplicates
    root.handlers = [h for h in root.handlers if not isinstance(h.formatter, _JsonFormatter)]

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.propagate = False


def get_pipeline_logger(name: str) -> logging.Logger:
    """Return a logger under the 'pipeline.*' namespace."""
    return logging.getLogger(f"pipeline.{name}")


def log_event(
    logger: logging.Logger,
    level: int,
    *,
    source_id: Optional[int] = None,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured log record with event + optional source_id + extra fields."""
    payload: dict[str, Any] = {"event": event}
    if source_id is not None:
        payload["source_id"] = source_id
    payload.update(fields)
    logger.log(level, payload)
