"""Structured (JSON) log formatter for production deployments.

Usage
-----
Activated by setting LOG_FORMAT=json in the environment (or .env).
The application wires this up in api/app.py after reading settings.

When JSON format is active, every log record emitted by the
travel_ai_search.* logger namespace is serialised as one line of
newline-delimited JSON (NDJSON), e.g.:

    {"ts":"2026-08-15T12:00:00Z","level":"INFO","logger":"...","msg":"..."}

For the POST /search structured log the record also carries pipeline
fields (strategy, took_ms, qu_took_ms, etc.) that appear as top-level
JSON keys, making the log directly queryable by tools like jq, Loki,
CloudWatch Logs Insights, or Splunk.

Human-readable format (default, LOG_FORMAT=text)
-------------------------------------------------
When LOG_FORMAT is anything other than "json" the standard Python
formatter is used, which is easier to read during local development.
"""

from __future__ import annotations

import json
import logging
import time

# Keys present on every LogRecord that are NOT caller-supplied extras.
# We exclude these from the JSON payload to avoid noise.
_STDLIB_KEYS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class StructuredFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON.

    Standard fields: ts (ISO-8601 UTC), level, logger, msg.
    Any keys passed via extra={} in logging calls appear as top-level
    JSON fields.  Non-serialisable values are coerced to str.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STDLIB_KEYS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_structured_logging(namespace: str, level: str) -> None:
    """Wire a StructuredFormatter onto the given logger namespace.

    Call once at application startup when LOG_FORMAT=json.  The handler is
    added to the root logger so that it receives records propagated from all
    child loggers; the existing basicConfig StreamHandler is removed first to
    avoid duplicate output.
    """
    root = logging.getLogger()
    # Remove any plain-text handlers installed by basicConfig.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root.addHandler(handler)

    logging.getLogger(namespace).setLevel(level.upper())
