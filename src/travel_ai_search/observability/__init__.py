"""Observability module (Milestone 15).

Provides two capabilities:

1. In-memory Prometheus-compatible metrics (metrics.py):
   - Counter and Histogram implementations
   - GET /metrics renders in Prometheus text exposition format

2. Structured log formatter (logging.py):
   - StructuredFormatter emits NDJSON records
   - Activated by LOG_FORMAT=json environment variable
"""
