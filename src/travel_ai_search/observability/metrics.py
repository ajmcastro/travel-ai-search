"""In-memory Prometheus-compatible metrics registry.

Implements the Prometheus text exposition format (v0.0.4) without external
dependencies. The registry is a module-level singleton; call the module-level
metric objects from route handlers.

Prometheus text format example
-------------------------------
    # HELP travel_search_requests_total Total search requests processed.
    # TYPE travel_search_requests_total counter
    travel_search_requests_total{strategy="hybrid"} 42
    # HELP travel_search_latency_ms Search latency in milliseconds.
    # TYPE travel_search_latency_ms histogram
    travel_search_latency_ms_bucket{le="50"} 18
    travel_search_latency_ms_bucket{le="+Inf"} 42
    travel_search_latency_ms_count 42
    travel_search_latency_ms_sum 5312.0

Thread safety
-------------
All mutable state is protected by a threading.Lock so the registry is safe
for concurrent FastAPI requests.  For high-throughput production use, a
proper prometheus_client library (which uses atomic operations) should be
preferred; this implementation prioritises clarity over performance.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

# Default latency buckets in milliseconds (1 ms → 10 s)
_DEFAULT_BUCKETS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Convert a label dict to a sorted, hashable tuple."""
    return tuple(sorted(labels.items()))


def _labels_str(labels: dict[str, str]) -> str:
    """Render label dict as Prometheus label selector string, e.g. {a="b",c="d"}."""
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + pairs + "}"


class Counter:
    """A monotonically increasing counter with optional labels."""

    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None, amount: float = 1.0) -> None:
        """Increment the counter by *amount* for the given label set."""
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] += amount

    def render(self) -> str:
        """Return Prometheus text lines for this counter."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            snapshot = dict(self._values)
        for key, value in sorted(snapshot.items()):
            label_str = _labels_str(dict(key))
            lines.append(f"{self.name}{label_str} {value:g}")
        if not snapshot:
            lines.append(f"{self.name} 0")
        return "\n".join(lines)


class Histogram:
    """A histogram that tracks the distribution of observed values.

    Uses explicit upper-bound buckets (like Prometheus).  An implicit +Inf
    bucket always exists.  The _count and _sum metrics are also maintained.
    """

    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float] = _DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.help_text = help_text
        self._buckets: tuple[float, ...] = tuple(sorted(buckets))
        # _counts[key][bucket_index] = cumulative count up to bucket upper bound
        self._counts: dict[tuple[tuple[str, str], ...], list[float]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._totals: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def _ensure_key(self, key: tuple[tuple[str, str], ...]) -> None:
        """Initialise bucket list for a new label set (call with lock held)."""
        if key not in self._counts:
            self._counts[key] = [0.0] * len(self._buckets)

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record one observation of *value* for the given label set."""
        key = _labels_key(labels or {})
        with self._lock:
            self._ensure_key(key)
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[key][i] += 1.0
            self._sums[key] += value
            self._totals[key] += 1.0

    def render(self) -> str:
        """Return Prometheus text lines for this histogram."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            counts_snap = {k: list(v) for k, v in self._counts.items()}
            sums_snap = dict(self._sums)
            totals_snap = dict(self._totals)

        for key in sorted(counts_snap):
            label_dict = dict(key)
            label_str = _labels_str(label_dict)
            bucket_counts = counts_snap[key]
            # observe() already writes cumulative counts per bucket — use directly.
            for i, bound in enumerate(self._buckets):
                bucket_labels = _labels_str({**label_dict, "le": str(bound)})
                lines.append(f"{self.name}_bucket{bucket_labels} {bucket_counts[i]:g}")
            # +Inf bucket (= total observations)
            inf_labels = _labels_str({**label_dict, "le": "+Inf"})
            lines.append(f"{self.name}_bucket{inf_labels} {totals_snap.get(key, 0):g}")
            lines.append(f"{self.name}_count{label_str} {totals_snap.get(key, 0):g}")
            lines.append(f"{self.name}_sum{label_str} {sums_snap.get(key, 0.0):.3f}")

        if not counts_snap:
            lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
            lines.append(f"{self.name}_count 0")
            lines.append(f"{self.name}_sum 0.000")

        return "\n".join(lines)


class MetricsRegistry:
    """Holds all registered metrics and renders them together."""

    def __init__(self) -> None:
        self._metrics: list[Counter | Histogram] = []

    def counter(self, name: str, help_text: str) -> Counter:
        m = Counter(name, help_text)
        self._metrics.append(m)
        return m

    def histogram(
        self,
        name: str,
        help_text: str,
        buckets: Sequence[float] = _DEFAULT_BUCKETS,
    ) -> Histogram:
        m = Histogram(name, help_text, buckets=buckets)
        self._metrics.append(m)
        return m

    def render_all(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        return "\n".join(m.render() for m in self._metrics) + "\n"


# ── Module-level singleton registry ───────────────────────────────────────────

_registry = MetricsRegistry()

search_requests_total: Counter = _registry.counter(
    "travel_search_requests_total",
    "Total number of POST /search requests, labelled by strategy.",
)

search_latency_ms: Histogram = _registry.histogram(
    "travel_search_latency_ms",
    "End-to-end search request latency in milliseconds.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000),
)

search_fallbacks_total: Counter = _registry.counter(
    "travel_search_fallbacks_total",
    "Total fallbacks triggered in the search pipeline, labelled by reason.",
)
