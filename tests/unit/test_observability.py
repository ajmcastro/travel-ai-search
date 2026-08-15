"""Unit tests for the in-memory metrics registry (Milestone 15).

Verifies Counter and Histogram semantics and the Prometheus text-format output,
without any HTTP or OpenSearch infrastructure.
"""

from __future__ import annotations

from travel_ai_search.observability.metrics import (
    Counter,
    Histogram,
    MetricsRegistry,
)

# ── Counter ───────────────────────────────────────────────────────────────────


class TestCounter:
    def test_starts_at_zero(self) -> None:
        c = Counter("test_total", "Test counter.")
        assert "test_total 0" in c.render()

    def test_inc_default_amount(self) -> None:
        c = Counter("req_total", "Requests.")
        c.inc()
        c.inc()
        assert "req_total 2" in c.render()

    def test_inc_custom_amount(self) -> None:
        c = Counter("bytes_total", "Bytes.")
        c.inc(amount=100.0)
        assert "bytes_total 100" in c.render()

    def test_inc_with_labels(self) -> None:
        c = Counter("req_total", "Requests.")
        c.inc({"strategy": "hybrid"})
        c.inc({"strategy": "hybrid"})
        c.inc({"strategy": "lexical_fallback"})
        rendered = c.render()
        assert 'req_total{strategy="hybrid"} 2' in rendered
        assert 'req_total{strategy="lexical_fallback"} 1' in rendered

    def test_different_label_sets_tracked_independently(self) -> None:
        c = Counter("events_total", "Events.")
        c.inc({"type": "a"})
        c.inc({"type": "b"})
        c.inc({"type": "b"})
        rendered = c.render()
        assert 'events_total{type="a"} 1' in rendered
        assert 'events_total{type="b"} 2' in rendered

    def test_render_includes_help_and_type_lines(self) -> None:
        c = Counter("my_counter", "My help text.")
        rendered = c.render()
        assert "# HELP my_counter My help text." in rendered
        assert "# TYPE my_counter counter" in rendered

    def test_render_no_labels_no_braces(self) -> None:
        c = Counter("plain_total", "Plain.")
        c.inc()
        rendered = c.render()
        # Label-free metric should not have { } in its value line
        assert "plain_total{" not in rendered
        assert "plain_total 1" in rendered

    def test_concurrent_increments_are_safe(self) -> None:
        import threading

        c = Counter("concurrent_total", "Concurrent.")
        threads = [threading.Thread(target=lambda: c.inc()) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert "concurrent_total 100" in c.render()


# ── Histogram ─────────────────────────────────────────────────────────────────


class TestHistogram:
    def test_starts_empty_with_zero_count(self) -> None:
        h = Histogram("latency_ms", "Latency.", buckets=[10, 50, 100])
        rendered = h.render()
        assert "latency_ms_count 0" in rendered
        assert "latency_ms_sum 0.000" in rendered

    def test_observe_increments_count_and_sum(self) -> None:
        h = Histogram("latency_ms", "Latency.", buckets=[10, 50, 100])
        h.observe(30.0)
        h.observe(80.0)
        rendered = h.render()
        assert "latency_ms_count 2" in rendered
        assert "latency_ms_sum 110.000" in rendered

    def test_bucket_counts_are_cumulative(self) -> None:
        h = Histogram("dur_ms", "Duration.", buckets=[10, 50, 100])
        h.observe(5.0)  # ≤10, ≤50, ≤100
        h.observe(30.0)  # ≤50, ≤100
        h.observe(75.0)  # ≤100
        rendered = h.render()
        # le=10: only 5 ms observation → count 1
        assert 'dur_ms_bucket{le="10"} 1' in rendered
        # le=50: 5 and 30 → count 2
        assert 'dur_ms_bucket{le="50"} 2' in rendered
        # le=100: all three → count 3
        assert 'dur_ms_bucket{le="100"} 3' in rendered
        # +Inf: same as total
        assert 'dur_ms_bucket{le="+Inf"} 3' in rendered

    def test_value_above_all_buckets_only_counts_inf(self) -> None:
        h = Histogram("size_ms", "Size.", buckets=[10, 50])
        h.observe(999.0)
        rendered = h.render()
        assert 'size_ms_bucket{le="10"} 0' in rendered
        assert 'size_ms_bucket{le="50"} 0' in rendered
        assert 'size_ms_bucket{le="+Inf"} 1' in rendered

    def test_observe_with_labels(self) -> None:
        h = Histogram("req_ms", "Request ms.", buckets=[50, 200])
        h.observe(30.0, {"endpoint": "/search"})
        h.observe(150.0, {"endpoint": "/search"})
        rendered = h.render()
        assert 'req_ms_count{endpoint="/search"} 2' in rendered

    def test_render_includes_help_and_type_lines(self) -> None:
        h = Histogram("my_hist", "My histogram help.")
        rendered = h.render()
        assert "# HELP my_hist My histogram help." in rendered
        assert "# TYPE my_hist histogram" in rendered

    def test_bucket_boundary_value_is_included(self) -> None:
        h = Histogram("exact_ms", "Exact.", buckets=[50.0])
        h.observe(50.0)  # exactly on the boundary — should be counted in the ≤50 bucket
        rendered = h.render()
        assert 'exact_ms_bucket{le="50.0"} 1' in rendered


# ── MetricsRegistry ───────────────────────────────────────────────────────────


class TestMetricsRegistry:
    def test_render_all_includes_all_registered_metrics(self) -> None:
        reg = MetricsRegistry()
        reg.counter("c_total", "Counter.")
        reg.histogram("h_ms", "Histogram.")
        rendered = reg.render_all()
        assert "c_total" in rendered
        assert "h_ms" in rendered

    def test_render_all_ends_with_newline(self) -> None:
        reg = MetricsRegistry()
        reg.counter("x_total", "X.")
        assert reg.render_all().endswith("\n")

    def test_counter_and_histogram_registered_separately(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("a_total", "A.")
        h = reg.histogram("b_ms", "B.")
        c.inc()
        h.observe(42.0)
        rendered = reg.render_all()
        assert "# TYPE a_total counter" in rendered
        assert "# TYPE b_ms histogram" in rendered

    def test_empty_registry_renders_without_error(self) -> None:
        reg = MetricsRegistry()
        rendered = reg.render_all()
        assert rendered == "\n"
