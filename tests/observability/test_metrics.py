"""Unit tests for Metrics instruments and percentile Histograms."""

import pytest

from enterprise_orchestrator.observability.metrics.models import Counter, Gauge, Histogram
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry


def test_counter_and_gauge_operations() -> None:
    c = Counter("test_counter")
    c.inc()
    c.inc(5.5)
    assert c.value == 6.5

    with pytest.raises(ValueError):
        c.inc(-1.0)

    g = Gauge("test_gauge")
    g.set(10.0)
    g.inc(2.0)
    g.dec(5.0)
    assert g.value == 7.0


def test_histogram_percentile_calculations() -> None:
    h = Histogram("latency_seconds")
    # Record 100 observations from 1.0 to 100.0
    for i in range(1, 101):
        h.observe(float(i))

    summary = h.get_summary()

    assert summary["count"] == 100.0
    assert summary["min"] == 1.0
    assert summary["max"] == 100.0
    assert summary["mean"] == 50.5
    # p50 of 1..100 is 50.5
    assert abs(summary["p50"] - 50.5) < 0.1
    # p90 of 1..100 is 90.1
    assert abs(summary["p90"] - 90.1) < 0.5
    # p95 of 1..100 is 95.05
    assert abs(summary["p95"] - 95.05) < 0.5
    # p99 of 1..100 is 99.01
    assert abs(summary["p99"] - 99.01) < 0.5


def test_metrics_registry_snapshot() -> None:
    reg = MetricsRegistry()
    reg.counter("runs_started_total").inc(3)
    reg.gauge("active_runs").set(2)
    reg.histogram("run_duration_seconds").observe(1.25)

    snapshot = reg.get_snapshot()

    assert snapshot.counters["runs_started_total"] == 3.0
    assert snapshot.gauges["active_runs"] == 2.0
    assert snapshot.histograms["run_duration_seconds"]["count"] == 1.0
    assert snapshot.histograms["run_duration_seconds"]["p50"] == 1.25
