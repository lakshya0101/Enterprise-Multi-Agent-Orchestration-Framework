"""Unit tests for PrometheusTextSerializer and metrics text formatting."""

import pytest

from enterprise_orchestrator.observability.metrics.models import (
    Counter,
    Gauge,
    Histogram,
    MetricSnapshot,
)
from enterprise_orchestrator.observability.metrics.prometheus import (
    PrometheusTextSerializer,
    escape_help_string,
    escape_label_value,
    format_float,
    sanitize_metric_name,
)
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry


def test_sanitize_metric_name():
    """Verify Prometheus metric name sanitization."""
    assert sanitize_metric_name("runs_started_total") == "runs_started_total"
    assert sanitize_metric_name("my-custom.metric@123") == "my_custom_metric_123"
    assert sanitize_metric_name("99_bottles") == "_99_bottles"
    assert sanitize_metric_name(":leading_colon") == "_:leading_colon"
    assert sanitize_metric_name("") == "unnamed_metric"


def test_escape_help_string_and_labels():
    """Verify proper escaping of HELP strings and label values."""
    assert escape_help_string("Help with \\ backslash and \n newline") == "Help with \\\\ backslash and \\n newline"
    assert escape_label_value('Label "with" quotes and \n newline') == 'Label \\"with\\" quotes and \\n newline'


def test_format_float():
    """Verify float formatting for Prometheus exposition."""
    assert format_float(42.0) == "42"
    assert format_float(3.1415) == "3.1415"
    assert format_float(float("nan")) == "Nan"
    assert format_float(float("inf")) == "+Inf"
    assert format_float(float("-inf")) == "-Inf"


def test_prometheus_serializer_empty_registry():
    """Verify Prometheus serialization of default empty registry."""
    registry = MetricsRegistry()
    serializer = PrometheusTextSerializer()
    output = serializer.serialize_registry(registry)

    # Standard metrics are initialized with 0 values
    assert "# TYPE runs_started_total counter" in output
    assert "runs_started_total 0" in output
    assert "# TYPE active_runs gauge" in output
    assert "active_runs 0" in output
    assert "# TYPE run_duration_seconds summary" in output
    assert 'run_duration_seconds{quantile="0.5"} 0' in output
    assert 'run_duration_seconds{quantile="0.9"} 0' in output
    assert 'run_duration_seconds{quantile="0.95"} 0' in output
    assert 'run_duration_seconds{quantile="0.99"} 0' in output
    assert "run_duration_seconds_sum 0" in output
    assert "run_duration_seconds_count 0" in output


def test_prometheus_serializer_with_observations():
    """Verify formatting of populated counters, gauges, and histograms."""
    registry = MetricsRegistry()
    serializer = PrometheusTextSerializer()

    # 1. Update counter
    c = registry.counter("runs_started_total")
    c.inc(5.0)

    # 2. Update gauge
    g = registry.gauge("active_runs")
    g.set(3.0)

    # 3. Add histogram observations
    h = registry.histogram("run_duration_seconds")
    for val in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
        h.observe(val)

    output = serializer.serialize_registry(registry)

    # Verify counter
    assert "# HELP runs_started_total Total workflow runs started" in output
    assert "# TYPE runs_started_total counter" in output
    assert "runs_started_total 5" in output

    # Verify gauge
    assert "# HELP active_runs Currently active in-flight runs" in output
    assert "# TYPE active_runs gauge" in output
    assert "active_runs 3" in output

    # Verify summary quantiles and aggregates
    assert "# HELP run_duration_seconds End-to-end workflow execution duration in seconds" in output
    assert "# TYPE run_duration_seconds summary" in output
    assert 'run_duration_seconds{quantile="0.5"}' in output
    assert 'run_duration_seconds{quantile="0.9"}' in output
    assert 'run_duration_seconds{quantile="0.95"}' in output
    assert 'run_duration_seconds{quantile="0.99"}' in output
    assert "run_duration_seconds_sum 25" in output
    assert "run_duration_seconds_count 6" in output


def test_prometheus_serializer_custom_metric():
    """Verify custom dynamic metrics and custom descriptions."""
    registry = MetricsRegistry()
    c = registry.counter("custom_events_total", description="Count of custom events")
    c.inc(12)

    serializer = PrometheusTextSerializer()
    output = serializer.serialize_registry(registry)

    assert "# HELP custom_events_total Count of custom events" in output
    assert "# TYPE custom_events_total counter" in output
    assert "custom_events_total 12" in output


def test_prometheus_serializer_direct_snapshot():
    """Verify serializing a standalone MetricSnapshot."""
    snapshot = MetricSnapshot(
        counters={"test_counter": 10.5},
        gauges={"test_gauge": -2.0},
        histograms={"test_hist": {"p50": 1.5, "p90": 2.5, "p95": 2.8, "p99": 2.99, "sum": 15.0, "count": 10.0}},
    )
    serializer = PrometheusTextSerializer()
    output = serializer.serialize_snapshot(snapshot)

    assert "test_counter 10.5" in output
    assert "test_gauge -2" in output
    assert 'test_hist{quantile="0.5"} 1.5' in output
    assert 'test_hist{quantile="0.99"} 2.99' in output
    assert "test_hist_sum 15" in output
    assert "test_hist_count 10" in output
