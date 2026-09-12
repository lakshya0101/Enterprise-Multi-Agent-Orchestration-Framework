"""Metrics registry managing global and per-run telemetry instruments."""

import threading
from typing import Any, Dict, Optional

from enterprise_orchestrator.observability.metrics.models import (
    Counter,
    Gauge,
    Histogram,
    MetricSnapshot,
)


class MetricsRegistry:
    """Thread-safe registry for managing counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Initialize core framework operational metrics."""
        # Counters
        self.counter("runs_started_total", "Total workflow runs started")
        self.counter("runs_completed_total", "Total workflow runs successfully completed")
        self.counter("runs_failed_total", "Total workflow runs failed")
        self.counter("llm_requests_total", "Total LLM generation requests")
        self.counter("llm_fallbacks_total", "Total provider fallback triggers")
        self.counter("human_escalations_total", "Total human-in-the-loop escalations")
        self.counter("human_decisions_approved_total", "Total human approvals")
        self.counter("human_decisions_rejected_total", "Total human rejections")
        self.counter("human_decisions_modified_total", "Total human modifications")
        self.counter("validator_retries_total", "Total critic-requested task retries")
        self.counter("tool_executions_total", "Total tool executions")
        self.counter("retrieval_queries_total", "Total semantic retrieval queries")

        # Gauges
        self.gauge("active_runs", "Currently active in-flight runs")
        self.gauge("pending_human_reviews", "Currently pending human reviews")

        # Histograms
        self.histogram("run_duration_seconds", "End-to-end workflow execution duration in seconds")
        self.histogram("step_count", "Execution steps per run")
        self.histogram("llm_duration_seconds", "LLM request latency in seconds")
        self.histogram("planner_duration_seconds", "Planner agent latency in seconds")
        self.histogram("retrieval_duration_seconds", "Retrieval agent latency in seconds")
        self.histogram("tool_duration_seconds", "Tool execution latency in seconds")
        self.histogram("validator_duration_seconds", "Validator agent latency in seconds")
        self.histogram("prompt_tokens", "Prompt token count distribution")
        self.histogram("completion_tokens", "Completion token count distribution")

    def counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description)
            return self._histograms[name]

    def get_snapshot(self) -> MetricSnapshot:
        """Produce a complete structured snapshot of current metrics."""
        with self._lock:
            counters_dict = {k: c.value for k, c in self._counters.items()}
            gauges_dict = {k: g.value for k, g in self._gauges.items()}
            histograms_dict = {k: h.get_summary() for k, h in self._histograms.items()}

        return MetricSnapshot(
            counters=counters_dict,
            gauges=gauges_dict,
            histograms=histograms_dict,
        )

    def clear(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._init_standard_metrics()


_GLOBAL_METRICS = MetricsRegistry()


def get_global_metrics() -> MetricsRegistry:
    """Accessor for the singleton global metrics registry."""
    return _GLOBAL_METRICS
