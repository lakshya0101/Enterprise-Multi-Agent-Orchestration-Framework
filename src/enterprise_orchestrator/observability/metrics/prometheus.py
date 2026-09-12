"""Prometheus metrics text exposition serializer."""

import io
import math
import re
from typing import Dict, List, Optional

from enterprise_orchestrator.observability.metrics.models import MetricSnapshot
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry


def sanitize_metric_name(name: str) -> str:
    """Sanitize metric name to adhere to Prometheus naming specification [a-zA-Z_:][a-zA-Z0-9_:]*."""
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if sanitized and (sanitized[0].isdigit() or sanitized[0] == ":"):
        sanitized = "_" + sanitized
    return sanitized or "unnamed_metric"


def escape_help_string(help_text: str) -> str:
    """Escape backslashes and newlines in HELP docstrings."""
    return help_text.replace("\\", "\\\\").replace("\n", "\\n")


def escape_label_value(val: str) -> str:
    """Escape backslashes, newlines, and double quotes in label values."""
    return val.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def format_float(val: float) -> str:
    """Format floating point numbers for Prometheus text output."""
    if math.isnan(val):
        return "Nan"
    if math.isinf(val):
        return "+Inf" if val > 0 else "-Inf"
    # Integer-like floats
    if val.is_integer():
        return str(int(val))
    return str(val)


class PrometheusTextSerializer:
    """Pure-Python serializer converting MetricsRegistry / MetricSnapshot to Prometheus exposition text."""

    def __init__(self, metric_descriptions: Optional[Dict[str, str]] = None) -> None:
        self.metric_descriptions = metric_descriptions or {
            "runs_started_total": "Total workflow runs started",
            "runs_completed_total": "Total workflow runs successfully completed",
            "runs_failed_total": "Total workflow runs failed",
            "llm_requests_total": "Total LLM generation requests",
            "llm_fallbacks_total": "Total provider fallback triggers",
            "human_escalations_total": "Total human-in-the-loop escalations",
            "human_decisions_approved_total": "Total human approvals",
            "human_decisions_rejected_total": "Total human rejections",
            "human_decisions_modified_total": "Total human modifications",
            "validator_retries_total": "Total critic-requested task retries",
            "tool_executions_total": "Total tool executions",
            "retrieval_queries_total": "Total semantic retrieval queries",
            "active_runs": "Currently active in-flight runs",
            "pending_human_reviews": "Currently pending human reviews",
            "run_duration_seconds": "End-to-end workflow execution duration in seconds",
            "step_count": "Execution steps per run",
            "llm_duration_seconds": "LLM request latency in seconds",
            "planner_duration_seconds": "Planner agent latency in seconds",
            "retrieval_duration_seconds": "Retrieval agent latency in seconds",
            "tool_duration_seconds": "Tool execution latency in seconds",
            "validator_duration_seconds": "Validator agent latency in seconds",
            "prompt_tokens": "Prompt token count distribution",
            "completion_tokens": "Completion token count distribution",
        }

    def serialize_registry(self, registry: MetricsRegistry) -> str:
        """Serialize a live MetricsRegistry instance into Prometheus text exposition format."""
        snapshot = registry.get_snapshot()
        # Collect any custom descriptions dynamically from registry instruments
        descriptions = dict(self.metric_descriptions)
        with registry._lock:
            for name, c in registry._counters.items():
                if c.description:
                    descriptions[name] = c.description
            for name, g in registry._gauges.items():
                if g.description:
                    descriptions[name] = g.description
            for name, h in registry._histograms.items():
                if h.description:
                    descriptions[name] = h.description

        return self.serialize_snapshot(snapshot, descriptions)

    def serialize_snapshot(
        self,
        snapshot: MetricSnapshot,
        descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Serialize a MetricSnapshot model into Prometheus text format."""
        desc_map = descriptions or self.metric_descriptions
        buffer = io.StringIO()

        # 1. Counters
        for name, value in sorted(snapshot.counters.items()):
            clean_name = sanitize_metric_name(name)
            help_text = desc_map.get(name, f"Counter metric {name}")
            buffer.write(f"# HELP {clean_name} {escape_help_string(help_text)}\n")
            buffer.write(f"# TYPE {clean_name} counter\n")
            buffer.write(f"{clean_name} {format_float(value)}\n\n")

        # 2. Gauges
        for name, value in sorted(snapshot.gauges.items()):
            clean_name = sanitize_metric_name(name)
            help_text = desc_map.get(name, f"Gauge metric {name}")
            buffer.write(f"# HELP {clean_name} {escape_help_string(help_text)}\n")
            buffer.write(f"# TYPE {clean_name} gauge\n")
            buffer.write(f"{clean_name} {format_float(value)}\n\n")

        # 3. Histograms exported as Prometheus Summaries with process-local quantiles
        for name, summary in sorted(snapshot.histograms.items()):
            clean_name = sanitize_metric_name(name)
            help_text = desc_map.get(name, f"Summary distribution for {name}")
            buffer.write(f"# HELP {clean_name} {escape_help_string(help_text)}\n")
            buffer.write(f"# TYPE {clean_name} summary\n")

            # Quantiles: p50, p90, p95, p99
            quantiles = [
                ("0.5", summary.get("p50", 0.0)),
                ("0.9", summary.get("p90", 0.0)),
                ("0.95", summary.get("p95", 0.0)),
                ("0.99", summary.get("p99", 0.0)),
            ]
            for q_label, q_val in quantiles:
                buffer.write(f'{clean_name}{{quantile="{q_label}"}} {format_float(q_val)}\n')

            # Sum and Count
            sum_val = summary.get("sum", 0.0)
            count_val = summary.get("count", 0.0)
            buffer.write(f"{clean_name}_sum {format_float(sum_val)}\n")
            buffer.write(f"{clean_name}_count {format_float(count_val)}\n\n")

        return buffer.getvalue()
