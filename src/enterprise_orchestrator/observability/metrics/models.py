"""Metrics models including Counters, Gauges, and mathematical Histograms with exact percentiles."""

import math
import threading
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Supported metric instrument types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Counter:
    """Thread-safe monotonic incremental counter."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("Counter increments must be non-negative.")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Gauge:
    """Thread-safe variable gauge."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Histogram:
    """Thread-safe distribution histogram computing exact percentiles (p50, p90, p95, p99)."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._observations: List[float] = []
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._observations.append(float(value))

    def get_summary(self) -> Dict[str, float]:
        """Compute statistical percentiles and aggregates from recorded observations."""
        with self._lock:
            vals = sorted(self._observations)

        count = len(vals)
        if count == 0:
            return {
                "count": 0.0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        total = sum(vals)
        mean = total / count

        def _calc_percentile(p: float) -> float:
            if count == 1:
                return vals[0]
            k = (count - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return vals[int(k)]
            d0 = vals[int(f)] * (c - k)
            d1 = vals[int(c)] * (k - f)
            return d0 + d1

        return {
            "count": float(count),
            "sum": round(total, 4),
            "mean": round(mean, 4),
            "min": round(vals[0], 4),
            "max": round(vals[-1], 4),
            "p50": round(_calc_percentile(0.50), 4),
            "p90": round(_calc_percentile(0.90), 4),
            "p95": round(_calc_percentile(0.95), 4),
            "p99": round(_calc_percentile(0.99), 4),
        }


class MetricSnapshot(BaseModel):
    """Structured JSON-serializable snapshot of registered metrics."""

    counters: Dict[str, float] = Field(default_factory=dict)
    gauges: Dict[str, float] = Field(default_factory=dict)
    histograms: Dict[str, Dict[str, float]] = Field(default_factory=dict)
