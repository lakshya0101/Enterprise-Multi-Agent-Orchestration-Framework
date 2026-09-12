"""Benchmark package exports."""

import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from benchmarks.runner import BenchmarkRunner, print_benchmark_summary

__all__ = ["BenchmarkRunner", "print_benchmark_summary"]
