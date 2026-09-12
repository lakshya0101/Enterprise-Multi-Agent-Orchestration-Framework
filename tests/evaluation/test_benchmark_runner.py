"""Integration tests for BenchmarkRunner and deterministic benchmark execution."""

import pytest

from benchmarks.runner import BenchmarkRunner


@pytest.mark.asyncio
async def test_benchmark_runner_scenario_discovery_and_execution() -> None:
    runner = BenchmarkRunner()
    scenarios = runner.load_scenarios()

    assert len(scenarios) >= 7

    # Run all deterministic benchmarks
    results = await runner.run_all()

    assert len(results) == len(scenarios)
    for res in results:
        assert res.passed is True, f"Scenario '{res.benchmark_id}' failed with score {res.overall_score}"
        assert res.overall_score >= 0.75
