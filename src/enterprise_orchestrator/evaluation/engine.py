"""Evaluation engine executing registered scorers against workflow runs."""

import time
from typing import List, Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.evaluation.base import BaseScorer
from enterprise_orchestrator.evaluation.models import (
    BenchmarkScenario,
    EvaluationResult,
    ScorerResult,
)
from enterprise_orchestrator.evaluation.scorers import (
    HITLComplianceScorer,
    PlanFeasibilityScorer,
    RetrievalScorer,
    ValidatorFidelityScorer,
)


class EvaluationEngine:
    """Orchestrates comprehensive offline evaluation using modular scorers."""

    def __init__(self, scorers: Optional[List[BaseScorer]] = None) -> None:
        self.scorers = scorers or [
            PlanFeasibilityScorer(),
            RetrievalScorer(),
            ValidatorFidelityScorer(),
            HITLComplianceScorer(),
        ]

    async def evaluate(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> EvaluationResult:
        """Run all registered scorers on an orchestration state snapshot."""
        start_time = time.perf_counter()
        results: List[ScorerResult] = []

        for scorer in self.scorers:
            try:
                res = await scorer.score(state, scenario)
                results.append(res)
            except Exception as e:
                results.append(
                    ScorerResult(
                        scorer_name=scorer.name,
                        score=0.0,
                        passed=False,
                        error=str(e),
                        details={"exception": str(e)},
                    )
                )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        if results:
            overall_score = sum(r.score for r in results) / len(results)
            all_passed = all(r.passed for r in results)
        else:
            overall_score = 1.0
            all_passed = True

        min_score = scenario.min_required_score if scenario else 0.75
        passed = all_passed and (overall_score >= min_score)

        return EvaluationResult(
            benchmark_id=scenario.benchmark_id if scenario else None,
            run_id=state.run_id,
            overall_score=round(overall_score, 4),
            passed=passed,
            scorer_results=results,
            duration_ms=round(duration_ms, 2),
            metadata={
                "scenario_name": scenario.name if scenario else None,
                "execution_status": state.metadata.status.value,
                "current_step": state.current_step,
            },
        )
