"""Validator fidelity and critic behavior scorer."""

from typing import Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.evaluation.base import BaseScorer
from enterprise_orchestrator.evaluation.models import BenchmarkScenario, ScorerResult


class ValidatorFidelityScorer(BaseScorer):
    """Evaluates critic fidelity, retry budget bounding, and escalation verdicts."""

    @property
    def name(self) -> str:
        return "validator_fidelity"

    async def score(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> ScorerResult:
        if not state.plan or not state.plan.tasks:
            return ScorerResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "No tasks to validate."},
            )

        tasks = state.plan.tasks
        total_tasks = len(tasks)
        retry_bounded = True
        failed_within_budget = True

        for t in tasks:
            if t.retry_count > t.max_retries:
                retry_bounded = False
            if t.status.value == "failed" and t.retry_count < t.max_retries and not state.requires_human:
                # If failed without exhausting retries or explicit escalation
                failed_within_budget = False

        score = 1.0
        if not retry_bounded:
            score -= 0.5
        if not failed_within_budget:
            score -= 0.2

        score = max(0.0, score)
        return ScorerResult(
            scorer_name=self.name,
            score=round(score, 4),
            passed=(score >= 0.8),
            details={
                "total_tasks": total_tasks,
                "retry_bounded": retry_bounded,
                "failed_within_budget": failed_within_budget,
                "total_retries_across_tasks": sum(t.retry_count for t in tasks),
            },
        )
