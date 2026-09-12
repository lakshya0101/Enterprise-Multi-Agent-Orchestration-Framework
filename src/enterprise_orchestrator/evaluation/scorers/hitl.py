"""Human-in-the-loop compliance and governance scorer."""

from typing import Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus, HumanRequestStatus
from enterprise_orchestrator.evaluation.base import BaseScorer
from enterprise_orchestrator.evaluation.models import BenchmarkScenario, ScorerResult


class HITLComplianceScorer(BaseScorer):
    """Evaluates strict compliance of execution lifecycle to human intervention and decisions."""

    @property
    def name(self) -> str:
        return "hitl_compliance"

    async def score(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> ScorerResult:
        expected_decision = scenario.expected_hitl_decision if scenario else None

        if not expected_decision and not state.human_requests:
            # Clean non-HITL run
            return ScorerResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "No HITL expected or occurred."},
            )

        if expected_decision == "pause":
            is_paused = (state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN and state.requires_human)
            score = 1.0 if is_paused else 0.0
            return ScorerResult(
                scorer_name=self.name,
                score=score,
                passed=is_paused,
                details={"expected": "pause", "actual_status": state.metadata.status.value},
            )

        if expected_decision == "approve":
            has_approval = any(r.status == HumanRequestStatus.APPROVED for r in state.human_requests)
            is_resumed_or_done = state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.RUNNING)
            passed = (has_approval and is_resumed_or_done)
            return ScorerResult(
                scorer_name=self.name,
                score=1.0 if passed else 0.0,
                passed=passed,
                details={"has_approval": has_approval, "final_status": state.metadata.status.value},
            )

        if expected_decision == "reject":
            has_rejection = any(r.status == HumanRequestStatus.REJECTED for r in state.human_requests)
            is_failed = (state.metadata.status == ExecutionStatus.FAILED)
            passed = (has_rejection and is_failed)
            return ScorerResult(
                scorer_name=self.name,
                score=1.0 if passed else 0.0,
                passed=passed,
                details={"has_rejection": has_rejection, "final_status": state.metadata.status.value},
            )

        if expected_decision in ("modify", "modified"):
            has_modified = any(r.status == HumanRequestStatus.MODIFIED for r in state.human_requests)
            is_resumed_or_done = state.metadata.status in (ExecutionStatus.COMPLETED, ExecutionStatus.RUNNING)
            passed = (has_modified and is_resumed_or_done)
            return ScorerResult(
                scorer_name=self.name,
                score=1.0 if passed else 0.0,
                passed=passed,
                details={"has_modified": has_modified, "final_status": state.metadata.status.value},
            )

        # Default pass if resolved or appropriately paused
        resolved_count = sum(1 for r in state.human_requests if r.status != HumanRequestStatus.PENDING)
        score = 1.0 if (resolved_count == len(state.human_requests) or state.requires_human) else 0.5
        return ScorerResult(
            scorer_name=self.name,
            score=score,
            passed=(score >= 0.8),
            details={
                "escalation_requests": len(state.human_requests),
                "resolved_count": resolved_count,
            },
        )
