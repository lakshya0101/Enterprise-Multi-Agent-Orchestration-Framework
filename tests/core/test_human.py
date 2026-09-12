"""Tests for human escalation and decision models."""

from enterprise_orchestrator.core.human import HumanDecision, HumanEscalationRequest
from enterprise_orchestrator.core.types import HumanRequestStatus


def test_human_escalation_request_model():
    """Verify human escalation request fields and defaults."""
    req = HumanEscalationRequest(
        run_id="run-999",
        task_id="task-123",
        reason="Ambiguous business requirement",
        requested_action="Clarify metric calculation formula",
        context={"formula": "A / B vs A / (B + C)"},
    )
    assert req.status == HumanRequestStatus.PENDING
    assert req.run_id == "run-999"
    assert req.task_id == "task-123"
    assert req.resolved_at is None


def test_human_decision_model():
    """Verify human decision recording."""
    decision = HumanDecision(
        request_id="req-555",
        status=HumanRequestStatus.MODIFIED,
        decision_note="Use formula A / (B + C)",
        modified_payload={"formula": "A / (B + C)"},
        reviewer_id="reviewer_01",
    )
    assert decision.request_id == "req-555"
    assert decision.status == HumanRequestStatus.MODIFIED
    assert decision.modified_payload["formula"] == "A / (B + C)"
