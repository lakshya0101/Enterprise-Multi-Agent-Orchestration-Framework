"""Tests for OrchestrationState model, transitions, and serialization."""

from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus, HumanRequestStatus, ValidationStatus
from enterprise_orchestrator.validation.models import ValidationResult


def test_initial_state_creation():
    """Verify factory initialization of state."""
    state = OrchestrationState.create_initial(
        request="Analyze financial dataset for Q3",
        correlation_id="corr-12345",
        session_id="sess-abc",
        max_retries=5,
    )
    assert state.request == "Analyze financial dataset for Q3"
    assert state.metadata.correlation_id == "corr-12345"
    assert state.metadata.session_id == "sess-abc"
    assert state.metadata.status == ExecutionStatus.RUNNING
    assert state.current_step == 0
    assert state.requires_human is False
    assert state.max_retries == 5


def test_state_updates_and_metadata_touch():
    """Verify state mutation methods update metadata timestamps."""
    state = OrchestrationState.create_initial(request="Test request")
    initial_updated = state.metadata.updated_at

    state.add_agent_output("planner", {"tasks": ["step1", "step2"]})
    assert "planner" in state.agent_outputs
    assert state.metadata.updated_at >= initial_updated

    state.add_tool_result("calc_tool", {"result": 42})
    assert state.tool_results["calc_tool"]["result"] == 42

    state.add_retrieved_context({"doc_id": "doc-1", "content": "Context info"})
    assert len(state.retrieved_context) == 1

    validation = ValidationResult(is_valid=True, status=ValidationStatus.VALID, score=0.95)
    state.add_validation_result(validation)
    assert len(state.validation_results) == 1

    state.record_error({"code": "TEST_ERR", "message": "Non-fatal error"})
    assert len(state.errors) == 1

    state.increment_step(current_agent="retriever")
    assert state.current_step == 1
    assert state.metadata.current_step == 1
    assert state.metadata.current_agent == "retriever"


def test_state_human_escalation_flow():
    """Verify pausing for human intervention and resuming with decision."""
    state = OrchestrationState.create_initial(request="Delete production database")
    req = state.request_human_escalation(
        reason="High risk destructive operation requires confirmation",
        requested_action="Approve or Reject deletion",
        context={"target": "prod_db"},
    )
    assert state.requires_human is True
    assert state.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN
    assert len(state.human_requests) == 1
    assert req.status == HumanRequestStatus.PENDING

    # Apply decision
    decision = HumanDecision(
        request_id=req.request_id,
        status=HumanRequestStatus.APPROVED,
        decision_note="Approved by Ops Lead",
        reviewer_id="user_admin",
    )
    state.apply_human_decision(decision)
    assert state.requires_human is False
    assert state.metadata.status == ExecutionStatus.RUNNING
    assert state.human_requests[0].status == HumanRequestStatus.APPROVED


def test_state_completion_and_failure():
    """Verify completion and failure state changes."""
    state = OrchestrationState.create_initial(request="Summary task")
    state.complete_run(final_response="All tasks completed successfully.")
    assert state.final_response == "All tasks completed successfully."
    assert state.metadata.status == ExecutionStatus.COMPLETED

    state_fail = OrchestrationState.create_initial(request="Failed task")
    state_fail.fail_run("Fatal unrecoverable error.")
    assert state_fail.metadata.status == ExecutionStatus.FAILED
    assert len(state_fail.errors) == 1


def test_state_serialization_roundtrip():
    """Verify full JSON/Dict serialization and deserialization."""
    state = OrchestrationState.create_initial(request="Serialization test")
    state.add_agent_output("agent1", {"key": "value"})

    serialized = state.model_dump()
    reconstructed = OrchestrationState.model_validate(serialized)

    assert reconstructed.run_id == state.run_id
    assert reconstructed.request == state.request
    assert reconstructed.agent_outputs == {"agent1": {"key": "value"}}
    assert reconstructed.metadata.status == ExecutionStatus.RUNNING
