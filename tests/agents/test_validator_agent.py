"""Unit tests for ValidatorAgent critic evaluation and quality-gate routing."""

from unittest.mock import AsyncMock

import pytest

from enterprise_orchestrator.agents.result import AgentStatus
from enterprise_orchestrator.agents.validator import CriticEvaluationSchema, ValidatorAgent
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import Task
from enterprise_orchestrator.core.types import ValidationStatus
from enterprise_orchestrator.errors.exceptions import ProviderError
from enterprise_orchestrator.providers.models import LLMResponse
from enterprise_orchestrator.providers.router import LLMRouter


@pytest.fixture
def mock_router():
    router = LLMRouter()
    router.generate_structured = AsyncMock()
    return router


@pytest.mark.asyncio
async def test_validator_valid_outcome(mock_router):
    """Verify happy path validation where candidate output satisfies all criteria."""
    critique = CriticEvaluationSchema(
        is_valid=True,
        status=ValidationStatus.VALID,
        score=0.98,
        feedback="The output directly addresses the question with verified facts.",
        issues=[],
        needs_retry=False,
        needs_human_review=False,
    )
    mock_resp = LLMResponse(content="{}", model="gemini-2.5-flash", provider="gemini")
    mock_router.generate_structured.return_value = (critique, mock_resp)

    validator = ValidatorAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Calculate total expenses")
    task = Task(
        title="Check Financial Calculation",
        input_data={
            "target_output": {"total": 50000, "breakdown": [20000, 30000]},
            "acceptance_criteria": "Total must equal the sum of items.",
        },
    )

    result = await validator.execute(state, task)

    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "validator_agent"
    val_data = result.output["validation"]
    assert val_data["is_valid"] is True
    assert val_data["status"] == "valid"
    assert val_data["score"] == 0.98
    assert len(state.validation_results) == 1
    assert state.validation_results[0].is_valid is True


@pytest.mark.asyncio
async def test_validator_retryable_outcome(mock_router):
    """Verify critic identifying retryable defect."""
    critique = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_RETRY,
        score=0.6,
        feedback="Output format omitted required percentage field.",
        issues=["Missing 'percentage_growth' key"],
        needs_retry=True,
        needs_human_review=False,
    )
    mock_resp = LLMResponse(content="{}", model="gemini-2.5-flash", provider="gemini")
    mock_router.generate_structured.return_value = (critique, mock_resp)

    validator = ValidatorAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Analyze metrics")
    task = Task(
        title="Check Metrics Output",
        input_data={
            "target_output": {"revenue": 100},
            "acceptance_criteria": "Must include percentage_growth.",
        },
    )

    result = await validator.execute(state, task)
    assert result.status == AgentStatus.SUCCESS
    val_data = result.output["validation"]
    assert val_data["is_valid"] is False
    assert val_data["status"] == "needs_retry"
    assert val_data["needs_retry"] is True


@pytest.mark.asyncio
async def test_validator_human_escalation_outcome(mock_router):
    """Verify critic triggering human escalation for high-risk or ambiguous output."""
    critique = CriticEvaluationSchema(
        is_valid=False,
        status=ValidationStatus.NEEDS_HUMAN_REVIEW,
        score=0.3,
        feedback="High-risk transaction discrepancy detected. Requires compliance officer approval.",
        issues=["Unexplained $50,000 ledger difference"],
        needs_retry=False,
        needs_human_review=True,
    )
    mock_resp = LLMResponse(content="{}", model="gemini-2.5-flash", provider="gemini")
    mock_router.generate_structured.return_value = (critique, mock_resp)

    validator = ValidatorAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Reconcile ledger")
    task = Task(
        title="Verify Reconciliation",
        input_data={"target_output": {"difference": 50000}},
    )

    result = await validator.execute(state, task)
    assert result.status == AgentStatus.ESCALATE_HUMAN
    assert result.requires_human is True
    assert "compliance officer approval" in result.error


@pytest.mark.asyncio
async def test_validator_missing_target_output_failure(mock_router):
    """Verify failure when no target output exists in state or task."""
    validator = ValidatorAgent(router=mock_router)
    state = OrchestrationState(request="Analyze")

    result = await validator.execute(state)
    assert result.status == AgentStatus.FAILURE
    assert "No target output" in result.error


@pytest.mark.asyncio
async def test_validator_llm_failure_propagation(mock_router):
    """Verify provider error propagation from LLMRouter."""
    mock_router.generate_structured.side_effect = ProviderError(
        message="Critic LLM rate limit",
        provider_name="groq",
        retryable=True,
    )
    validator = ValidatorAgent(router=mock_router)
    state = OrchestrationState.create_initial(request="Validate output")
    task = Task(title="Test Task", input_data={"target_output": "data"})

    result = await validator.execute(state, task)
    assert result.status == AgentStatus.RETRYABLE_FAILURE
    assert "Critic evaluation provider error" in result.error
    assert result.retryable is True
