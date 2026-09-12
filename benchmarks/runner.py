"""Deterministic offline benchmark runner executing evaluation scenarios."""

import asyncio
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Type

_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pydantic import BaseModel

from enterprise_orchestrator.agents.planner import PlannerAgent
from enterprise_orchestrator.agents.retrieval import Document, InMemoryDocumentStore, RetrievalAgent
from enterprise_orchestrator.agents.tool_execution import ToolExecutionAgent
from enterprise_orchestrator.agents.validator import ValidatorAgent
from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import (
    ExecutionStatus,
    HumanRequestStatus,
    TaskStatus,
    TaskType,
)
from enterprise_orchestrator.evaluation.engine import EvaluationEngine
from enterprise_orchestrator.evaluation.models import (
    BenchmarkScenario,
    EvaluationResult,
)
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.observability.audit.in_memory import InMemoryAuditSink
from enterprise_orchestrator.observability.metrics.registry import MetricsRegistry
from enterprise_orchestrator.observability.tracing.in_memory import InMemoryTracer
from enterprise_orchestrator.orchestration.runtime import OrchestrationRuntime
from enterprise_orchestrator.providers.base import BaseLLMProvider
from enterprise_orchestrator.providers.models import LLMRequest, LLMResponse, TokenUsage
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.tools.base import BaseTool, ToolMetadata
from enterprise_orchestrator.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    pass


class DummyOutput(BaseModel):
    result: str = "ok"


class CalcTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(name="calc", description="calc tool"))

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return DummyOutput(result="ratio: 1.45")


class EchoTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(name="echo", description="echo tool"))

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        return DummyOutput(result="echo: ok")


class FaultyTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(name="faulty_tool", description="faulty tool"))

    @property
    def input_schema(self) -> Type[BaseModel]:
        return DummyInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return DummyOutput

    async def execute(self, input_data: BaseModel) -> BaseModel:
        raise RuntimeError("Script crashed")


class BenchmarkFakeProvider(BaseLLMProvider):
    """Deterministic LLM Provider for offline benchmark evaluation."""

    def __init__(self) -> None:
        super().__init__(provider_name="benchmark_fake", default_model="fake-model")
        self.retry_counter = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="OK", model="fake-model", provider="benchmark_fake")

    async def generate_structured(self, request: LLMRequest, schema: Any) -> Tuple[Any, LLMResponse]:
        schema_name = getattr(schema, "__name__", "")

        if "Plan" in schema_name or "ExecutionPlan" in schema_name:
            req_text = request.messages[-1].content.lower()
            if "financial" in req_text:
                plan = {
                    "rationale": "3-step financial analysis",
                    "tasks": [
                        {
                            "id": "task-1",
                            "title": "Fetch Data",
                            "task_type": "retrieval",
                            "status": "pending",
                            "dependencies": [],
                            "input_data": {"query": "vacation policy"},
                        },
                        {
                            "id": "task-2",
                            "title": "Calculate Ratios",
                            "task_type": "tool_execution",
                            "status": "pending",
                            "dependencies": ["task-1"],
                            "input_data": {"tool_name": "calc", "parameters": {}},
                        },
                        {
                            "id": "task-3",
                            "title": "Summarize",
                            "task_type": "synthesis",
                            "status": "pending",
                            "dependencies": ["task-2"],
                            "input_data": {},
                        },
                    ],
                }
            elif "faulty" in req_text:
                plan = {
                    "rationale": "Faulty script execution",
                    "tasks": [
                        {
                            "id": "task-1",
                            "title": "Run Script",
                            "task_type": "tool_execution",
                            "status": "pending",
                            "dependencies": [],
                            "input_data": {"tool_name": "faulty_tool", "parameters": {}},
                        }
                    ],
                }
            else:
                plan = {
                    "rationale": "Standard single-task plan",
                    "tasks": [
                        {
                            "id": "task-1",
                            "title": "Primary Task",
                            "task_type": "retrieval" if ("vacation" in req_text or "retrieve" in req_text) else "tool_execution",
                            "status": "pending",
                            "dependencies": [],
                            "input_data": {"query": "vacation policy"} if ("vacation" in req_text or "retrieve" in req_text) else {"tool_name": "echo", "parameters": {}},
                        }
                    ],
                }
            val_model = schema.model_validate(plan)
            return val_model, LLMResponse(content="{}", model="fake-model", provider="benchmark_fake")

        elif "Critic" in schema_name or "Validation" in schema_name or "Validator" in schema_name:
            req_text = request.messages[-1].content.lower()
            if "strict schema" in req_text:
                self.retry_counter += 1
                if self.retry_counter == 1:
                    val = {
                        "status": "needs_retry",
                        "is_valid": False,
                        "score": 0.4,
                        "feedback": "Retry needed for schema",
                        "issues": ["Format flaw"],
                        "needs_retry": True,
                        "needs_human_review": False,
                    }
                else:
                    val = {
                        "status": "valid",
                        "is_valid": True,
                        "score": 1.0,
                        "feedback": "Valid on retry",
                        "issues": [],
                        "needs_retry": False,
                        "needs_human_review": False,
                    }
            elif "deploy configuration" in req_text:
                self.retry_counter += 1
                if self.retry_counter == 1:
                    val = {
                        "status": "needs_human_review",
                        "is_valid": False,
                        "score": 0.5,
                        "feedback": "Escalate to human review",
                        "issues": ["Risk ceiling"],
                        "needs_retry": False,
                        "needs_human_review": True,
                    }
                else:
                    val = {
                        "status": "valid",
                        "is_valid": True,
                        "score": 1.0,
                        "feedback": "Valid after approved modification",
                        "issues": [],
                        "needs_retry": False,
                        "needs_human_review": False,
                    }
            elif "transfer funds" in req_text or "delete production" in req_text:
                val = {
                    "status": "needs_human_review",
                    "is_valid": False,
                    "score": 0.5,
                    "feedback": "Escalate to human review",
                    "issues": ["Risk ceiling"],
                    "needs_retry": False,
                    "needs_human_review": True,
                }
            else:
                val = {
                    "status": "valid",
                    "is_valid": True,
                    "score": 1.0,
                    "feedback": "All criteria met",
                    "issues": [],
                    "needs_retry": False,
                    "needs_human_review": False,
                }

            val_model = schema.model_validate(val)
            return val_model, LLMResponse(content="{}", model="fake-model", provider="benchmark_fake")

        val_model = schema.model_validate({})
        return val_model, LLMResponse(content="{}", model="fake-model", provider="benchmark_fake")

    async def health_check(self) -> bool:
        return True


class BenchmarkRunner:
    """Discovers, runs, and evaluates deterministic benchmark suites."""

    def __init__(self, scenarios_dir: Optional[str] = None) -> None:
        if scenarios_dir:
            self.scenarios_dir = scenarios_dir
        else:
            self.scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
        self.engine = EvaluationEngine()

    def load_scenarios(self) -> List[BenchmarkScenario]:
        """Load all versioned JSON benchmark scenarios."""
        pattern = os.path.join(self.scenarios_dir, "*.json")
        files = glob.glob(pattern)
        scenarios: List[BenchmarkScenario] = []
        for fpath in sorted(files):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                scenarios.append(BenchmarkScenario.model_validate(data))
        return scenarios

    async def run_scenario(self, scenario: BenchmarkScenario) -> EvaluationResult:
        """Execute and score a single benchmark scenario."""
        provider = BenchmarkFakeProvider()
        router = LLMRouter(providers={provider.provider_name: provider}, default_provider=provider.provider_name)

        doc_store = InMemoryDocumentStore()
        await doc_store.add_documents([
            Document(id="doc_policy_001", content="Enterprise Vacation Policy: 25 days annual leave.", metadata={}),
        ])

        tool_reg = ToolRegistry()
        tool_reg.register(CalcTool())
        tool_reg.register(EchoTool())
        tool_reg.register(FaultyTool())

        tracer = InMemoryTracer()
        audit_sink = InMemoryAuditSink()
        metrics = MetricsRegistry()

        runtime = OrchestrationRuntime(
            router=router,
            document_store=doc_store,
            tool_registry=tool_reg,
            tracer=tracer,
            audit_sink=audit_sink,
            metrics=metrics,
        )

        state = await runtime.run(
            request=scenario.request,
            correlation_id=f"corr-{scenario.benchmark_id}",
            session_id=f"sess-{scenario.benchmark_id}",
            custom_context=scenario.custom_context,
        )

        # If scenario involves HITL resumption
        if state.requires_human and scenario.expected_hitl_decision:
            req_id = state.human_requests[-1].request_id
            if scenario.expected_hitl_decision == "approve":
                decision_status = HumanRequestStatus.APPROVED
            elif scenario.expected_hitl_decision in ("modify", "modified"):
                decision_status = HumanRequestStatus.MODIFIED
            else:
                decision_status = HumanRequestStatus.REJECTED

            decision = HumanDecision(
                request_id=req_id,
                status=decision_status,
                decision_note=f"Benchmark decision: {scenario.expected_hitl_decision}",
                reviewer_id="benchmark_reviewer",
                modified_payload={"parameters": {}},
            )
            state = await runtime.resume_with_human_decision(state, decision)

        eval_result = await self.engine.evaluate(state, scenario)
        return eval_result

    async def run_all(self) -> List[EvaluationResult]:
        """Execute all discovered benchmark scenarios."""
        scenarios = self.load_scenarios()
        results: List[EvaluationResult] = []
        for s in scenarios:
            res = await self.run_scenario(s)
            results.append(res)
        return results


def print_benchmark_summary(results: List[EvaluationResult]) -> None:
    """Format and print benchmark results table."""
    print("\n" + "=" * 80)
    print("           ENTERPRISE MULTI-AGENT FRAMEWORK BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Benchmark ID':<35} | {'Score':<8} | {'Status':<8} | {'Duration (ms)':<12}")
    print("-" * 80)

    total_score = 0.0
    passed_count = 0

    for r in results:
        status_str = "PASSED" if r.passed else "FAILED"
        print(f"{r.benchmark_id or 'unknown':<35} | {r.overall_score:<8.2f} | {status_str:<8} | {r.duration_ms:<12.2f}")
        total_score += r.overall_score
        if r.passed:
            passed_count += 1

    avg_score = total_score / len(results) if results else 0.0
    print("-" * 80)
    print(f"Total Scenarios: {len(results)} | Passed: {passed_count}/{len(results)} | Average Score: {avg_score:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = asyncio.run(runner.run_all())
    print_benchmark_summary(results)
    if not all(r.passed for r in results):
        sys.exit(1)
