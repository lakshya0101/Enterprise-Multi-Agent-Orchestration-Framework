"""Centralized LangGraph supervisor orchestration runtime engine with non-invasive telemetry."""

import time
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from enterprise_orchestrator.agents.planner import PlannerAgent
from enterprise_orchestrator.agents.retrieval import BaseDocumentStore, InMemoryDocumentStore, RetrievalAgent
from enterprise_orchestrator.agents.tool_execution import ToolExecutionAgent
from enterprise_orchestrator.agents.validator import ValidatorAgent
from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus
from enterprise_orchestrator.memory.base import BaseStateStore, Checkpoint
from enterprise_orchestrator.memory.in_memory import InMemoryStateStore
from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent, AuditEventType
from enterprise_orchestrator.observability.metrics.registry import (
    MetricsRegistry,
    get_global_metrics,
)
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.observability.tracing.models import SpanKind
from enterprise_orchestrator.orchestration.graph_state import GraphState
from enterprise_orchestrator.orchestration.nodes import SupervisorWorkflowNodes
from enterprise_orchestrator.providers.router import LLMRouter
from enterprise_orchestrator.tools.registry import ToolRegistry


class OrchestrationRuntime:
    """Central orchestration runtime managing LangGraph StateGraph compilation and execution."""

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        planner: Optional[PlannerAgent] = None,
        retriever: Optional[RetrievalAgent] = None,
        tool_executor: Optional[ToolExecutionAgent] = None,
        validator: Optional[ValidatorAgent] = None,
        document_store: Optional[BaseDocumentStore] = None,
        tool_registry: Optional[ToolRegistry] = None,
        state_store: Optional[BaseStateStore] = None,
        tracer: Optional[BaseTracer] = None,
        audit_sink: Optional[BaseAuditSink] = None,
        metrics: Optional[MetricsRegistry] = None,
        max_steps: int = 50,
    ) -> None:
        # Default dependency wiring if not explicitly injected
        self.router = router or LLMRouter()
        self.document_store = document_store or InMemoryDocumentStore()
        self.tool_registry = tool_registry or ToolRegistry()
        self.state_store = state_store or InMemoryStateStore()
        self.tracer = tracer
        self.audit_sink = audit_sink
        self.metrics = metrics or get_global_metrics()

        self.planner = planner or PlannerAgent(router=self.router)
        self.retriever = retriever or RetrievalAgent(document_store=self.document_store)
        self.tool_executor = tool_executor or ToolExecutionAgent(tool_registry=self.tool_registry)
        self.validator = validator or ValidatorAgent(router=self.router)

        self.max_steps = max_steps
        self.nodes = SupervisorWorkflowNodes(
            planner=self.planner,
            retriever=self.retriever,
            tool_executor=self.tool_executor,
            validator=self.validator,
            tracer=self.tracer,
            audit_sink=self.audit_sink,
            metrics=self.metrics,
        )

        self._graph = self._build_graph()
        self._compiled_graph = self._graph.compile()

    def _safe_audit(self, event: AuditEvent) -> None:
        if self.audit_sink is not None:
            try:
                self.audit_sink.record(event)
            except Exception:
                pass

    def _build_graph(self) -> StateGraph:
        """Construct the centralized supervisor StateGraph with routing and conditional branches."""
        builder = StateGraph(GraphState)

        # 1. Register Graph Nodes
        builder.add_node("supervisor", self.nodes.supervisor_node)
        builder.add_node("planner", self.nodes.planner_node)
        builder.add_node("retrieval", self.nodes.retrieval_node)
        builder.add_node("tool", self.nodes.tool_node)
        builder.add_node("validator", self.nodes.validator_node)
        builder.add_node("human_gate", self.nodes.human_gate_node)
        builder.add_node("synthesis", self.nodes.synthesis_node)
        builder.add_node("failed", self.nodes.failed_node)

        # 2. Add Fixed and Return Edges
        builder.add_edge(START, "supervisor")
        builder.add_edge("planner", "supervisor")
        builder.add_edge("retrieval", "validator")
        builder.add_edge("tool", "validator")
        builder.add_edge("synthesis", END)
        builder.add_edge("failed", END)

        # 3. Conditional Supervisor Dispatch Routing
        def supervisor_router(state: GraphState) -> str:
            action = state.get("next_action", "failed")
            if action == "planner":
                return "planner"
            elif action == "retrieval":
                return "retrieval"
            elif action == "tool":
                return "tool"
            elif action == "human_gate":
                return "human_gate"
            elif action == "synthesis":
                return "synthesis"
            elif action == "failed":
                return "failed"
            return "failed"

        builder.add_conditional_edges(
            "supervisor",
            supervisor_router,
            {
                "planner": "planner",
                "retrieval": "retrieval",
                "tool": "tool",
                "human_gate": "human_gate",
                "synthesis": "synthesis",
                "failed": "failed",
            },
        )

        # 4. Conditional Validator Routing
        def validator_router(state: GraphState) -> str:
            action = state.get("next_action")
            if action == "human_gate":
                return "human_gate"
            return "supervisor"

        builder.add_conditional_edges(
            "validator",
            validator_router,
            {
                "supervisor": "supervisor",
                "human_gate": "human_gate",
            },
        )

        # 5. Conditional Human Gate Routing
        def human_gate_router(state: GraphState) -> str:
            action = state.get("next_action")
            if action == "failed":
                return "failed"
            elif action == "supervisor":
                return "supervisor"
            return END

        builder.add_conditional_edges(
            "human_gate",
            human_gate_router,
            {
                "supervisor": "supervisor",
                "failed": "failed",
                END: END,
            },
        )

        return builder

    @property
    def compiled_graph(self) -> Any:
        """Accessor for the compiled LangGraph instance."""
        return self._compiled_graph

    async def run(
        self,
        request: str,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationState:
        """Execute end-to-end multi-agent orchestration workflow for a request."""
        state = OrchestrationState.create_initial(
            request=request,
            correlation_id=correlation_id,
            session_id=session_id,
            custom_context=custom_context,
        )

        start_time = time.perf_counter()
        self.metrics.counter("runs_started_total").inc()
        self.metrics.gauge("active_runs").inc()

        self._safe_audit(
            AuditEvent(
                run_id=state.run_id,
                correlation_id=correlation_id,
                session_id=session_id,
                event_type=AuditEventType.RUN_STARTED,
                actor="client",
                component="runtime",
                action="start_run",
                outcome="SUCCESS",
                metadata={"request": request[:200]},
            )
        )

        initial_graph_state: GraphState = {"state": state}

        # Save initial checkpoint
        await self.state_store.save_state(state)
        await self.state_store.save_checkpoint(
            Checkpoint(run_id=state.run_id, step=0, state=state)
        )

        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="OrchestrationRuntime.run",
                kind=SpanKind.WORKFLOW,
                run_id=state.run_id,
                correlation_id=correlation_id,
                session_id=session_id,
                attributes={"request": request[:200]},
            )

        try:
            # Execute LangGraph workflow with recursion limit guardrails
            output_state = await self._compiled_graph.ainvoke(
                initial_graph_state,
                config={"recursion_limit": self.max_steps},
            )

            final_state: OrchestrationState = output_state["state"]

            # Persist final state snapshot
            await self.state_store.save_state(final_state)
            await self.state_store.save_checkpoint(
                Checkpoint(run_id=final_state.run_id, step=final_state.current_step, state=final_state)
            )

            if self.tracer and span:
                self.tracer.end_span(span)

            return final_state

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            state.fail_run(f"Workflow execution exception: {e}")
            await self.state_store.save_state(state)
            raise

        finally:
            duration = time.perf_counter() - start_time
            self.metrics.gauge("active_runs").dec()
            self.metrics.histogram("run_duration_seconds").observe(duration)

            # Check final status if state object is available
            st = output_state["state"] if "output_state" in locals() and isinstance(output_state, dict) and "state" in output_state else state
            self.metrics.histogram("step_count").observe(st.current_step)

            if st.metadata.status == ExecutionStatus.COMPLETED:
                self.metrics.counter("runs_completed_total").inc()
                self._safe_audit(
                    AuditEvent(
                        run_id=st.run_id,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        event_type=AuditEventType.RUN_COMPLETED,
                        actor="runtime",
                        component="runtime",
                        action="complete_run",
                        outcome="SUCCESS",
                        metadata={"current_step": st.current_step},
                    )
                )
            elif st.metadata.status == ExecutionStatus.FAILED:
                self.metrics.counter("runs_failed_total").inc()
                self._safe_audit(
                    AuditEvent(
                        run_id=st.run_id,
                        correlation_id=correlation_id,
                        session_id=session_id,
                        event_type=AuditEventType.RUN_FAILED,
                        actor="runtime",
                        component="runtime",
                        action="fail_run",
                        outcome="FAILURE",
                        metadata={"errors": st.errors},
                    )
                )
            elif st.metadata.status == ExecutionStatus.PAUSED_FOR_HUMAN:
                self.metrics.gauge("pending_human_reviews").inc()

    async def resume_with_human_decision(
        self,
        state: OrchestrationState,
        decision: HumanDecision,
    ) -> OrchestrationState:
        """Resume a paused workflow with a submitted human decision."""
        graph_state: GraphState = {
            "state": state,
            "human_decision": decision,
        }

        self.metrics.gauge("pending_human_reviews").dec()

        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="OrchestrationRuntime.resume_with_human_decision",
                kind=SpanKind.HUMAN_GATE,
                run_id=state.run_id,
                correlation_id=state.metadata.correlation_id,
                session_id=state.metadata.session_id,
                attributes={"decision_status": decision.status.value},
            )

        try:
            # Invoke graph directly starting from human_gate / supervisor
            output_state = await self._compiled_graph.ainvoke(
                graph_state,
                config={"recursion_limit": self.max_steps},
            )

            final_state: OrchestrationState = output_state["state"]
            await self.state_store.save_state(final_state)
            await self.state_store.save_checkpoint(
                Checkpoint(run_id=final_state.run_id, step=final_state.current_step, state=final_state)
            )

            if self.tracer and span:
                self.tracer.end_span(span)

            return final_state

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            raise
