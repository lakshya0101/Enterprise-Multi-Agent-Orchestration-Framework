"""Graph node implementations coordinating specialized agents within the LangGraph supervisor workflow."""

import time
from typing import Any, Dict, Optional

from enterprise_orchestrator.agents.planner import PlannerAgent
from enterprise_orchestrator.agents.result import AgentStatus
from enterprise_orchestrator.agents.retrieval import RetrievalAgent
from enterprise_orchestrator.agents.tool_execution import ToolExecutionAgent
from enterprise_orchestrator.agents.validator import ValidatorAgent
from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.task import ExecutionPlan, Task
from enterprise_orchestrator.core.types import (
    ExecutionStatus,
    HumanRequestStatus,
    TaskStatus,
    TaskType,
    ValidationStatus,
)
from enterprise_orchestrator.observability.audit.base import BaseAuditSink
from enterprise_orchestrator.observability.audit.models import AuditEvent, AuditEventType
from enterprise_orchestrator.observability.metrics.registry import (
    MetricsRegistry,
    get_global_metrics,
)
from enterprise_orchestrator.observability.tracing.base import BaseTracer
from enterprise_orchestrator.observability.tracing.models import SpanKind
from enterprise_orchestrator.orchestration.graph_state import GraphState


class SupervisorWorkflowNodes:
    """Encapsulates all node logic executed by the LangGraph supervisor state graph."""

    def __init__(
        self,
        planner: PlannerAgent,
        retriever: RetrievalAgent,
        tool_executor: ToolExecutionAgent,
        validator: ValidatorAgent,
        tracer: Optional[BaseTracer] = None,
        audit_sink: Optional[BaseAuditSink] = None,
        metrics: Optional[MetricsRegistry] = None,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.tool_executor = tool_executor
        self.validator = validator
        self.tracer = tracer
        self.audit_sink = audit_sink
        self.metrics = metrics or get_global_metrics()

    def _safe_audit(self, event: AuditEvent) -> None:
        """Safely record an audit event without raising exceptions."""
        if self.audit_sink is not None:
            try:
                self.audit_sink.record(event)
            except Exception:
                pass

    async def supervisor_node(self, graph_state: GraphState) -> GraphState:
        """Inspect orchestration state and determine next task routing or terminal state."""
        state = graph_state["state"]
        state.increment_step(current_agent="supervisor")

        if graph_state.get("human_decision") is not None:
            return {**graph_state, "next_action": "human_gate"}

        if state.metadata.status == ExecutionStatus.FAILED:
            return {**graph_state, "next_action": "failed"}

        if state.requires_human:
            return {**graph_state, "next_action": "human_gate"}

        # If no plan exists yet, route to planner
        if state.plan is None:
            return {**graph_state, "next_action": "planner"}

        # Find next ready task in DAG
        ready_tasks = state.plan.get_ready_tasks()
        if ready_tasks:
            task = ready_tasks[0]
            action = "tool"
            if task.task_type == TaskType.RETRIEVAL:
                action = "retrieval"
            elif task.task_type == TaskType.TOOL_EXECUTION:
                action = "tool"
            elif task.task_type == TaskType.PLANNING:
                action = "planner"
            elif task.task_type == TaskType.SYNTHESIS:
                action = "synthesis"

            return {
                **graph_state,
                "current_task_id": task.id,
                "next_action": action,
            }

        # If no ready tasks available
        if state.plan.all_completed():
            return {**graph_state, "next_action": "synthesis"}

        if state.plan.has_failed_tasks():
            return {**graph_state, "next_action": "failed"}

        # Dependency deadlock or unresolvable blocked tasks
        state.record_error({"error": "Dependency deadlock: tasks remain pending with unmet dependencies."})
        return {**graph_state, "next_action": "failed"}

    async def planner_node(self, graph_state: GraphState) -> GraphState:
        """Invoke PlannerAgent to generate a structured DAG ExecutionPlan."""
        state = graph_state["state"]
        state.increment_step(current_agent="planner_agent")

        start_time = time.perf_counter()
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="planner_node",
                kind=SpanKind.AGENT,
                run_id=state.run_id,
                correlation_id=state.metadata.correlation_id,
                session_id=state.metadata.session_id,
            )

        try:
            result = await self.planner.execute(state)
            duration = time.perf_counter() - start_time
            self.metrics.histogram("planner_duration_seconds").observe(duration)

            if result.status == AgentStatus.SUCCESS and result.output and "execution_plan" in result.output:
                plan_data = result.output["execution_plan"]
                state.plan = ExecutionPlan.model_validate(plan_data)
                state.add_agent_output(agent_name="planner_agent", output=plan_data)

                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.PLAN_CREATED,
                        actor="planner_agent",
                        component="planner",
                        action="generate_execution_plan",
                        outcome="SUCCESS",
                        metadata={
                            "tasks_count": len(state.plan.tasks),
                            "rationale": state.plan.rationale,
                        },
                    )
                )

                if self.tracer and span:
                    self.tracer.end_span(span)
                return {**graph_state, "next_action": "supervisor"}

            # Planner failure
            state.fail_run(f"Planner failed: {result.error}")
            self._safe_audit(
                AuditEvent(
                    run_id=state.run_id,
                    correlation_id=state.metadata.correlation_id,
                    session_id=state.metadata.session_id,
                    event_type=AuditEventType.PLAN_CREATED,
                    actor="planner_agent",
                    component="planner",
                    action="generate_execution_plan",
                    outcome="FAILURE",
                    metadata={"error": result.error},
                )
            )
            if self.tracer and span:
                self.tracer.end_span(span, error=Exception(result.error))
            return {**graph_state, "error": result.error, "next_action": "failed"}

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            raise

    async def retrieval_node(self, graph_state: GraphState) -> GraphState:
        """Invoke RetrievalAgent to fetch relevant context for the active task."""
        state = graph_state["state"]
        task_id = graph_state.get("current_task_id")
        task = state.plan.get_task(task_id) if (state.plan and task_id) else None

        if not task:
            state.fail_run(f"Active task ID '{task_id}' not found in plan.")
            return {**graph_state, "next_action": "failed"}

        task.mark_in_progress()
        state.increment_step(current_agent="retrieval_agent")
        self.metrics.counter("retrieval_queries_total").inc()

        start_time = time.perf_counter()
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="retrieval_node",
                kind=SpanKind.RETRIEVAL,
                run_id=state.run_id,
                attributes={"task_id": task.id, "task_title": task.title},
            )

        try:
            result = await self.retriever.execute(state, task)
            duration = time.perf_counter() - start_time
            self.metrics.histogram("retrieval_duration_seconds").observe(duration)

            if result.status == AgentStatus.SUCCESS:
                task.output_data = result.output
                state.add_agent_output(agent_name="retrieval_agent", output=result.output)

                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.RETRIEVAL_EXECUTED,
                        actor="retrieval_agent",
                        component="retrieval",
                        action="query_documents",
                        outcome="SUCCESS",
                        metadata={"task_id": task.id, "result_summary": str(result.output)[:200]},
                    )
                )
                if self.tracer and span:
                    self.tracer.end_span(span)
                return {**graph_state, "next_action": "validator"}

            task.mark_failed(result.error or "Retrieval failed")
            state.record_error({"task_id": task.id, "error": result.error})
            if self.tracer and span:
                self.tracer.end_span(span, error=Exception(result.error))
            return {**graph_state, "next_action": "validator"}

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            raise

    async def tool_node(self, graph_state: GraphState) -> GraphState:
        """Invoke ToolExecutionAgent to run a registered tool for the active task."""
        state = graph_state["state"]
        task_id = graph_state.get("current_task_id")
        task = state.plan.get_task(task_id) if (state.plan and task_id) else None

        if not task:
            state.fail_run(f"Active task ID '{task_id}' not found in plan.")
            return {**graph_state, "next_action": "failed"}

        task.mark_in_progress()
        state.increment_step(current_agent="tool_execution_agent")
        self.metrics.counter("tool_executions_total").inc()

        start_time = time.perf_counter()
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="tool_node",
                kind=SpanKind.TOOL,
                run_id=state.run_id,
                attributes={"task_id": task.id, "task_title": task.title},
            )

        try:
            result = await self.tool_executor.execute(state, task)
            duration = time.perf_counter() - start_time
            self.metrics.histogram("tool_duration_seconds").observe(duration)

            if result.status == AgentStatus.SUCCESS:
                task.output_data = result.output
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.TOOL_INVOKED,
                        actor="tool_execution_agent",
                        component="tools",
                        action="execute_tool",
                        outcome="SUCCESS",
                        metadata={"task_id": task.id},
                    )
                )
                if self.tracer and span:
                    self.tracer.end_span(span)
                return {**graph_state, "next_action": "validator"}

            task.mark_failed(result.error or "Tool execution failed")
            state.record_error({"task_id": task.id, "error": result.error})
            self._safe_audit(
                AuditEvent(
                    run_id=state.run_id,
                    correlation_id=state.metadata.correlation_id,
                    session_id=state.metadata.session_id,
                    event_type=AuditEventType.TOOL_INVOKED,
                    actor="tool_execution_agent",
                    component="tools",
                    action="execute_tool",
                    outcome="FAILURE",
                    metadata={"task_id": task.id, "error": result.error},
                )
            )
            if self.tracer and span:
                self.tracer.end_span(span, error=Exception(result.error))
            return {**graph_state, "next_action": "validator"}

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            raise

    async def validator_node(self, graph_state: GraphState) -> GraphState:
        """Invoke ValidatorAgent to inspect task output and enforce quality gate verdicts."""
        state = graph_state["state"]
        task_id = graph_state.get("current_task_id")
        task = state.plan.get_task(task_id) if (state.plan and task_id) else None

        if not task:
            return {**graph_state, "next_action": "supervisor"}

        state.increment_step(current_agent="validator_agent")

        start_time = time.perf_counter()
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                name="validator_node",
                kind=SpanKind.VALIDATOR,
                run_id=state.run_id,
                attributes={"task_id": task.id},
            )

        try:
            result = await self.validator.execute(state, task)
            duration = time.perf_counter() - start_time
            self.metrics.histogram("validator_duration_seconds").observe(duration)

            # 1. Human review required outcome
            if result.status == AgentStatus.ESCALATE_HUMAN or result.requires_human:
                self.metrics.counter("human_escalations_total").inc()
                state.request_human_escalation(
                    reason=result.error or "Validation critic requested human review.",
                    requested_action="Review and approve task output or provide guidance.",
                    context={"task_id": task.id, "output": task.output_data},
                    task_id=task.id,
                )
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.HUMAN_ESCALATED,
                        actor="validator_agent",
                        component="validator",
                        action="escalate_to_human",
                        outcome="ESCALATED",
                        metadata={"task_id": task.id, "reason": result.error},
                    )
                )
                if self.tracer and span:
                    self.tracer.end_span(span)
                return {**graph_state, "next_action": "human_gate"}

            # 2. Inspect validation dictionary
            val_data = (result.output or {}).get("validation", {})
            status_val = val_data.get("status")

            if status_val == ValidationStatus.VALID or (result.status == AgentStatus.SUCCESS and val_data.get("is_valid")):
                task.mark_completed(task.output_data or {})
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.VALIDATION_VERDICT,
                        actor="validator_agent",
                        component="validator",
                        action="verify_task_output",
                        outcome="SUCCESS",
                        metadata={"task_id": task.id, "verdict": "VALID"},
                    )
                )
                if self.tracer and span:
                    self.tracer.end_span(span)
                return {**graph_state, "current_task_id": None, "next_action": "supervisor"}

            elif status_val == ValidationStatus.NEEDS_RETRY or val_data.get("needs_retry"):
                self.metrics.counter("validator_retries_total").inc()
                task.retry_count += 1
                if task.can_retry():
                    task.status = TaskStatus.PENDING
                    self._safe_audit(
                        AuditEvent(
                            run_id=state.run_id,
                            correlation_id=state.metadata.correlation_id,
                            session_id=state.metadata.session_id,
                            event_type=AuditEventType.RETRY_TRIGGERED,
                            actor="validator_agent",
                            component="validator",
                            action="retry_task",
                            outcome="RETRY",
                            metadata={"task_id": task.id, "retry_count": task.retry_count},
                        )
                    )
                    if self.tracer and span:
                        self.tracer.end_span(span)
                    return {**graph_state, "next_action": "supervisor"}
                else:
                    task.mark_failed("Retry budget exhausted after validation failure.")
                    state.record_error({"task_id": task.id, "error": "Retry budget exhausted after validation failure."})
                    if self.tracer and span:
                        self.tracer.end_span(span, error=Exception("Retry budget exhausted."))
                    return {**graph_state, "next_action": "supervisor"}

            task.mark_failed(f"Task output rejected by critic: {val_data.get('feedback')}")
            if self.tracer and span:
                self.tracer.end_span(span)
            return {**graph_state, "next_action": "supervisor"}

        except Exception as e:
            if self.tracer and span:
                self.tracer.end_span(span, error=e)
            raise

    async def human_gate_node(self, graph_state: GraphState) -> GraphState:
        """Pause or resolve human intervention decisions."""
        state = graph_state["state"]
        decision: Optional[HumanDecision] = graph_state.get("human_decision")
        task_id = graph_state.get("current_task_id")

        if decision is not None and not task_id and state.human_requests:
            for req in state.human_requests:
                if req.request_id == decision.request_id:
                    task_id = req.task_id or req.context.get("task_id")
                    break

        task = state.plan.get_task(task_id) if (state.plan and task_id) else None

        if decision is not None:
            state.apply_human_decision(decision)

            if decision.status == HumanRequestStatus.APPROVED:
                self.metrics.counter("human_decisions_approved_total").inc()
                if task:
                    task.mark_completed(task.output_data or {})
                state.requires_human = False
                state.metadata.status = ExecutionStatus.RUNNING
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.HUMAN_DECISION_APPLIED,
                        actor="human_reviewer",
                        component="human_gate",
                        action="approve_decision",
                        outcome="APPROVED",
                        metadata={"decision_note": decision.decision_note, "reviewer_id": decision.reviewer_id},
                    )
                )
                return {
                    **graph_state,
                    "human_decision": None,
                    "current_task_id": None,
                    "next_action": "supervisor",
                }

            elif decision.status == HumanRequestStatus.MODIFIED:
                self.metrics.counter("human_decisions_modified_total").inc()
                if task and decision.modified_payload:
                    task.input_data.update(decision.modified_payload)
                    task.status = TaskStatus.PENDING
                state.requires_human = False
                state.metadata.status = ExecutionStatus.RUNNING
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.HUMAN_DECISION_APPLIED,
                        actor="human_reviewer",
                        component="human_gate",
                        action="modify_decision",
                        outcome="MODIFIED",
                        metadata={"decision_note": decision.decision_note, "reviewer_id": decision.reviewer_id},
                    )
                )
                return {
                    **graph_state,
                    "human_decision": None,
                    "current_task_id": None,
                    "next_action": "supervisor",
                }

            elif decision.status == HumanRequestStatus.REJECTED:
                self.metrics.counter("human_decisions_rejected_total").inc()
                if task:
                    task.mark_failed(f"Rejected by human reviewer: {decision.decision_note}")
                state.fail_run(f"Workflow terminated by human reviewer: {decision.decision_note}")
                self._safe_audit(
                    AuditEvent(
                        run_id=state.run_id,
                        correlation_id=state.metadata.correlation_id,
                        session_id=state.metadata.session_id,
                        event_type=AuditEventType.HUMAN_DECISION_APPLIED,
                        actor="human_reviewer",
                        component="human_gate",
                        action="reject_decision",
                        outcome="REJECTED",
                        metadata={"decision_note": decision.decision_note, "reviewer_id": decision.reviewer_id},
                    )
                )
                return {
                    **graph_state,
                    "human_decision": None,
                    "current_task_id": None,
                    "next_action": "failed",
                }

        # Awaiting human input - pause status
        state.metadata.status = ExecutionStatus.PAUSED_FOR_HUMAN
        return {**graph_state, "next_action": "paused"}

    async def synthesis_node(self, graph_state: GraphState) -> GraphState:
        """Synthesize final verified response from all completed task outputs."""
        state = graph_state["state"]
        state.increment_step(current_agent="synthesis_node")

        summary_lines = [f"Summary for Request: '{state.request}'\n"]
        if state.plan:
            for t in state.plan.tasks:
                if t.task_type == TaskType.SYNTHESIS and t.status != TaskStatus.COMPLETED:
                    t.mark_completed(output=t.output_data or {"status": "synthesized"})
                summary_lines.append(f"- Task '{t.title}' ({t.id}): {t.status.value}")
                if t.output_data:
                    summary_lines.append(f"  Output: {t.output_data}")

        final_text = "\n".join(summary_lines)
        state.complete_run(final_response=final_text)
        return {**graph_state, "next_action": "complete"}

    async def failed_node(self, graph_state: GraphState) -> GraphState:
        """Finalize workflow in failed state."""
        state = graph_state["state"]
        if state.metadata.status != ExecutionStatus.FAILED:
            state.fail_run("Workflow encountered unrecoverable errors.")
        return {**graph_state, "next_action": "complete"}
