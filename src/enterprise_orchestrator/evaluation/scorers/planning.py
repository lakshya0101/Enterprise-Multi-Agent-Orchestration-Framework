"""Plan feasibility and DAG structure scorer."""

from collections import deque
from typing import Dict, List, Optional, Set

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import ExecutionStatus, TaskStatus
from enterprise_orchestrator.evaluation.base import BaseScorer
from enterprise_orchestrator.evaluation.models import BenchmarkScenario, ScorerResult


class PlanFeasibilityScorer(BaseScorer):
    """Evaluates DAG validity, topological acyclicity, and task completeness."""

    @property
    def name(self) -> str:
        return "plan_feasibility"

    async def score(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> ScorerResult:
        if state.plan is None:
            # If state failed before planning or scenario expects failure
            if state.metadata.status == ExecutionStatus.FAILED:
                return ScorerResult(
                    scorer_name=self.name,
                    score=0.5,
                    passed=False,
                    details={"reason": "No execution plan generated; run failed."},
                )
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "Execution plan is missing."},
            )

        tasks = state.plan.tasks
        if not tasks:
            return ScorerResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "Plan contains zero tasks."},
            )

        task_ids = {t.id for t in tasks}
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
        adj_list: Dict[str, List[str]] = {t.id: [] for t in tasks}
        invalid_deps: List[str] = []

        for t in tasks:
            for dep in t.dependencies:
                if dep not in task_ids:
                    invalid_deps.append(f"Task '{t.id}' references unknown dependency '{dep}'")
                else:
                    adj_list[dep].append(t.id)
                    in_degree[t.id] += 1

        # Kahn's Algorithm for cycle detection
        queue = deque([t_id for t_id, deg in in_degree.items() if deg == 0])
        visited_count = 0
        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        is_acyclic = (visited_count == len(tasks))
        has_valid_deps = (len(invalid_deps) == 0)

        # Check task statuses on completion
        completed_tasks = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        completion_ratio = completed_tasks / len(tasks)

        score_components = []
        score_components.append(1.0 if is_acyclic else 0.0)
        score_components.append(1.0 if has_valid_deps else 0.0)

        if state.metadata.status == ExecutionStatus.COMPLETED:
            score_components.append(completion_ratio)
        else:
            # If paused or in-progress, give partial credit for valid tasks
            score_components.append(1.0 if any(t.status in (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS, TaskStatus.PENDING) for t in tasks) else 0.5)

        overall_score = sum(score_components) / len(score_components)
        passed = (is_acyclic and has_valid_deps and overall_score >= 0.75)

        return ScorerResult(
            scorer_name=self.name,
            score=round(overall_score, 4),
            passed=passed,
            details={
                "task_count": len(tasks),
                "is_acyclic": is_acyclic,
                "invalid_dependencies": invalid_deps,
                "completed_tasks": completed_tasks,
                "completion_ratio": round(completion_ratio, 4),
            },
        )
