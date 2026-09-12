"""Retrieval quality and relevance scorer."""

from typing import Any, Dict, List, Optional, Set

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.core.types import TaskType
from enterprise_orchestrator.evaluation.base import BaseScorer
from enterprise_orchestrator.evaluation.models import BenchmarkScenario, ScorerResult


class RetrievalScorer(BaseScorer):
    """Evaluates semantic retrieval precision, hit-rate (Hit@k), and relevance."""

    @property
    def name(self) -> str:
        return "retrieval_relevance"

    async def score(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> ScorerResult:
        if not state.plan:
            return ScorerResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "No plan present; skipping retrieval score."},
            )

        retrieval_tasks = [t for t in state.plan.tasks if t.task_type == TaskType.RETRIEVAL]
        if not retrieval_tasks:
            # If no retrieval tasks in workflow, retrieval is neutrally passed
            return ScorerResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "No retrieval tasks present in execution plan."},
            )

        ground_truth_ids: Set[str] = set()
        if scenario and scenario.expected_ground_truth_doc_ids:
            ground_truth_ids = set(scenario.expected_ground_truth_doc_ids)

        hits_at_1 = 0
        hits_at_3 = 0
        total_evaluable = 0
        retrieved_docs_count = 0

        for task in retrieval_tasks:
            output = task.output_data or {}
            docs = output.get("matches") or output.get("documents") or []
            retrieved_docs_count += len(docs)

            if ground_truth_ids:
                total_evaluable += 1
                doc_ids = [d.get("id") or d.get("document_id") for d in docs if isinstance(d, dict)]

                if doc_ids and doc_ids[0] in ground_truth_ids:
                    hits_at_1 += 1
                if any(d_id in ground_truth_ids for d_id in doc_ids[:3]):
                    hits_at_3 += 1

        if ground_truth_ids and total_evaluable > 0:
            hit_1_rate = hits_at_1 / total_evaluable
            hit_3_rate = hits_at_3 / total_evaluable
            score = (hit_1_rate * 0.4) + (hit_3_rate * 0.6)
            passed = score >= 0.6
            return ScorerResult(
                scorer_name=self.name,
                score=round(score, 4),
                passed=passed,
                details={
                    "retrieval_tasks": len(retrieval_tasks),
                    "hit_at_1_rate": round(hit_1_rate, 4),
                    "hit_at_3_rate": round(hit_3_rate, 4),
                    "total_retrieved": retrieved_docs_count,
                },
            )

        # If no ground truth specified, score based on retrieval task completion
        completed_retrievals = sum(1 for t in retrieval_tasks if t.status.value == "completed")
        score = completed_retrievals / len(retrieval_tasks)
        return ScorerResult(
            scorer_name=self.name,
            score=round(score, 4),
            passed=(score >= 0.8),
            details={
                "retrieval_tasks": len(retrieval_tasks),
                "completed": completed_retrievals,
                "total_retrieved": retrieved_docs_count,
            },
        )
