"""Abstract base scorer interface for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Optional

from enterprise_orchestrator.core.state import OrchestrationState
from enterprise_orchestrator.evaluation.models import BenchmarkScenario, ScorerResult


class BaseScorer(ABC):
    """Abstract base class for all evaluation scorers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier name for this scorer."""
        pass

    @abstractmethod
    async def score(
        self,
        state: OrchestrationState,
        scenario: Optional[BenchmarkScenario] = None,
    ) -> ScorerResult:
        """Evaluate orchestration state and return normalized ScorerResult."""
        pass
