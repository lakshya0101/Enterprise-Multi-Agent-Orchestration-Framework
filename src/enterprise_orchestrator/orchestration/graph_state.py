"""LangGraph state schema adapter linking OrchestrationState to the graph execution cycle."""

from typing import Any, Dict, Optional, TypedDict

from enterprise_orchestrator.core.human import HumanDecision
from enterprise_orchestrator.core.state import OrchestrationState


class GraphState(TypedDict, total=False):
    """Execution state passing through the LangGraph supervisor workflow."""

    state: OrchestrationState
    current_task_id: Optional[str]
    next_action: Optional[str]
    human_decision: Optional[HumanDecision]
    error: Optional[str]
