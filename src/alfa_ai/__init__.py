"""Public API for the AlfaLearn consultation service."""

from .contracts import (
    AIRequest,
    AIResponse,
    LearningProfile,
    Recommendation,
    ReminderProposal,
    Source,
)
from .recommendations import recommend_materials
from .service import AIService
from .settings import Settings

__all__ = [
    "AIRequest",
    "AIResponse",
    "AIService",
    "LearningProfile",
    "Recommendation",
    "ReminderProposal",
    "Settings",
    "Source",
    "recommend_materials",
]
