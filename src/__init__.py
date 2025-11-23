"""Conversational Recommendation System Package."""

from .conversational_system import ConversationalRecommendationSystem
from .types import Item, User, Interaction, Recommendation, InteractionType, ConversationState
from .data import DataGenerator, DataLoader
from .conversation.engine import ConversationEngine
from .models.recommenders import (
    BaseRecommender, PopularityRecommender, ContentBasedRecommender,
    CollaborativeFilteringRecommender, HybridRecommender
)
from .utils.evaluation import RecommendationEvaluator

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "A modern conversational recommendation system with multiple AI approaches"

__all__ = [
    "ConversationalRecommendationSystem",
    "Item", "User", "Interaction", "Recommendation", "InteractionType", "ConversationState",
    "DataGenerator", "DataLoader",
    "ConversationEngine",
    "BaseRecommender", "PopularityRecommender", "ContentBasedRecommender",
    "CollaborativeFilteringRecommender", "HybridRecommender",
    "RecommendationEvaluator"
]
