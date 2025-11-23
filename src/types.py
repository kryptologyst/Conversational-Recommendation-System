"""Core data structures and types for the conversational recommendation system."""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class InteractionType(Enum):
    """Types of user-item interactions."""
    VIEW = "view"
    LIKE = "like"
    PURCHASE = "purchase"
    RATING = "rating"


class ConversationState(Enum):
    """States in the conversational flow."""
    GREETING = "greeting"
    PREFERENCE_COLLECTION = "preference_collection"
    RECOMMENDATION = "recommendation"
    FEEDBACK = "feedback"
    CLOSING = "closing"


@dataclass
class Item:
    """Represents an item in the recommendation system."""
    item_id: str
    title: str
    category: str
    price: float
    brand: str
    description: str
    features: Dict[str, Union[str, float, int]]
    tags: List[str]


@dataclass
class User:
    """Represents a user in the recommendation system."""
    user_id: str
    preferences: Dict[str, Union[str, float, int]]
    interaction_history: List[Tuple[str, str, InteractionType, float]]
    demographic_info: Optional[Dict[str, str]] = None


@dataclass
class Interaction:
    """Represents a user-item interaction."""
    user_id: str
    item_id: str
    interaction_type: InteractionType
    timestamp: float
    rating: Optional[float] = None
    context: Optional[Dict[str, Union[str, float]]] = None


@dataclass
class ConversationTurn:
    """Represents a turn in the conversational flow."""
    turn_id: int
    user_id: str
    system_message: str
    user_response: str
    state: ConversationState
    timestamp: float
    extracted_preferences: Optional[Dict[str, Union[str, float]]] = None


@dataclass
class Recommendation:
    """Represents a recommendation result."""
    item_id: str
    score: float
    explanation: str
    reasoning: List[str]


@dataclass
class ConversationSession:
    """Represents a complete conversation session."""
    session_id: str
    user_id: str
    turns: List[ConversationTurn]
    final_recommendations: List[Recommendation]
    user_satisfaction: Optional[float] = None
    session_duration: Optional[float] = None


class RecommendationMetrics:
    """Container for recommendation evaluation metrics."""
    
    def __init__(self):
        self.precision_at_k: Dict[int, float] = {}
        self.recall_at_k: Dict[int, float] = {}
        self.ndcg_at_k: Dict[int, float] = {}
        self.map_at_k: Dict[int, float] = {}
        self.hit_rate_at_k: Dict[int, float] = {}
        self.coverage: float = 0.0
        self.diversity: float = 0.0
        self.novelty: float = 0.0
        self.popularity_bias: float = 0.0


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
