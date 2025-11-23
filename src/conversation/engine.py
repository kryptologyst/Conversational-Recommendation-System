"""Conversational engine for interactive recommendation system."""

import re
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from ..types import (
    ConversationState, ConversationTurn, ConversationSession, 
    User, Item, Recommendation, InteractionType
)


class ConversationEngine:
    """Handles the conversational flow for recommendation collection."""
    
    def __init__(self, items: List[Item]):
        """Initialize the conversation engine with available items."""
        self.items = items
        self.current_state = ConversationState.GREETING
        self.session_id = None
        self.user_id = None
        self.turns = []
        self.extracted_preferences = {}
        
    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation session."""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.user_id = user_id
        self.current_state = ConversationState.GREETING
        self.turns = []
        self.extracted_preferences = {}
        
        greeting_message = self._get_greeting_message()
        return greeting_message
    
    def process_user_response(self, user_response: str) -> Tuple[str, bool]:
        """Process user response and return system message and conversation status."""
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=len(self.turns) + 1,
            user_id=self.user_id,
            system_message="",  # Will be filled after processing
            user_response=user_response,
            state=self.current_state,
            timestamp=datetime.now().timestamp()
        )
        
        # Process based on current state
        if self.current_state == ConversationState.GREETING:
            system_message, next_state = self._process_greeting_response(user_response)
        elif self.current_state == ConversationState.PREFERENCE_COLLECTION:
            system_message, next_state = self._process_preference_response(user_response)
        elif self.current_state == ConversationState.RECOMMENDATION:
            system_message, next_state = self._process_recommendation_response(user_response)
        elif self.current_state == ConversationState.FEEDBACK:
            system_message, next_state = self._process_feedback_response(user_response)
        else:
            system_message = "Thank you for using our recommendation system!"
            next_state = ConversationState.CLOSING
        
        # Update turn with system message
        turn.system_message = system_message
        turn.extracted_preferences = self.extracted_preferences.copy()
        
        # Add turn to conversation
        self.turns.append(turn)
        
        # Update state
        self.current_state = next_state
        
        # Check if conversation is complete
        is_complete = next_state == ConversationState.CLOSING
        
        return system_message, is_complete
    
    def _get_greeting_message(self) -> str:
        """Get the initial greeting message."""
        greetings = [
            "Hello! I'm here to help you find the perfect products. Let's start by learning about your preferences.",
            "Hi there! I'd love to recommend some great products for you. Tell me, what brings you here today?",
            "Welcome! I'm your personal shopping assistant. What kind of products are you looking for?"
        ]
        return random.choice(greetings)
    
    def _process_greeting_response(self, response: str) -> Tuple[str, ConversationState]:
        """Process the initial greeting response."""
        # Extract basic intent
        if any(word in response.lower() for word in ["looking", "need", "want", "searching"]):
            next_message = "Great! Let me ask you a few questions to understand your preferences better. What category of products interests you most?"
        else:
            next_message = "I'd be happy to help you find products you'll love! What type of products are you interested in?"
        
        return next_message, ConversationState.PREFERENCE_COLLECTION
    
    def _process_preference_response(self, response: str) -> Tuple[str, ConversationState]:
        """Process preference collection responses."""
        # Extract category preference
        categories = [item.category for item in self.items]
        detected_category = self._extract_category(response, categories)
        
        if detected_category:
            self.extracted_preferences["category"] = detected_category
            next_message = f"Perfect! I see you're interested in {detected_category}. What's your preferred price range? (low/medium/high)"
        else:
            # Try to extract any category mentioned
            mentioned_categories = [cat for cat in categories if cat.lower() in response.lower()]
            if mentioned_categories:
                self.extracted_preferences["category"] = mentioned_categories[0]
                next_message = f"I found {mentioned_categories[0]} in your response. What's your preferred price range? (low/medium/high)"
            else:
                next_message = "I didn't catch that. Could you tell me what category of products you're interested in? (e.g., Electronics, Clothing, Books)"
                return next_message, ConversationState.PREFERENCE_COLLECTION
        
        return next_message, ConversationState.PREFERENCE_COLLECTION
    
    def _process_recommendation_response(self, response: str) -> Tuple[str, ConversationState]:
        """Process responses during recommendation phase."""
        if any(word in response.lower() for word in ["yes", "sure", "ok", "good", "like"]):
            next_message = "Excellent! Would you like me to explain why I recommended these items, or do you have any other preferences to consider?"
            return next_message, ConversationState.FEEDBACK
        elif any(word in response.lower() for word in ["no", "not", "different", "other"]):
            next_message = "No problem! Let me ask you a few more questions to better understand your preferences. What specific features are most important to you?"
            return next_message, ConversationState.PREFERENCE_COLLECTION
        else:
            next_message = "I'd be happy to help further! Would you like me to explain my recommendations or adjust them based on additional preferences?"
            return next_message, ConversationState.FEEDBACK
    
    def _process_feedback_response(self, response: str) -> Tuple[str, ConversationState]:
        """Process feedback responses."""
        if any(word in response.lower() for word in ["explain", "why", "reason", "reasoning"]):
            next_message = "I'll explain my recommendations based on your preferences and similar users' choices. Would you like to see more options or are you satisfied with these recommendations?"
        elif any(word in response.lower() for word in ["more", "other", "different", "alternatives"]):
            next_message = "I can show you more options! Let me know if you'd like to refine your preferences or see items from different categories."
        else:
            next_message = "Thank you for your feedback! Is there anything else I can help you with today?"
        
        return next_message, ConversationState.CLOSING
    
    def _extract_category(self, text: str, categories: List[str]) -> Optional[str]:
        """Extract category from user text."""
        text_lower = text.lower()
        
        # Direct category matches
        for category in categories:
            if category.lower() in text_lower:
                return category
        
        # Synonym mapping
        synonyms = {
            "electronics": ["tech", "technology", "electronic", "gadgets", "devices"],
            "clothing": ["clothes", "fashion", "apparel", "wear", "garments"],
            "books": ["book", "literature", "reading", "novels", "texts"],
            "home & garden": ["home", "garden", "household", "outdoor", "furniture"],
            "sports": ["sport", "fitness", "exercise", "athletic", "outdoor activities"],
            "beauty": ["cosmetics", "makeup", "skincare", "personal care"],
            "toys": ["toy", "games", "children", "kids", "play"],
            "automotive": ["car", "vehicle", "auto", "automobile", "driving"],
            "health": ["medical", "wellness", "fitness", "supplements"],
            "food": ["food", "beverages", "drinks", "snacks", "groceries"]
        }
        
        for category, syns in synonyms.items():
            if any(syn in text_lower for syn in syns):
                # Find the actual category name
                for actual_category in categories:
                    if actual_category.lower() == category:
                        return actual_category
        
        return None
    
    def extract_price_preference(self, text: str) -> Optional[str]:
        """Extract price preference from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["low", "cheap", "affordable", "budget", "inexpensive"]):
            return "low"
        elif any(word in text_lower for word in ["medium", "moderate", "average", "mid"]):
            return "medium"
        elif any(word in text_lower for word in ["high", "expensive", "premium", "luxury", "top"]):
            return "high"
        
        # Extract numeric ranges
        price_patterns = [
            r'\$?(\d+)\s*-\s*\$?(\d+)',  # $100-$500
            r'under\s*\$?(\d+)',  # under $100
            r'over\s*\$?(\d+)',  # over $500
            r'around\s*\$?(\d+)',  # around $200
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                if 'under' in pattern:
                    return "low"
                elif 'over' in pattern:
                    return "high"
                else:
                    return "medium"
        
        return None
    
    def get_conversation_summary(self) -> ConversationSession:
        """Get a summary of the conversation session."""
        return ConversationSession(
            session_id=self.session_id,
            user_id=self.user_id,
            turns=self.turns,
            final_recommendations=[],  # Will be filled by recommendation engine
            session_duration=datetime.now().timestamp() - self.turns[0].timestamp if self.turns else 0
        )
