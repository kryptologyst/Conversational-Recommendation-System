"""Main conversational recommendation system."""

import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .types import Item, User, Interaction, Recommendation, set_random_seeds
from .data import DataGenerator, DataLoader
from .conversation.engine import ConversationEngine
from .models.recommenders import (
    PopularityRecommender, ContentBasedRecommender, 
    CollaborativeFilteringRecommender, HybridRecommender
)
from .utils.evaluation import RecommendationEvaluator


class ConversationalRecommendationSystem:
    """Main conversational recommendation system."""
    
    def __init__(self, seed: int = 42):
        """Initialize the conversational recommendation system."""
        set_random_seeds(seed)
        self.seed = seed
        
        # Initialize components
        self.data_generator = DataGenerator(seed)
        self.data_loader = DataLoader()
        self.conversation_engine = None
        self.recommenders = {}
        self.evaluator = RecommendationEvaluator()
        
        # Data
        self.items = []
        self.users = []
        self.interactions = []
        
        # Current session
        self.current_user = None
        self.current_session = None
        
    def generate_synthetic_data(self, n_items: int = 100, n_users: int = 1000, n_interactions: int = 10000) -> None:
        """Generate synthetic data for the system."""
        print("Generating synthetic data...")
        
        self.items = self.data_generator.generate_items(n_items)
        self.users = self.data_generator.generate_users(n_users)
        self.interactions = self.data_generator.generate_interactions(
            self.users, self.items, n_interactions
        )
        
        print(f"Generated {len(self.items)} items, {len(self.users)} users, {len(self.interactions)} interactions")
        
    def load_data(self, items_file: str, interactions_file: str) -> None:
        """Load data from files."""
        print("Loading data from files...")
        
        self.items = self.data_loader.load_items_from_csv(items_file)
        self.interactions = self.data_loader.load_interactions_from_csv(interactions_file)
        
        print(f"Loaded {len(self.items)} items and {len(self.interactions)} interactions")
        
    def initialize_recommenders(self) -> None:
        """Initialize and fit all recommendation models."""
        print("Initializing recommendation models...")
        
        # Initialize recommenders
        self.recommenders = {
            'popularity': PopularityRecommender(),
            'content_based': ContentBasedRecommender(),
            'collaborative': CollaborativeFilteringRecommender(),
        }
        
        # Create hybrid recommender
        self.recommenders['hybrid'] = HybridRecommender([
            self.recommenders['popularity'],
            self.recommenders['content_based'],
            self.recommenders['collaborative']
        ], weights=[0.2, 0.4, 0.4])
        
        # Fit all models
        for name, recommender in self.recommenders.items():
            print(f"Fitting {name} model...")
            recommender.fit(self.interactions, self.items, self.users)
        
        print("All models fitted successfully!")
        
    def initialize_conversation_engine(self) -> None:
        """Initialize the conversation engine."""
        self.conversation_engine = ConversationEngine(self.items)
        
    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation with a user."""
        if not self.conversation_engine:
            self.initialize_conversation_engine()
        
        # Find or create user
        self.current_user = self._get_or_create_user(user_id)
        
        # Start conversation
        greeting = self.conversation_engine.start_conversation(user_id)
        return greeting
    
    def process_user_input(self, user_input: str) -> Tuple[str, bool, Optional[List[Recommendation]]]:
        """Process user input and return system response."""
        if not self.conversation_engine:
            raise ValueError("Conversation engine not initialized")
        
        # Process user response
        system_response, is_complete = self.conversation_engine.process_user_response(user_input)
        
        recommendations = None
        
        # Generate recommendations if we have enough preferences
        if len(self.conversation_engine.extracted_preferences) >= 2:
            recommendations = self._generate_recommendations()
            if recommendations:
                # Add recommendations to conversation
                rec_text = "Here are my recommendations for you:\n"
                for i, rec in enumerate(recommendations[:5], 1):
                    item = self._get_item_by_id(rec.item_id)
                    rec_text += f"{i}. {item.title} - {rec.explanation}\n"
                
                system_response += "\n\n" + rec_text
        
        return system_response, is_complete, recommendations
    
    def _generate_recommendations(self) -> Optional[List[Recommendation]]:
        """Generate recommendations based on current conversation state."""
        if not self.current_user or not self.conversation_engine:
            return None
        
        # Use hybrid recommender for best results
        recommender = self.recommenders.get('hybrid')
        if not recommender:
            return None
        
        preferences = self.conversation_engine.extracted_preferences.copy()
        
        # Extract price preference from conversation
        if not preferences.get('price_range'):
            for turn in self.conversation_engine.turns:
                price_pref = self.conversation_engine.extract_price_preference(turn.user_response)
                if price_pref:
                    preferences['price_range'] = price_pref
                    break
        
        # Generate recommendations
        recommendations = recommender.recommend(
            self.current_user.user_id,
            n_recommendations=10,
            preferences=preferences
        )
        
        # Filter by preferences if available
        filtered_recommendations = self._filter_recommendations_by_preferences(recommendations, preferences)
        
        return filtered_recommendations
    
    def _filter_recommendations_by_preferences(self, recommendations: List[Recommendation], preferences: Dict) -> List[Recommendation]:
        """Filter recommendations based on user preferences."""
        filtered = []
        
        for rec in recommendations:
            item = self._get_item_by_id(rec.item_id)
            if not item:
                continue
            
            # Filter by category
            if 'category' in preferences:
                if item.category.lower() != preferences['category'].lower():
                    continue
            
            # Filter by price range
            if 'price_range' in preferences:
                price_range = preferences['price_range']
                if price_range == 'low' and item.price >= 300:
                    continue
                elif price_range == 'medium' and (item.price < 300 or item.price > 800):
                    continue
                elif price_range == 'high' and item.price <= 800:
                    continue
            
            filtered.append(rec)
        
        return filtered
    
    def _get_or_create_user(self, user_id: str) -> User:
        """Get existing user or create a new one."""
        for user in self.users:
            if user.user_id == user_id:
                return user
        
        # Create new user
        new_user = User(
            user_id=user_id,
            preferences={},
            interaction_history=[]
        )
        self.users.append(new_user)
        return new_user
    
    def _get_item_by_id(self, item_id: str) -> Optional[Item]:
        """Get item by ID."""
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None
    
    def evaluate_models(self, test_size: float = 0.2) -> Dict[str, str]:
        """Evaluate all recommendation models."""
        print("Evaluating models...")
        
        # Split data for evaluation
        train_interactions, test_interactions = self._split_interactions(test_size)
        
        # Retrain models on training data
        temp_recommenders = {}
        for name, recommender_class in [
            ('popularity', PopularityRecommender),
            ('content_based', ContentBasedRecommender),
            ('collaborative', CollaborativeFilteringRecommender)
        ]:
            temp_recommenders[name] = recommender_class()
            temp_recommenders[name].fit(train_interactions, self.items, self.users)
        
        # Create hybrid
        temp_recommenders['hybrid'] = HybridRecommender([
            temp_recommenders['popularity'],
            temp_recommenders['content_based'],
            temp_recommenders['collaborative']
        ])
        
        # Generate recommendations for test users
        test_users = list(set(interaction.user_id for interaction in test_interactions))
        model_results = {}
        
        for model_name, recommender in temp_recommenders.items():
            recommendations = {}
            for user_id in test_users[:100]:  # Limit for performance
                try:
                    recs = recommender.recommend(user_id, n_recommendations=10)
                    recommendations[user_id] = recs
                except:
                    continue
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(recommendations, test_interactions)
            model_results[model_name] = metrics
        
        # Generate comparison report
        report = self.evaluator.generate_report(model_results)
        
        return {
            'report': report,
            'model_results': model_results
        }
    
    def _split_interactions(self, test_size: float) -> Tuple[List[Interaction], List[Interaction]]:
        """Split interactions into train and test sets."""
        # Sort by timestamp
        sorted_interactions = sorted(self.interactions, key=lambda x: x.timestamp)
        
        split_idx = int(len(sorted_interactions) * (1 - test_size))
        train_interactions = sorted_interactions[:split_idx]
        test_interactions = sorted_interactions[split_idx:]
        
        return train_interactions, test_interactions
    
    def get_conversation_summary(self) -> Optional[Dict]:
        """Get summary of current conversation."""
        if not self.conversation_engine:
            return None
        
        session = self.conversation_engine.get_conversation_summary()
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'turns': len(session.turns),
            'duration': session.session_duration,
            'preferences': self.conversation_engine.extracted_preferences
        }
    
    def save_data(self, items_file: str, interactions_file: str) -> None:
        """Save current data to files."""
        self.data_loader.save_items_to_csv(self.items, items_file)
        self.data_loader.save_interactions_to_csv(self.interactions, interactions_file)
        print(f"Data saved to {items_file} and {interactions_file}")
