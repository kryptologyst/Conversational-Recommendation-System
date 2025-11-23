"""Tests for the conversational recommendation system."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.types import Item, User, Interaction, InteractionType, set_random_seeds
from src.data import DataGenerator
from src.conversation.engine import ConversationEngine
from src.models.recommenders import PopularityRecommender, ContentBasedRecommender
from src.conversational_system import ConversationalRecommendationSystem


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_generate_items(self):
        """Test item generation."""
        generator = DataGenerator(seed=42)
        items = generator.generate_items(n_items=10)
        
        assert len(items) == 10
        assert all(isinstance(item, Item) for item in items)
        assert all(item.item_id for item in items)
        assert all(item.title for item in items)
        assert all(item.price > 0 for item in items)
    
    def test_generate_users(self):
        """Test user generation."""
        generator = DataGenerator(seed=42)
        users = generator.generate_users(n_users=5)
        
        assert len(users) == 5
        assert all(isinstance(user, User) for user in users)
        assert all(user.user_id for user in users)
        assert all('preferred_categories' in user.preferences for user in users)
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = DataGenerator(seed=42)
        items = generator.generate_items(n_items=5)
        users = generator.generate_users(n_users=3)
        interactions = generator.generate_interactions(users, items, n_interactions=10)
        
        assert len(interactions) == 10
        assert all(isinstance(interaction, Interaction) for interaction in interactions)
        assert all(interaction.user_id in [u.user_id for u in users] for interaction in interactions)
        assert all(interaction.item_id in [i.item_id for i in items] for interaction in interactions)


class TestConversationEngine:
    """Test conversation engine functionality."""
    
    def test_conversation_initialization(self):
        """Test conversation engine initialization."""
        items = [Item(
            item_id="test_item",
            title="Test Item",
            category="Electronics",
            price=100.0,
            brand="TestBrand",
            description="Test description",
            features={},
            tags=["test"]
        )]
        
        engine = ConversationEngine(items)
        assert engine.items == items
        assert engine.current_state.value == "greeting"
    
    def test_start_conversation(self):
        """Test starting a conversation."""
        items = [Item(
            item_id="test_item",
            title="Test Item",
            category="Electronics",
            price=100.0,
            brand="TestBrand",
            description="Test description",
            features={},
            tags=["test"]
        )]
        
        engine = ConversationEngine(items)
        greeting = engine.start_conversation("test_user")
        
        assert isinstance(greeting, str)
        assert len(greeting) > 0
        assert engine.user_id == "test_user"
        assert engine.session_id is not None
    
    def test_process_user_response(self):
        """Test processing user responses."""
        items = [Item(
            item_id="test_item",
            title="Test Item",
            category="Electronics",
            price=100.0,
            brand="TestBrand",
            description="Test description",
            features={},
            tags=["test"]
        )]
        
        engine = ConversationEngine(items)
        engine.start_conversation("test_user")
        
        response, is_complete = engine.process_user_response("I'm looking for electronics")
        
        assert isinstance(response, str)
        assert isinstance(is_complete, bool)
        assert len(engine.turns) == 1


class TestRecommenders:
    """Test recommendation models."""
    
    def test_popularity_recommender(self):
        """Test popularity-based recommender."""
        # Create test data
        items = [
            Item("item1", "Item 1", "Electronics", 100.0, "Brand1", "Desc1", {}, []),
            Item("item2", "Item 2", "Electronics", 200.0, "Brand2", "Desc2", {}, [])
        ]
        
        users = [
            User("user1", {}, [], None)
        ]
        
        interactions = [
            Interaction("user1", "item1", InteractionType.PURCHASE, 1234567890),
            Interaction("user1", "item1", InteractionType.LIKE, 1234567891),
            Interaction("user1", "item2", InteractionType.VIEW, 1234567892)
        ]
        
        recommender = PopularityRecommender()
        recommender.fit(interactions, items, users)
        
        recommendations = recommender.recommend("user1", n_recommendations=2)
        
        assert len(recommendations) <= 2
        assert all(isinstance(rec, type(recommendations[0])) for rec in recommendations)
        assert all(rec.score >= 0 for rec in recommendations)
    
    def test_content_based_recommender(self):
        """Test content-based recommender."""
        # Create test data
        items = [
            Item("item1", "Electronics Laptop", "Electronics", 100.0, "Brand1", "High quality laptop", {}, ["portable"]),
            Item("item2", "Electronics Phone", "Electronics", 200.0, "Brand2", "Smart phone", {}, ["wireless"])
        ]
        
        users = [
            User("user1", {}, [], None)
        ]
        
        interactions = [
            Interaction("user1", "item1", InteractionType.PURCHASE, 1234567890)
        ]
        
        recommender = ContentBasedRecommender()
        recommender.fit(interactions, items, users)
        
        recommendations = recommender.recommend("user1", n_recommendations=2, preferences={"category": "Electronics"})
        
        assert len(recommendations) <= 2
        assert all(isinstance(rec, type(recommendations[0])) for rec in recommendations)


class TestConversationalSystem:
    """Test the main conversational system."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = ConversationalRecommendationSystem(seed=42)
        assert system.seed == 42
        assert len(system.items) == 0
        assert len(system.users) == 0
        assert len(system.interactions) == 0
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        system = ConversationalRecommendationSystem(seed=42)
        system.generate_synthetic_data(n_items=5, n_users=3, n_interactions=10)
        
        assert len(system.items) == 5
        assert len(system.users) == 3
        assert len(system.interactions) == 10
    
    def test_recommender_initialization(self):
        """Test recommender initialization."""
        system = ConversationalRecommendationSystem(seed=42)
        system.generate_synthetic_data(n_items=5, n_users=3, n_interactions=10)
        system.initialize_recommenders()
        
        assert len(system.recommenders) == 4  # popularity, content_based, collaborative, hybrid
        assert all(recommender.is_fitted for recommender in system.recommenders.values())
    
    def test_conversation_flow(self):
        """Test basic conversation flow."""
        system = ConversationalRecommendationSystem(seed=42)
        system.generate_synthetic_data(n_items=5, n_users=3, n_interactions=10)
        system.initialize_recommenders()
        
        # Start conversation
        greeting = system.start_conversation("test_user")
        assert isinstance(greeting, str)
        assert len(greeting) > 0
        
        # Process user input
        response, is_complete, recommendations = system.process_user_input("I want electronics")
        assert isinstance(response, str)
        assert isinstance(is_complete, bool)
        assert recommendations is None or isinstance(recommendations, list)


class TestTypes:
    """Test type definitions and utilities."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        # This should not raise an exception
        set_random_seeds(42)
        assert True  # If we get here, no exception was raised
    
    def test_item_creation(self):
        """Test Item creation."""
        item = Item(
            item_id="test",
            title="Test Item",
            category="Electronics",
            price=100.0,
            brand="TestBrand",
            description="Test description",
            features={"feature1": "value1"},
            tags=["tag1", "tag2"]
        )
        
        assert item.item_id == "test"
        assert item.title == "Test Item"
        assert item.category == "Electronics"
        assert item.price == 100.0
        assert item.brand == "TestBrand"
        assert item.description == "Test description"
        assert item.features == {"feature1": "value1"}
        assert item.tags == ["tag1", "tag2"]
    
    def test_user_creation(self):
        """Test User creation."""
        user = User(
            user_id="test_user",
            preferences={"category": "Electronics"},
            interaction_history=[],
            demographic_info={"age": "25-35"}
        )
        
        assert user.user_id == "test_user"
        assert user.preferences == {"category": "Electronics"}
        assert user.interaction_history == []
        assert user.demographic_info == {"age": "25-35"}
    
    def test_interaction_creation(self):
        """Test Interaction creation."""
        interaction = Interaction(
            user_id="test_user",
            item_id="test_item",
            interaction_type=InteractionType.PURCHASE,
            timestamp=1234567890,
            rating=4.5
        )
        
        assert interaction.user_id == "test_user"
        assert interaction.item_id == "test_item"
        assert interaction.interaction_type == InteractionType.PURCHASE
        assert interaction.timestamp == 1234567890
        assert interaction.rating == 4.5


if __name__ == "__main__":
    pytest.main([__file__])
