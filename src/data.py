"""Data generation and loading utilities for the conversational recommendation system."""

import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

from .types import Item, User, Interaction, InteractionType, set_random_seeds


class DataGenerator:
    """Generates synthetic data for the conversational recommendation system."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        set_random_seeds(seed)
        self.seed = seed
        
    def generate_items(self, n_items: int = 100) -> List[Item]:
        """Generate synthetic items with realistic attributes."""
        categories = [
            "Electronics", "Clothing", "Books", "Home & Garden", 
            "Sports", "Beauty", "Toys", "Automotive", "Health", "Food"
        ]
        
        brands = [
            "TechCorp", "StyleBrand", "BookHouse", "GardenPro", "SportMax",
            "BeautyLux", "ToyWorld", "AutoTech", "HealthPlus", "FoodFresh"
        ]
        
        items = []
        for i in range(n_items):
            category = random.choice(categories)
            brand = random.choice(brands)
            
            # Generate realistic price based on category
            base_prices = {
                "Electronics": (50, 2000),
                "Clothing": (10, 500),
                "Books": (5, 50),
                "Home & Garden": (20, 800),
                "Sports": (15, 600),
                "Beauty": (5, 200),
                "Toys": (10, 300),
                "Automotive": (100, 5000),
                "Health": (10, 300),
                "Food": (2, 100)
            }
            
            price_range = base_prices.get(category, (10, 100))
            price = round(random.uniform(*price_range), 2)
            
            # Generate features based on category
            features = self._generate_features(category, price)
            
            # Generate tags
            tags = self._generate_tags(category, brand)
            
            item = Item(
                item_id=f"item_{i:04d}",
                title=f"{brand} {category} Product {i+1}",
                category=category,
                price=price,
                brand=brand,
                description=f"A high-quality {category.lower()} product from {brand}.",
                features=features,
                tags=tags
            )
            items.append(item)
            
        return items
    
    def _generate_features(self, category: str, price: float) -> Dict[str, float]:
        """Generate item features based on category."""
        features = {"price": price}
        
        if category == "Electronics":
            features.update({
                "battery_life": random.uniform(2, 24),
                "screen_size": random.uniform(4, 15),
                "storage_gb": random.choice([32, 64, 128, 256, 512]),
                "ram_gb": random.choice([2, 4, 8, 16, 32])
            })
        elif category == "Clothing":
            features.update({
                "size": random.choice(["XS", "S", "M", "L", "XL", "XXL"]),
                "material_quality": random.uniform(1, 10),
                "style_rating": random.uniform(1, 10)
            })
        elif category == "Books":
            features.update({
                "pages": random.randint(50, 800),
                "difficulty_level": random.uniform(1, 10),
                "popularity_score": random.uniform(1, 10)
            })
        else:
            features.update({
                "quality_rating": random.uniform(1, 10),
                "durability": random.uniform(1, 10),
                "ease_of_use": random.uniform(1, 10)
            })
            
        return features
    
    def _generate_tags(self, category: str, brand: str) -> List[str]:
        """Generate relevant tags for items."""
        tag_pools = {
            "Electronics": ["portable", "wireless", "smart", "premium", "compact"],
            "Clothing": ["casual", "formal", "trendy", "comfortable", "stylish"],
            "Books": ["educational", "fiction", "non-fiction", "bestseller", "classic"],
            "Home & Garden": ["eco-friendly", "durable", "modern", "traditional", "space-saving"],
            "Sports": ["outdoor", "indoor", "professional", "beginner-friendly", "high-performance"],
            "Beauty": ["natural", "organic", "long-lasting", "gentle", "luxury"],
            "Toys": ["educational", "interactive", "safe", "fun", "creative"],
            "Automotive": ["reliable", "efficient", "high-performance", "eco-friendly", "premium"],
            "Health": ["natural", "effective", "safe", "fast-acting", "premium"],
            "Food": ["organic", "fresh", "healthy", "delicious", "premium"]
        }
        
        base_tags = tag_pools.get(category, ["quality", "reliable", "popular"])
        tags = random.sample(base_tags, min(3, len(base_tags)))
        tags.append(brand.lower())
        
        return tags
    
    def generate_users(self, n_users: int = 1000) -> List[User]:
        """Generate synthetic users with preferences."""
        users = []
        
        for i in range(n_users):
            # Generate demographic info
            age_groups = ["18-25", "26-35", "36-45", "46-55", "55+"]
            genders = ["Male", "Female", "Other"]
            locations = ["Urban", "Suburban", "Rural"]
            
            demographic_info = {
                "age_group": random.choice(age_groups),
                "gender": random.choice(genders),
                "location": random.choice(locations)
            }
            
            # Generate preferences
            preferences = {
                "preferred_categories": random.sample(
                    ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"], 
                    random.randint(1, 3)
                ),
                "price_sensitivity": random.uniform(0, 1),
                "brand_loyalty": random.uniform(0, 1),
                "quality_preference": random.uniform(0, 1)
            }
            
            user = User(
                user_id=f"user_{i:04d}",
                preferences=preferences,
                interaction_history=[],
                demographic_info=demographic_info
            )
            users.append(user)
            
        return users
    
    def generate_interactions(
        self, 
        users: List[User], 
        items: List[Item], 
        n_interactions: int = 10000
    ) -> List[Interaction]:
        """Generate synthetic user-item interactions."""
        interactions = []
        
        # Create popularity distribution for items
        item_popularity = np.random.pareto(1.5, len(items))
        item_popularity = item_popularity / item_popularity.sum()
        
        start_time = datetime.now() - timedelta(days=365)
        
        for _ in range(n_interactions):
            user = random.choice(users)
            item_idx = np.random.choice(len(items), p=item_popularity)
            item = items[item_idx]
            
            # Generate interaction type based on user preferences
            interaction_types = [InteractionType.VIEW, InteractionType.LIKE, InteractionType.PURCHASE]
            weights = [0.7, 0.2, 0.1]  # Most interactions are views
            
            # Adjust weights based on user preferences
            if item.category in user.preferences.get("preferred_categories", []):
                weights = [0.5, 0.3, 0.2]  # More purchases for preferred categories
            
            interaction_type = random.choices(interaction_types, weights=weights)[0]
            
            # Generate timestamp (more recent interactions are more likely)
            days_ago = random.expovariate(0.1)  # Exponential decay
            timestamp = start_time + timedelta(days=days_ago)
            
            # Generate rating for purchase interactions
            rating = None
            if interaction_type == InteractionType.PURCHASE:
                rating = random.uniform(3.0, 5.0)  # Generally positive ratings
            elif interaction_type == InteractionType.LIKE:
                rating = random.uniform(4.0, 5.0)
            
            interaction = Interaction(
                user_id=user.user_id,
                item_id=item.item_id,
                interaction_type=interaction_type,
                timestamp=timestamp.timestamp(),
                rating=rating
            )
            
            interactions.append(interaction)
            
        return interactions


class DataLoader:
    """Loads and processes data for the recommendation system."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader."""
        self.data_dir = data_dir
    
    def load_items_from_csv(self, filepath: str) -> List[Item]:
        """Load items from a CSV file."""
        df = pd.read_csv(filepath)
        items = []
        
        for _, row in df.iterrows():
            item = Item(
                item_id=str(row['item_id']),
                title=str(row['title']),
                category=str(row['category']),
                price=float(row['price']),
                brand=str(row['brand']),
                description=str(row.get('description', '')),
                features=eval(row.get('features', '{}')),
                tags=eval(row.get('tags', '[]'))
            )
            items.append(item)
            
        return items
    
    def load_interactions_from_csv(self, filepath: str) -> List[Interaction]:
        """Load interactions from a CSV file."""
        df = pd.read_csv(filepath)
        interactions = []
        
        for _, row in df.iterrows():
            interaction = Interaction(
                user_id=str(row['user_id']),
                item_id=str(row['item_id']),
                interaction_type=InteractionType(row['interaction_type']),
                timestamp=float(row['timestamp']),
                rating=row.get('rating'),
                context=eval(row.get('context', '{}')) if row.get('context') else None
            )
            interactions.append(interaction)
            
        return interactions
    
    def save_items_to_csv(self, items: List[Item], filepath: str) -> None:
        """Save items to a CSV file."""
        data = []
        for item in items:
            data.append({
                'item_id': item.item_id,
                'title': item.title,
                'category': item.category,
                'price': item.price,
                'brand': item.brand,
                'description': item.description,
                'features': str(item.features),
                'tags': str(item.tags)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def save_interactions_to_csv(self, interactions: List[Interaction], filepath: str) -> None:
        """Save interactions to a CSV file."""
        data = []
        for interaction in interactions:
            data.append({
                'user_id': interaction.user_id,
                'item_id': interaction.item_id,
                'interaction_type': interaction.interaction_type.value,
                'timestamp': interaction.timestamp,
                'rating': interaction.rating,
                'context': str(interaction.context) if interaction.context else None
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
