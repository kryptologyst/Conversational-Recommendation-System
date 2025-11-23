"""Recommendation models for the conversational recommendation system."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import implicit
from scipy.sparse import csr_matrix

from ..types import Item, User, Interaction, Recommendation, InteractionType


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    def __init__(self, name: str):
        """Initialize the recommender."""
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, interactions: List[Interaction], items: List[Item], users: List[User]) -> None:
        """Fit the recommendation model."""
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        preferences: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate recommendations for a user."""
        pass
    
    def explain_recommendation(self, user_id: str, item_id: str) -> List[str]:
        """Provide explanation for a recommendation."""
        return [f"Recommended by {self.name} model"]


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommender."""
    
    def __init__(self):
        super().__init__("Popularity")
        self.item_popularity = {}
        
    def fit(self, interactions: List[Interaction], items: List[Item], users: List[User]) -> None:
        """Fit the popularity model."""
        # Calculate item popularity scores
        item_counts = {}
        for interaction in interactions:
            if interaction.interaction_type in [InteractionType.PURCHASE, InteractionType.LIKE]:
                item_counts[interaction.item_id] = item_counts.get(interaction.item_id, 0) + 1
        
        total_interactions = sum(item_counts.values())
        self.item_popularity = {
            item_id: count / total_interactions 
            for item_id, count in item_counts.items()
        }
        self.is_fitted = True
        
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        preferences: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate popularity-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Sort items by popularity
        sorted_items = sorted(
            self.item_popularity.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        recommendations = []
        for item_id, score in sorted_items[:n_recommendations]:
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                explanation=f"Popular item with {score:.3f} popularity score",
                reasoning=[f"High popularity score: {score:.3f}"]
            ))
        
        return recommendations


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using TF-IDF and cosine similarity."""
    
    def __init__(self):
        super().__init__("Content-Based")
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.item_features = None
        self.item_similarity = None
        self.items = None
        
    def fit(self, interactions: List[Interaction], items: List[Item], users: List[User]) -> None:
        """Fit the content-based model."""
        self.items = {item.item_id: item for item in items}
        
        # Create item descriptions for TF-IDF
        item_descriptions = []
        item_ids = []
        
        for item in items:
            # Combine title, description, category, and tags
            description = f"{item.title} {item.description} {item.category} {' '.join(item.tags)}"
            item_descriptions.append(description)
            item_ids.append(item.item_id)
        
        # Fit TF-IDF vectorizer
        self.item_features = self.tfidf_vectorizer.fit_transform(item_descriptions)
        
        # Compute item similarity matrix
        self.item_similarity = cosine_similarity(self.item_features)
        
        self.is_fitted = True
        
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        preferences: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate content-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # If no preferences, return popular items
        if not preferences:
            return self._get_popular_items(n_recommendations)
        
        # Create preference vector
        preference_vector = self._create_preference_vector(preferences)
        
        # Compute similarity scores
        scores = cosine_similarity(preference_vector, self.item_features)[0]
        
        # Get top recommendations
        item_ids = list(self.items.keys())
        scored_items = list(zip(item_ids, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in scored_items[:n_recommendations]:
            item = self.items[item_id]
            recommendations.append(Recommendation(
                item_id=item_id,
                score=float(score),
                explanation=f"Matches your preferences for {preferences.get('category', 'items')}",
                reasoning=[f"Content similarity: {score:.3f}", f"Category: {item.category}"]
            ))
        
        return recommendations
    
    def _create_preference_vector(self, preferences: Dict) -> np.ndarray:
        """Create a preference vector from user preferences."""
        # Create a text description of preferences
        pref_text = ""
        if "category" in preferences:
            pref_text += f"{preferences['category']} "
        if "price_range" in preferences:
            pref_text += f"{preferences['price_range']} price "
        
        # Transform to vector
        pref_vector = self.tfidf_vectorizer.transform([pref_text])
        return pref_vector
    
    def _get_popular_items(self, n_recommendations: int) -> List[Recommendation]:
        """Get popular items when no preferences are available."""
        # Simple popularity based on item features
        item_scores = np.mean(self.item_similarity, axis=1)
        item_ids = list(self.items.keys())
        
        scored_items = list(zip(item_ids, item_scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in scored_items[:n_recommendations]:
            item = self.items[item_id]
            recommendations.append(Recommendation(
                item_id=item_id,
                score=float(score),
                explanation="Popular item based on content features",
                reasoning=[f"Content popularity: {score:.3f}"]
            ))
        
        return recommendations


class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommender using matrix factorization."""
    
    def __init__(self, factors: int = 50, regularization: float = 0.01):
        super().__init__("Collaborative Filtering")
        self.factors = factors
        self.regularization = regularization
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def fit(self, interactions: List[Interaction], items: List[Item], users: List[User]) -> None:
        """Fit the collaborative filtering model."""
        # Create user and item mappings
        user_ids = list(set(interaction.user_id for interaction in interactions))
        item_ids = list(set(interaction.item_id for interaction in interactions))
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Create interaction matrix
        rows, cols, data = [], [], []
        for interaction in interactions:
            if interaction.interaction_type in [InteractionType.PURCHASE, InteractionType.LIKE]:
                user_idx = self.user_mapping[interaction.user_id]
                item_idx = self.item_mapping[interaction.item_id]
                
                # Weight different interaction types
                weight = 1.0
                if interaction.interaction_type == InteractionType.PURCHASE:
                    weight = 2.0
                elif interaction.rating:
                    weight = interaction.rating / 5.0
                
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(weight)
        
        # Create sparse matrix
        interaction_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(user_ids), len(item_ids))
        )
        
        # Fit ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            random_state=42
        )
        self.model.fit(interaction_matrix)
        
        self.is_fitted = True
        
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        preferences: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate collaborative filtering recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            # Cold start - return popular items
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_mapping[user_id]
        item_scores = self.model.recommend(user_idx, None, N=n_recommendations)
        
        recommendations = []
        for item_idx, score in item_scores:
            item_id = self.reverse_item_mapping[item_idx]
            recommendations.append(Recommendation(
                item_id=item_id,
                score=float(score),
                explanation="Recommended by similar users",
                reasoning=[f"Collaborative score: {score:.3f}", "Based on similar users' preferences"]
            ))
        
        return recommendations
    
    def _get_popular_items(self, n_recommendations: int) -> List[Recommendation]:
        """Get popular items for cold start users."""
        # Return items with highest average scores
        item_scores = np.mean(self.model.item_factors, axis=1)
        top_items = np.argsort(item_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for item_idx in top_items:
            item_id = self.reverse_item_mapping[item_idx]
            score = item_scores[item_idx]
            recommendations.append(Recommendation(
                item_id=item_id,
                score=float(score),
                explanation="Popular item among all users",
                reasoning=[f"Popularity score: {score:.3f}"]
            ))
        
        return recommendations


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple approaches."""
    
    def __init__(self, recommenders: List[BaseRecommender], weights: Optional[List[float]] = None):
        super().__init__("Hybrid")
        self.recommenders = recommenders
        self.weights = weights or [1.0] * len(recommenders)
        
    def fit(self, interactions: List[Interaction], items: List[Item], users: List[User]) -> None:
        """Fit all component recommenders."""
        for recommender in self.recommenders:
            recommender.fit(interactions, items, users)
        self.is_fitted = True
        
    def recommend(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        preferences: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate hybrid recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from all models
        all_recommendations = {}
        
        for recommender, weight in zip(self.recommenders, self.weights):
            recs = recommender.recommend(user_id, n_recommendations * 2, preferences)
            for rec in recs:
                if rec.item_id not in all_recommendations:
                    all_recommendations[rec.item_id] = {
                        'score': 0.0,
                        'explanations': [],
                        'reasoning': []
                    }
                
                all_recommendations[rec.item_id]['score'] += rec.score * weight
                all_recommendations[rec.item_id]['explanations'].append(rec.explanation)
                all_recommendations[rec.item_id]['reasoning'].extend(rec.reasoning)
        
        # Normalize scores
        max_score = max(item['score'] for item in all_recommendations.values())
        if max_score > 0:
            for item_data in all_recommendations.values():
                item_data['score'] /= max_score
        
        # Sort and return top recommendations
        sorted_items = sorted(
            all_recommendations.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        recommendations = []
        for item_id, data in sorted_items[:n_recommendations]:
            recommendations.append(Recommendation(
                item_id=item_id,
                score=data['score'],
                explanation=f"Hybrid recommendation combining {len(data['explanations'])} models",
                reasoning=data['reasoning']
            ))
        
        return recommendations
