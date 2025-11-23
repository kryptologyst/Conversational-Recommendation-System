"""Evaluation metrics and utilities for recommendation systems."""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd

from ..types import Recommendation, Interaction, InteractionType, RecommendationMetrics


class RecommendationEvaluator:
    """Evaluates recommendation models using various metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = RecommendationMetrics()
        
    def evaluate_model(
        self,
        recommendations: Dict[str, List[Recommendation]],
        test_interactions: List[Interaction],
        k_values: List[int] = [5, 10, 20]
    ) -> RecommendationMetrics:
        """Evaluate a recommendation model."""
        # Group interactions by user
        user_interactions = defaultdict(list)
        for interaction in test_interactions:
            if interaction.interaction_type in [InteractionType.PURCHASE, InteractionType.LIKE]:
                user_interactions[interaction.user_id].append(interaction.item_id)
        
        # Calculate metrics for each k
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            map_scores = []
            hit_rates = []
            
            for user_id, relevant_items in user_interactions.items():
                if user_id not in recommendations:
                    continue
                    
                user_recs = recommendations[user_id][:k]
                recommended_items = [rec.item_id for rec in user_recs]
                
                # Calculate metrics
                precision = self._precision_at_k(recommended_items, relevant_items, k)
                recall = self._recall_at_k(recommended_items, relevant_items, len(relevant_items))
                ndcg = self._ndcg_at_k(recommended_items, relevant_items, k)
                map_score = self._map_at_k(recommended_items, relevant_items, k)
                hit_rate = self._hit_rate_at_k(recommended_items, relevant_items)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                ndcg_scores.append(ndcg)
                map_scores.append(map_score)
                hit_rates.append(hit_rate)
            
            # Average metrics
            self.metrics.precision_at_k[k] = np.mean(precision_scores) if precision_scores else 0.0
            self.metrics.recall_at_k[k] = np.mean(recall_scores) if recall_scores else 0.0
            self.metrics.ndcg_at_k[k] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            self.metrics.map_at_k[k] = np.mean(map_scores) if map_scores else 0.0
            self.metrics.hit_rate_at_k[k] = np.mean(hit_rates) if hit_rates else 0.0
        
        # Calculate additional metrics
        self.metrics.coverage = self._calculate_coverage(recommendations)
        self.metrics.diversity = self._calculate_diversity(recommendations)
        self.metrics.novelty = self._calculate_novelty(recommendations, user_interactions)
        self.metrics.popularity_bias = self._calculate_popularity_bias(recommendations, user_interactions)
        
        return self.metrics
    
    def _precision_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        recommended_set = set(recommended[:k])
        relevant_set = set(relevant)
        intersection = recommended_set.intersection(relevant_set)
        return len(intersection) / k
    
    def _recall_at_k(self, recommended: List[str], relevant: Set[str], total_relevant: int) -> float:
        """Calculate Recall@K."""
        if total_relevant == 0:
            return 0.0
        recommended_set = set(recommended)
        relevant_set = set(relevant)
        intersection = recommended_set.intersection(relevant_set)
        return len(intersection) / total_relevant
    
    def _ndcg_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Calculate NDCG@K."""
        if k == 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _map_at_k(self, recommended: List[str], relevant: Set[str], k: int) -> float:
        """Calculate MAP@K."""
        if k == 0 or len(relevant) == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant)
    
    def _hit_rate_at_k(self, recommended: List[str], relevant: Set[str]) -> float:
        """Calculate Hit Rate@K."""
        recommended_set = set(recommended)
        relevant_set = set(relevant)
        intersection = recommended_set.intersection(relevant_set)
        return 1.0 if len(intersection) > 0 else 0.0
    
    def _calculate_coverage(self, recommendations: Dict[str, List[Recommendation]]) -> float:
        """Calculate catalog coverage."""
        all_items = set()
        recommended_items = set()
        
        for user_recs in recommendations.values():
            for rec in user_recs:
                recommended_items.add(rec.item_id)
        
        # Assuming we have access to all items somehow
        # For now, return the ratio of unique recommended items
        return len(recommended_items) / max(len(all_items), 1)
    
    def _calculate_diversity(self, recommendations: Dict[str, List[Recommendation]]) -> float:
        """Calculate intra-list diversity using cosine similarity."""
        diversity_scores = []
        
        for user_recs in recommendations.values():
            if len(user_recs) < 2:
                continue
                
            # Simple diversity based on item IDs (in practice, would use item features)
            items = [rec.item_id for rec in user_recs]
            diversity = len(set(items)) / len(items)
            diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_novelty(self, recommendations: Dict[str, List[Recommendation]], user_interactions: Dict[str, List[str]]) -> float:
        """Calculate novelty of recommendations."""
        novelty_scores = []
        
        for user_id, user_recs in recommendations.items():
            if user_id not in user_interactions:
                continue
                
            user_items = set(user_interactions[user_id])
            recommended_items = [rec.item_id for rec in user_recs]
            
            # Novelty = ratio of new items
            new_items = set(recommended_items) - user_items
            novelty = len(new_items) / len(recommended_items) if recommended_items else 0
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def _calculate_popularity_bias(self, recommendations: Dict[str, List[Recommendation]], user_interactions: Dict[str, List[str]]) -> float:
        """Calculate popularity bias in recommendations."""
        # Count item popularity
        item_popularity = defaultdict(int)
        for user_items in user_interactions.values():
            for item in user_items:
                item_popularity[item] += 1
        
        if not item_popularity:
            return 0.0
        
        # Calculate average popularity of recommended items
        avg_popularity_scores = []
        for user_recs in recommendations.values():
            rec_items = [rec.item_id for rec in user_recs]
            popularity_scores = [item_popularity[item] for item in rec_items]
            if popularity_scores:
                avg_popularity_scores.append(np.mean(popularity_scores))
        
        if not avg_popularity_scores:
            return 0.0
        
        # Normalize by max possible popularity
        max_popularity = max(item_popularity.values())
        return np.mean(avg_popularity_scores) / max_popularity if max_popularity > 0 else 0.0
    
    def compare_models(self, model_results: Dict[str, RecommendationMetrics]) -> pd.DataFrame:
        """Compare multiple models and return results as DataFrame."""
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {
                'Model': model_name,
                'Precision@5': metrics.precision_at_k.get(5, 0.0),
                'Recall@5': metrics.recall_at_k.get(5, 0.0),
                'NDCG@5': metrics.ndcg_at_k.get(5, 0.0),
                'MAP@5': metrics.map_at_k.get(5, 0.0),
                'HitRate@5': metrics.hit_rate_at_k.get(5, 0.0),
                'Precision@10': metrics.precision_at_k.get(10, 0.0),
                'Recall@10': metrics.recall_at_k.get(10, 0.0),
                'NDCG@10': metrics.ndcg_at_k.get(10, 0.0),
                'MAP@10': metrics.map_at_k.get(10, 0.0),
                'HitRate@10': metrics.hit_rate_at_k.get(10, 0.0),
                'Coverage': metrics.coverage,
                'Diversity': metrics.diversity,
                'Novelty': metrics.novelty,
                'PopularityBias': metrics.popularity_bias
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, model_results: Dict[str, RecommendationMetrics]) -> str:
        """Generate a text report comparing models."""
        report = "Recommendation Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        df = self.compare_models(model_results)
        
        # Sort by NDCG@10
        df_sorted = df.sort_values('NDCG@10', ascending=False)
        
        report += "Model Rankings (by NDCG@10):\n"
        report += "-" * 30 + "\n"
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            report += f"{i}. {row['Model']}: NDCG@10 = {row['NDCG@10']:.4f}\n"
        
        report += "\nDetailed Metrics:\n"
        report += "-" * 20 + "\n"
        report += df_sorted.to_string(index=False, float_format='%.4f')
        
        return report
