"""
Tests for clustering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from clustering import CustomerSegmentation
from data_preprocessing import engineer_features, prepare_clustering_data


class TestClustering:
    """Test cases for clustering functions."""
    
    @pytest.fixture
    def sample_data_for_clustering(self):
        """Create sample data for clustering tests."""
        np.random.seed(42)
        data = {
            'Income': np.random.normal(50000, 20000, 100),
            'MntTotal': np.random.normal(1000, 300, 100),
            'marital_Married': np.random.choice([0, 1], 100),
            'marital_Together': np.random.choice([0, 1], 100),
            'Kidhome': np.random.choice([0, 1, 2], 100),
            'Teenhome': np.random.choice([0, 1], 100),
            'Recency': np.random.randint(1, 100, 100)
        }
        df = pd.DataFrame(data)
        return engineer_features(df)
    
    def test_customer_segmentation_initialization(self):
        """Test CustomerSegmentation class initialization."""
        segmentation = CustomerSegmentation(random_state=42)
        assert segmentation.random_state == 42
        assert segmentation.model is None
        assert segmentation.labels is None
    
    def test_find_optimal_clusters(self, sample_data_for_clustering):
        """Test optimal cluster finding."""
        X_scaled, _, _ = prepare_clustering_data(sample_data_for_clustering)
        segmentation = CustomerSegmentation()
        
        results = segmentation.find_optimal_clusters(X_scaled, max_k=5)
        
        # Check if results contain expected keys
        assert 'wcss' in results
        assert 'silhouette_scores' in results
        
        # Check if results have correct lengths
        assert len(results['wcss']) == 5
        assert len(results['silhouette_scores']) == 4  # k=2 to k=5
    
    def test_perform_kmeans_clustering(self, sample_data_for_clustering):
        """Test K-means clustering."""
        X_scaled, _, _ = prepare_clustering_data(sample_data_for_clustering)
        segmentation = CustomerSegmentation()
        
        labels, metrics = segmentation.perform_kmeans_clustering(X_scaled, n_clusters=3)
        
        # Check if labels are created
        assert labels is not None
        assert len(labels) == len(X_scaled)
        
        # Check if metrics are calculated
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        
        # Check if model is trained
        assert segmentation.model is not None
    
    def test_analyze_clusters(self, sample_data_for_clustering):
        """Test cluster analysis."""
        X_scaled, feature_names, _ = prepare_clustering_data(sample_data_for_clustering)
        segmentation = CustomerSegmentation()
        labels, _ = segmentation.perform_kmeans_clustering(X_scaled, n_clusters=3)
        
        cluster_summary, insights = segmentation.analyze_clusters(
            sample_data_for_clustering, labels, feature_names
        )
        
        # Check if cluster summary is created
        assert cluster_summary is not None
        assert len(cluster_summary) == 3  # 3 clusters
        
        # Check if insights are generated
        assert insights is not None
        assert len(insights) == 3
        
        # Check insight structure
        for cluster in [0, 1, 2]:
            assert 'size' in insights[cluster]
            assert 'percentage' in insights[cluster]
            assert 'avg_income' in insights[cluster]


def test_clustering_with_different_k():
    """Test clustering with different numbers of clusters."""
    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.randn(50, 3)
    
    segmentation = CustomerSegmentation()
    
    for k in [2, 3, 4]:
        labels, metrics = segmentation.perform_kmeans_clustering(X, n_clusters=k)
        assert len(np.unique(labels)) == k
        assert metrics['silhouette_score'] <= 1.0  # Silhouette score range
        