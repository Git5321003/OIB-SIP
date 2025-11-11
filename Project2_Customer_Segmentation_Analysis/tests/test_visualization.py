"""
Tests for visualization module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization import plot_elbow_method, plot_cluster_analysis, plot_cluster_profiles


class TestVisualization:
    """Test cases for visualization functions."""
    
    @pytest.fixture
    def sample_cluster_results(self):
        """Create sample cluster results for testing."""
        return {
            'wcss': [1000, 500, 300, 200, 150, 120],
            'silhouette_scores': [0.5, 0.6, 0.7, 0.65, 0.6]
        }
    
    @pytest.fixture
    def sample_clustered_data(self):
        """Create sample clustered data for testing."""
        np.random.seed(42)
        data = {
            'Income': np.random.normal(50000, 20000, 100),
            'MntTotal': np.random.normal(1000, 300, 100),
            'Cluster': np.random.choice([0, 1, 2], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_cluster_profiles(self):
        """Create sample cluster profiles for testing."""
        return {
            0: {
                'size': 40,
                'percentage': 40.0,
                'demographics': {
                    'avg_income': 60000,
                    'relationship_rate': 75.0,
                    'avg_children': 1.2
                },
                'spending_behavior': {
                    'total_spending': 1200,
                    'wine_spending': 400,
                    'fruit_spending': 150
                }
            },
            1: {
                'size': 35,
                'percentage': 35.0,
                'demographics': {
                    'avg_income': 35000,
                    'relationship_rate': 25.0,
                    'avg_children': 0.8
                },
                'spending_behavior': {
                    'total_spending': 600,
                    'wine_spending': 150,
                    'fruit_spending': 80
                }
            },
            2: {
                'size': 25,
                'percentage': 25.0,
                'demographics': {
                    'avg_income': 80000,
                    'relationship_rate': 40.0,
                    'avg_children': 0.5
                },
                'spending_behavior': {
                    'total_spending': 1800,
                    'wine_spending': 600,
                    'fruit_spending': 200
                }
            }
        }
    
    def test_plot_elbow_method(self, sample_cluster_results, tmp_path):
        """Test elbow method plotting."""
        save_path = tmp_path / "test_elbow.png"
        
        # Test that function runs without errors
        try:
            plot_elbow_method(sample_cluster_results, save_path=str(save_path))
            assert save_path.exists()
        except Exception as e:
            pytest.fail(f"plot_elbow_method failed with error: {e}")
    
    def test_plot_cluster_analysis(self, sample_clustered_data, tmp_path):
        """Test cluster analysis plotting."""
        save_path = tmp_path / "test_cluster_analysis.png"
        
        try:
            plot_cluster_analysis(sample_clustered_data, save_path=str(save_path))
            assert save_path.exists()
        except Exception as e:
            pytest.fail(f"plot_cluster_analysis failed with error: {e}")
    
    def test_plot_cluster_profiles(self, sample_cluster_profiles, tmp_path):
        """Test cluster profiles plotting."""
        save_path = tmp_path / "test_cluster_profiles.png"
        
        try:
            plot_cluster_profiles(sample_cluster_profiles, save_path=str(save_path))
            assert save_path.exists()
        except Exception as e:
            pytest.fail(f"plot_cluster_profiles failed with error: {e}")
    
    def test_visualization_with_empty_data(self):
        """Test visualization functions with empty data."""
        # Test with empty cluster results
        empty_results = {'wcss': [], 'silhouette_scores': []}
        try:
            plot_elbow_method(empty_results)
            # Should handle empty data gracefully
        except Exception:
            pass  # Some errors are expected with empty data
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        try:
            plot_cluster_analysis(empty_df)
        except Exception:
            pass


def test_visualization_figure_creation():
    """Test that visualizations create matplotlib figures."""
    # Create simple test data
    results = {
        'wcss': [100, 50, 30],
        'silhouette_scores': [0.5, 0.6]
    }
    
    # Test elbow method
    plot_elbow_method(results)
    assert plt.gcf().get_size_inches().prod() > 0
    plt.close('all')
    
    # Test that we can create multiple plots without interference
    data = pd.DataFrame({
        'Income': [1, 2, 3],
        'MntTotal': [4, 5, 6],
        'Cluster': [0, 1, 0]
    })
    
    plot_cluster_analysis(data)
    assert plt.gcf().get_size_inches().prod() > 0
    plt.close('all')
    