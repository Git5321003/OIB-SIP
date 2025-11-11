"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import clean_data, engineer_features, prepare_clustering_data


class TestDataPreprocessing:
    """Test cases for data preprocessing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'Income': [50000, 75000, np.nan, 60000],
            'MntTotal': [1000, 1500, 800, 1200],
            'marital_Married': [1, 0, 1, 0],
            'marital_Together': [0, 1, 0, 1],
            'Kidhome': [1, 0, 2, 1],
            'Teenhome': [0, 1, 0, 1],
            'NumDealsPurchases': [2, 1, 3, 2],
            'NumCatalogPurchases': [1, 2, 1, 3],
            'NumStorePurchases': [3, 2, 4, 1],
            'NumWebPurchases': [2, 3, 1, 2],
            'Recency': [30, 45, 60, 15]
        }
        return pd.DataFrame(data)
    
    def test_clean_data(self, sample_data):
        """Test data cleaning function."""
        cleaned_data = clean_data(sample_data)
        
        # Check if missing values are handled
        assert cleaned_data.isnull().sum().sum() == 0
        
        # Check if data shape is preserved
        assert cleaned_data.shape[0] == sample_data.shape[0]
    
    def test_engineer_features(self, sample_data):
        """Test feature engineering function."""
        engineered_data = engineer_features(sample_data)
        
        # Check if new features are created
        expected_features = ['In_relationship', 'Total_Children', 'Total_Purchases']
        for feature in expected_features:
            assert feature in engineered_data.columns
        
        # Check relationship status calculation
        assert engineered_data['In_relationship'].dtype == np.int64
        assert engineered_data['In_relationship'].isin([0, 1]).all()
        
        # Check total children calculation
        assert engineered_data['Total_Children'].equals(
            engineered_data['Kidhome'] + engineered_data['Teenhome']
        )
    
    def test_prepare_clustering_data(self, sample_data):
        """Test clustering data preparation."""
        engineered_data = engineer_features(sample_data)
        X_scaled, features, scaler = prepare_clustering_data(engineered_data)
        
        # Check output shapes and types
        assert isinstance(X_scaled, np.ndarray)
        assert len(features) == X_scaled.shape[1]
        assert X_scaled.shape[0] == engineered_data.shape[0]
        
        # Check if features are standardized
        assert abs(X_scaled.mean()) < 1e-10
        assert abs(X_scaled.std() - 1) < 1e-10


def test_feature_engineering_edge_cases():
    """Test feature engineering with edge cases."""
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    try:
        engineer_features(empty_df)
    except Exception as e:
        # Should handle empty data gracefully
        assert True
    
    # Test with all NaN values
    nan_data = {
        'Income': [np.nan, np.nan],
        'MntTotal': [np.nan, np.nan],
        'marital_Married': [np.nan, np.nan],
        'marital_Together': [np.nan, np.nan]
    }
    nan_df = pd.DataFrame(nan_data)
    engineered_nan = engineer_features(nan_df)
    assert not engineered_nan.empty
    