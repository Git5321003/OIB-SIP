"""
Utility functions for the Wine Quality Prediction package.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Tuple

def validate_wine_features(features: List[float], expected_length: int = 11) -> bool:
    """
    Validate that wine features have the correct format and range.
    
    Args:
        features: List of chemical feature values
        expected_length: Expected number of features (default: 11)
    
    Returns:
        bool: True if features are valid
    """
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    
    # Basic range validation (you can expand this based on your data)
    if any(x < 0 for x in features[:3]):  # acidity values should be positive
        raise ValueError("Acidity values should be positive")
    
    return True

def create_sample_wine_data() -> dict:
    """
    Create sample wine data for testing and demonstration.
    
    Returns:
        dict: Sample wine feature data
    """
    return {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }

def save_model_results(results: dict, filename: str = "model_results.csv"):
    """
    Save model results to a CSV file.
    
    Args:
        results: Dictionary containing model results
        filename: Output filename
    """
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': result['accuracy'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        }
        for name, result in results.items()
    ])
    
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def load_model_results(filename: str = "model_results.csv") -> pd.DataFrame:
    """
    Load model results from a CSV file.
    
    Args:
        filename: Input filename
    
    Returns:
        pd.DataFrame: Loaded results
    """
    return pd.read_csv(filename)

# Update __init__.py to include utils if you create this file
# from .utils import validate_wine_features, create_sample_wine_data, save_model_results, load_model_results
