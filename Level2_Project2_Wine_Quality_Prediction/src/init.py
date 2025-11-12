"""
Wine Quality Prediction Package

A machine learning package for predicting wine quality based on chemical characteristics.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Package imports for easier access
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator

# Define what gets imported with "from src import *"
__all__ = [
    'DataPreprocessor',
    'ModelTrainer', 
    'ModelEvaluator',
]

# Package metadata
__package_name__ = "wine_quality_predictor"
__description__ = "A machine learning package for wine quality prediction using chemical features"
__keywords__ = ["wine", "quality", "prediction", "machine learning", "classification"]

# Dependencies
__dependencies__ = [
    "pandas>=1.5.0",
    "numpy>=1.21.0", 
    "scikit-learn>=1.0.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0"
]

def get_version():
    """Return the package version."""
    return __version__

def get_authors():
    """Return package authors."""
    return [__author__]

def get_dependencies():
    """Return package dependencies."""
    return __dependencies__

def package_info():
    """Print comprehensive package information."""
    info = f"""
Wine Quality Prediction Package
------------------------------
Version: {__version__}
Author: {__author__}
Description: {__description__}

Available Classes:
- DataPreprocessor: Handles data loading, cleaning, and preprocessing
- ModelTrainer: Manages model training and cross-validation  
- ModelEvaluator: Provides comprehensive model evaluation and visualization

Dependencies: {', '.join(__dependencies__)}
    """
    print(info)

# Optional: Initialize package-level configuration
class Config:
    """Package configuration settings."""
    
    # Default random state for reproducibility
    RANDOM_STATE = 42
    
    # Default test size for train-test split
    TEST_SIZE = 0.2
    
    # Supported models
    SUPPORTED_MODELS = ['Random Forest', 'SGD Classifier', 'SVC']
    
    # Feature names from the dataset
    FEATURE_NAMES = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    
    # Target variable
    TARGET_NAME = 'quality'

# Create package-level logger (optional)
import logging

def setup_logging(level=logging.INFO):
    """Setup basic logging configuration for the package."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger for this package
    logger = logging.getLogger(__name__)
    return logger

# Initialize package logger
logger = setup_logging()

# Package initialization message
logger.info(f"Initialized {__package_name__} version {__version__}")

# Clean up namespace
del logging
