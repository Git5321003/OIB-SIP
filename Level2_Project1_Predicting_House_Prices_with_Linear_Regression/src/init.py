"""
Housing Price Prediction Package
-------------------------------
A comprehensive package for predicting house prices using linear regression.
Includes data preprocessing, modeling, and visualization modules.
"""

__version__ = "1.0.0"
__author__ = "Real Estate Analytics Team"
__email__ = "analytics@realestate.com"

from . import data_preprocessing
from . import model
from . import visualization

__all__ = ["data_preprocessing", "model", "visualization"]

print(f"Initializing Housing Price Prediction Package v{__version__}")