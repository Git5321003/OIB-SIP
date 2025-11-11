"""
Models Package
-------------
Contains saved machine learning models and model management utilities.
"""

__version__ = "1.0.0"
__author__ = "Real Estate Analytics Team"

import os
import joblib
import pandas as pd
from datetime import datetime

MODELS_DIR = os.path.dirname(__file__)

def list_saved_models():
    """
    List all saved models in the models directory.
    
    Returns:
        list: List of model filenames
    """
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') or f.endswith('.joblib')]
    return sorted(models)

def get_model_info(model_path):
    """
    Get information about a saved model.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        dict: Model information
    """
    try:
        model_data = joblib.load(model_path)
        info = {
            'filename': os.path.basename(model_path),
            'file_size': f"{os.path.getsize(model_path) / 1024 / 1024:.2f} MB",
            'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path)),
            'model_type': type(model_data.get('model', None)).__name__,
            'feature_names': model_data.get('feature_names', []),
            'model_trained': model_data.get('model_trained', False)
        }
        return info
    except Exception as e:
        return {'error': str(e)}

print(f"Models package initialized. Models directory: {MODELS_DIR}")