"""
Model Manager
------------
Handles model saving, loading, versioning, and management.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages machine learning models including versioning, storage, and retrieval.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir (str): Directory to store models
        """
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "models_metadata.json")
        self._ensure_directories()
        self._load_metadata()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Models directory: {os.path.abspath(self.models_dir)}")
    
    def _load_metadata(self):
        """Load models metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}. Creating new metadata.")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save models metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def save_model(self, model_data: Dict, model_name: str, 
                  version: str = "v1.0.0", description: str = "",
                  metrics: Optional[Dict] = None, tags: Optional[List[str]] = None):
        """
        Save a trained model with metadata.
        
        Args:
            model_data (dict): Model data including the actual model
            model_name (str): Name of the model
            version (str): Model version
            description (str): Model description
            metrics (dict): Model performance metrics
            tags (list): List of tags for the model
            
        Returns:
            str: Path to the saved model file
        """
        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{version}_{timestamp}.joblib"
            filepath = os.path.join(self.models_dir, filename)
            
            # Add metadata to model data
            model_data['metadata'] = {
                'model_name': model_name,
                'version': version,
                'description': description,
                'saved_at': datetime.now().isoformat(),
                'metrics': metrics or {},
                'tags': tags or [],
                'filename': filename
            }
            
            # Save model
            joblib.dump(model_data, filepath)
            
            # Update metadata
            if model_name not in self.metadata:
                self.metadata[model_name] = {}
            
            self.metadata[model_name][version] = {
                'filename': filename,
                'saved_at': datetime.now().isoformat(),
                'description': description,
                'metrics': metrics or {},
                'tags': tags or [],
                'file_path': filepath
            }
            
            self._save_metadata()
            
            logger.info(f"Model saved successfully: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_name: str, version: str = "latest"):
        """
        Load a saved model.
        
        Args:
            model_name (str): Name of the model
            version (str): Model version or 'latest'
            
        Returns:
            dict: Model data
        """
        try:
            if model_name not in self.metadata:
                raise ValueError(f"Model '{model_name}' not found in metadata")
            
            if version == "latest":
                # Get the latest version
                versions = list(self.metadata[model_name].keys())
                if not versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                version = sorted(versions)[-1]  # Get latest version
            
            if version not in self.metadata[model_name]:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            
            model_info = self.metadata[model_name][version]
            filepath = model_info['file_path']
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            model_data = joblib.load(filepath)
            logger.info(f"Model loaded successfully: {filepath}")
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def list_models(self) -> pd.DataFrame:
        """
        List all saved models.
        
        Returns:
            pd.DataFrame: DataFrame with model information
        """
        models_list = []
        
        for model_name, versions in self.metadata.items():
            for version, info in versions.items():
                models_list.append({
                    'model_name': model_name,
                    'version': version,
                    'filename': info['filename'],
                    'saved_at': info['saved_at'],
                    'description': info['description'],
                    'metrics': info.get('metrics', {}),
                    'tags': ', '.join(info.get('tags', [])),
                    'file_size': f"{os.path.getsize(info['file_path']) / 1024 / 1024:.2f} MB"
                })
        
        return pd.DataFrame(models_list)
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict:
        """
        Get detailed information about a model.
        
        Args:
            model_name (str): Name of the model
            version (str): Model version
            
        Returns:
            dict: Model information
        """
        if version == "latest":
            versions = list(self.metadata.get(model_name, {}).keys())
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            version = sorted(versions)[-1]
        
        if model_name not in self.metadata or version not in self.metadata[model_name]:
            raise ValueError(f"Model '{model_name}' version '{version}' not found")
        
        return self.metadata[model_name][version]
    
    def delete_model(self, model_name: str, version: str):
        """
        Delete a saved model.
        
        Args:
            model_name (str): Name of the model
            version (str): Model version
        """
        try:
            if model_name not in self.metadata or version not in self.metadata[model_name]:
                raise ValueError(f"Model '{model_name}' version '{version}' not found")
            
            model_info = self.metadata[model_name][version]
            filepath = model_info['file_path']
            
            # Delete file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove from metadata
            del self.metadata[model_name][version]
            
            # Remove model name if no versions left
            if not self.metadata[model_name]:
                del self.metadata[model_name]
            
            self._save_metadata()
            
            logger.info(f"Model deleted: {model_name} version {version}")
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            raise
    
    def export_model_card(self, model_name: str, version: str, output_path: str):
        """
        Export a model card with detailed information.
        
        Args:
            model_name (str): Name of the model
            version (str): Model version
            output_path (str): Path to save the model card
        """
        try:
            model_info = self.get_model_info(model_name, version)
            model_data = self.load_model(model_name, version)
            
            model_card = {
                'model_name': model_name,
                'version': version,
                'exported_at': datetime.now().isoformat(),
                'model_info': model_info,
                'model_type': type(model_data.get('model')).__name__,
                'feature_names': model_data.get('feature_names', []),
                'performance_metrics': model_info.get('metrics', {}),
                'training_metrics': model_data.get('train_metrics'),
                'testing_metrics': model_data.get('test_metrics')
            }
            
            with open(output_path, 'w') as f:
                json.dump(model_card, f, indent=2, default=str)
            
            logger.info(f"Model card exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting model card: {e}")
            raise
    
    def get_latest_model(self, model_name: str):
        """
        Get the latest version of a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: (model_data, model_info)
        """
        model_info = self.get_model_info(model_name, "latest")
        model_data = self.load_model(model_name, "latest")
        return model_data, model_info


# Utility functions
def create_model_package(model, feature_names, train_metrics, test_metrics, 
                       feature_importance, scaler=None):
    """
    Create a standardized model package for saving.
    
    Args:
        model: Trained model object
        feature_names (list): List of feature names
        train_metrics (dict): Training metrics
        test_metrics (dict): Testing metrics
        feature_importance (pd.DataFrame): Feature importance
        scaler: Fitted scaler object (optional)
        
    Returns:
        dict: Model package ready for saving
    """
    model_package = {
        'model': model,
        'feature_names': feature_names,
        'model_trained': True,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance,
        'saved_at': datetime.now().isoformat(),
        'model_type': type(model).__name__
    }
    
    if scaler is not None:
        model_package['scaler'] = scaler
    
    return model_package


def load_model_simple(filepath):
    """
    Simple function to load a model from file.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        dict: Model data
    """
    try:
        model_data = joblib.load(filepath)
        logger.info(f"Model loaded from: {filepath}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def save_model_simple(model_data, filepath):
    """
    Simple function to save a model to file.
    
    Args:
        model_data (dict): Model data to save
        filepath (str): Path to save the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
    