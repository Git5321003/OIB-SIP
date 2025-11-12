import pickle
import joblib
import json
import os
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator

class ModelManager:
    """
    Manages saving, loading, and versioning of trained models.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        """Create models directory if it doesn't exist."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model: BaseEstimator, model_name: str, 
                  version: str = None, metadata: dict = None):
        """
        Save a trained model with versioning and metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version (default: timestamp)
            metadata: Additional model metadata
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_dir = os.path.join(self.models_dir, model_name, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model using joblib (better for scikit-learn models)
        model_path = os.path.join(version_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'version': version,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__
        })
        
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_path}")
        return version_dir
    
    def load_model(self, model_name: str, version: str = "latest"):
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model
            version: Model version or 'latest'
            
        Returns:
            tuple: (model, metadata)
        """
        if version == "latest":
            version = self.get_latest_version(model_name)
        
        model_dir = os.path.join(self.models_dir, model_name, version)
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Model loaded from: {model_path}")
        return model, metadata
    
    def get_latest_version(self, model_name: str):
        """Get the latest version of a model."""
        model_base_dir = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_base_dir):
            raise FileNotFoundError(f"No models found for: {model_name}")
        
        versions = [d for d in os.listdir(model_base_dir) 
                   if os.path.isdir(os.path.join(model_base_dir, d))]
        if not versions:
            raise FileNotFoundError(f"No versions found for: {model_name}")
        
        return sorted(versions)[-1]  # Latest version
    
    def list_models(self):
        """List all saved models and their versions."""
        models_info = {}
        
        if not os.path.exists(self.models_dir):
            return models_info
        
        for model_name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.isdir(model_path):
                versions = []
                for version in os.listdir(model_path):
                    version_path = os.path.join(model_path, version)
                    if os.path.isdir(version_path):
                        metadata_path = os.path.join(version_path, "metadata.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            versions.append({
                                'version': version,
                                'saved_at': metadata.get('saved_at', 'Unknown'),
                                'model_type': metadata.get('model_type', 'Unknown')
                            })
                models_info[model_name] = versions
        
        return models_info
    
    def save_training_results(self, results: dict, filename: str = "training_results.json"):
        """Save training results to JSON."""
        results_path = os.path.join(self.models_dir, filename)
        
        # Convert any non-serializable objects
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'accuracy': float(result['accuracy']),
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'saved_at': datetime.now().isoformat()
            }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Training results saved to: {results_path}")
    
    def save_feature_importance(self, feature_importance: dict, model_name: str):
        """Save feature importance analysis."""
        importance_path = os.path.join(self.models_dir, f"{model_name}_feature_importance.json")
        
        with open(importance_path, 'w') as f:
            json.dump(feature_importance, f, indent=2)
        
        print(f"Feature importance saved to: {importance_path}")
    
    def delete_model(self, model_name: str, version: str):
        """Delete a specific model version."""
        model_dir = os.path.join(self.models_dir, model_name, version)
        
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
            print(f"Deleted model: {model_name} version {version}")
        else:
            print(f"Model not found: {model_name} version {version}")
            