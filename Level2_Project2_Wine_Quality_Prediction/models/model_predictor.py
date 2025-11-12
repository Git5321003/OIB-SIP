import pandas as pd
import numpy as np
from .model_manager import ModelManager
from .data_preprocessing import DataPreprocessor

class ModelPredictor:
    """
    Handles loading saved models and making predictions.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.model_manager = ModelManager(models_dir)
        self.loaded_models = {}
        self.current_model = None
        self.current_metadata = None
    
    def load_model_for_prediction(self, model_name: str, version: str = "latest"):
        """
        Load a specific model for making predictions.
        
        Args:
            model_name: Name of the model to load
            version: Model version (default: latest)
        """
        model, metadata = self.model_manager.load_model(model_name, version)
        self.loaded_models[model_name] = model
        self.current_model = model
        self.current_metadata = metadata
        
        print(f"Loaded {model_name} version {metadata['version']}")
        print(f"Model type: {metadata['model_type']}")
        print(f"Training accuracy: {metadata.get('accuracy', 'Unknown')}")
        
        return model, metadata
    
    def predict_quality(self, features, feature_names=None):
        """
        Predict wine quality for given features.
        
        Args:
            features: List or array of feature values
            feature_names: Names of features (for DataFrame conversion)
            
        Returns:
            dict: Prediction results
        """
        if self.current_model is None:
            raise ValueError("No model loaded. Call load_model_for_prediction first.")
        
        # Convert to numpy array and ensure correct shape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = self.current_model.predict(features_array)[0]
            quality = "Good" if prediction == 1 else "Bad"
            
            # Get prediction probabilities/confidence
            if hasattr(self.current_model, 'predict_proba'):
                probabilities = self.current_model.predict_proba(features_array)[0]
                confidence = max(probabilities)
            else:
                decision_scores = self.current_model.decision_function(features_array)
                confidence = abs(decision_scores[0])
            
            result = {
                'prediction': prediction,
                'quality': quality,
                'confidence': float(confidence),
                'features_used': len(features)
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, features_list):
        """
        Make predictions for multiple wine samples.
        
        Args:
            features_list: List of feature arrays
            
        Returns:
            list: Prediction results for each sample
        """
        results = []
        for features in features_list:
            try:
                result = self.predict_quality(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    
    def get_model_info(self):
        """Get information about the currently loaded model."""
        if self.current_metadata is None:
            return "No model loaded"
        
        info = f"""
Model Information:
-----------------
Name: {self.current_metadata.get('model_name', 'Unknown')}
Version: {self.current_metadata.get('version', 'Unknown')}
Type: {self.current_metadata.get('model_type', 'Unknown')}
Saved: {self.current_metadata.get('saved_at', 'Unknown')}
Training Accuracy: {self.current_metadata.get('accuracy', 'Unknown'):.4f}
        """
        return info
    