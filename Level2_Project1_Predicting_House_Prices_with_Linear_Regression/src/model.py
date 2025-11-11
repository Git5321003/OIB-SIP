"""
Model Module
-----------
Contains the Linear Regression model implementation and evaluation functions.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HousingPriceModel:
    """
    A class for housing price prediction using Linear Regression.
    
    Attributes:
        model (LinearRegression): Trained linear regression model
        feature_names (list): Names of features used in the model
        model_trained (bool): Whether the model has been trained
    """
    
    def __init__(self):
        """Initialize the HousingPriceModel."""
        self.model = LinearRegression()
        self.feature_names = None
        self.model_trained = False
        self.train_metrics = None
        self.test_metrics = None
        self.feature_importance = None
        
    def train(self, X_train, y_train, feature_names=None):
        """
        Train the linear regression model.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training target
            feature_names (list): Names of features
        """
        try:
            logger.info("Training Linear Regression model...")
            self.model.fit(X_train, y_train)
            self.model_trained = True
            self.feature_names = feature_names
            
            # Calculate feature importance
            if feature_names is not None:
                self._calculate_feature_importance(feature_names)
            
            logger.info("Model training completed successfully")
            logger.info(f"Model intercept: {self.model.intercept_:.2f}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X (array-like): Input features
            
        Returns:
            array: Predicted values
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Please call train() first.")
            
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model performance on training and testing sets.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training target
            X_test (array-like): Testing features
            y_test (array-like): Testing target
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Please call train() first.")
            
        try:
            # Make predictions
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)
            
            # Calculate metrics
            self.train_metrics = self._calculate_metrics(y_train, y_train_pred, 'Training')
            self.test_metrics = self._calculate_metrics(y_test, y_test_pred, 'Testing')
            
            # Combine metrics
            metrics_df = pd.DataFrame([self.train_metrics, self.test_metrics])
            
            logger.info("Model evaluation completed")
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_metrics(self, y_true, y_pred, set_name):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            set_name (str): Name of the dataset
            
        Returns:
            dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'Set': set_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
    
    def _calculate_feature_importance(self, feature_names):
        """
        Calculate feature importance based on coefficients.
        
        Args:
            feature_names (list): Names of features
        """
        if self.model.coef_ is None:
            raise ValueError("Model coefficients not available")
            
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.model.coef_,
            'Absolute_Coefficient': np.abs(self.model.coef_)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        logger.info("Feature importance calculated")
    
    def get_feature_importance(self, top_n=None):
        """
        Get feature importance.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train model first.")
            
        if top_n is None:
            return self.feature_importance
        else:
            return self.feature_importance.head(top_n)
    
    def get_model_summary(self):
        """
        Get comprehensive model summary.
        
        Returns:
            dict: Model summary
        """
        if not self.model_trained:
            raise ValueError("Model not trained.")
            
        summary = {
            'model_type': 'Linear Regression',
            'trained': self.model_trained,
            'intercept': self.model.intercept_,
            'num_features': len(self.model.coef_) if self.model.coef_ is not None else 0,
            'feature_names': self.feature_names
        }
        
        if self.train_metrics is not None and self.test_metrics is not None:
            summary['training_r2'] = self.train_metrics['R²']
            summary['testing_r2'] = self.test_metrics['R²']
            summary['testing_rmse'] = self.test_metrics['RMSE']
        
        return summary
    
    def predict_single(self, features, scaler=None):
        """
        Predict price for a single property.
        
        Args:
            features (dict or array): Property features
            scaler (StandardScaler): Fitted scaler for feature normalization
            
        Returns:
            float: Predicted price
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Please call train() first.")
            
        try:
            # If features is a dictionary, convert to array
            if isinstance(features, dict):
                if self.feature_names is None:
                    raise ValueError("Feature names not available")
                
                # Create array in correct order
                feature_array = np.array([features[feature] for feature in self.feature_names]).reshape(1, -1)
            else:
                feature_array = features.reshape(1, -1)
            
            # Scale features if scaler is provided
            if scaler is not None:
                feature_array = scaler.transform(feature_array)
            
            prediction = self.predict(feature_array)[0]
            return prediction
            
        except Exception as e:
            logger.error(f"Error in single prediction: {e}")
            raise
    
    def save_model(self, file_path):
        """
        Save the trained model to a file.
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.model_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_trained': self.model_trained,
                'feature_importance': self.feature_importance,
                'train_metrics': self.train_metrics,
                'test_metrics': self.test_metrics
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, file_path):
        """
        Load a trained model from a file.
        
        Args:
            file_path (str): Path to the saved model
        """
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_trained = model_data['model_trained']
            self.feature_importance = model_data['feature_importance']
            self.train_metrics = model_data['train_metrics']
            self.test_metrics = model_data['test_metrics']
            
            logger.info(f"Model loaded successfully from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Utility functions
def calculate_residuals(y_true, y_pred):
    """
    Calculate residuals between true and predicted values.
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        array: Residuals
    """
    return y_true - y_pred


def calculate_prediction_interval(y_pred, residuals, confidence=0.95):
    """
    Calculate prediction intervals for regression.
    
    Args:
        y_pred (array-like): Predicted values
        residuals (array-like): Residuals from training
        confidence (float): Confidence level (0.95 for 95%)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    residual_std = np.std(residuals)
    z_score = 1.96  # For 95% confidence
    
    margin_of_error = z_score * residual_std
    
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error
    
    return lower_bound, upper_bound


def compare_models_metrics(metrics_dict):
    """
    Compare metrics from multiple models.
    
    Args:
        metrics_dict (dict): Dictionary of model metrics
        
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_df = pd.DataFrame(metrics_dict).T
    return comparison_df