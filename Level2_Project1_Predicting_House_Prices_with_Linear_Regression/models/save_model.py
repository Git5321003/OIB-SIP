"""
Model Saving Script
------------------
Script to save trained models with proper versioning and metadata.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.model import HousingPriceModel
from models.model_manager import ModelManager, create_model_package

def train_and_save_model():
    """
    Train a model and save it with proper versioning.
    """
    print("=" * 60)
    print("MODEL TRAINING AND SAVING SCRIPT")
    print("=" * 60)
    
    try:
        # Initialize components
        preprocessor = DataPreprocessor()
        model = HousingPriceModel()
        model_manager = ModelManager()
        
        # Load and preprocess data
        data_path = "../data/Housing.csv"
        print(f"Loading data from: {data_path}")
        
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names = \
            preprocessor.full_preprocessing_pipeline(data_path)
        
        # Train model
        print("Training model...")
        model.train(X_train_scaled, y_train, feature_names)
        
        # Evaluate model
        print("Evaluating model...")
        metrics_df = model.evaluate(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Print performance
        test_metrics = metrics_df[metrics_df['Set'] == 'Testing'].iloc[0]
        print(f"\nModel Performance:")
        print(f"  R² Score: {test_metrics['R²']:.4f}")
        print(f"  RMSE: ₹{test_metrics['RMSE']:,.2f}")
        print(f"  MAE: ₹{test_metrics['MAE']:,.2f}")
        
        # Create model package
        model_package = create_model_package(
            model=model.model,
            feature_names=feature_names,
            train_metrics=model.train_metrics,
            test_metrics=model.test_metrics,
            feature_importance=model.feature_importance,
            scaler=preprocessor.scaler
        )
        
        # Save model with metadata
        model_path = model_manager.save_model(
            model_data=model_package,
            model_name="housing_price_predictor",
            version="v1.0.0",
            description="Linear Regression model for housing price prediction in Delhi region",
            metrics={
                'r2_score': float(test_metrics['R²']),
                'rmse': float(test_metrics['RMSE']),
                'mae': float(test_metrics['MAE']),
                'mse': float(test_metrics['MSE'])
            },
            tags=['linear_regression', 'housing', 'delhi', 'price_prediction']
        )
        
        print(f"\n✅ Model saved successfully!")
        print(f"📁 Location: {model_path}")
        
        # Display saved models
        print(f"\n📊 Saved Models:")
        models_df = model_manager.list_models()
        print(models_df.to_string(index=False))
        
        return model_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def save_multiple_versions():
    """
    Save multiple versions of the model for testing purposes.
    """
    print("\n" + "=" * 60)
    print("SAVING MULTIPLE MODEL VERSIONS")
    print("=" * 60)
    
    try:
        model_manager = ModelManager()
        
        # Different versions with slight variations
        versions = [
            {
                'name': 'housing_price_predictor',
                'version': 'v1.0.0',
                'description': 'Initial Linear Regression model',
                'tags': ['linear_regression', 'baseline']
            },
            {
                'name': 'housing_price_predictor',
                'version': 'v1.1.0', 
                'description': 'Linear Regression with feature engineering',
                'tags': ['linear_regression', 'feature_engineered']
            },
            {
                'name': 'premium_predictor',
                'version': 'v1.0.0',
                'description': 'Model optimized for premium properties',
                'tags': ['premium', 'optimized']
            }
        ]
        
        for version_info in versions:
            # Create dummy model data for demonstration
            dummy_model_data = {
                'model': "dummy_model_object",
                'feature_names': ['area', 'bedrooms', 'bathrooms'],
                'model_trained': True,
                'train_metrics': {'R²': 0.68, 'RMSE': 1200000},
                'test_metrics': {'R²': 0.65, 'RMSE': 1250000},
                'feature_importance': "dummy_importance"
            }
            
            model_path = model_manager.save_model(
                model_data=dummy_model_data,
                model_name=version_info['name'],
                version=version_info['version'],
                description=version_info['description'],
                metrics={'r2_score': 0.65, 'rmse': 1250000},
                tags=version_info['tags']
            )
            
            print(f"✅ Saved {version_info['name']} {version_info['version']}")
        
        print(f"\n📊 All Saved Models:")
        models_df = model_manager.list_models()
        print(models_df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Train and save the main model
    model_path = train_and_save_model()
    
    # Save multiple versions for demonstration
    save_multiple_versions()
    
    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    