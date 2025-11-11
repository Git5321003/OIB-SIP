"""
Model Loading Script
-------------------
Script to load and use saved models for predictions.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_manager import ModelManager
from src.data_preprocessing import create_sample_property, create_multiple_sample_properties

def load_and_predict():
    """
    Load a saved model and make predictions.
    """
    print("=" * 60)
    print("MODEL LOADING AND PREDICTION SCRIPT")
    print("=" * 60)
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # List available models
        print("📊 Available Models:")
        models_df = model_manager.list_models()
        if models_df.empty:
            print("No models found. Please train and save a model first.")
            return
        
        print(models_df[['model_name', 'version', 'saved_at', 'description']].to_string(index=False))
        
        # Load the latest housing price predictor
        print(f"\n🔄 Loading latest housing price predictor...")
        model_data, model_info = model_manager.get_latest_model("housing_price_predictor")
        
        print(f"✅ Model loaded successfully!")
        print(f"📝 Description: {model_info['description']}")
        print(f"📅 Saved at: {model_info['saved_at']}")
        print(f"📈 Metrics: {model_info.get('metrics', {})}")
        
        # Extract components from model data
        model = model_data['model']
        feature_names = model_data['feature_names']
        scaler = model_data.get('scaler')
        
        print(f"\n🔧 Model Details:")
        print(f"   Model Type: {type(model).__name__}")
        print(f"   Number of Features: {len(feature_names)}")
        print(f"   Feature Names: {feature_names}")
        print(f"   Scaler Included: {'Yes' if scaler else 'No'}")
        
        return model_data, model_info
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def make_predictions(model_data):
    """
    Make predictions using the loaded model.
    
    Args:
        model_data (dict): Loaded model data
    """
    print("\n" + "=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    try:
        model = model_data['model']
        feature_names = model_data['feature_names']
        scaler = model_data.get('scaler')
        
        # Create sample properties
        sample_properties = create_multiple_sample_properties()
        
        print("🏠 PREDICTED PRICES FOR SAMPLE PROPERTIES:\n")
        
        for prop in sample_properties:
            # Prepare features for prediction (remove 'name' key)
            features = {k: v for k, v in prop.items() if k != 'name'}
            
            # Ensure all required features are present
            feature_vector = np.array([features[feature] for feature in feature_names]).reshape(1, -1)
            
            # Scale features if scaler is available
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            # Make prediction
            predicted_price = model.predict(feature_vector)[0]
            
            # Display results
            print(f"🔹 {prop['name']}:")
            print(f"   📏 Area: {prop['area']} sq ft")
            print(f"   🛏️  Bedrooms: {prop['bedrooms']}")
            print(f"   🚿 Bathrooms: {prop['bathrooms']}")
            print(f"   🏢 Stories: {prop['stories']}")
            print(f"   ❄️  AC: {'Yes' if prop['airconditioning'] else 'No'}")
            print(f"   📍 Preferred Area: {'Yes' if prop['prefarea'] else 'No'}")
            print(f"   🛋️  Furnishing: {'Furnished' if prop['furnishingstatus'] == 2 else 'Semi-Furnished' if prop['furnishingstatus'] == 1 else 'Unfurnished'}")
            print(f"   💰 Predicted Price: ₹{predicted_price:,.2f}")
            print()
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")

def model_performance_report(model_info):
    """
    Generate a performance report for the loaded model.
    
    Args:
        model_info (dict): Model information
    """
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 60)
    
    metrics = model_info.get('metrics', {})
    
    if metrics:
        print("📊 Performance Metrics:")
        for metric, value in metrics.items():
            if metric == 'r2_score':
                print(f"   R² Score: {value:.4f} ({value*100:.1f}% variance explained)")
            elif metric == 'rmse':
                print(f"   RMSE: ₹{value:,.2f}")
            elif metric == 'mae':
                print(f"   MAE: ₹{value:,.2f}")
            elif metric == 'mse':
                print(f"   MSE: {value:,.0f}")
    
    # Performance interpretation
    r2_score = metrics.get('r2_score', 0)
    if r2_score > 0.7:
        performance_level = "EXCELLENT"
    elif r2_score > 0.6:
        performance_level = "GOOD"
    elif r2_score > 0.5:
        performance_level = "MODERATE"
    else:
        performance_level = "NEEDS IMPROVEMENT"
    
    print(f"\n🎯 Performance Level: {performance_level}")
    
    if performance_level in ["MODERATE", "NEEDS IMPROVEMENT"]:
        print("💡 Suggestions for improvement:")
        print("   - Try feature engineering")
        print("   - Consider non-linear models")
        print("   - Collect more data")
        print("   - Handle outliers better")

def export_model_card(model_data, model_info):
    """
    Export a detailed model card.
    
    Args:
        model_data (dict): Model data
        model_info (dict): Model information
    """
    try:
        model_manager = ModelManager()
        
        output_path = os.path.join("models", "model_card.json")
        model_manager.export_model_card(
            model_name="housing_price_predictor",
            version=model_info.get('version', 'v1.0.0'),
            output_path=output_path
        )
        
        print(f"📄 Model card exported to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error exporting model card: {e}")

if __name__ == "__main__":
    # Load model
    model_data, model_info = load_and_predict()
    
    if model_data and model_info:
        # Make predictions
        make_predictions(model_data)
        
        # Show performance report
        model_performance_report(model_info)
        
        # Export model card
        export_model_card(model_data, model_info)
        
        print("\n" + "=" * 60)
        print("✅ SCRIPT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\n❌ Failed to load model. Please check if models are available.")
        