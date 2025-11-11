#!/usr/bin/env python3
"""
Housing Price Prediction - Main Application
------------------------------------------
A comprehensive machine learning application for predicting house prices
using linear regression.

Author: Real Estate Analytics Team
Version: 1.0.0
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add src and models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from src.data_preprocessing import DataPreprocessor, create_sample_property, create_multiple_sample_properties
from src.model import HousingPriceModel
from src.visualization import HousingVisualizer, create_comprehensive_eda_plots
from models.model_manager import ModelManager, create_model_package


class HousingPriceApp:
    """
    Main application class for housing price prediction.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.preprocessor = None
        self.model = None
        self.visualizer = HousingVisualizer()
        self.model_manager = ModelManager()
        self.data_loaded = False
        self.model_trained = False
        
    def print_header(self):
        """Print application header."""
        print("=" * 70)
        print("🏠 HOUSING PRICE PREDICTION SYSTEM")
        print("=" * 70)
        print("📊 Multiple Linear Regression for Real Estate Pricing")
        print("📍 Delhi Region Property Analysis")
        print("🤖 Machine Learning Powered")
        print("=" * 70)
    
    def load_data(self, data_path="data/Housing.csv"):
        """
        Load and preprocess housing data.
        
        Args:
            data_path (str): Path to the data file
        """
        print("\n📁 LOADING DATA...")
        print("-" * 40)
        
        try:
            # Check if data file exists
            if not os.path.exists(data_path):
                print(f"❌ Data file not found: {data_path}")
                print("💡 Please ensure the data file exists in the data/ directory")
                return False
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor()
            
            # Run full preprocessing pipeline
            X_train_scaled, X_test_scaled, y_train, y_test, feature_names = \
                self.preprocessor.full_preprocessing_pipeline(data_path)
            
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
            self.feature_names = feature_names
            
            # Data inspection
            inspection = self.preprocessor.inspect_data()
            print(f"✅ Data loaded successfully!")
            print(f"   📊 Dataset shape: {inspection['shape']}")
            print(f"   🔢 Numerical features: {len(inspection['numerical_columns'])}")
            print(f"   🏷️  Categorical features: {len(inspection['categorical_columns'])}")
            print(f"   🎯 Target variable: price")
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        if not self.data_loaded:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("-" * 40)
        
        try:
            # Create comprehensive EDA plots
            numerical_features = self.preprocessor.numerical_cols
            categorical_features = self.preprocessor.categorical_cols
            
            print("📈 Generating visualizations...")
            
            # Price distribution
            self.visualizer.plot_price_distribution(self.preprocessor.df)
            
            # Numerical features distribution
            self.visualizer.plot_numerical_features_distribution(
                self.preprocessor.df, numerical_features
            )
            
            # Correlation heatmap
            numerical_cols = numerical_features + ['price']
            self.visualizer.plot_correlation_heatmap(
                self.preprocessor.df, numerical_cols
            )
            
            # Top features vs price
            correlation_with_price = self.preprocessor.df[numerical_cols].corr()['price'].sort_values(ascending=False)
            top_features = correlation_with_price.index[1:4]  # Top 3 excluding price
            
            for feature in top_features:
                self.visualizer.plot_feature_vs_price(self.preprocessor.df, feature)
            
            print("✅ EDA completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during EDA: {e}")
    
    def train_model(self):
        """Train the housing price prediction model."""
        if not self.data_loaded:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n🤖 TRAINING MODEL...")
        print("-" * 40)
        
        try:
            # Initialize model
            self.model = HousingPriceModel()
            
            # Train model
            self.model.train(self.X_train, self.y_train, self.feature_names)
            
            # Evaluate model
            metrics_df = self.model.evaluate(self.X_train, self.y_train, self.X_test, self.y_test)
            
            # Display results
            test_metrics = metrics_df[metrics_df['Set'] == 'Testing'].iloc[0]
            print("✅ Model trained successfully!")
            print(f"   📊 R² Score: {test_metrics['R²']:.4f}")
            print(f"   💰 RMSE: ₹{test_metrics['RMSE']:,.2f}")
            print(f"   📏 MAE: ₹{test_metrics['MAE']:,.2f}")
            
            # Feature importance
            feature_importance = self.model.get_feature_importance(top_n=5)
            print(f"\n🎯 Top 5 Most Important Features:")
            for _, row in feature_importance.iterrows():
                impact = "📈 Increases" if row['Coefficient'] > 0 else "📉 Decreases"
                print(f"   {row['Feature']}: {impact} price")
            
            self.model_trained = True
            
            # Model diagnostics
            self._model_diagnostics()
            
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def _model_diagnostics(self):
        """Perform model diagnostics and visualization."""
        if not self.model_trained:
            return
        
        print("\n🔧 MODEL DIAGNOSTICS")
        print("-" * 40)
        
        try:
            # Make predictions for diagnostics
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            # Calculate residuals
            train_residuals = self.y_train - y_train_pred
            test_residuals = self.y_test - y_test_pred
            
            # Residual analysis plots
            self.visualizer.plot_residual_analysis(
                self.y_test, y_test_pred, test_residuals
            )
            
            # Feature importance plot
            feature_importance = self.model.get_feature_importance()
            self.visualizer.plot_feature_importance(feature_importance)
            
            # Actual vs Predicted plot
            self.visualizer.plot_actual_vs_predicted(self.y_test, y_test_pred)
            
            print("✅ Model diagnostics completed!")
            
        except Exception as e:
            print(f"❌ Error in model diagnostics: {e}")
    
    def save_model(self, model_name="housing_price_predictor", version="v1.0.0"):
        """Save the trained model."""
        if not self.model_trained:
            print("❌ No trained model. Please train model first.")
            return
        
        print("\n💾 SAVING MODEL...")
        print("-" * 40)
        
        try:
            # Get model metrics
            test_metrics = self.model.test_metrics
            
            # Create model package
            model_package = create_model_package(
                model=self.model.model,
                feature_names=self.feature_names,
                train_metrics=self.model.train_metrics,
                test_metrics=self.model.test_metrics,
                feature_importance=self.model.feature_importance,
                scaler=self.preprocessor.scaler
            )
            
            # Save model with metadata
            model_path = self.model_manager.save_model(
                model_data=model_package,
                model_name=model_name,
                version=version,
                description="Linear Regression model for housing price prediction in Delhi region",
                metrics={
                    'r2_score': float(test_metrics['R²']),
                    'rmse': float(test_metrics['RMSE']),
                    'mae': float(test_metrics['MAE']),
                    'mse': float(test_metrics['MSE'])
                },
                tags=['linear_regression', 'housing', 'delhi', 'price_prediction']
            )
            
            print(f"✅ Model saved successfully!")
            print(f"   📁 Location: {model_path}")
            print(f"   🏷️  Name: {model_name}")
            print(f"   🔢 Version: {version}")
            
            # List all saved models
            self._list_saved_models()
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def _list_saved_models(self):
        """List all saved models."""
        try:
            models_df = self.model_manager.list_models()
            if not models_df.empty:
                print(f"\n📊 SAVED MODELS:")
                print("-" * 30)
                for _, row in models_df.iterrows():
                    print(f"   🏷️  {row['model_name']} {row['version']}")
                    print(f"   📝 {row['description']}")
                    print(f"   📅 {row['saved_at']}")
                    print()
            else:
                print("   No models saved yet.")
                
        except Exception as e:
            print(f"❌ Error listing models: {e}")
    
    def load_saved_model(self, model_name="housing_price_predictor", version="latest"):
        """Load a saved model."""
        print("\n📂 LOADING SAVED MODEL...")
        print("-" * 40)
        
        try:
            model_data, model_info = self.model_manager.get_latest_model(model_name)
            
            print(f"✅ Model loaded successfully!")
            print(f"   🏷️  Name: {model_name}")
            print(f"   🔢 Version: {model_info['version']}")
            print(f"   📝 Description: {model_info['description']}")
            print(f"   📅 Saved: {model_info['saved_at']}")
            
            # Update application state
            self.model = HousingPriceModel()
            self.model.model = model_data['model']
            self.model.feature_names = model_data['feature_names']
            self.model.model_trained = model_data['model_trained']
            self.model.feature_importance = model_data['feature_importance']
            self.model.train_metrics = model_data['train_metrics']
            self.model.test_metrics = model_data['test_metrics']
            
            self.model_trained = True
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def make_predictions(self):
        """Make predictions using the trained model."""
        if not self.model_trained:
            print("❌ No trained model. Please train or load a model first.")
            return
        
        print("\n🎯 MAKING PREDICTIONS")
        print("-" * 40)
        
        try:
            # Create sample properties
            sample_properties = create_multiple_sample_properties()
            
            print("🏠 PREDICTED PRICES FOR SAMPLE PROPERTIES:\n")
            
            for prop in sample_properties:
                # Prepare features for prediction
                features = {k: v for k, v in prop.items() if k != 'name'}
                
                # Make prediction
                predicted_price = self.model.predict_single(
                    features, 
                    self.preprocessor.scaler if hasattr(self, 'preprocessor') else None
                )
                
                # Display results
                print(f"🔹 {prop['name'].upper()}:")
                print(f"   📏 Area: {prop['area']} sq ft")
                print(f"   🛏️  Bedrooms: {prop['bedrooms']}")
                print(f"   🚿 Bathrooms: {prop['bathrooms']}")
                print(f"   🏢 Stories: {prop['stories']}")
                print(f"   ❄️  AC: {'Yes' if prop['airconditioning'] else 'No'}")
                print(f"   📍 Preferred Area: {'Yes' if prop['prefarea'] else 'No'}")
                print(f"   🛋️  Furnishing: {'Furnished' if prop['furnishingstatus'] == 2 else 'Semi-Furnished' if prop['furnishingstatus'] == 1 else 'Unfurnished'}")
                print(f"   🅿️  Parking: {prop['parking']} spaces")
                print(f"   💰 Predicted Price: ₹{predicted_price:,.2f}")
                print("   " + "─" * 40)
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
    
    def predict_custom_property(self):
        """Predict price for a custom property."""
        if not self.model_trained:
            print("❌ No trained model. Please train or load a model first.")
            return
        
        print("\n🏡 CUSTOM PROPERTY PREDICTION")
        print("-" * 40)
        
        try:
            # Get feature information
            feature_names = self.model.feature_names
            
            print("📝 Enter property details:")
            print("   (Enter values for the following features)\n")
            
            custom_property = {}
            
            for feature in feature_names:
                if feature == 'area':
                    value = input(f"   📏 Enter area (sq ft): ")
                elif feature == 'bedrooms':
                    value = input(f"   🛏️  Enter number of bedrooms: ")
                elif feature == 'bathrooms':
                    value = input(f"   🚿 Enter number of bathrooms: ")
                elif feature == 'stories':
                    value = input(f"   🏢 Enter number of stories: ")
                elif feature == 'mainroad':
                    value = input(f"   🛣️  On main road? (1 for yes, 0 for no): ")
                elif feature == 'guestroom':
                    value = input(f"   🛌 Has guest room? (1 for yes, 0 for no): ")
                elif feature == 'basement':
                    value = input(f"   🏠 Has basement? (1 for yes, 0 for no): ")
                elif feature == 'hotwaterheating':
                    value = input(f"   🔥 Hot water heating? (1 for yes, 0 for no): ")
                elif feature == 'airconditioning':
                    value = input(f"   ❄️  Air conditioning? (1 for yes, 0 for no): ")
                elif feature == 'parking':
                    value = input(f"   🅿️  Number of parking spaces: ")
                elif feature == 'prefarea':
                    value = input(f"   📍 In preferred area? (1 for yes, 0 for no): ")
                elif feature == 'furnishingstatus':
                    value = input(f"   🛋️  Furnishing (0=unfurnished, 1=semi-furnished, 2=furnished): ")
                else:
                    value = input(f"   {feature}: ")
                
                try:
                    custom_property[feature] = float(value)
                except ValueError:
                    print(f"   ❌ Invalid input for {feature}. Using default value 0.")
                    custom_property[feature] = 0.0
            
            # Make prediction
            predicted_price = self.model.predict_single(
                custom_property, 
                self.preprocessor.scaler if hasattr(self, 'preprocessor') else None
            )
            
            print(f"\n🎯 PREDICTION RESULT:")
            print("   " + "=" * 30)
            print(f"   💰 Estimated Price: ₹{predicted_price:,.2f}")
            print("   " + "=" * 30)
            
        except Exception as e:
            print(f"❌ Error in custom prediction: {e}")
    
    def business_insights(self):
        """Generate business insights from the model."""
        if not self.model_trained:
            print("❌ No trained model. Please train or load a model first.")
            return
        
        print("\n💡 BUSINESS INSIGHTS")
        print("-" * 40)
        
        try:
            # Get feature importance
            feature_importance = self.model.get_feature_importance(top_n=8)
            
            print("🎯 TOP FACTORS AFFECTING HOUSE PRICES:")
            print("   " + "─" * 35)
            
            for idx, (_, row) in enumerate(feature_importance.iterrows(), 1):
                impact = "INCREASES" if row['Coefficient'] > 0 else "DECREASES"
                print(f"   {idx}. {row['Feature'].upper()}: {impact} price")
            
            print(f"\n📊 MODEL PERFORMANCE:")
            print("   " + "─" * 20)
            test_metrics = self.model.test_metrics
            print(f"   R² Score: {test_metrics['R²']:.4f} ({test_metrics['R²']*100:.1f}% variance explained)")
            print(f"   Average Error: ±₹{test_metrics['RMSE']:,.2f}")
            
            print(f"\n💼 RECOMMENDATIONS:")
            print("   " + "─" * 15)
            print("   ✅ Focus on properties with larger areas")
            print("   ✅ Prioritize air conditioning installation")
            print("   ✅ Target properties in preferred areas")
            print("   ✅ Consider adding more bathrooms")
            print("   ✅ Furnished properties yield better returns")
            print("   ✅ Parking spaces add significant value")
            
        except Exception as e:
            print(f"❌ Error generating insights: {e}")
    
    def run_full_pipeline(self, data_path="data/Housing.csv"):
        """Run the complete pipeline from data loading to predictions."""
        print("\n🚀 RUNNING COMPLETE PIPELINE")
        print("=" * 50)
        
        steps = [
            ("Loading Data", self.load_data, [data_path]),
            ("Exploratory Analysis", self.explore_data, []),
            ("Training Model", self.train_model, []),
            ("Saving Model", self.save_model, []),
            ("Making Predictions", self.make_predictions, []),
            ("Business Insights", self.business_insights, [])
        ]
        
        for step_name, step_func, step_args in steps:
            print(f"\n📋 STEP: {step_name}")
            print("-" * 30)
            try:
                if step_args:
                    step_func(*step_args)
                else:
                    step_func()
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                continue


def main():
    """Main function to run the housing price prediction application."""
    parser = argparse.ArgumentParser(description='Housing Price Prediction System')
    parser.add_argument('--data', '-d', default='data/Housing.csv', 
                       help='Path to housing data CSV file')
    parser.add_argument('--pipeline', '-p', action='store_true',
                       help='Run complete pipeline automatically')
    parser.add_argument('--predict', '-r', action='store_true',
                       help='Make predictions using saved model')
    parser.add_argument('--custom', '-c', action='store_true',
                       help='Predict price for custom property')
    parser.add_argument('--insights', '-i', action='store_true',
                       help='Show business insights')
    
    args = parser.parse_args()
    
    # Initialize application
    app = HousingPriceApp()
    app.print_header()
    
    if args.pipeline:
        # Run complete pipeline
        app.run_full_pipeline(args.data)
    
    elif args.predict:
        # Load saved model and make predictions
        if app.load_saved_model():
            app.make_predictions()
    
    elif args.custom:
        # Predict custom property
        if app.load_saved_model():
            app.predict_custom_property()
    
    elif args.insights:
        # Show business insights
        if app.load_saved_model():
            app.business_insights()
    
    else:
        # Interactive mode
        app.interactive_mode()


    def interactive_mode(self):
        """Run application in interactive mode."""
        while True:
            print(f"\n🎮 INTERACTIVE MENU")
            print("-" * 30)
            print("1. 📁 Load Data")
            print("2. 🔍 Explore Data")
            print("3. 🤖 Train Model")
            print("4. 💾 Save Model")
            print("5. 📂 Load Saved Model")
            print("6. 🎯 Make Predictions (Sample)")
            print("7. 🏡 Custom Prediction")
            print("8. 💡 Business Insights")
            print("9. 🚀 Run Complete Pipeline")
            print("0. 🚪 Exit")
            print("-" * 30)
            
            choice = input("Enter your choice (0-9): ").strip()
            
            if choice == '1':
                data_path = input("Enter data path [data/Housing.csv]: ").strip()
                if not data_path:
                    data_path = "data/Housing.csv"
                self.load_data(data_path)
            
            elif choice == '2':
                self.explore_data()
            
            elif choice == '3':
                self.train_model()
            
            elif choice == '4':
                model_name = input("Enter model name [housing_price_predictor]: ").strip()
                version = input("Enter version [v1.0.0]: ").strip()
                self.save_model(
                    model_name or "housing_price_predictor",
                    version or "v1.0.0"
                )
            
            elif choice == '5':
                model_name = input("Enter model name [housing_price_predictor]: ").strip()
                version = input("Enter version [latest]: ").strip()
                self.load_saved_model(
                    model_name or "housing_price_predictor",
                    version or "latest"
                )
            
            elif choice == '6':
                self.make_predictions()
            
            elif choice == '7':
                self.predict_custom_property()
            
            elif choice == '8':
                self.business_insights()
            
            elif choice == '9':
                data_path = input("Enter data path [data/Housing.csv]: ").strip()
                self.run_full_pipeline(data_path or "data/Housing.csv")
            
            elif choice == '0':
                print("\n👋 Thank you for using Housing Price Prediction System!")
                print("🎯 Happy analyzing!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
    