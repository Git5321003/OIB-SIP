#!/usr/bin/env python3
"""
Wine Quality Prediction - Main Script

A comprehensive machine learning pipeline for predicting wine quality
based on chemical characteristics using multiple classifiers.

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
import os
import json
from datetime import datetime

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.model_manager import ModelManager
from src.model_predictor import ModelPredictor
from src.results_manager import ResultsManager
from src.visualization import WineQualityVisualizer

# Suppress warnings
warnings.filterwarnings('ignore')

class WineQualityPredictor:
    """
    Main class for the Wine Quality Prediction pipeline.
    """
    
    def __init__(self, data_path: str = 'data/WineQT.csv', 
                 save_models: bool = True, 
                 save_results: bool = True):
        """
        Initialize the Wine Quality Predictor.
        
        Args:
            data_path: Path to the wine dataset
            save_models: Whether to save trained models
            save_results: Whether to save results and plots
        """
        self.data_path = data_path
        self.save_models = save_models
        self.save_results = save_results
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_importance = {}
        
        # Initialize components
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer(save_models=save_models)
        self.evaluator = ModelEvaluator(save_results=save_results)
        self.visualizer = WineQualityVisualizer()
        
        if save_models:
            self.model_manager = ModelManager()
            self.model_predictor = ModelPredictor()
        
        if save_results:
            self.results_manager = ResultsManager()
        
        print("🚀 Wine Quality Predictor Initialized")
        print(f"📁 Data path: {data_path}")
        print(f"💾 Save models: {save_models}")
        print(f"📊 Save results: {save_results}")
    
    def load_and_explore_data(self):
        """
        Load the dataset and perform initial exploration.
        """
        print("\n" + "="*60)
        print("📊 STEP 1: LOADING AND EXPLORING DATA")
        print("="*60)
        
        try:
            # Load data
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📐 Dataset shape: {self.df.shape}")
            
            # Basic information
            print("\n📋 Dataset Info:")
            print(f"   - Columns: {list(self.df.columns)}")
            print(f"   - Missing values: {self.df.isnull().sum().sum()}")
            print(f"   - Duplicates: {self.df.duplicated().sum()}")
            
            # Quality distribution
            print(f"\n🍷 Quality Distribution:")
            quality_counts = self.df['quality'].value_counts().sort_index()
            for quality, count in quality_counts.items():
                print(f"   - Quality {quality}: {count} samples ({count/len(self.df)*100:.1f}%)")
            
            # Basic statistics
            print(f"\n📈 Basic Statistics:")
            print(f"   - Quality range: {self.df['quality'].min()} to {self.df['quality'].max()}")
            print(f"   - Average alcohol: {self.df['alcohol'].mean():.2f}%")
            print(f"   - Average pH: {self.df['pH'].mean():.2f}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive exploratory data analysis with visualizations.
        """
        print("\n" + "="*60)
        print("🔍 STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # 1. Quality Distribution
        print("📊 Creating quality distribution plots...")
        fig1 = self.visualizer.plot_quality_distribution(self.df)
        if self.save_results:
            self.results_manager.save_plot(fig1, "quality_distribution")
        plt.show()
        
        # 2. Correlation Heatmap
        print("📈 Creating correlation heatmap...")
        fig2 = self.visualizer.plot_correlation_heatmap(self.df)
        if self.save_results:
            self.results_manager.save_plot(fig2, "correlation_heatmap")
        plt.show()
        
        # 3. Feature Distributions
        print("📉 Creating feature distribution plots...")
        fig3 = self.visualizer.plot_feature_distributions(self.df)
        if self.save_results:
            self.results_manager.save_plot(fig3, "feature_distributions")
        plt.show()
        
        # 4. Quality vs Key Features
        print("🔗 Creating quality vs features plots...")
        key_features = ['alcohol', 'volatile acidity', 'citric acid', 'sulphates']
        fig4 = self.visualizer.plot_quality_vs_features(self.df, features=key_features)
        if self.save_results:
            self.results_manager.save_plot(fig4, "quality_vs_features")
        plt.show()
        
        # 5. Boxplots by Quality
        print("📦 Creating boxplots by quality...")
        fig5 = self.visualizer.plot_boxplots_by_quality(self.df)
        if self.save_results:
            self.results_manager.save_plot(fig5, "boxplots_by_quality")
        plt.show()
        
        print("✅ Exploratory Data Analysis completed!")
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning.
        """
        print("\n" + "="*60)
        print("⚙️ STEP 3: DATA PREPROCESSING")
        print("="*60)
        
        # Prepare features and target
        self.X = self.df.drop(['quality', 'Id'], axis=1)
        self.y = self.df['quality']
        
        # Convert to binary classification (Good: 6-10, Bad: 0-5)
        self.y_binary = (self.y > 5).astype(int)
        
        print(f"🎯 Target variable created:")
        print(f"   - Bad quality (0-5): {(self.y_binary == 0).sum()} samples")
        print(f"   - Good quality (6-10): {(self.y_binary == 1).sum()} samples")
        print(f"   - Good wine proportion: {self.y_binary.mean():.2%}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_binary, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_binary
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📊 Data splitting completed:")
        print(f"   - Training set: {self.X_train.shape[0]} samples")
        print(f"   - Test set: {self.X_test.shape[0]} samples")
        print(f"   - Number of features: {self.X_train.shape[1]}")
        
        # Save feature names for later use
        self.feature_names = self.X.columns.tolist()
        
        return self.X_train_scaled, self.X_test_scaled, self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models.
        """
        print("\n" + "="*60)
        print("🤖 STEP 4: MODEL TRAINING")
        print("="*60)
        
        # Initialize models
        self.model_trainer.initialize_models()
        
        print("🧠 Training the following models:")
        for model_name in self.model_trainer.models.keys():
            print(f"   - {model_name}")
        
        # Train models
        self.results = self.model_trainer.train_models(
            (self.X_train_scaled, self.X_train),
            (self.X_test_scaled, self.X_test),
            self.y_train,
            self.y_test,
            save_trained_models=self.save_models
        )
        
        # Extract feature importance from Random Forest
        if 'Random_Forest' in self.results:
            rf_model = self.results['Random_Forest']['model']
            if hasattr(rf_model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, rf_model.feature_importances_))
                
                # Save feature importance
                if self.save_models:
                    self.model_trainer.save_feature_importance(self.feature_names)
        
        print("✅ Model training completed!")
        
        return self.results
    
    def evaluate_models(self):
        """
        Evaluate and compare all trained models.
        """
        print("\n" + "="*60)
        print("📈 STEP 5: MODEL EVALUATION")
        print("="*60)
        
        # Generate comprehensive evaluation report
        self.evaluator.generate_report(self.results, self.y_test)
        
        # Create visualizations
        print("\n🎨 Creating evaluation visualizations...")
        
        # Model comparison
        self.evaluator.plot_model_comparison(self.results, self.y_test)
        
        # Confusion matrices
        self.evaluator.plot_confusion_matrices(self.results, self.y_test)
        
        # ROC curves
        self.evaluator.plot_roc_curves(self.results, self.y_test)
        
        # Feature importance
        if self.feature_importance:
            self.evaluator.plot_feature_importance(self.feature_importance)
        
        print("✅ Model evaluation completed!")
    
    def analyze_feature_importance(self):
        """
        Perform detailed feature importance analysis.
        """
        if not self.feature_importance:
            print("⚠️ No feature importance data available.")
            return
        
        print("\n" + "="*60)
        print("🔍 STEP 6: FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Create feature importance visualization
        fig = self.visualizer.plot_feature_importance(self.feature_importance)
        if self.save_results:
            self.results_manager.save_plot(fig, "feature_importance")
        plt.show()
        
        # Display top features
        importance_df = pd.DataFrame({
            'Feature': list(self.feature_importance.keys()),
            'Importance': list(self.feature_importance.values())
        }).sort_values('Importance', ascending=False)
        
        print("\n🏆 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"   {i:2d}. {row['Feature']:20} {row['Importance']:.4f}")
        
        # Save feature importance data
        if self.save_results:
            self.results_manager.save_metrics(
                dict(importance_df.head(10).values), 
                "top_features_importance"
            )
            self.results_manager.save_table(importance_df, "feature_importance_ranking")
        
        return importance_df
    
    def predict_new_wine(self, wine_features, feature_names=None):
        """
        Predict quality for a new wine sample.
        
        Args:
            wine_features: List of feature values for the new wine
            feature_names: Names of the features (optional)
        """
        print("\n" + "="*60)
        print("🍷 STEP 7: PREDICTING NEW WINE QUALITY")
        print("="*60)
        
        if not self.results:
            print("❌ No trained models available. Please train models first.")
            return
        
        # Get the best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"🎯 Using best model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        
        # Prepare features
        if feature_names is None:
            feature_names = self.feature_names
        
        if len(wine_features) != len(feature_names):
            print(f"❌ Expected {len(feature_names)} features, got {len(wine_features)}")
            return
        
        # Create feature dictionary for display
        feature_dict = dict(zip(feature_names, wine_features))
        
        print("\n📋 Wine Features:")
        for feature, value in feature_dict.items():
            print(f"   - {feature:20} {value}")
        
        # Make prediction
        try:
            if best_model_name in ['SGD_Classifier', 'SVC']:
                wine_features_scaled = self.scaler.transform([wine_features])
                prediction = best_model.predict(wine_features_scaled)[0]
                if hasattr(best_model, 'predict_proba'):
                    probability = best_model.predict_proba(wine_features_scaled)[0, 1]
                else:
                    probability = best_model.decision_function(wine_features_scaled)[0]
            else:
                prediction = best_model.predict([wine_features])[0]
                probability = best_model.predict_proba([wine_features])[0, 1]
            
            quality = "GOOD 🎉" if prediction == 1 else "BAD 👎"
            confidence = probability if prediction == 1 else (1 - probability)
            
            print(f"\n🔮 Prediction Results:")
            print(f"   - Quality: {quality}")
            print(f"   - Confidence: {confidence:.1%}")
            print(f"   - Model: {best_model_name}")
            
            return prediction, probability
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None, None
    
    def demonstrate_saved_models(self):
        """
        Demonstrate loading and using saved models.
        """
        if not self.save_models:
            print("⚠️ Model saving was disabled. Cannot demonstrate saved models.")
            return
        
        print("\n" + "="*60)
        print("💾 DEMONSTRATING SAVED MODELS")
        print("="*60)
        
        try:
            # List available models
            available_models = self.model_manager.list_models()
            
            if not available_models:
                print("❌ No saved models found.")
                return
            
            print("📁 Available saved models:")
            for model_name, versions in available_models.items():
                print(f"   - {model_name}:")
                for version in versions[:3]:  # Show only first 3 versions
                    print(f"     * {version['version']} ({version['saved_at'][:10]})")
            
            # Load the best model (Random Forest)
            if 'Random_Forest' in available_models:
                print(f"\n🔄 Loading Random Forest model...")
                model, metadata = self.model_predictor.load_model_for_prediction('Random_Forest')
                
                # Make prediction with loaded model
                example_wine = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
                result = self.model_predictor.predict_quality(example_wine)
                
                print(f"\n🎯 Prediction with loaded model:")
                print(f"   - Quality: {result['quality']}")
                print(f"   - Confidence: {result['confidence']:.1%}")
                
        except Exception as e:
            print(f"❌ Error demonstrating saved models: {e}")
    
    def create_interactive_dashboard(self):
        """
        Create interactive visualizations using Plotly.
        """
        print("\n" + "="*60)
        print("📊 CREATING INTERACTIVE DASHBOARD")
        print("="*60)
        
        try:
            if self.feature_importance:
                fig_heatmap, fig_importance, fig_comparison = \
                    self.visualizer.create_interactive_plots(self.df, self.results, self.feature_importance)
                
                print("✅ Interactive plots created!")
                print("   - Correlation Heatmap")
                print("   - Feature Importance")
                print("   - Model Comparison")
                print("\n💡 Note: Interactive plots will open in your web browser.")
                
                # Uncomment the lines below to display interactive plots
                # fig_heatmap.show()
                # fig_importance.show()
                # fig_comparison.show()
                
            else:
                print("⚠️ No feature importance data available for interactive plots.")
                
        except ImportError:
            print("❌ Plotly not installed. Install with: pip install plotly")
        except Exception as e:
            print(f"❌ Error creating interactive plots: {e}")
    
    def run_complete_pipeline(self):
        """
        Run the complete wine quality prediction pipeline.
        """
        print("🍷 WINE QUALITY PREDICTION PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        print(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Exploratory Data Analysis
            self.exploratory_data_analysis()
            
            # Step 3: Data Preprocessing
            self.preprocess_data()
            
            # Step 4: Model Training
            self.train_models()
            
            # Step 5: Model Evaluation
            self.evaluate_models()
            
            # Step 6: Feature Importance Analysis
            self.analyze_feature_importance()
            
            # Step 7: Example Prediction
            example_wine = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
            self.predict_new_wine(example_wine)
            
            # Step 8: Demonstrate saved models
            self.demonstrate_saved_models()
            
            # Step 9: Interactive dashboard
            self.create_interactive_dashboard()
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print("\n" + "="*60)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"⏰ Execution time: {execution_time:.2f} seconds")
            print(f"📊 Models trained: {len(self.results)}")
            print(f"🎯 Best accuracy: {max([r['accuracy'] for r in self.results.values()]):.3f}")
            
            if self.save_models:
                print(f"💾 Models saved to: models/")
            if self.save_results:
                print(f"📈 Results saved to: results/")
            
            print("\n🎉 Thank you for using the Wine Quality Predictor!")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            raise

def main():
    """
    Main function to run the Wine Quality Prediction pipeline.
    """
    # Configuration
    DATA_PATH = 'data/WineQT.csv'
    SAVE_MODELS = True
    SAVE_RESULTS = True
    
    # Initialize the predictor
    predictor = WineQualityPredictor(
        data_path=DATA_PATH,
        save_models=SAVE_MODELS,
        save_results=SAVE_RESULTS
    )
    
    # Run the complete pipeline
    predictor.run_complete_pipeline()

if __name__ == "__main__":
    main()
    