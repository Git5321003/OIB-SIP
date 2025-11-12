from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from .model_manager import ModelManager
import numpy as np

class ModelTrainer:
    def __init__(self, save_models: bool = True):
        self.models = {}
        self.results = {}
        self.model_manager = ModelManager() if save_models else None
        
    def initialize_models(self):
        """Initialize the three classifier models"""
        self.models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10
            ),
            'SGD_Classifier': SGDClassifier(
                random_state=42,
                max_iter=1000,
                tol=1e-3
            ),
            'SVC': SVC(
                random_state=42,
                kernel='rbf',
                probability=True
            )
        }
    
    def train_models(self, X_train, X_test, y_train, y_test, use_scaled=True, 
                    save_trained_models: bool = True):
        """Train all models and evaluate performance"""
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Choose appropriate data (scaled or unscaled)
            if use_scaled and name in ['SGD_Classifier', 'SVC']:
                X_train_used = X_train[0] if isinstance(X_train, tuple) else X_train
                X_test_used = X_test[0] if isinstance(X_test, tuple) else X_test
            else:
                X_train_used = X_train[1] if isinstance(X_train, tuple) else X_train
                X_test_used = X_test[1] if isinstance(X_test, tuple) else X_test
            
            # Train model
            model.fit(X_train_used, y_train)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict(X_test_used)
                y_proba = model.predict_proba(X_test_used)[:, 1]
            else:
                y_pred = model.predict(X_test_used)
                y_proba = model.decision_function(X_test_used)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_used, y_train, cv=5, scoring='accuracy')
            
            # Prepare metadata
            metadata = {
                'accuracy': float(accuracy),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'feature_count': X_train_used.shape[1],
                'training_samples': X_train_used.shape[0]
            }
            
            # Save model if requested
            if save_trained_models and self.model_manager:
                self.model_manager.save_model(model, name, metadata=metadata)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_proba,
                'metadata': metadata
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save training results
        if save_trained_models and self.model_manager:
            self.model_manager.save_training_results(self.results)
        
        return self.results
    
    def get_best_model(self):
        """Return the best performing model"""
        if not self.results:
            return None
            
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        return best_model_name, self.results[best_model_name]
    
    def save_feature_importance(self, feature_names):
        """Save feature importance for tree-based models"""
        if 'Random_Forest' in self.models and self.model_manager:
            rf_model = self.models['Random_Forest']
            if hasattr(rf_model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, rf_model.feature_importances_))
                self.model_manager.save_feature_importance(importance_dict, 'Random_Forest')
                