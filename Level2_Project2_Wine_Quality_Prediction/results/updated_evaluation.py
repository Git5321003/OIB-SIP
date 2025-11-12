import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from .results_manager import ResultsManager, VisualizationGenerator

class ModelEvaluator:
    def __init__(self, save_results: bool = True):
        self.save_results = save_results
        if save_results:
            self.results_manager = ResultsManager()
            self.viz_generator = VisualizationGenerator(self.results_manager)
        
    def plot_model_comparison(self, results, y_true=None):
        """Create comprehensive model comparison visualizations."""
        if not self.save_results:
            print("Results saving is disabled. Enable with save_results=True")
            return
        
        # Create all comparison plots
        self.viz_generator.create_model_comparison_plot(results)
        self.viz_generator.create_performance_summary(results)
        
        if y_true is not None:
            self.viz_generator.create_roc_curves(results, y_true)
            self.viz_generator.create_precision_recall_curves(results, y_true)
    
    def plot_confusion_matrices(self, results, y_true):
        """Plot confusion matrices for all models."""
        if not self.save_results:
            # Create basic plot without saving
            n_models = len(results)
            fig, axes = plt.subplots(1, n_models, figsize=(15, 5))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, result) in enumerate(results.items()):
                y_pred = result['predictions']
                cm = confusion_matrix(y_true, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name}')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.show()
        else:
            self.viz_generator.create_confusion_matrices(results, y_true)
    
    def plot_roc_curves(self, results, y_true):
        """Plot ROC curves for all models."""
        if not self.save_results:
            # Create basic plot without saving
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for model_name, result in results.items():
                y_proba = result['probabilities']
                
                if y_proba is not None:
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, lw=2, 
                           label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            self.viz_generator.create_roc_curves(results, y_true)
    
    def plot_feature_importance(self, feature_importance, model_name="Random Forest"):
        """Plot feature importance."""
        if not self.save_results:
            # Create basic plot without saving
            importance_df = pd.DataFrame({
                'Feature': list(feature_importance.keys()),
                'Importance': list(feature_importance.values())
            }).sort_values('Importance', ascending=True)
            
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['Feature'], importance_df['Importance'])
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
        else:
            self.viz_generator.create_feature_importance_plot(feature_importance, model_name)
    
    def generate_report(self, results, y_true):
        """Generate comprehensive evaluation report."""
        print("=== COMPREHENSIVE MODEL EVALUATION ===\n")
        
        # Create results table
        results_data = []
        for model_name, result in results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                'Best Model': '★' if model_name == max(results, key=lambda x: results[x]['accuracy']) else ''
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # Best model details
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Accuracy: {best_result['accuracy']:.4f}")
        print(f"Best CV Score: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
        
        # Detailed classification report for best model
        print(f"\nDetailed Classification Report for {best_model_name}:")
        print(classification_report(y_true, best_result['predictions'], 
                                 target_names=['Bad Quality', 'Good Quality']))
        
        # Save results if enabled
        if self.save_results:
            # Save metrics
            metrics = {
                'best_model': best_model_name,
                'best_accuracy': float(best_result['accuracy']),
                'best_cv_score': float(best_result['cv_mean']),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'models_tested': list(results.keys())
            }
            
            self.results_manager.save_metrics(metrics, "evaluation_summary")
            self.results_manager.save_table(results_df, "model_performance")
            