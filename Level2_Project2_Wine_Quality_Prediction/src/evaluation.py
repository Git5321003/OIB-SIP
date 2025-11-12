import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class ModelEvaluator:
    def __init__(self):
        self.figsize = (12, 8)
        
    def plot_model_comparison(self, results):
        """Create visual comparison of model performance"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Accuracy comparison
        bars = ax1.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        # CV scores with error bars
        x_pos = range(len(models))
        ax2.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
               color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
        ax2.set_title('Cross-Validation Scores')
        ax2.set_ylabel('CV Score (Mean ± 2 STD)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, results, y_true):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['predictions']
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {model_name}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, results, y_true):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            y_proba = result['probabilities']
            
            # For models that don't have predict_proba, use decision function
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()
    
    def generate_report(self, results, y_true):
        """Generate comprehensive evaluation report"""
        print("=== COMPREHENSIVE MODEL EVALUATION ===\n")
        
        # Create results table
        results_data = []
        for model_name, result in results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}",
                'Best Model': model_name == max(results, key=lambda x: results[x]['accuracy'])
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
        