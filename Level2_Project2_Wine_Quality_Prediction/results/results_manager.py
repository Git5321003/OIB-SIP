import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class ResultsManager:
    """
    Manages saving and organizing all results, plots, and metrics.
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        self.tables_dir = os.path.join(results_dir, "tables")
        self.ensure_directories_exist()
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def ensure_directories_exist(self):
        """Create all necessary directories."""
        directories = [self.results_dir, self.plots_dir, self.metrics_dir, self.tables_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_plot(self, fig, filename: str, dpi: int = 300, 
                  bbox_inches: str = 'tight', **kwargs):
        """
        Save matplotlib figure with consistent settings.
        
        Args:
            fig: matplotlib figure object
            filename: Name of the file (without extension)
            dpi: Resolution for saving
            bbox_inches: Bounding box setting
        """
        filepath = os.path.join(self.plots_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        plt.close(fig)  # Close figure to free memory
        print(f"Plot saved: {filepath}")
        return filepath
    
    def save_metrics(self, metrics_dict: dict, filename: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics_dict: Dictionary containing metrics
            filename: Name of the file (without extension)
        """
        filepath = os.path.join(self.metrics_dir, f"{filename}.json")
        
        # Convert numpy types to Python native types for JSON serialization
        serializable_dict = self._make_json_serializable(metrics_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_dict, f, indent=2)
        
        print(f"Metrics saved: {filepath}")
        return filepath
    
    def save_table(self, df: pd.DataFrame, filename: str):
        """
        Save DataFrame as CSV and HTML.
        
        Args:
            df: pandas DataFrame
            filename: Name of the file (without extension)
        """
        # Save as CSV
        csv_path = os.path.join(self.tables_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save as HTML with better formatting
        html_path = os.path.join(self.tables_dir, f"{filename}.html")
        styled_df = df.style.set_properties(**{
            'background-color': '#f8f9fa',
            'color': 'black',
            'border-color': 'white'
        })
        
        with open(html_path, 'w') as f:
            f.write(styled_df.to_html())
        
        print(f"Table saved: {csv_path} and {html_path}")
        return csv_path, html_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

class VisualizationGenerator:
    """
    Generates various plots and visualizations for model evaluation.
    """
    
    def __init__(self, results_manager: ResultsManager):
        self.rm = results_manager
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
    def create_model_comparison_plot(self, results: dict, figsize: tuple = (12, 8)):
        """
        Create comprehensive model comparison visualization.
        
        Args:
            results: Dictionary containing model results
            figsize: Figure size
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        models = list(results.keys())
        
        # 1. Accuracy Comparison
        accuracies = [results[model]['accuracy'] for model in models]
        bars1 = ax1.bar(models, accuracies, color=self.colors[:len(models)], alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cross-Validation Scores
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        x_pos = np.arange(len(models))
        
        bars2 = ax2.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                       color=self.colors[:len(models)], alpha=0.8)
        ax2.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CV Score (Mean ± STD)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylim(0, 1)
        
        # 3. Training Time Comparison (if available)
        if 'training_time' in next(iter(results.values())):
            training_times = [results[model].get('training_time', 0) for model in models]
            bars3 = ax3.bar(models, training_times, color=self.colors[:len(models)], alpha=0.8)
            ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Time (seconds)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
        else:
            # Feature importance for best model
            best_model = max(results, key=lambda x: results[x]['accuracy'])
            if hasattr(results[best_model]['model'], 'feature_importances_'):
                feature_importance = results[best_model]['model'].feature_importances_
                # For demonstration - you would need to pass feature names
                features = [f'Feature {i+1}' for i in range(len(feature_importance))]
                sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
                ax3.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                ax3.set_yticks(range(len(sorted_idx)))
                ax3.set_yticklabels([features[i] for i in sorted_idx])
                ax3.set_title(f'Feature Importance - {best_model}', fontsize=14, fontweight='bold')
        
        # 4. Model Performance Summary
        summary_data = {
            'Model': models,
            'Accuracy': accuracies,
            'CV Mean': cv_means,
            'CV Std': cv_stds
        }
        summary_df = pd.DataFrame(summary_data)
        
        ax4.axis('off')
        table = ax4.table(cellText=summary_df.round(4).values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        self.rm.save_plot(fig, "model_comparison")
        return fig
    
    def create_confusion_matrices(self, results: dict, y_true, figsize: tuple = (15, 5)):
        """
        Create confusion matrices for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels
            figsize: Figure size
        """
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['predictions']
            cm = confusion_matrix(y_true, y_pred)
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar=False, ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', 
                              fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
            # Add class labels
            classes = ['Bad', 'Good']
            axes[idx].set_xticklabels(classes)
            axes[idx].set_yticklabels(classes)
        
        plt.tight_layout()
        
        # Save the plot
        self.rm.save_plot(fig, "confusion_matrices")
        return fig
    
    def create_feature_importance_plot(self, feature_importance: dict, 
                                    model_name: str = "Random Forest",
                                    top_n: int = 10,
                                    figsize: tuple = (12, 8)):
        """
        Create feature importance visualization.
        
        Args:
            feature_importance: Dictionary of feature importances
            model_name: Name of the model
            top_n: Number of top features to show
            figsize: Figure size
        """
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True).tail(top_n)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Horizontal bar plot
        bars = ax1.barh(importance_df['Feature'], importance_df['Importance'],
                       color=self.colors[0], alpha=0.8)
        ax1.set_title(f'Top {top_n} Feature Importance\n{model_name}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance Score', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        # 2. Pie chart for top features
        top_features = importance_df.tail(6)  # Top 6 for pie chart
        ax2.pie(top_features['Importance'], labels=top_features['Feature'],
               autopct='%1.1f%%', startangle=90, colors=self.colors)
        ax2.set_title('Feature Importance Distribution\n(Top 6 Features)', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        self.rm.save_plot(fig, "feature_importance")
        
        # Save feature importance data
        self.rm.save_metrics(feature_importance, "feature_importance")
        self.rm.save_table(importance_df, "feature_importance_ranking")
        
        return fig, importance_df
    
    def create_roc_curves(self, results: dict, y_true, figsize: tuple = (10, 8)):
        """
        Create ROC curves for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        roc_data = {}
        
        for model_name, result in results.items():
            y_proba = result['probabilities']
            
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                roc_data[model_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', 
                   fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        self.rm.save_plot(fig, "roc_curves")
        self.rm.save_metrics(roc_data, "roc_curves_data")
        
        return fig
    
    def create_precision_recall_curves(self, results: dict, y_true, figsize: tuple = (10, 8)):
        """
        Create Precision-Recall curves for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pr_data = {}
        
        for model_name, result in results.items():
            y_proba = result['probabilities']
            
            if y_proba is not None:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)
                
                ax.plot(recall, precision, lw=2,
                       label=f'{model_name} (AUC = {pr_auc:.3f})')
                
                pr_data[model_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'auc': float(pr_auc)
                }
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        self.rm.save_plot(fig, "precision_recall_curves")
        self.rm.save_metrics(pr_data, "precision_recall_data")
        
        return fig
    
    def create_performance_summary(self, results: dict, figsize: tuple = (12, 6)):
        """
        Create a comprehensive performance summary table.
        
        Args:
            results: Dictionary containing model results
            figsize: Figure size
        """
        summary_data = []
        
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Mean': f"{result['cv_mean']:.4f}",
                'CV Std': f"±{result['cv_std']:.4f}",
                'Best Model': '★' if model_name == max(results, key=lambda x: results[x]['accuracy']) else ''
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create a figure for the table
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['#f0f0f0'] * len(summary_df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot and table
        self.rm.save_plot(fig, "performance_summary")
        self.rm.save_table(summary_df, "performance_summary")
        
        return fig, summary_df
    