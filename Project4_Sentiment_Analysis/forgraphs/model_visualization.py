import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

class ModelPerformanceVisualizer:
    def __init__(self):
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_confusion_matrix_comparison(self, models_results, X_test, y_test):
        """Plot confusion matrices for multiple models"""
        n_models = len(models_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        sentiment_names = ['Negative', 'Neutral', 'Positive']
        
        for idx, (model_name, model_info) in enumerate(models_results.items()):
            if idx >= 4:  # Limit to 4 models
                break
                
            model = model_info['pipeline']
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=sentiment_names,
                       yticklabels=sentiment_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'Confusion Matrix - {model_name}', 
                              fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        
        # Hide empty subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison_bar_chart(self, results_dict):
        """Create bar chart comparing model performance"""
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['accuracy'] for model in models]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create gradient colors based on accuracy
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        
        bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{accuracy:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=10)
        
        ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, models_results, X_train, X_test, y_train, y_test):
        """Plot learning curves for models"""
        from sklearn.model_selection import learning_curve
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (model_name, model_info) in enumerate(models_results.items()):
            if idx >= 4:
                break
                
            model = model_info['pipeline']
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy'
            )
            
            # Calculate mean and standard deviation
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            axes[idx].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
            axes[idx].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
            axes[idx].plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            axes[idx].plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            
            axes[idx].set_title(f'Learning Curve - {model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel("Training examples")
            axes[idx].set_ylabel("Accuracy")
            axes[idx].legend(loc="best")
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_comparison(self, models_results, X_test, y_test):
        """Plot precision and recall comparison for all models"""
        models = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for model_name, model_info in models_results.items():
            model = model_info['pipeline']
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate macro averages
            precision_scores.append(report['macro avg']['precision'])
            recall_scores.append(report['macro avg']['recall'])
            f1_scores.append(report['macro avg']['f1-score'])
            models.append(model_name)
        
        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width, precision_scores, width, label='Precision', 
                      alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recall_scores, width, label='Recall', 
                      alpha=0.8, color='lightgreen')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', 
                      alpha=0.8, color='salmon')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_performance_radar_chart(self, models_results, X_test, y_test):
        """Create radar chart comparing model performance across metrics"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Speed']
        models_data = {}
        
        for model_name, model_info in models_results.items():
            model = model_info['pipeline']
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = report['macro avg']['precision']
            recall = report['macro avg']['recall']
            f1 = report['macro avg']['f1-score']
            
            # Simulate speed (inverse of model complexity)
            speed = 1.0 / (len(str(model.named_steps['classifier'].get_params())) * 0.01)
            
            models_data[model_name] = [accuracy, precision, recall, f1, speed]
        
        # Normalize data for radar chart
        normalized_data = {}
        for model_name, metrics_values in models_data.items():
            normalized_data[model_name] = metrics_values
        
        # Create radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for model_name, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([])
        ax.set_title('Model Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()