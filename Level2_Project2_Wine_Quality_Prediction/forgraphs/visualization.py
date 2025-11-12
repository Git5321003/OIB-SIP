import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class WineQualityVisualizer:
    """
    Comprehensive visualization class for Wine Quality Prediction project.
    """
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        self.set_plot_style()
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6A8EAE']
        
    def set_plot_style(self):
        """Set consistent plotting style."""
        if self.style == "seaborn":
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
        elif self.style == "ggplot":
            plt.style.use('ggplot')
        else:
            plt.style.use('default')
            
        # Set consistent font sizes
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
    
    def plot_quality_distribution(self, df, figsize=(12, 5)):
        """
        Plot wine quality distribution.
        
        Args:
            df: DataFrame with wine data
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(df['quality'], bins=10, edgecolor='black', alpha=0.7, color=self.colors[0])
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Wine Quality Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df['quality'], patch_artist=True,
                   boxprops=dict(facecolor=self.colors[1], alpha=0.7))
        ax2.set_ylabel('Quality Score')
        ax2.set_title('Wine Quality Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, df, figsize=(14, 10)):
        """
        Plot correlation heatmap of all features.
        
        Args:
            df: DataFrame with wine data
            figsize: Figure size
        """
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_distributions(self, df, features=None, figsize=(20, 15)):
        """
        Plot distribution of all features.
        
        Args:
            df: DataFrame with wine data
            features: List of features to plot (default: all except quality and Id)
            figsize: Figure size
        """
        if features is None:
            features = [col for col in df.columns if col not in ['quality', 'Id']]
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(df[feature], bins=30, alpha=0.7, 
                           color=self.colors[i % len(self.colors)], edgecolor='black')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].grid(True, alpha=0.3)
                
                # Add mean line
                mean_val = df[feature].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_quality_vs_features(self, df, features=None, figsize=(20, 15)):
        """
        Plot relationship between quality and other features.
        
        Args:
            df: DataFrame with wine data
            features: List of features to plot against quality
            figsize: Figure size
        """
        if features is None:
            features = ['alcohol', 'volatile acidity', 'citric acid', 'sulphates']
        
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i < len(axes):
                # Scatter plot
                scatter = axes[i].scatter(df[feature], df['quality'], 
                                        c=df['quality'], cmap='viridis', 
                                        alpha=0.6, s=50)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Quality')
                axes[i].set_title(f'Quality vs {feature}')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(df[feature], df['quality'], 1)
                p = np.poly1d(z)
                axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[i], label='Quality')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_boxplots_by_quality(self, df, features=None, figsize=(20, 15)):
        """
        Plot boxplots of features grouped by quality.
        
        Args:
            df: DataFrame with wine data
            features: List of features to plot
            figsize: Figure size
        """
        if features is None:
            features = ['alcohol', 'volatile acidity', 'citric acid', 
                       'sulphates', 'total sulfur dioxide', 'density']
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if i < len(axes):
                # Create boxplot
                df.boxplot(column=feature, by='quality', ax=axes[i], 
                          patch_artist=True,
                          boxprops=dict(facecolor=self.colors[i % len(self.colors)], alpha=0.7))
                axes[i].set_title(f'{feature} by Quality')
                axes[i].set_xlabel('Quality Score')
                axes[i].set_ylabel(feature)
                
                # Remove automatic title
                axes[i].get_figure().suptitle('')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions by Wine Quality', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self, results, figsize=(15, 10)):
        """
        Create comprehensive model comparison visualization.
        
        Args:
            results: Dictionary containing model results
            figsize: Figure size
        """
        models = list(results.keys())
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Accuracy bars
        ax2 = fig.add_subplot(gs[0, 1])  # CV scores
        ax3 = fig.add_subplot(gs[1, 0])  # Performance metrics
        ax4 = fig.add_subplot(gs[1, 1])  # Summary table
        
        # 1. Accuracy Comparison
        accuracies = [results[model]['accuracy'] for model in models]
        bars1 = ax1.bar(models, accuracies, color=self.colors[:len(models)], alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, accuracy in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cross-Validation Scores
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        
        bars2 = ax2.bar(range(len(models)), cv_means, yerr=cv_stds, capsize=5,
                       color=self.colors[:len(models)], alpha=0.8)
        ax2.set_title('Cross-Validation Performance', fontweight='bold')
        ax2.set_ylabel('CV Score (Mean ± STD)')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylim(0, 1)
        
        # 3. Performance Metrics Radar Chart (simplified)
        metrics = ['Accuracy', 'CV Mean', '1 - CV Std']
        metric_values = []
        
        for model in models:
            model_metrics = [
                results[model]['accuracy'],
                results[model]['cv_mean'],
                1 - results[model]['cv_std']  # Inverse of std for better visualization
            ]
            metric_values.append(model_metrics)
        
        # Plot lines for each model
        for i, model in enumerate(models):
            ax3.plot(metrics, metric_values[i], 'o-', label=model, 
                   color=self.colors[i], linewidth=2, markersize=8)
        
        ax3.set_title('Model Performance Metrics', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Performance Summary Table
        summary_data = []
        for model_name in models:
            summary_data.append([
                model_name,
                f"{results[model_name]['accuracy']:.4f}",
                f"{results[model_name]['cv_mean']:.4f}",
                f"{results[model_name]['cv_std']:.4f}"
            ])
        
        ax4.axis('off')
        table = ax4.table(cellText=summary_data,
                        colLabels=['Model', 'Accuracy', 'CV Mean', 'CV Std'],
                        cellLoc='center',
                        loc='center',
                        colColours=['#f0f0f0'] * 4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(self, results, y_true, figsize=(18, 6)):
        """
        Plot confusion matrices for all models.
        
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
            
            # Plot confusion matrix
            im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', 
                              fontweight='bold')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Bad', 'Good'])
            axes[idx].set_yticklabels(['Bad', 'Good'])
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, results, y_true, figsize=(12, 8)):
        """
        Plot ROC curves for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, result in results.items():
            y_proba = result['probabilities']
            
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, lw=2, 
                       label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', 
                   fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance, model_name="Random Forest", 
                              top_n=15, figsize=(14, 8)):
        """
        Plot feature importance.
        
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
                     fontweight='bold')
        ax1.set_xlabel('Importance Score', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Pie chart for top features
        top_features = importance_df.tail(6)  # Top 6 for pie chart
        wedges, texts, autotexts = ax2.pie(top_features['Importance'], 
                                         labels=top_features['Feature'],
                                         autopct='%1.1f%%', startangle=90, 
                                         colors=self.colors[:6])
        
        # Improve pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('Feature Importance Distribution\n(Top 6 Features)', 
                     fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curves(self, results, y_true, figsize=(12, 8)):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, result in results.items():
            y_proba = result['probabilities']
            
            if y_proba is not None:
                precision, recall, _ = precision_recall_curve(y_true, y_proba)
                pr_auc = auc(recall, precision)
                
                ax.plot(recall, precision, lw=2,
                       label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, train_sizes, train_scores, test_scores, 
                           model_name, figsize=(10, 6)):
        """
        Plot learning curves.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            test_scores: Test scores
            model_name: Name of the model
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curves
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1,
                       color=self.colors[0])
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                       test_scores_mean + test_scores_std, alpha=0.1,
                       color=self.colors[1])
        
        ax.plot(train_sizes, train_scores_mean, 'o-', color=self.colors[0],
               label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color=self.colors[1],
               label="Cross-validation score")
        
        ax.set_xlabel("Training examples", fontweight='bold')
        ax.set_ylabel("Score", fontweight='bold')
        ax.set_title(f"Learning Curves - {model_name}", fontsize=16, fontweight='bold')
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_interactive_plots(self, df, results, feature_importance):
        """
        Create interactive plots using Plotly.
        
        Args:
            df: DataFrame with wine data
            results: Model results
            feature_importance: Feature importance dictionary
        """
        # 1. Interactive Correlation Heatmap
        corr_matrix = df.corr()
        fig_heatmap = px.imshow(corr_matrix, 
                              title="Interactive Correlation Heatmap",
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
        fig_heatmap.update_layout(width=800, height=600)
        
        # 2. Interactive Feature Importance
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(importance_df.tail(15), 
                              x='Importance', y='Feature',
                              orientation='h',
                              title='Interactive Feature Importance',
                              color='Importance',
                              color_continuous_scale='viridis')
        fig_importance.update_layout(width=800, height=600)
        
        # 3. Interactive Model Comparison
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        fig_comparison = px.bar(x=models, y=accuracies,
                              title='Interactive Model Accuracy Comparison',
                              labels={'x': 'Model', 'y': 'Accuracy'},
                              color=accuracies,
                              color_continuous_scale='teal')
        fig_comparison.update_layout(width=800, height=500)
        
        return fig_heatmap, fig_importance, fig_comparison
    