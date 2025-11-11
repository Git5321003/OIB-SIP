"""
Visualization Module
-------------------
Contains functions for data visualization and model results plotting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousingVisualizer:
    """
    A class for creating visualizations for housing price analysis.
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Default figure size
        """
        self.figsize = figsize
    
    def plot_price_distribution(self, df, figsize=None):
        """
        Plot distribution of house prices.
        
        Args:
            df (pd.DataFrame): Housing dataset
            figsize (tuple): Figure size
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(df['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_title('Distribution of House Prices', fontweight='bold')
        ax1.set_xlabel('Price (in 10 millions)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(df['price'])
        ax2.set_title('Box Plot of House Prices', fontweight='bold')
        ax2.set_ylabel('Price (in 10 millions)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Price distribution plot created")
    
    def plot_numerical_features_distribution(self, df, numerical_features, figsize=(15, 10)):
        """
        Plot distribution of numerical features.
        
        Args:
            df (pd.DataFrame): Housing dataset
            numerical_features (list): List of numerical feature names
            figsize (tuple): Figure size
        """
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.ravel() if n_features > 1 else [axes]
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                axes[i].hist(df[feature], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
                axes[i].set_title(f'Distribution of {feature.title()}', fontweight='bold')
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Numerical features distribution plot created")
    
    def plot_correlation_heatmap(self, df, numerical_cols, figsize=(12, 8)):
        """
        Plot correlation heatmap for numerical features.
        
        Args:
            df (pd.DataFrame): Housing dataset
            numerical_cols (list): List of numerical column names
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        correlation_matrix = df[numerical_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, mask=mask, fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap of Numerical Features', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        logger.info("Correlation heatmap created")
    
    def plot_feature_vs_price(self, df, feature, figsize=None):
        """
        Plot relationship between a feature and price.
        
        Args:
            df (pd.DataFrame): Housing dataset
            feature (str): Feature name
            figsize (tuple): Figure size
        """
        if figsize is None:
            figsize = self.figsize
            
        plt.figure(figsize=figsize)
        
        if df[feature].dtype == 'object':
            # Categorical feature
            avg_prices = df.groupby(feature)['price'].mean().sort_values()
            
            bars = plt.bar(range(len(avg_prices)), avg_prices.values,
                         color=plt.cm.Set3(np.arange(len(avg_prices))))
            
            plt.title(f'Average Price by {feature}', fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Average Price (in 10 millions)')
            plt.xticks(range(len(avg_prices)), avg_prices.index, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, price in zip(bars, avg_prices.values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'₹{price/1000000:.1f}M', ha='center', va='bottom', 
                        fontweight='bold')
        else:
            # Numerical feature
            plt.scatter(df[feature], df['price'], alpha=0.6, color='coral')
            plt.title(f'{feature.title()} vs Price', fontweight='bold')
            plt.xlabel(feature.title())
            plt.ylabel('Price (in 10 millions)')
            plt.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = df[feature].corr(df['price'])
            plt.annotate(f'Correlation: {correlation:.3f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Feature vs price plot created for {feature}")
    
    def plot_feature_importance(self, feature_importance_df, figsize=None):
        """
        Plot feature importance from model.
        
        Args:
            feature_importance_df (pd.DataFrame): Feature importance dataframe
            figsize (tuple): Figure size
        """
        if figsize is None:
            figsize = (12, 8)
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Coefficient values (with colors for positive/negative)
        colors = ['green' if x > 0 else 'red' for x in feature_importance_df['Coefficient']]
        
        bars1 = ax1.barh(feature_importance_df['Feature'], 
                        feature_importance_df['Coefficient'], 
                        color=colors)
        ax1.set_title('Feature Coefficients in Linear Regression', 
                     fontweight='bold', fontsize=14)
        ax1.set_xlabel('Coefficient Value')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    ha='left' if width > 0 else 'right', 
                    va='center', fontweight='bold')
        
        # Absolute coefficient values (importance)
        bars2 = ax2.barh(feature_importance_df['Feature'], 
                        feature_importance_df['Absolute_Coefficient'], 
                        color='steelblue')
        ax2.set_title('Absolute Feature Importance', 
                     fontweight='bold', fontsize=14)
        ax2.set_xlabel('Absolute Coefficient Value')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Feature importance plot created")
    
    def plot_residual_analysis(self, y_true, y_pred, residuals, figsize=(15, 10)):
        """
        Plot residual analysis for model diagnostics.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            residuals (array-like): Residuals
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals vs Predicted', fontweight='bold')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightblue')
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot of Residuals', fontweight='bold')
        
        # Actual vs Predicted
        axes[1, 0].scatter(y_true, y_pred, alpha=0.6, color='green')
        axes[1, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_title('Actual vs Predicted', fontweight='bold')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time/index (if applicable)
        axes[1, 1].plot(residuals, alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Residuals over Index', fontweight='bold')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Hide the last subplot
        axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Residual analysis plots created")
    
    def plot_model_metrics_comparison(self, metrics_df, figsize=None):
        """
        Plot comparison of model metrics.
        
        Args:
            metrics_df (pd.DataFrame): Metrics dataframe
            figsize (tuple): Figure size
        """
        if figsize is None:
            figsize = (12, 8)
            
        metrics_to_plot = ['RMSE', 'MAE', 'R²']
        colors = ['skyblue', 'lightcoral']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes) - 1:  # Leave last subplot for actual vs predicted
                values = metrics_df[metric].values
                sets = metrics_df['Set'].values
                
                bars = axes[i].bar(sets, values, color=colors[:len(values)])
                axes[i].set_title(f'{metric} Comparison', fontweight='bold')
                axes[i].set_ylabel(metric)
                
                if metric == 'R²':
                    axes[i].set_ylim(0, 1)
                
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom', 
                               fontweight='bold')
        
        # Hide the last subplot if not used
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Model metrics comparison plot created")
    
    def plot_actual_vs_predicted(self, y_true, y_pred, figsize=None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            figsize (tuple): Figure size
        """
        if figsize is None:
            figsize = self.figsize
            
        plt.figure(figsize=figsize)
        
        plt.scatter(y_true, y_pred, alpha=0.6, color='purple', label='Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        plt.title('Actual vs Predicted Values', fontweight='bold', fontsize=14)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.annotate(f'R² = {r2:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Actual vs predicted plot created")


# Utility functions
def create_comprehensive_eda_plots(df, numerical_features, categorical_features):
    """
    Create comprehensive EDA plots for housing data.
    
    Args:
        df (pd.DataFrame): Housing dataset
        numerical_features (list): Numerical feature names
        categorical_features (list): Categorical feature names
    """
    visualizer = HousingVisualizer()
    
    # Price distribution
    visualizer.plot_price_distribution(df)
    
    # Numerical features distribution
    visualizer.plot_numerical_features_distribution(df, numerical_features)
    
    # Correlation heatmap
    numerical_cols = numerical_features + ['price']
    visualizer.plot_correlation_heatmap(df, numerical_cols)
    
    # Feature vs price plots for top correlated features
    correlation_with_price = df[numerical_cols].corr()['price'].sort_values(ascending=False)
    top_features = correlation_with_price.index[1:4]  # Top 3 excluding price itself
    
    for feature in top_features:
        visualizer.plot_feature_vs_price(df, feature)
    
    logger.info("Comprehensive EDA plots created")


def save_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save plot to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure object
        filename (str): Output filename
        dpi (int): Resolution
        bbox_inches (str): Bounding box inches
    """
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Plot saved as {filename}")