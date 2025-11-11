"""
Graphs and Visualizations for Housing Price Prediction
----------------------------------------------------
Comprehensive visualization module with all graphs used in the project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set global style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class HousingGraphs:
    """
    A comprehensive class for creating all graphs in the housing price prediction project.
    """
    
    def __init__(self):
        """Initialize the graphing class."""
        self.figsize = (12, 8)
    
    def set_style(self, style='seaborn-v0_8', palette='husl'):
        """
        Set the plotting style.
        
        Args:
            style (str): Matplotlib style
            palette (str): Seaborn color palette
        """
        plt.style.use(style)
        sns.set_palette(palette)
    
    def price_distribution_comprehensive(self, df):
        """
        Create comprehensive price distribution plots.
        
        Args:
            df (pd.DataFrame): Housing dataset
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram with KDE
        ax1.hist(df['price'], bins=30, density=True, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Histogram')
        df['price'].plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
        ax1.set_title('Price Distribution with KDE', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Price (in 10 millions)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2.boxplot(df['price'])
        ax2.set_title('Price Box Plot', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Price (in 10 millions)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Violin plot
        sns.violinplot(y=df['price'], ax=ax3, color='lightgreen')
        ax3.set_title('Price Violin Plot', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Price (in 10 millions)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plot for normality check
        stats.probplot(df['price'], dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot for Normality Check', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("📊 Price Statistics:")
        print(f"   Mean: ₹{df['price'].mean():,.2f}")
        print(f"   Median: ₹{df['price'].median():,.2f}")
        print(f"   Std Dev: ₹{df['price'].std():,.2f}")
        print(f"   Min: ₹{df['price'].min():,.2f}")
        print(f"   Max: ₹{df['price'].max():,.2f}")
    
    def numerical_features_distribution(self, df, numerical_features):
        """
        Create distribution plots for all numerical features.
        
        Args:
            df (pd.DataFrame): Housing dataset
            numerical_features (list): List of numerical feature names
        """
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_features))
        
        for i, feature in enumerate(numerical_features):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(df[feature], bins=20, alpha=0.7, color=colors[i], 
                           edgecolor='black', density=True)
                df[feature].plot(kind='kde', ax=axes[i], color='red', linewidth=2)
                
                axes[i].set_title(f'Distribution of {feature.title()}', 
                                fontweight='bold', fontsize=12)
                axes[i].set_xlabel(feature.title())
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = df[feature].mean()
                median_val = df[feature].median()
                axes[i].axvline(mean_val, color='blue', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].axvline(median_val, color='green', linestyle='--', 
                              label=f'Median: {median_val:.2f}')
                axes[i].legend()
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis_comprehensive(self, df, numerical_cols):
        """
        Create comprehensive correlation analysis.
        
        Args:
            df (pd.DataFrame): Housing dataset
            numerical_cols (list): List of numerical column names
        """
        # 1. Correlation Heatmap
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        correlation_matrix = df[numerical_cols].corr()
        
        # Heatmap with annotations
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, fmt='.2f', ax=ax1,
                   cbar_kws={"shrink": .8})
        ax1.set_title('Correlation Heatmap', fontweight='bold', fontsize=16)
        
        # 2. Correlation with Price (Bar plot)
        price_correlations = correlation_matrix['price'].drop('price').sort_values()
        colors = ['red' if x < 0 else 'green' for x in price_correlations.values]
        
        bars = ax2.barh(range(len(price_correlations)), price_correlations.values, 
                       color=colors, alpha=0.7)
        ax2.set_yticks(range(len(price_correlations)))
        ax2.set_yticklabels(price_correlations.index)
        ax2.set_title('Correlation with Price', fontweight='bold', fontsize=16)
        ax2.set_xlabel('Correlation Coefficient')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, price_correlations.values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{value:.2f}', ha='left' if width > 0 else 'right', 
                    va='center', fontweight='bold')
        
        # 3. Top correlated features scatter plots
        top_correlated = price_correlations.nlargest(4)
        for i, (feature, corr) in enumerate(top_correlated.items()):
            row, col = i // 2, i % 2
            ax = axes[1, col] if i < 2 else axes[1, col]
            
            ax.scatter(df[feature], df['price'], alpha=0.6, color=f'C{i}')
            ax.set_xlabel(feature.title())
            ax.set_ylabel('Price')
            ax.set_title(f'{feature} vs Price (r={corr:.2f})', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df[feature], df['price'], 1)
            p = np.poly1d(z)
            ax.plot(df[feature], p(df[feature]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("🔍 Correlation Insights:")
        print(f"   Strongest positive correlation: {top_correlated.index[0]} ({top_correlated.iloc[0]:.3f})")
        print(f"   Strongest negative correlation: {price_correlations.idxmin()} ({price_correlations.min():.3f})")
    
    def categorical_features_analysis(self, df, categorical_features):
        """
        Analyze categorical features and their relationship with price.
        
        Args:
            df (pd.DataFrame): Housing dataset
            categorical_features (list): List of categorical feature names
        """
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                # Create subplot for each categorical feature
                ax = axes[i]
                
                # Calculate statistics
                value_counts = df[feature].value_counts()
                avg_prices = df.groupby(feature)['price'].agg(['mean', 'median', 'std'])
                
                # Create bar plot for average prices
                bars = ax.bar(range(len(avg_prices)), avg_prices['mean'], 
                            color=plt.cm.Pastel1(range(len(avg_prices))),
                            alpha=0.7, edgecolor='black')
                
                ax.set_title(f'Average Price by {feature}', fontweight='bold', fontsize=14)
                ax.set_xlabel(feature)
                ax.set_ylabel('Average Price (in 10 millions)')
                ax.set_xticks(range(len(avg_prices)))
                ax.set_xticklabels(avg_prices.index, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, (idx, row) in zip(bars, avg_prices.iterrows()):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'₹{height/1000000:.1f}M\n(n={value_counts[idx]})',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def feature_vs_price_scatter(self, df, features):
        """
        Create scatter plots for features vs price.
        
        Args:
            df (pd.DataFrame): Housing dataset
            features (list): List of feature names to plot against price
        """
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Create scatter plot
                scatter = ax.scatter(df[feature], df['price'], alpha=0.6, 
                                   c=df['price'], cmap='viridis', s=50)
                
                ax.set_title(f'{feature.title()} vs Price', fontweight='bold', fontsize=14)
                ax.set_xlabel(feature.title())
                ax.set_ylabel('Price (in 10 millions)')
                ax.grid(True, alpha=0.3)
                
                # Add correlation coefficient
                correlation = df[feature].corr(df['price'])
                ax.annotate(f'r = {correlation:.3f}', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", 
                                   ec="gray", alpha=0.8),
                           fontweight='bold')
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label='Price')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def boxplots_by_category(self, df, categorical_features):
        """
        Create boxplots of price by categorical features.
        
        Args:
            df (pd.DataFrame): Housing dataset
            categorical_features (list): List of categorical feature names
        """
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.ravel() if n_features > 1 else [axes]
        
        for i, feature in enumerate(categorical_features):
            if i < len(axes):
                # Create boxplot
                df.boxplot(column='price', by=feature, ax=axes[i])
                axes[i].set_title(f'Price Distribution by {feature}', 
                                fontweight='bold', fontsize=14)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Price (in 10 millions)')
                
                # Remove automatic title
                axes[i].get_figure().suptitle('')
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_plot(self, feature_importance_df, top_n=10):
        """
        Create feature importance visualization.
        
        Args:
            feature_importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to show
        """
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. Coefficient values (with colors for positive/negative)
        colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
        
        bars1 = ax1.barh(top_features['Feature'], top_features['Coefficient'], 
                        color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Feature Coefficients in Linear Regression', 
                     fontweight='bold', fontsize=16)
        ax1.set_xlabel('Coefficient Value')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    ha='left' if width > 0 else 'right', 
                    va='center', fontweight='bold')
        
        # 2. Absolute coefficient values (importance)
        bars2 = ax2.barh(top_features['Feature'], 
                        top_features['Absolute_Coefficient'], 
                        color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Absolute Feature Importance', 
                     fontweight='bold', fontsize=16)
        ax2.set_xlabel('Absolute Coefficient Value')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print feature importance insights
        print("🎯 Feature Importance Insights:")
        top_positive = top_features[top_features['Coefficient'] > 0].iloc[0]
        top_negative = top_features[top_features['Coefficient'] < 0].iloc[0] if len(top_features[top_features['Coefficient'] < 0]) > 0 else None
        
        print(f"   Most positive impact: {top_positive['Feature']} (coef: {top_positive['Coefficient']:.2f})")
        if top_negative is not None:
            print(f"   Most negative impact: {top_negative['Feature']} (coef: {top_negative['Coefficient']:.2f})")
    
    def residual_analysis_comprehensive(self, y_true, y_pred, residuals):
        """
        Create comprehensive residual analysis plots.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            residuals (array-like): Residuals
        """
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax1.set_title('Residuals vs Predicted Values', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of residuals
        ax2.hist(residuals, bins=30, density=True, alpha=0.7, color='lightblue', 
                edgecolor='black', label='Histogram')
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
        ax2.set_title('Distribution of Residuals', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals', fontweight='bold', fontsize=14)
        
        # 4. Actual vs Predicted
        ax4.scatter(y_true, y_pred, alpha=0.6, color='green')
        ax4.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax4.set_title('Actual vs Predicted Values', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals over index (for time series-like analysis)
        ax5.plot(residuals, alpha=0.7, color='purple')
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('Residuals over Index', fontweight='bold', fontsize=14)
        ax5.set_xlabel('Index')
        ax5.set_ylabel('Residuals')
        ax5.grid(True, alpha=0.3)
        
        # 6. Scale-Location plot (sqrt(|residuals|) vs predicted)
        ax6.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.6, color='orange')
        ax6.set_title('Scale-Location Plot', fontweight='bold', fontsize=14)
        ax6.set_xlabel('Predicted Values')
        ax6.set_ylabel('√|Standardized Residuals|')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print residual statistics
        print("📊 Residual Analysis Statistics:")
        print(f"   Mean of residuals: {residuals.mean():.2f}")
        print(f"   Standard deviation: {residuals.std():.2f}")
        print(f"   Skewness: {stats.skew(residuals):.2f}")
        print(f"   Kurtosis: {stats.kurtosis(residuals):.2f}")
        
        # Normality test
        _, p_value = stats.normaltest(residuals)
        print(f"   Normality test p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("   ✅ Residuals appear normally distributed")
        else:
            print("   ⚠️ Residuals may not be normally distributed")
    
    def model_performance_comparison(self, metrics_dict):
        """
        Compare performance of multiple models.
        
        Args:
            metrics_dict (dict): Dictionary with model names as keys and metrics as values
        """
        models = list(metrics_dict.keys())
        metrics = ['RMSE', 'MAE', 'R²']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=colors, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric} Comparison', fontweight='bold', fontsize=14)
            axes[i].set_ylabel(metric)
            
            if metric == 'R²':
                axes[i].set_ylim(0, 1)
            
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def prediction_comparison(self, y_true, y_pred, sample_size=50):
        """
        Compare actual vs predicted values for a sample.
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            sample_size (int): Number of samples to show
        """
        # Select random sample
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Side-by-side comparison
        x_pos = np.arange(len(y_true_sample))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, y_true_sample, width, 
                       label='Actual', alpha=0.7, color='blue')
        bars2 = ax1.bar(x_pos + width/2, y_pred_sample, width, 
                       label='Predicted', alpha=0.7, color='red')
        
        ax1.set_title('Actual vs Predicted Values (Sample)', fontweight='bold', fontsize=16)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot with perfect prediction line
        ax2.scatter(y_true, y_pred, alpha=0.6, color='green', s=50)
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        # Add R² score
        r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
        ax2.annotate(f'R² = {r2:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontweight='bold')
        
        ax2.set_title('Actual vs Predicted (All Data)', fontweight='bold', fontsize=16)
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self, df, numerical_features, categorical_features, 
                        feature_importance_df, y_true, y_pred, residuals):
        """
        Create a comprehensive dashboard of all important graphs.
        
        Args:
            df (pd.DataFrame): Housing dataset
            numerical_features (list): Numerical feature names
            categorical_features (list): Categorical feature names
            feature_importance_df (pd.DataFrame): Feature importance
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            residuals (array-like): Residuals
        """
        print("🚀 CREATING COMPREHENSIVE DASHBOARD")
        print("=" * 50)
        
        # 1. Data Overview
        print("\n📊 1. DATA OVERVIEW")
        self.price_distribution_comprehensive(df)
        
        # 2. Feature Analysis
        print("\n🔍 2. FEATURE ANALYSIS")
        self.numerical_features_distribution(df, numerical_features)
        self.categorical_features_analysis(df, categorical_features)
        
        # 3. Correlation Analysis
        print("\n📈 3. CORRELATION ANALYSIS")
        numerical_cols = numerical_features + ['price']
        self.correlation_analysis_comprehensive(df, numerical_cols)
        
        # 4. Feature Importance
        print("\n🎯 4. FEATURE IMPORTANCE")
        self.feature_importance_plot(feature_importance_df)
        
        # 5. Model Diagnostics
        print("\n🔧 5. MODEL DIAGNOSTICS")
        self.residual_analysis_comprehensive(y_true, y_pred, residuals)
        self.prediction_comparison(y_true, y_pred)
        
        print("\n✅ DASHBOARD COMPLETED!")


# Utility functions
def save_all_graphs(df, numerical_features, categorical_features, 
                   feature_importance_df, y_true, y_pred, residuals, 
                   save_dir='graphs'):
    """
    Save all graphs to files.
    
    Args:
        df (pd.DataFrame): Housing dataset
        numerical_features (list): Numerical feature names
        categorical_features (list): Categorical feature names
        feature_importance_df (pd.DataFrame): Feature importance
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        residuals (array-like): Residuals
        save_dir (str): Directory to save graphs
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    graphs = HousingGraphs()
    
    # Save each graph
    graphs_list = [
        ('price_distribution.png', graphs.price_distribution_comprehensive, [df]),
        ('numerical_features.png', graphs.numerical_features_distribution, [df, numerical_features]),
        ('categorical_features.png', graphs.categorical_features_analysis, [df, categorical_features]),
        ('correlation_analysis.png', graphs.correlation_analysis_comprehensive, [df, numerical_features + ['price']]),
        ('feature_importance.png', graphs.feature_importance_plot, [feature_importance_df]),
        ('residual_analysis.png', graphs.residual_analysis_comprehensive, [y_true, y_pred, residuals]),
        ('prediction_comparison.png', graphs.prediction_comparison, [y_true, y_pred])
    ]
    
    for filename, graph_func, args in graphs_list:
        try:
            plt.figure(figsize=(12, 8))
            graph_func(*args)
            plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
    
    print(f"\n🎉 All graphs saved to: {save_dir}/")


def create_simple_plots(df, target_col='price'):
    """
    Create simple version of key plots for quick analysis.
    
    Args:
        df (pd.DataFrame): Housing dataset
        target_col (str): Target column name
    """
    graphs = HousingGraphs()
    
    # Quick correlation heatmap
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    graphs.correlation_analysis_comprehensive(df, numerical_cols)
    
    # Quick feature vs target
    top_features = df[numerical_cols].corr()[target_col].nlargest(4).index[1:]
    graphs.feature_vs_price_scatter(df, top_features)
    