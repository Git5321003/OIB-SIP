"""
Quick script to generate all graphs for the Wine Quality Prediction project.
"""

import pandas as pd
import matplotlib.pyplot as plt
from src.visualization import WineQualityVisualizer

def create_all_graphs():
    """Generate all graphs for the project."""
    
    # Load data
    df = pd.read_csv('data/WineQT.csv')
    
    # Initialize visualizer
    viz = WineQualityVisualizer()
    
    print("Creating all graphs for Wine Quality Prediction Project...")
    
    # 1. Basic Distribution Plots
    print("1. Creating distribution plots...")
    fig1 = viz.plot_quality_distribution(df)
    plt.savefig('results/plots/quality_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    print("2. Creating correlation heatmap...")
    fig2 = viz.plot_correlation_heatmap(df)
    plt.savefig('results/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Distributions
    print("3. Creating feature distributions...")
    fig3 = viz.plot_feature_distributions(df)
    plt.savefig('results/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Quality vs Features
    print("4. Creating quality vs features plots...")
    fig4 = viz.plot_quality_vs_features(df)
    plt.savefig('results/plots/quality_vs_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Boxplots by Quality
    print("5. Creating boxplots by quality...")
    fig5 = viz.plot_boxplots_by_quality(df)
    plt.savefig('results/plots/boxplots_by_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All graphs created successfully!")
    print("Graphs saved to: results/plots/")

if __name__ == "__main__":
    create_all_graphs()
    