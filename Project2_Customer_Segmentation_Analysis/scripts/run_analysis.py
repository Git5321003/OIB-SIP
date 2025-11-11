#!/usr/bin/env python3
"""
Main script to run complete customer segmentation analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, clean_data, engineer_features, prepare_clustering_data
from src.clustering import CustomerSegmentation
from src.visualization import plot_elbow_method, plot_cluster_analysis, plot_cluster_profiles
from src.utils import create_directories, save_results, print_cluster_insights, validate_data


def main():
    """Run complete customer segmentation analysis."""
    print("🚀 Starting Customer Segmentation Analysis")
    print("=" * 50)
    
    # Configuration
    DATA_PATH = "data/raw/Marketing-Analytics-Customer-Segmentation.csv"
    N_CLUSTERS = 4
    
    # Create directories
    create_directories()
    
    # Step 1: Load and preprocess data
    print("\n📊 Step 1: Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    if df is None:
        return
    
    if not validate_data(df):
        return
    
    df_clean = clean_data(df)
    df_engineered = engineer_features(df_clean)
    
    # Step 2: Prepare data for clustering
    print("\n🔧 Step 2: Preparing data for clustering...")
    X_scaled, feature_names, scaler = prepare_clustering_data(df_engineered)
    
    # Step 3: Find optimal number of clusters
    print("\n🎯 Step 3: Finding optimal number of clusters...")
    segmentation = CustomerSegmentation()
    cluster_results = segmentation.find_optimal_clusters(X_scaled)
    
    # Visualize elbow method
    plot_elbow_method(cluster_results, 
                     save_path="reports/figures/clustering_results/elbow_method.png")
    
    # Step 4: Perform clustering
    print(f"\n🔍 Step 4: Performing K-means clustering with {N_CLUSTERS} clusters...")
    labels, metrics = segmentation.perform_kmeans_clustering(X_scaled, n_clusters=N_CLUSTERS)
    
    # Add cluster labels to dataframe
    df_final = df_engineered.copy()
    df_final['Cluster'] = labels
    
    # Step 5: Analyze clusters
    print("\n📈 Step 5: Analyzing cluster characteristics...")
    cluster_summary, insights = segmentation.analyze_clusters(df_engineered, labels, feature_names)
    
    # Create detailed cluster profiles
    cluster_profiles = {}
    for cluster in range(N_CLUSTERS):
        cluster_data = df_final[df_final['Cluster'] == cluster]
        cluster_profiles[cluster] = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_final) * 100,
            'demographics': {
                'avg_income': cluster_data['Income'].mean(),
                'relationship_rate': cluster_data['In_relationship'].mean() * 100,
                'avg_children': cluster_data['Total_Children'].mean()
            },
            'spending_behavior': {
                'total_spending': cluster_data['MntTotal'].mean(),
                'wine_spending': cluster_data['MntWines'].mean(),
                'fruit_spending': cluster_data['MntFruits'].mean()
            }
        }
    
    # Step 6: Visualize results
    print("\n🎨 Step 6: Generating visualizations...")
    plot_cluster_analysis(df_final, 
                         save_path="reports/figures/clustering_results/cluster_analysis.png")
    plot_cluster_profiles(cluster_profiles,
                         save_path="reports/figures/clustering_results/cluster_profiles.png")
    
    # Step 7: Save results
    print("\n💾 Step 7: Saving results...")
    save_results(df_final, cluster_summary)
    
    # Step 8: Print insights
    print_cluster_insights(insights)
    
    # Print final summary
    print("\n" + "=" * 50)
    print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"📈 Model Performance:")
    print(f"  • Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"  • Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}")
    print(f"  • Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")


if __name__ == "__main__":
    main()
    