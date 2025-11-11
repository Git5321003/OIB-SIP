#!/usr/bin/env python3
"""
Script to deploy the trained clustering model.
"""

import sys
import os
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import load_data, clean_data, engineer_features


def deploy_model():
    """Deploy the trained clustering model."""
    print("🚀 Deploying Customer Segmentation Model")
    print("=" * 40)
    
    # Load and preprocess new data
    DATA_PATH = "data/raw/Marketing-Analytics-Customer-Segmentation.csv"
    
    print("\n📊 Loading data...")
    df = load_data(DATA_PATH)
    if df is None:
        return
    
    df_clean = clean_data(df)
    df_engineered = engineer_features(df_clean)
    
    # Load trained model (in a real scenario, this would be loaded from a file)
    print("\n🔧 Loading trained model...")
    # For now, we'll train a new model (in practice, load from joblib file)
    from src.clustering import CustomerSegmentation
    from src.data_preprocessing import prepare_clustering_data
    
    X_scaled, feature_names, scaler = prepare_clustering_data(df_engineered)
    
    segmentation = CustomerSegmentation()
    labels, metrics = segmentation.perform_kmeans_clustering(X_scaled, n_clusters=4)
    
    # Save the model
    model_path = "models/customer_segmentation_model.joblib"
    os.makedirs("models", exist_ok=True)
    
    model_artifacts = {
        'model': segmentation.model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics
    }
    
    joblib.dump(model_artifacts, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Create deployment report
    create_deployment_report(df_engineered, labels, metrics, model_path)


def create_deployment_report(df, labels, metrics, model_path):
    """Create a deployment report."""
    report_path = "reports/deployment_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("CUSTOMER SEGMENTATION MODEL DEPLOYMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL INFORMATION:\n")
        f.write(f"- Model Type: K-means Clustering\n")
        f.write(f"- Number of Clusters: 4\n")
        f.write(f"- Model File: {model_path}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"- Silhouette Score: {metrics['silhouette_score']:.3f}\n")
        f.write(f"- Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.3f}\n")
        f.write(f"- Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}\n\n")
        
        f.write("CLUSTER DISTRIBUTION:\n")
        cluster_sizes = df.groupby(labels).size()
        for cluster, size in cluster_sizes.items():
            percentage = (size / len(df)) * 100
            f.write(f"- Cluster {cluster}: {size} customers ({percentage:.1f}%)\n")
        
        f.write(f"\nTotal Customers: {len(df)}\n")
        f.write(f"Deployment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Deployment report saved to: {report_path}")


if __name__ == "__main__":
    deploy_model()
    