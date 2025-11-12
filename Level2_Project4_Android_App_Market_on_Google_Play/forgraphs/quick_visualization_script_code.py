#!/usr/bin/env python3
"""
Quick script to generate essential visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_quick_overview(df):
    """Create a quick overview of the dataset"""
    
    print("Creating quick overview visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create overview figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Category distribution
    df['Category'].value_counts().head(10).plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Top 10 Categories')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Rating distribution
    df['Rating'].hist(bins=30, ax=axes[0,1])
    axes[0,1].set_title('Rating Distribution')
    axes[0,1].set_xlabel('Rating')
    
    # 3. Free vs Paid
    df['Type'].value_counts().plot(kind='pie', ax=axes[0,2], autopct='%1.1f%%')
    axes[0,2].set_title('Free vs Paid Apps')
    
    # 4. Price distribution (paid apps only)
    paid_apps = df[df['Type'] == 'Paid']
    if len(paid_apps) > 0:
        paid_apps['Price'].hist(bins=20, ax=axes[1,0])
        axes[1,0].set_title('Price Distribution (Paid Apps)')
        axes[1,0].set_xlabel('Price ($)')
    
    # 5. Installs distribution
    df['Installs'].hist(bins=50, ax=axes[1,1], log=True)
    axes[1,1].set_title('Installs Distribution (Log Scale)')
    axes[1,1].set_xlabel('Installs')
    
    # 6. Content rating
    df['Content Rating'].value_counts().plot(kind='bar', ax=axes[1,2])
    axes[1,2].set_title('Content Rating Distribution')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/quick_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Quick overview created: visualizations/quick_overview.png")

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('data/apps_cleaned.csv')
    create_quick_overview(df)
    
