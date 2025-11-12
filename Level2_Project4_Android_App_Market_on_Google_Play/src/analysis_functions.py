import pandas as pd
import numpy as np
from collections import Counter

def perform_comprehensive_analysis(df):
    """
    Perform comprehensive analysis on the cleaned dataset
    
    Parameters:
    df (pd.DataFrame): Cleaned dataset
    
    Returns:
    dict: Analysis results
    """
    
    results = {}
    
    # Basic statistics
    results['basic_stats'] = get_basic_statistics(df)
    
    # Category analysis
    results['category_analysis'] = analyze_categories(df)
    
    # Rating analysis
    results['rating_analysis'] = analyze_ratings(df)
    
    # Pricing analysis
    results['pricing_analysis'] = analyze_pricing(df)
    
    # Size analysis
    results['size_analysis'] = analyze_sizes(df)
    
    # Installation analysis
    results['installation_analysis'] = analyze_installations(df)
    
    # Content rating analysis
    results['content_rating_analysis'] = analyze_content_ratings(df)
    
    return results

def get_basic_statistics(df):
    """Get basic statistics about the dataset"""
    
    stats = {
        'total_apps': len(df),
        'total_categories': df['Category'].nunique(),
        'total_genres': df['Genres'].nunique(),
        'free_apps': len(df[df['Type'] == 'Free']),
        'paid_apps': len(df[df['Type'] == 'Paid']),
        'avg_rating': df['Rating'].mean(),
        'avg_size_mb': df['Size'].mean(),
        'total_installs': df['Installs'].sum(),
        'date_range': {
            'earliest_update': df['Last Updated'].min(),
            'latest_update': df['Last Updated'].max()
        }
    }
    
    return stats

def analyze_categories(df):
    """Analyze app distribution across categories"""
    
    category_stats = df.groupby('Category').agg({
        'App': 'count',
        'Rating': ['mean', 'median', 'std'],
        'Installs': ['mean', 'sum'],
        'Price': ['mean', 'sum'],
        'Reviews': 'sum'
    }).round(2)
    
    category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns.values]
    category_stats = category_stats.rename(columns={'App_count': 'App_Count'})
    category_stats = category_stats.sort_values('App_Count', ascending=False)
    
    return category_stats.reset_index()

def analyze_ratings(df):
    """Analyze rating patterns and distributions"""
    
    rating_stats = {
        'distribution': df['Rating'].describe(),
        'rating_by_category': df.groupby('Category')['Rating'].agg(['mean', 'median', 'count']),
        'rating_vs_installs': df.groupby(pd.cut(df['Rating'], bins=np.arange(0, 5.5, 0.5)))['Installs'].mean(),
        'high_rated_apps': df[df['Rating'] >= 4.5]['Category'].value_counts(),
        'low_rated_apps': df[df['Rating'] <= 2.0]['Category'].value_counts()
    }
    
    return rating_stats

def analyze_pricing(df):
    """Analyze pricing strategies and trends"""
    
    paid_apps = df[df['Type'] == 'Paid']
    
    pricing_stats = {
        'free_vs_paid': {
            'count': df['Type'].value_counts().to_dict(),
            'percentage': (df['Type'].value_counts(normalize=True) * 100).round(2).to_dict()
        },
        'paid_apps_stats': {
            'count': len(paid_apps),
            'avg_price': paid_apps['Price'].mean(),
            'max_price': paid_apps['Price'].max(),
            'price_distribution': paid_apps['Price'].describe(),
            'price_by_category': paid_apps.groupby('Category')['Price'].mean().sort_values(ascending=False)
        },
        'revenue_estimate': (paid_apps['Price'] * paid_apps['Installs']).sum()
    }
    
    return pricing_stats

def analyze_sizes(df):
    """Analyze app size patterns"""
    
    size_stats = {
        'overall_stats': df['Size'].describe(),
        'size_by_category': df.groupby('Category')['Size'].mean().sort_values(ascending=False),
        'size_vs_rating': df.groupby(pd.cut(df['Size'], bins=[0, 10, 50, 100, 200, float('inf')]))['Rating'].mean(),
        'size_trends': df.groupby(df['Last Updated'].dt.year)['Size'].mean()
    }
    
    return size_stats

def analyze_installations(df):
    """Analyze installation patterns"""
    
    installation_stats = {
        'overall_install_stats': df['Installs'].describe(),
        'installs_by_category': df.groupby('Category')['Installs'].agg(['mean', 'sum', 'count']).sort_values('sum', ascending=False),
        'installs_vs_rating': df.groupby(pd.cut(df['Rating'], bins=[0, 2, 3, 4, 4.5, 5]))['Installs'].mean(),
        'top_installed_apps': df.nlargest(10, 'Installs')[['App', 'Category', 'Installs', 'Rating']],
        'install_concentration': (df['Installs'].sum(), df.nlargest(100, 'Installs')['Installs'].sum() / df['Installs'].sum() * 100)
    }
    
    return installation_stats

def analyze_content_ratings(df):
    """Analyze content rating distributions"""
    
    content_stats = {
        'distribution': df['Content Rating'].value_counts(),
        'percentage': (df['Content Rating'].value_counts(normalize=True) * 100).round(2),
        'rating_by_content': df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False),
        'installs_by_content': df.groupby('Content Rating')['Installs'].sum().sort_values(ascending=False)
    }
    
    return content_stats
