import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_all_visualizations(df, analysis_results):
    """Create all visualizations for the analysis"""
    
    # Create visualization directories
    viz_dirs = [
        'visualizations/category_distribution',
        'visualizations/rating_analysis', 
        'visualizations/pricing_insights',
        'visualizations/popularity_metrics',
        'visualizations/sentiment_analysis'
    ]
    
    for dir_path in viz_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create visualizations
    create_category_visualizations(df, analysis_results)
    create_rating_visualizations(df, analysis_results)
    create_pricing_visualizations(df, analysis_results)
    create_popularity_visualizations(df, analysis_results)
    create_comprehensive_dashboard(df, analysis_results)

def create_category_visualizations(df, analysis_results):
    """Create visualizations for category analysis"""
    
    category_stats = analysis_results['category_analysis']
    
    # App distribution by category
    plt.figure(figsize=(12, 8))
    category_counts = df['Category'].value_counts()
    
    plt.subplot(2, 2, 1)
    category_counts.head(15).plot(kind='bar')
    plt.title('Top 15 Categories by App Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.subplot(2, 2, 2)
    df['Category'].value_counts().tail(15).plot(kind='bar')
    plt.title('Bottom 15 Categories by App Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.subplot(2, 2, 3)
    category_stats.set_index('Category')['Installs_mean'].nlargest(15).plot(kind='bar')
    plt.title('Top 15 Categories by Average Installs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.subplot(2, 2, 4)
    category_stats.set_index('Category')['Rating_mean'].nlargest(15).plot(kind='bar')
    plt.title('Top 15 Categories by Average Rating')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('visualizations/category_distribution/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive category plot
    fig = px.treemap(category_stats.head(20),
                    path=['Category'],
                    values='App_Count',
                    title='App Distribution Across Categories (Top 20)')
    fig.write_html('visualizations/category_distribution/category_treemap.html')

def create_rating_visualizations(df, analysis_results):
    """Create visualizations for rating analysis"""
    
    # Rating distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['Rating'].hist(bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of App Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 2)
    df.boxplot(column='Rating', by='Content Rating', grid=False)
    plt.title('Rating Distribution by Content Rating')
    plt.suptitle('')  # Remove automatic title
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 3)
    rating_by_category = df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
    rating_by_category.head(15).plot(kind='bar')
    plt.title('Top 15 Categories by Average Rating')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=df, x='Rating', y='Reviews', alpha=0.6)
    plt.title('Rating vs Number of Reviews')
    
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df, x='Rating', y='Installs', alpha=0.6)
    plt.title('Rating vs Installs')
    
    plt.subplot(2, 3, 6)
    df['Rating'].value_counts().sort_index().plot(kind='line', marker='o')
    plt.title('Rating Frequency Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('visualizations/rating_analysis/rating_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_pricing_visualizations(df, analysis_results):
    """Create visualizations for pricing analysis"""
    
    paid_apps = df[df['Type'] == 'Paid']
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Free vs Paid Apps Distribution')
    
    plt.subplot(2, 3, 2)
    paid_apps['Price'].hist(bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Paid App Prices')
    plt.xlabel('Price ($)')
    
    plt.subplot(2, 3, 3)
    price_by_category = paid_apps.groupby('Category')['Price'].mean().sort_values(ascending=False)
    price_by_category.head(15).plot(kind='bar')
    plt.title('Top 15 Categories by Average Price')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=paid_apps, x='Price', y='Rating', alpha=0.6)
    plt.title('Price vs Rating for Paid Apps')
    
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=paid_apps, x='Price', y='Installs', alpha=0.6)
    plt.title('Price vs Installs for Paid Apps')
    
    plt.subplot(2, 3, 6)
    paid_apps.groupby('Content Rating')['Price'].mean().plot(kind='bar')
    plt.title('Average Price by Content Rating')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/pricing_insights/pricing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_popularity_visualizations(df, analysis_results):
    """Create visualizations for popularity metrics"""
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    df['Installs'].hist(bins=50, alpha=0.7, edgecolor='black', log=True)
    plt.title('Distribution of Installs (Log Scale)')
    plt.xlabel('Installs')
    
    plt.subplot(2, 3, 2)
    installs_by_category = df.groupby('Category')['Installs'].sum().sort_values(ascending=False)
    installs_by_category.head(15).plot(kind='bar')
    plt.title('Top 15 Categories by Total Installs')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 3)
    sns.heatmap(df[['Rating', 'Reviews', 'Installs', 'Price']].corr(), 
                annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    plt.subplot(2, 3, 4)
    df.groupby('Size_Category')['Installs'].mean().plot(kind='bar')
    plt.title('Average Installs by App Size Category')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 5)
    df.groupby('Content Rating')['Installs'].sum().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Installs Distribution by Content Rating')
    
    plt.subplot(2, 3, 6)
    top_apps = df.nlargest(10, 'Installs')
    plt.barh(range(len(top_apps)), top_apps['Installs'])
    plt.yticks(range(len(top_apps)), top_apps['App'])
    plt.title('Top 10 Most Installed Apps')
    plt.xlabel('Installs')
    
    plt.tight_layout()
    plt.savefig('visualizations/popularity_metrics/popularity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard(df, analysis_results):
    """Create an interactive comprehensive dashboard"""
    
    # Interactive scatter plot: Rating vs Installs by Category
    fig = px.scatter(df, x='Rating', y='Installs', color='Category',
                    size='Reviews', hover_data=['App', 'Type'],
                    title='Rating vs Installs by Category',
                    log_y=True, size_max=60)
    fig.write_html('visualizations/comprehensive_dashboard.html')
    
    # Category performance matrix
    category_performance = df.groupby('Category').agg({
        'App': 'count',
        'Rating': 'mean',
        'Installs': 'mean',
        'Price': 'mean'
    }).reset_index()
    
    fig = px.scatter(category_performance, x='Rating', y='Installs',
                    size='App', color='Price', hover_name='Category',
                    title='Category Performance Matrix',
                    size_max=60)
    fig.write_html('visualizations/category_performance.html')
    