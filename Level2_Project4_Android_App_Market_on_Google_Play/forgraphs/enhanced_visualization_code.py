import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud
import squarify

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def create_all_visualizations(df, analysis_results):
    """Create all visualizations for the analysis"""
    
    print("🎨 Creating comprehensive visualizations...")
    
    # Create visualization directories
    viz_dirs = [
        'visualizations/category_distribution',
        'visualizations/rating_analysis', 
        'visualizations/pricing_insights',
        'visualizations/popularity_metrics',
        'visualizations/sentiment_analysis',
        'visualizations/comprehensive'
    ]
    
    for dir_path in viz_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create visualizations
    create_category_visualizations(df, analysis_results)
    create_rating_visualizations(df, analysis_results)
    create_pricing_visualizations(df, analysis_results)
    create_popularity_visualizations(df, analysis_results)
    create_size_analysis_visualizations(df, analysis_results)
    create_content_rating_visualizations(df, analysis_results)
    create_temporal_visualizations(df, analysis_results)
    create_comprehensive_dashboards(df, analysis_results)
    
    print("✅ All visualizations created successfully!")

def create_category_visualizations(df, analysis_results):
    """Create visualizations for category analysis"""
    
    print("📊 Creating category visualizations...")
    
    # 1. Top Categories by App Count
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    top_categories = df['Category'].value_counts().head(15)
    bars = plt.bar(range(len(top_categories)), top_categories.values)
    plt.title('Top 15 Categories by Number of Apps', fontsize=14, fontweight='bold')
    plt.xticks(range(len(top_categories)), top_categories.index, rotation=45, ha='right')
    plt.ylabel('Number of Apps')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Category Distribution Treemap
    plt.subplot(2, 2, 2)
    category_counts = df['Category'].value_counts()
    squarify.plot(sizes=category_counts.values, 
                  label=category_counts.index, 
                  alpha=0.8,
                  text_kwargs={'fontsize':8})
    plt.title('Category Distribution Treemap', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 3. Average Rating by Category
    plt.subplot(2, 2, 3)
    rating_by_category = df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(15)
    colors = plt.cm.viridis(np.linspace(0, 1, len(rating_by_category)))
    bars = plt.bar(range(len(rating_by_category)), rating_by_category.values, color=colors)
    plt.title('Top 15 Categories by Average Rating', fontsize=14, fontweight='bold')
    plt.xticks(range(len(rating_by_category)), rating_by_category.index, rotation=45, ha='right')
    plt.ylabel('Average Rating')
    plt.ylim(3.5, 5.0)
    
    # 4. Average Installs by Category
    plt.subplot(2, 2, 4)
    installs_by_category = df.groupby('Category')['Installs'].mean().sort_values(ascending=False).head(15)
    plt.bar(range(len(installs_by_category)), installs_by_category.values)
    plt.title('Top 15 Categories by Average Installs', fontsize=14, fontweight='bold')
    plt.xticks(range(len(installs_by_category)), installs_by_category.index, rotation=45, ha='right')
    plt.ylabel('Average Installs (Log Scale)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('visualizations/category_distribution/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive category plot
    category_stats = analysis_results['category_analysis']
    fig = px.treemap(category_stats.head(20),
                    path=['Category'],
                    values='App_Count',
                    color='Rating_mean',
                    color_continuous_scale='Viridis',
                    title='App Distribution Across Categories (Top 20) - Color shows Average Rating')
    fig.update_layout(width=1000, height=600)
    fig.write_html('visualizations/category_distribution/category_treemap.html')
    
    # Category performance scatter plot
    fig = px.scatter(category_stats, 
                    x='App_Count', 
                    y='Rating_mean',
                    size='Installs_mean',
                    color='Installs_mean',
                    hover_name='Category',
                    title='Category Performance: App Count vs Average Rating',
                    labels={'App_Count': 'Number of Apps', 'Rating_mean': 'Average Rating'})
    fig.update_layout(width=1000, height=600)
    fig.write_html('visualizations/category_distribution/category_performance.html')

def create_rating_visualizations(df, analysis_results):
    """Create visualizations for rating analysis"""
    
    print("⭐ Creating rating visualizations...")
    
    plt.figure(figsize=(20, 12))
    
    # 1. Rating Distribution
    plt.subplot(2, 3, 1)
    df['Rating'].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of App Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Rating by Content Rating
    plt.subplot(2, 3, 2)
    rating_by_content = df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(rating_by_content)), rating_by_content.values)
    plt.title('Average Rating by Content Rating', fontsize=14, fontweight='bold')
    plt.xticks(range(len(rating_by_content)), rating_by_content.index, rotation=45)
    plt.ylabel('Average Rating')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 3. Rating vs Reviews Scatter
    plt.subplot(2, 3, 3)
    plt.scatter(df['Rating'], df['Reviews'], alpha=0.6, c=df['Installs'], cmap='viridis', s=50)
    plt.colorbar(label='Installs')
    plt.title('Rating vs Number of Reviews', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Reviews (Log Scale)')
    plt.yscale('log')
    
    # 4. Rating vs Installs
    plt.subplot(2, 3, 4)
    plt.scatter(df['Rating'], df['Installs'], alpha=0.6, c=df['Price'], cmap='plasma', s=50)
    plt.colorbar(label='Price ($)')
    plt.title('Rating vs Installs', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Installs (Log Scale)')
    plt.yscale('log')
    
    # 5. Rating Distribution by Type
    plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x='Type', y='Rating')
    plt.title('Rating Distribution: Free vs Paid Apps', fontsize=14, fontweight='bold')
    plt.xlabel('App Type')
    plt.ylabel('Rating')
    
    # 6. Cumulative Rating Distribution
    plt.subplot(2, 3, 6)
    sorted_ratings = np.sort(df['Rating'].dropna())
    y_vals = np.arange(len(sorted_ratings)) / float(len(sorted_ratings))
    plt.plot(sorted_ratings, y_vals, linewidth=2)
    plt.title('Cumulative Distribution of Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Cumulative Proportion')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/rating_analysis/rating_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive rating distribution
    fig = px.histogram(df, x='Rating', nbins=30, 
                      title='Interactive Rating Distribution',
                      opacity=0.7)
    fig.update_layout(width=800, height=500)
    fig.write_html('visualizations/rating_analysis/rating_distribution_interactive.html')

def create_pricing_visualizations(df, analysis_results):
    """Create visualizations for pricing analysis"""
    
    print("💰 Creating pricing visualizations...")
    
    paid_apps = df[df['Type'] == 'Paid']
    
    plt.figure(figsize=(20, 12))
    
    # 1. Free vs Paid Distribution
    plt.subplot(2, 3, 1)
    type_counts = df['Type'].value_counts()
    colors = ['#ff9999', '#66b3ff']
    plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Free vs Paid Apps Distribution', fontsize=14, fontweight='bold')
    
    # 2. Price Distribution of Paid Apps
    plt.subplot(2, 3, 2)
    paid_apps['Price'].hist(bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Price Distribution of Paid Apps', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 3. Average Price by Category
    plt.subplot(2, 3, 3)
    price_by_category = paid_apps.groupby('Category')['Price'].mean().sort_values(ascending=False).head(15)
    bars = plt.bar(range(len(price_by_category)), price_by_category.values, color='orange')
    plt.title('Top 15 Categories by Average Price', fontsize=14, fontweight='bold')
    plt.xticks(range(len(price_by_category)), price_by_category.index, rotation=45, ha='right')
    plt.ylabel('Average Price ($)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}', ha='center', va='bottom')
    
    # 4. Price vs Rating
    plt.subplot(2, 3, 4)
    plt.scatter(paid_apps['Price'], paid_apps['Rating'], alpha=0.6, 
               c=paid_apps['Installs'], cmap='viridis', s=50)
    plt.colorbar(label='Installs')
    plt.title('Price vs Rating for Paid Apps', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Rating')
    
    # 5. Price vs Installs
    plt.subplot(2, 3, 5)
    plt.scatter(paid_apps['Price'], paid_apps['Installs'], alpha=0.6, 
               c=paid_apps['Rating'], cmap='plasma', s=50)
    plt.colorbar(label='Rating')
    plt.title('Price vs Installs for Paid Apps', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Installs (Log Scale)')
    plt.yscale('log')
    
    # 6. Revenue Potential by Category
    plt.subplot(2, 3, 6)
    revenue_by_category = (paid_apps.groupby('Category')
                         .apply(lambda x: (x['Price'] * x['Installs']).sum())
                         .sort_values(ascending=False).head(10))
    bars = plt.bar(range(len(revenue_by_category)), revenue_by_category.values, color='green')
    plt.title('Top 10 Categories by Revenue Potential', fontsize=14, fontweight='bold')
    plt.xticks(range(len(revenue_by_category)), revenue_by_category.index, rotation=45, ha='right')
    plt.ylabel('Estimated Revenue ($)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('visualizations/pricing_insights/pricing_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive pricing analysis
    fig = px.scatter(paid_apps, x='Price', y='Installs', 
                    color='Rating', size='Reviews',
                    hover_data=['App', 'Category'],
                    title='Paid Apps: Price vs Installs',
                    log_y=True)
    fig.update_layout(width=1000, height=600)
    fig.write_html('visualizations/pricing_insights/price_vs_installs.html')

def create_popularity_visualizations(df, analysis_results):
    """Create visualizations for popularity metrics"""
    
    print("📈 Creating popularity visualizations...")
    
    plt.figure(figsize=(20, 15))
    
    # 1. Installs Distribution
    plt.subplot(3, 3, 1)
    df['Installs'].hist(bins=50, alpha=0.7, color='lightgreen', edgecolor='black', log=True)
    plt.title('Distribution of Installs (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Installs')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Top 10 Most Installed Apps
    plt.subplot(3, 3, 2)
    top_apps = df.nlargest(10, 'Installs')
    y_pos = range(len(top_apps))
    plt.barh(y_pos, top_apps['Installs'])
    plt.yticks(y_pos, [app[:20] + '...' if len(app) > 20 else app for app in top_apps['App']])
    plt.xlabel('Installs')
    plt.title('Top 10 Most Installed Apps', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # 3. Installs by Category
    plt.subplot(3, 3, 3)
    installs_by_category = df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(15)
    bars = plt.bar(range(len(installs_by_category)), installs_by_category.values)
    plt.title('Top 15 Categories by Total Installs', fontsize=14, fontweight='bold')
    plt.xticks(range(len(installs_by_category)), installs_by_category.index, rotation=45, ha='right')
    plt.ylabel('Total Installs (Log Scale)')
    plt.yscale('log')
    
    # 4. Correlation Heatmap
    plt.subplot(3, 3, 4)
    numeric_cols = ['Rating', 'Reviews', 'Installs', 'Price', 'Size']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # 5. Reviews vs Installs
    plt.subplot(3, 3, 5)
    plt.scatter(df['Reviews'], df['Installs'], alpha=0.6, c=df['Rating'], cmap='viridis', s=30)
    plt.colorbar(label='Rating')
    plt.title('Reviews vs Installs', fontsize=14, fontweight='bold')
    plt.xlabel('Reviews (Log Scale)')
    plt.ylabel('Installs (Log Scale)')
    plt.xscale('log')
    plt.yscale('log')
    
    # 6. Installs by Content Rating
    plt.subplot(3, 3, 6)
    installs_by_content = df.groupby('Content Rating')['Installs'].sum()
    plt.pie(installs_by_content.values, labels=installs_by_content.index, autopct='%1.1f%%')
    plt.title('Installs Distribution by Content Rating', fontsize=14, fontweight='bold')
    
    # 7. Success Matrix: Rating vs Installs
    plt.subplot(3, 3, 7)
    success_matrix = pd.crosstab(
        pd.cut(df['Rating'], bins=[0, 3, 4, 4.5, 5]),
        pd.cut(df['Installs'], bins=[0, 1000, 10000, 100000, 1000000, float('inf')]),
        values=df['App'], aggfunc='count'
    )
    sns.heatmap(success_matrix, annot=True, fmt='g', cmap='YlOrBr')
    plt.title('Success Matrix: Rating vs Installs', fontsize=14, fontweight='bold')
    plt.xlabel('Installs Category')
    plt.ylabel('Rating Category')
    
    # 8. Top Genres by Popularity
    plt.subplot(3, 3, 8)
    # Extract primary genre
    df['Primary_Genre'] = df['Genres'].str.split(';').str[0]
    top_genres = df.groupby('Primary_Genre')['Installs'].sum().sort_values(ascending=False).head(10)
    plt.bar(range(len(top_genres)), top_genres.values)
    plt.title('Top 10 Genres by Total Installs', fontsize=14, fontweight='bold')
    plt.xticks(range(len(top_genres)), top_genres.index, rotation=45, ha='right')
    plt.ylabel('Total Installs (Log Scale)')
    plt.yscale('log')
    
    # 9. Popularity vs Price
    plt.subplot(3, 3, 9)
    plt.scatter(df['Price'], df['Installs'], alpha=0.3, c=df['Rating'], cmap='plasma')
    plt.colorbar(label='Rating')
    plt.title('Price vs Installs', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Installs (Log Scale)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('visualizations/popularity_metrics/popularity_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Interactive popularity dashboard
    fig = px.scatter(df, x='Rating', y='Installs',
                    color='Type', size='Reviews',
                    hover_data=['App', 'Category', 'Price'],
                    title='App Popularity: Rating vs Installs',
                    log_y=True)
    fig.update_layout(width=1000, height=600)
    fig.write_html('visualizations/popularity_metrics/popularity_dashboard.html')

def create_size_analysis_visualizations(df, analysis_results):
    """Create visualizations for app size analysis"""
    
    print("💾 Creating size analysis visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Size Distribution
    plt.subplot(2, 2, 1)
    df['Size'].hist(bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    plt.title('Distribution of App Sizes', fontsize=14, fontweight='bold')
    plt.xlabel('Size (MB)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Average Size by Category
    plt.subplot(2, 2, 2)
    size_by_category = df.groupby('Category')['Size'].mean().sort_values(ascending=False).head(15)
    bars = plt.bar(range(len(size_by_category)), size_by_category.values, color='orange')
    plt.title('Top 15 Categories by Average Size', fontsize=14, fontweight='bold')
    plt.xticks(range(len(size_by_category)), size_by_category.index, rotation=45, ha='right')
    plt.ylabel('Average Size (MB)')
    
    # 3. Size vs Rating
    plt.subplot(2, 2, 3)
    plt.scatter(df['Size'], df['Rating'], alpha=0.6, c=df['Installs'], cmap='viridis', s=30)
    plt.colorbar(label='Installs')
    plt.title('App Size vs Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Size (MB)')
    plt.ylabel('Rating')
    
    # 4. Size vs Installs
    plt.subplot(2, 2, 4)
    plt.scatter(df['Size'], df['Installs'], alpha=0.6, c=df['Rating'], cmap='plasma', s=30)
    plt.colorbar(label='Rating')
    plt.title('App Size vs Installs', fontsize=14, fontweight='bold')
    plt.xlabel('Size (MB)')
    plt.ylabel('Installs (Log Scale)')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('visualizations/size_analysis/size_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_content_rating_visualizations(df, analysis_results):
    """Create visualizations for content rating analysis"""
    
    print("🔞 Creating content rating visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. Content Rating Distribution
    plt.subplot(2, 2, 1)
    content_counts = df['Content Rating'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(content_counts)))
    plt.pie(content_counts.values, labels=content_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Content Rating Distribution', fontsize=14, fontweight='bold')
    
    # 2. Average Rating by Content Rating
    plt.subplot(2, 2, 2)
    rating_by_content = df.groupby('Content Rating')['Rating'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(rating_by_content)), rating_by_content.values, color='lightcoral')
    plt.title('Average Rating by Content Rating', fontsize=14, fontweight='bold')
    plt.xticks(range(len(rating_by_content)), rating_by_content.index)
    plt.ylabel('Average Rating')
    
    # 3. Installs by Content Rating
    plt.subplot(2, 2, 3)
    installs_by_content = df.groupby('Content Rating')['Installs'].sum().sort_values(ascending=False)
    bars = plt.bar(range(len(installs_by_content)), installs_by_content.values, color='lightgreen')
    plt.title('Total Installs by Content Rating', fontsize=14, fontweight='bold')
    plt.xticks(range(len(installs_by_content)), installs_by_content.index)
    plt.ylabel('Total Installs (Log Scale)')
    plt.yscale('log')
    
    # 4. Price by Content Rating
    plt.subplot(2, 2, 4)
    price_by_content = df.groupby('Content Rating')['Price'].mean().sort_values(ascending=False)
    bars = plt.bar(range(len(price_by_content)), price_by_content.values, color='gold')
    plt.title('Average Price by Content Rating', fontsize=14, fontweight='bold')
    plt.xticks(range(len(price_by_content)), price_by_content.index)
    plt.ylabel('Average Price ($)')
    
    plt.tight_layout()
    plt.savefig('visualizations/content_rating/content_rating_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_visualizations(df, analysis_results):
    """Create visualizations for temporal analysis"""
    
    print("📅 Creating temporal visualizations...")
    
    # Extract year and month from Last Updated
    df['Year'] = df['Last Updated'].dt.year
    df['Month'] = df['Last Updated'].dt.month
    
    plt.figure(figsize=(15, 10))
    
    # 1. Apps Published by Year
    plt.subplot(2, 2, 1)
    apps_by_year = df['Year'].value_counts().sort_index()
    plt.plot(apps_by_year.index, apps_by_year.values, marker='o', linewidth=2, markersize=8)
    plt.title('Number of Apps Updated by Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Apps Updated')
    plt.grid(True, alpha=0.3)
    
    # 2. Average Rating Trend
    plt.subplot(2, 2, 2)
    rating_trend = df.groupby('Year')['Rating'].mean()
    plt.plot(rating_trend.index, rating_trend.values, marker='s', linewidth=2, markersize=8, color='orange')
    plt.title('Average Rating Trend Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.grid(True, alpha=0.3)
    
    # 3. Average Size Trend
    plt.subplot(2, 2, 3)
    size_trend = df.groupby('Year')['Size'].mean()
    plt.plot(size_trend.index, size_trend.values, marker='^', linewidth=2, markersize=8, color='green')
    plt.title('Average App Size Trend Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Size (MB)')
    plt.grid(True, alpha=0.3)
    
    # 4. Price Trend
    plt.subplot(2, 2, 4)
    price_trend = df[df['Type'] == 'Paid'].groupby('Year')['Price'].mean()
    plt.plot(price_trend.index, price_trend.values, marker='d', linewidth=2, markersize=8, color='red')
    plt.title('Average Price Trend for Paid Apps', fontsize=14, fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Price ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/temporal_analysis/temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboards(df, analysis_results):
    """Create comprehensive interactive dashboards"""
    
    print("📊 Creating comprehensive dashboards...")
    
    # 1. Main Interactive Dashboard
    fig = px.scatter(df, 
                    x='Rating', 
                    y='Installs',
                    color='Category',
                    size='Reviews',
                    hover_name='App',
                    hover_data=['Type', 'Price', 'Content Rating'],
                    title='Comprehensive App Market Dashboard',
                    log_y=True,
                    size_max=60)
    fig.update_layout(width=1200, height=700)
    fig.write_html('visualizations/comprehensive/main_dashboard.html')
    
    # 2. Category Performance Dashboard
    category_stats = analysis_results['category_analysis']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('App Count by Category', 'Average Rating by Category', 
                       'Average Installs by Category', 'Average Price by Category'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # App Count
    fig.add_trace(
        go.Bar(x=category_stats['Category'], y=category_stats['App_Count'],
               name='App Count'),
        row=1, col=1
    )
    
    # Average Rating
    fig.add_trace(
        go.Bar(x=category_stats['Category'], y=category_stats['Rating_mean'],
               name='Avg Rating'),
        row=1, col=2
    )
    
    # Average Installs
    fig.add_trace(
        go.Bar(x=category_stats['Category'], y=category_stats['Installs_mean'],
               name='Avg Installs'),
        row=2, col=1
    )
    
    # Average Price
    fig.add_trace(
        go.Bar(x=category_stats['Category'], y=category_stats['Price_mean'],
               name='Avg Price'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Category Performance Dashboard", showlegend=False)
    fig.write_html('visualizations/comprehensive/category_dashboard.html')
    
    # 3. Correlation Matrix Interactive
    numeric_cols = ['Rating', 'Reviews', 'Installs', 'Price', 'Size']
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   color_continuous_scale='RdBu',
                   title='Feature Correlation Matrix',
                   aspect="auto")
    fig.update_layout(width=600, height=600)
    fig.write_html('visualizations/comprehensive/correlation_matrix.html')

def create_app_word_cloud(df):
    """Create word cloud from app names"""
    
    print("☁️ Creating app name word cloud...")
    
    # Combine all app names
    text = ' '.join(df['App'].dropna())
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in App Names', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/comprehensive/app_word_cloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    