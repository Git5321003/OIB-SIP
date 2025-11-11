import pandas as pd
import os
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.analyzer import DataAnalyzer
from src.visualizer import DataVisualizer

def main():
    print("=== Walmart Sales EDA Project ===")
    
    # Create directories
    os.makedirs('reports/figures', exist_ok=True)
    
    # Load data
    loader = DataLoader('data/Walmart.csv')
    data = loader.load_data()
    loader.get_basic_info()
    
    # Clean data
    cleaner = DataCleaner(data)
    cleaned_data = cleaner.clean_data()
    cleaned_data = cleaner.create_derived_features()
    
    # Analyze data
    analyzer = DataAnalyzer(cleaned_data)
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    stats = analyzer.get_descriptive_stats()
    print(stats)
    
    print("\n=== SALES TRENDS ANALYSIS ===")
    sales_trends = analyzer.analyze_sales_trends()
    print("Yearly Sales:")
    print(sales_trends['yearly_sales'])
    
    print("\n=== STORE PERFORMANCE ===")
    store_stats = analyzer.analyze_store_performance()
    print("Top 5 Stores by Average Sales:")
    print(store_stats.nlargest(5, 'Weekly_Sales_mean'))
    
    print("\n=== HOLIDAY IMPACT ===")
    holiday_impact = analyzer.analyze_holiday_impact()
    print(holiday_impact)
    
    print("\n=== CORRELATION ANALYSIS ===")
    correlations = analyzer.correlation_analysis()
    print(correlations)
    
    print("\n=== SEASONAL ANALYSIS ===")
    seasonal_analysis = analyzer.seasonal_analysis()
    print(seasonal_analysis)
    
    # Visualize data
    visualizer = DataVisualizer()
    
    print("\n=== GENERATING VISUALIZATIONS ===")
    visualizer.plot_sales_trend(cleaned_data, 'reports/figures/sales_trends.png')
    visualizer.plot_store_analysis(store_stats, 'reports/figures/store_analysis.png')
    visualizer.plot_correlation_heatmap(correlations, 'reports/figures/correlation_heatmap.png')
    visualizer.plot_boxplots(cleaned_data, 'reports/figures/boxplots.png')
    
    # Generate insights report
    generate_insights_report(cleaned_data, analyzer, store_stats)

def generate_insights_report(data, analyzer, store_stats):
    """Generate key insights from the analysis"""
    print("\n" + "="*50)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*50)
    
    # Overall sales performance
    total_sales = data['Weekly_Sales'].sum()
    avg_sales = data['Weekly_Sales'].mean()
    
    print(f"1. OVERALL PERFORMANCE:")
    print(f"   - Total Sales: ${total_sales:,.2f}")
    print(f"   - Average Weekly Sales per Store: ${avg_sales:,.2f}")
    print(f"   - Number of Stores: {data['Store'].nunique()}")
    print(f"   - Analysis Period: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    
    # Store performance insights
    best_store = store_stats.nlargest(1, 'Weekly_Sales_mean').index[0]
    worst_store = store_stats.nsmallest(1, 'Weekly_Sales_mean').index[0]
    
    print(f"\n2. STORE PERFORMANCE:")
    print(f"   - Best Performing Store: #{best_store}")
    print(f"   - Worst Performing Store: #{worst_store}")
    print(f"   - Performance Variation: {store_stats['Weekly_Sales_std'].std():.2f} (higher = more variability)")
    
    # Holiday impact
    holiday_sales = data[data['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
    non_holiday_sales = data[data['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
    holiday_boost = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100
    
    print(f"\n3. HOLIDAY IMPACT:")
    print(f"   - Sales Increase during Holidays: {holiday_boost:.1f}%")
    print(f"   - Holiday Weeks: {data['Holiday_Flag'].sum()} out of {len(data)} weeks")
    
    # Seasonal patterns
    seasonal_data = data.groupby('Season')['Weekly_Sales'].mean()
    best_season = seasonal_data.idxmax()
    worst_season = seasonal_data.idxmin()
    
    print(f"\n4. SEASONAL PATTERNS:")
    print(f"   - Best Season: {best_season}")
    print(f"   - Worst Season: {worst_season}")
    
    # Economic factors correlation
    correlations = analyzer.correlation_analysis()
    sales_corr = correlations.loc['Weekly_Sales']
    
    print(f"\n5. ECONOMIC FACTORS CORRELATION WITH SALES:")
    for factor, corr in sales_corr.items():
        if factor != 'Weekly_Sales':
            print(f"   - {factor}: {corr:.3f}")
    
    print(f"\n6. RECOMMENDATIONS:")
    print(f"   • Focus inventory planning around holiday periods")
    print(f"   • Investigate underperforming stores (#{worst_store}) for improvement opportunities")
    print(f"   • Optimize staffing and inventory for {best_season} season")
    print(f"   • Monitor economic indicators, especially those with strong sales correlation")
    print(f"   • Consider store-specific strategies based on local economic conditions")

if __name__ == "__main__":
    main()
    