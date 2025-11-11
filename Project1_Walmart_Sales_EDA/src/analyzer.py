import pandas as pd
import numpy as np
from scipy import stats

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    def get_descriptive_stats(self):
        """Calculate descriptive statistics"""
        numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        stats_dict = {}
        for col in numerical_cols:
            stats_dict[col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'skewness': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis()
            }
        
        return pd.DataFrame(stats_dict).T
    
    def analyze_sales_trends(self):
        """Analyze sales trends over time"""
        monthly_sales = self.data.groupby(['Year', 'Month'])['Weekly_Sales'].agg(['mean', 'sum', 'std']).reset_index()
        yearly_sales = self.data.groupby('Year')['Weekly_Sales'].agg(['mean', 'sum', 'std']).reset_index()
        
        return {
            'monthly_sales': monthly_sales,
            'yearly_sales': yearly_sales
        }
    
    def analyze_store_performance(self):
        """Analyze performance across different stores"""
        store_stats = self.data.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'sum', 'std', 'count'],
            'Holiday_Flag': 'mean',
            'Temperature': 'mean',
            'CPI': 'mean',
            'Unemployment': 'mean'
        }).round(2)
        
        store_stats.columns = ['_'.join(col).strip() for col in store_stats.columns.values]
        return store_stats
    
    def analyze_holiday_impact(self):
        """Analyze impact of holidays on sales"""
        holiday_sales = self.data.groupby('Holiday_Flag')['Weekly_Sales'].agg(['mean', 'std', 'count'])
        return holiday_sales
    
    def correlation_analysis(self):
        """Calculate correlations between variables"""
        numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        correlation_matrix = self.data[numerical_cols].corr()
        return correlation_matrix
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        seasonal_stats = self.data.groupby('Season').agg({
            'Weekly_Sales': ['mean', 'sum', 'std'],
            'Store': 'nunique'
        }).round(2)
        return seasonal_stats
    