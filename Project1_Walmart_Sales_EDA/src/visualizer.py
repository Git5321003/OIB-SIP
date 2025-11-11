import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

class DataVisualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_sales_trend(self, data, save_path=None):
        """Plot sales trends over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Monthly sales trend
        monthly_data = data.groupby(['Year', 'Month'])['Weekly_Sales'].mean().reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(DAY=1))
        
        axes[0,0].plot(monthly_data['Date'], monthly_data['Weekly_Sales'], marker='o')
        axes[0,0].set_title('Average Monthly Sales Trend')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Average Weekly Sales')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Yearly sales comparison
        yearly_data = data.groupby('Year')['Weekly_Sales'].sum().reset_index()
        axes[0,1].bar(yearly_data['Year'], yearly_data['Weekly_Sales'])
        axes[0,1].set_title('Total Yearly Sales')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Total Sales')
        
        # Seasonal sales pattern
        seasonal_data = data.groupby('Season')['Weekly_Sales'].mean().reset_index()
        axes[1,0].bar(seasonal_data['Season'], seasonal_data['Weekly_Sales'])
        axes[1,0].set_title('Average Sales by Season')
        axes[1,0].set_xlabel('Season')
        axes[1,0].set_ylabel('Average Weekly Sales')
        
        # Holiday vs Non-Holiday sales
        holiday_data = data.groupby('Holiday_Flag')['Weekly_Sales'].mean().reset_index()
        axes[1,1].bar(['Non-Holiday', 'Holiday'], holiday_data['Weekly_Sales'])
        axes[1,1].set_title('Average Sales: Holiday vs Non-Holiday')
        axes[1,1].set_ylabel('Average Weekly Sales')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_store_analysis(self, store_stats, save_path=None):
        """Plot store performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top 10 stores by average sales
        top_stores = store_stats.nlargest(10, 'Weekly_Sales_mean')
        axes[0,0].bar(range(len(top_stores)), top_stores['Weekly_Sales_mean'])
        axes[0,0].set_title('Top 10 Stores by Average Sales')
        axes[0,0].set_xlabel('Store Rank')
        axes[0,0].set_ylabel('Average Weekly Sales')
        
        # Store sales distribution
        axes[0,1].hist(store_stats['Weekly_Sales_mean'], bins=20, edgecolor='black')
        axes[0,1].set_title('Distribution of Store Average Sales')
        axes[0,1].set_xlabel('Average Weekly Sales')
        axes[0,1].set_ylabel('Frequency')
        
        # Sales vs Unemployment by store
        axes[1,0].scatter(store_stats['Unemployment_mean'], store_stats['Weekly_Sales_mean'])
        axes[1,0].set_title('Sales vs Unemployment Rate by Store')
        axes[1,0].set_xlabel('Average Unemployment Rate')
        axes[1,0].set_ylabel('Average Weekly Sales')
        
        # Sales vs CPI by store
        axes[1,1].scatter(store_stats['CPI_mean'], store_stats['Weekly_Sales_mean'])
        axes[1,1].set_title('Sales vs CPI by Store')
        axes[1,1].set_xlabel('Average CPI')
        axes[1,1].set_ylabel('Average Weekly Sales')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, correlation_matrix, save_path=None):
        """Plot correlation heatmap"""
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                  square=True, mask=mask, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_boxplots(self, data, save_path=None):
        """Plot boxplots for key variables"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        variables = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
        
        for var, pos in zip(variables, positions):
            axes[pos[0], pos[1]].boxplot(data[var])
            axes[pos[0], pos[1]].set_title(f'Boxplot of {var}')
            axes[pos[0], pos[1]].set_ylabel(var)
        
        # Remove empty subplot
        fig.delaxes(axes[1,2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        