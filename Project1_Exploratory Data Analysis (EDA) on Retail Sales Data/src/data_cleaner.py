import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None
    
    def clean_data(self):
        """Perform data cleaning operations"""
        df = self.data.copy()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        
        # Check for missing values
        print("Missing values before cleaning:")
        print(df.isnull().sum())
        
        # Handle missing values if any
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Check for outliers in numerical columns
        numerical_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        self.cleaned_data = df
        print(f"Data cleaned. Remaining rows: {self.cleaned_data.shape[0]}")
        return self.cleaned_data
    
    def create_derived_features(self):
        """Create additional features for analysis"""
        df = self.cleaned_data.copy()
        
        # Extract date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Week'] = df['Date'].dt.isocalendar().week
        
        # Create season feature
        df['Season'] = df['Month'].apply(self._get_season)
        
        # Create sales categories
        df['Sales_Category'] = pd.cut(df['Weekly_Sales'], 
                                    bins=[0, 1000000, 2000000, 3000000, float('inf')],
                                    labels=['Low', 'Medium', 'High', 'Very High'])
        
        self.cleaned_data = df
        return self.cleaned_data
    
    def _get_season(self, month):
        """Helper function to determine season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'