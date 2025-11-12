import pandas as pd
import numpy as np
import re

def clean_dataset(file_path):
    """
    Clean and preprocess the Google Play Store dataset
    
    Parameters:
    file_path (str): Path to the raw CSV file
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    
    # Load data
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Clean data types
    df = clean_data_types(df)
    
    # Feature engineering
    df = create_new_features(df)
    
    print(f"Final cleaned dataset shape: {df.shape}")
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    
    # Fill missing ratings with median of their category
    df['Rating'] = df.groupby('Category')['Rating'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # For remaining missing ratings, use overall median
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    
    # Fill missing size with median
    df['Size'] = df['Size'].fillna(df['Size'].median())
    
    # Fill missing Android version with mode
    df['Android Ver'] = df['Android Ver'].fillna(df['Android Ver'].mode()[0])
    
    return df

def clean_data_types(df):
    """Convert columns to proper data types"""
    
    # Clean Reviews column
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Clean Size column
    df['Size'] = df['Size'].apply(convert_size_to_mb)
    
    # Clean Installs column
    df['Installs'] = df['Installs'].apply(clean_installs)
    
    # Clean Price column
    df['Price'] = df['Price'].apply(clean_price)
    
    # Convert Last Updated to datetime
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')
    
    return df

def convert_size_to_mb(size):
    """Convert size string to megabytes"""
    if pd.isna(size) or size == 'Varies with device':
        return np.nan
    
    size = str(size).upper()
    if 'M' in size:
        return float(re.sub('[^0-9.]', '', size))
    elif 'K' in size:
        return float(re.sub('[^0-9.]', '', size)) / 1024
    else:
        return np.nan

def clean_installs(installs):
    """Convert installs string to numeric"""
    if pd.isna(installs):
        return 0
    
    installs = str(installs).replace(',', '').replace('+', '')
    if installs == 'Free':
        return 0
    
    try:
        return int(installs)
    except:
        return 0

def clean_price(price):
    """Convert price string to numeric"""
    if pd.isna(price) or price == '0':
        return 0.0
    
    price = str(price).replace('$', '')
    try:
        return float(price)
    except:
        return 0.0

def create_new_features(df):
    """Create new features for analysis"""
    
    # App age (days since last update)
    df['Days_Since_Update'] = (pd.Timestamp.now() - df['Last Updated']).dt.days
    
    # Size categories
    df['Size_Category'] = pd.cut(df['Size'], 
                               bins=[0, 10, 50, 100, float('inf')],
                               labels=['Small (<10MB)', 'Medium (10-50MB)', 
                                      'Large (50-100MB)', 'Very Large (>100MB)'])
    
    # Install categories
    df['Install_Category'] = pd.cut(df['Installs'],
                                  bins=[0, 1000, 10000, 100000, 1000000, 10000000, float('inf')],
                                  labels=['0-1K', '1K-10K', '10K-100K', '100K-1M', '1M-10M', '10M+'])
    
    # Free vs Paid
    df['Is_Free'] = df['Type'] == 'Free'
    
    return df
