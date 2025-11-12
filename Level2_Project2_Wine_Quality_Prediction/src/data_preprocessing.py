import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_clean(self, filepath):
        """Load data and handle basic cleaning"""
        df = pd.read_csv(filepath)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values if any
        df = df.dropna()
        
        self.feature_names = df.columns.tolist()
        return df
    
    def prepare_features(self, df, target_col='quality', exclude_cols=['Id']):
        """Prepare features and target variable"""
        # Exclude specified columns
        features = [col for col in df.columns if col not in [target_col] + exclude_cols]
        
        X = df[features]
        y = df[target_col]
        
        # Convert quality to binary classification
        y_binary = (y > 5).astype(int)
        
        return X, y_binary, features
    
    def split_and_scale(self, X, y, test_size=0.2, random_state=42):
        """Split data and scale features"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
    