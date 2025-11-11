import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Load the Walmart sales dataset"""
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_basic_info(self):
        """Get basic information about the dataset"""
        if self.data is not None:
            print("Dataset Info:")
            print(self.data.info())
            print("\nFirst 5 rows:")
            print(self.data.head())
            print("\nDataset Shape:", self.data.shape)
        else:
            print("No data loaded. Please load data first.")