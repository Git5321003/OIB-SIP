"""
Data Preprocessing Module
-----------------------
Handles data loading, cleaning, encoding, and preparation for the housing price prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    A class for preprocessing housing data for price prediction.
    
    Attributes:
        df (pd.DataFrame): Original dataset
        df_model (pd.DataFrame): Processed dataset ready for modeling
        scaler (StandardScaler): Fitted scaler for feature normalization
        X_train, X_test, y_train, y_test: Train-test split data
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.df = None
        self.df_model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_cols = None
        self.numerical_cols = None
        
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Dataset shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def inspect_data(self):
        """
        Perform initial data inspection.
        
        Returns:
            dict: Dictionary containing data inspection results
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        inspection_results = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'basic_stats': self.df.describe()
        }
        
        # Identify categorical and numerical columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        inspection_results['categorical_columns'] = self.categorical_cols
        inspection_results['numerical_columns'] = self.numerical_cols
        
        logger.info("Data inspection completed")
        return inspection_results
    
    def clean_data(self):
        """
        Clean the dataset by handling missing values and duplicates.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            # For this dataset, we assume no missing values based on initial inspection
        else:
            logger.info("No missing values found")
    
    def encode_categorical_variables(self):
        """
        Encode categorical variables for modeling.
        """
        if self.df is None:
            raise ValueError("No data loaded. Please call load_data() first.")
            
        self.df_model = self.df.copy()
        
        # Binary categorical variables
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea']
        
        for col in binary_cols:
            if col in self.df_model.columns:
                self.df_model[col] = self.df_model[col].map({'yes': 1, 'no': 0})
                logger.info(f"Encoded {col}: yes->1, no->0")
        
        # Furnishing status (ordinal encoding)
        if 'furnishingstatus' in self.df_model.columns:
            furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
            self.df_model['furnishingstatus'] = self.df_model['furnishingstatus'].map(furnishing_map)
            logger.info(f"Encoded furnishingstatus: {furnishing_map}")
        
        logger.info("Categorical variables encoded successfully")
    
    def prepare_features_target(self, target_column='price'):
        """
        Prepare features and target variable for modeling.
        
        Args:
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) features and target
        """
        if self.df_model is None:
            raise ValueError("Data not processed. Please call encode_categorical_variables() first.")
            
        if target_column not in self.df_model.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
            
        X = self.df_model.drop(target_column, axis=1)
        y = self.df_model[target_column]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        logger.info(f"Training set: {self.X_train.shape}, {self.y_train.shape}")
        logger.info(f"Testing set: {self.X_test.shape}, {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """
        Scale features using StandardScaler.
        
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not split. Please call split_data() first.")
            
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info("Features scaled successfully")
        return X_train_scaled, X_test_scaled
    
    def get_feature_names(self):
        """
        Get feature names after preprocessing.
        
        Returns:
            list: List of feature names
        """
        if self.df_model is None:
            raise ValueError("Data not processed. Please call encode_categorical_variables() first.")
            
        return [col for col in self.df_model.columns if col != 'price']
    
    def full_preprocessing_pipeline(self, file_path, test_size=0.2, random_state=42):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to the data file
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        """
        logger.info("Starting full preprocessing pipeline...")
        
        # Step 1: Load data
        self.load_data(file_path)
        
        # Step 2: Inspect data
        self.inspect_data()
        
        # Step 3: Clean data
        self.clean_data()
        
        # Step 4: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 5: Prepare features and target
        X, y = self.prepare_features_target()
        
        # Step 6: Split data
        self.split_data(X, y, test_size, random_state)
        
        # Step 7: Scale features
        X_train_scaled, X_test_scaled = self.scale_features()
        
        # Step 8: Get feature names
        feature_names = self.get_feature_names()
        
        logger.info("Full preprocessing pipeline completed successfully")
        
        return X_train_scaled, X_test_scaled, self.y_train, self.y_test, feature_names


# Utility functions
def create_sample_property():
    """
    Create a sample property for prediction demonstration.
    
    Returns:
        dict: Sample property features
    """
    sample_property = {
        'area': 7500,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'mainroad': 1,
        'guestroom': 0,
        'basement': 1,
        'hotwaterheating': 0,
        'airconditioning': 1,
        'parking': 2,
        'prefarea': 1,
        'furnishingstatus': 1  # semi-furnished
    }
    return sample_property


def create_multiple_sample_properties():
    """
    Create multiple sample properties for comparison.
    
    Returns:
        list: List of sample property dictionaries
    """
    sample_properties = [
        {
            'name': 'Premium Property',
            'area': 8000,
            'bedrooms': 4,
            'bathrooms': 3,
            'stories': 3,
            'mainroad': 1,
            'guestroom': 1,
            'basement': 1,
            'hotwaterheating': 0,
            'airconditioning': 1,
            'parking': 2,
            'prefarea': 1,
            'furnishingstatus': 2  # furnished
        },
        {
            'name': 'Standard Property',
            'area': 6000,
            'bedrooms': 3,
            'bathrooms': 2,
            'stories': 2,
            'mainroad': 1,
            'guestroom': 0,
            'basement': 0,
            'hotwaterheating': 0,
            'airconditioning': 1,
            'parking': 1,
            'prefarea': 0,
            'furnishingstatus': 1  # semi-furnished
        },
        {
            'name': 'Budget Property',
            'area': 4000,
            'bedrooms': 2,
            'bathrooms': 1,
            'stories': 1,
            'mainroad': 0,
            'guestroom': 0,
            'basement': 0,
            'hotwaterheating': 0,
            'airconditioning': 0,
            'parking': 0,
            'prefarea': 0,
            'furnishingstatus': 0  # unfurnished
        }
    ]
    return sample_properties
