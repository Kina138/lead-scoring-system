"""
Data preprocessing pipeline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_section, save_pickle, load_pickle
from src.utils.config import config


class DataPreprocessor:
    """Preprocessing pipeline for lead scoring data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_col = 'Converted'
        
        # Get config settings
        self.test_size = config.get('data.test_size', 0.2)
        self.random_state = config.get('data.random_state', 42)
        self.drop_cols = config.get('preprocessing.drop_columns', ['Prospect ID', 'Lead Number'])
    
    def fit_transform(self, df):
        """
        Fit preprocessor and transform data
        
        Args:
            df: Raw DataFrame
        
        Returns:
            DataFrame: Preprocessed data
        """
        print_section("Data Preprocessing")
        
        df = df.copy()
        
        # 1. Remove identifiers
        df = self._drop_columns(df)
        
        # 2. Handle missing values
        df = self._handle_missing_values(df)
        
        # 3. Feature engineering
        df = self._engineer_features(df)
        
        # 4. Encode categorical variables
        df = self._encode_categorical(df)
        
        # 5. Store feature names
        self.feature_names = [col for col in df.columns if col != self.target_col]
        
        print(f"✓ Preprocessing complete")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Samples: {len(df)}")
        
        return df
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        
        Args:
            df: Raw DataFrame
        
        Returns:
            DataFrame: Preprocessed data
        """
        df = df.copy()
        
        # Apply same transformations
        df = self._drop_columns(df)
        df = self._handle_missing_values(df)
        df = self._engineer_features(df)
        df = self._encode_categorical_transform(df)
        
        return df
    
    def _drop_columns(self, df):
        """Drop unnecessary columns"""
        cols_to_drop = [col for col in self.drop_cols if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"✓ Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        print("\nHandling missing values...")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from lists
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        
        # Numeric: fill with median
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Categorical: fill with mode or 'Unknown'
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        remaining_missing = df.isnull().sum().sum()
        print(f"✓ Missing values handled (remaining: {remaining_missing})")
        
        return df
    
    def _engineer_features(self, df):
        """Create new features"""
        print("\nEngineering features...")
        
        # Engagement score
        if all(col in df.columns for col in ['Total Time Spent on Website', 'TotalVisits']):
            df['engagement_score'] = (
                (df['Total Time Spent on Website'] / 100) + 
                (df['TotalVisits'] * 2)
            )
            print("  • Created engagement_score")
        
        # High engagement flag
        if 'Total Time Spent on Website' in df.columns:
            df['is_high_engagement'] = (df['Total Time Spent on Website'] > 500).astype(int)
            print("  • Created is_high_engagement")
        
        # Frequent visitor flag
        if 'TotalVisits' in df.columns:
            df['is_frequent_visitor'] = (df['TotalVisits'] > 3).astype(int)
            print("  • Created is_frequent_visitor")
        
        # Time bins
        if 'Total Time Spent on Website' in df.columns:
            df['time_category'] = pd.cut(
                df['Total Time Spent on Website'],
                bins=[0, 100, 500, 1000, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            ).astype(str)
            print("  • Created time_category")
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables (fit and transform)"""
        print("\nEncoding categorical variables...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from encoding
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✓ Encoded {len(categorical_cols)} categorical columns")
        
        return df
    
    def _encode_categorical_transform(self, df):
        """Encode categorical variables using fitted encoders"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        
        return df
    
    def split_data(self, df, scale=True):
        """
        Split data into train and test sets
        
        Args:
            df: Preprocessed DataFrame
            scale: Whether to scale features
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print_section("Splitting Data")
        
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Scale features
        if scale:
            print("\nScaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            print("✓ Features scaled")
        
        return X_train, X_test, y_train, y_test
    
    def save(self, filepath):
        """Save preprocessor"""
        save_pickle({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_col': self.target_col
        }, filepath)
    
    def load(self, filepath):
        """Load preprocessor"""
        data = load_pickle(filepath)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.target_col = data['target_col']
    
    def get_feature_info(self):
        """Get information about processed features"""
        print_section("Feature Information")
        print(f"Total features: {len(self.feature_names)}")
        print(f"\nFeature names:")
        for i, name in enumerate(self.feature_names, 1):
            print(f"  {i}. {name}")


def main():
    """Test preprocessor"""
    from src.data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load()
    
    if df is not None:
        # Preprocess
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        
        # Split
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        print(f"\nFinal shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Save preprocessor
        preprocessor.save('models/preprocessor.pkl')


if __name__ == "__main__":
    main()
