"""
Data loading and validation module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_section, ensure_dir


class DataLoader:
    """Load and validate lead scoring dataset"""
    
    def __init__(self, data_path='data/raw/Leads.csv'):
        self.data_path = data_path
        self.df = None
        self.metadata = {}
    
    def load(self):
        """Load dataset from CSV"""
        print_section("Loading Dataset")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.df):,} records from {self.data_path}")
            print(f"  Columns: {len(self.df.columns)}")
            
            self._generate_metadata()
            return self.df
            
        except FileNotFoundError:
            print(f"✗ Error: File not found at {self.data_path}")
            print("\n📥 Please download the dataset from:")
            print("   https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset")
            return None
        
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def _generate_metadata(self):
        """Generate dataset metadata"""
        if self.df is None:
            return
        
        self.metadata = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': dict(self.df.dtypes),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': dict(self.df.isnull().sum()),
            'missing_percentage': dict((self.df.isnull().sum() / len(self.df) * 100).round(2))
        }
    
    def get_basic_info(self):
        """Print basic dataset information"""
        if self.df is None:
            print("No data loaded")
            return
        
        print_section("Dataset Overview")
        print(f"Shape: {self.df.shape}")
        print(f"Memory: {self.metadata['memory_usage_mb']:.2f} MB")
        
        print("\nData Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        print("\nFirst few rows:")
        print(self.df.head())
    
    def check_target_distribution(self, target_col='Converted'):
        """Check target variable distribution"""
        if self.df is None or target_col not in self.df.columns:
            print(f"Target column '{target_col}' not found")
            return
        
        print_section("Target Distribution")
        
        counts = self.df[target_col].value_counts()
        percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        print(f"\n{target_col}:")
        for val in counts.index:
            count = counts[val]
            pct = percentages[val]
            label = "Converted" if val == 1 else "Not Converted"
            print(f"  {label} ({val}): {count:,} ({pct:.2f}%)")
        
        # Check for class imbalance
        ratio = percentages.max() / percentages.min()
        if ratio > 2:
            print(f"\n⚠ Class imbalance detected (ratio: {ratio:.2f}:1)")
            print("  Consider using class weights or sampling techniques")
    
    def check_missing_values(self, threshold=50):
        """
        Check for missing values
        
        Args:
            threshold: Percentage threshold to highlight columns
        """
        if self.df is None:
            return
        
        print_section("Missing Values Analysis")
        
        missing_pct = self.metadata['missing_percentage']
        
        # Filter columns with missing values
        missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
        
        if not missing_cols:
            print("✓ No missing values found!")
            return
        
        print(f"\nColumns with missing values: {len(missing_cols)}/{len(self.df.columns)}")
        
        # Sort by percentage
        sorted_missing = sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 columns with highest missing percentage:")
        for col, pct in sorted_missing[:15]:
            status = "⚠ HIGH" if pct > threshold else ""
            print(f"  {col:<40}: {pct:>6.2f}% {status}")
    
    def get_numeric_columns(self):
        """Get list of numeric columns"""
        if self.df is None:
            return []
        return list(self.df.select_dtypes(include=[np.number]).columns)
    
    def get_categorical_columns(self):
        """Get list of categorical columns"""
        if self.df is None:
            return []
        return list(self.df.select_dtypes(include=['object']).columns)
    
    def validate_dataset(self):
        """
        Validate dataset for ML readiness
        
        Returns:
            tuple: (is_valid, issues_list)
        """
        if self.df is None:
            return False, ["Dataset not loaded"]
        
        issues = []
        
        # Check for target variable
        if 'Converted' not in self.df.columns:
            issues.append("Missing target column 'Converted'")
        
        # Check for minimum rows
        if len(self.df) < 1000:
            issues.append(f"Dataset too small: {len(self.df)} rows (minimum 1000 recommended)")
        
        # Check for excessive missing values
        missing_pct = self.df.isnull().sum() / len(self.df) * 100
        high_missing_cols = missing_pct[missing_pct > 70].index.tolist()
        if high_missing_cols:
            issues.append(f"Columns with >70% missing: {len(high_missing_cols)}")
        
        # Check for duplicates
        n_duplicates = self.df.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"Found {n_duplicates} duplicate rows")
        
        is_valid = len(issues) == 0
        
        print_section("Dataset Validation")
        if is_valid:
            print("✓ Dataset is valid and ready for ML")
        else:
            print("✗ Issues found:")
            for issue in issues:
                print(f"  • {issue}")
        
        return is_valid, issues
    
    def get_feature_summary(self):
        """Get summary of features by category"""
        if self.df is None:
            return None
        
        summary = {
            'Identifiers': ['Prospect ID', 'Lead Number'],
            'Demographic': ['Country', 'City', 'Specialization', 'What is your current occupation'],
            'Engagement': ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit'],
            'Communication': ['Do Not Email', 'Do Not Call', 'Last Activity', 'Last Notable Activity'],
            'Source': ['Lead Origin', 'Lead Source'],
            'Target': ['Converted']
        }
        
        return summary


def main():
    """Test data loader"""
    loader = DataLoader()
    df = loader.load()
    
    if df is not None:
        loader.get_basic_info()
        loader.check_target_distribution()
        loader.check_missing_values()
        loader.validate_dataset()


if __name__ == "__main__":
    main()
