"""
Helper functions and utilities
"""
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(obj, filepath):
    """Save object as pickle file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✓ Saved to {filepath}")


def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"✓ Loaded from {filepath}")
    return obj


def save_model(model, filepath):
    """Save ML model using joblib"""
    ensure_dir(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath):
    """Load ML model using joblib"""
    model = joblib.load(filepath)
    print(f"✓ Model loaded from {filepath}")
    return model


def save_dataframe(df, filepath, index=False):
    """Save pandas DataFrame to CSV"""
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=index)
    print(f"✓ DataFrame saved to {filepath}")


def load_dataframe(filepath):
    """Load pandas DataFrame from CSV"""
    df = pd.read_csv(filepath)
    print(f"✓ DataFrame loaded from {filepath} | Shape: {df.shape}")
    return df


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title, char='=', width=60):
    """Print formatted section header"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")


def print_metrics(metrics_dict, title="Model Metrics"):
    """Print metrics in formatted table"""
    print_section(title)
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric:<20}: {value:.4f}")
        else:
            print(f"{metric:<20}: {value}")
    print('=' * 60)


def classify_segment(probability, high_threshold=0.25, low_threshold=0.08):
    """
    Classify lead into segment based on conversion probability
    
    Args:
        probability: Conversion probability (0-1)
        high_threshold: Threshold for high priority
        low_threshold: Threshold for low priority
    
    Returns:
        str: 'High', 'Medium', or 'Low'
    """
    if probability >= high_threshold:
        return 'High'
    elif probability >= low_threshold:
        return 'Medium'
    else:
        return 'Low'


def calculate_engagement_score(row):
    """
    Calculate composite engagement score from multiple features
    
    Args:
        row: DataFrame row with engagement features
    
    Returns:
        float: Engagement score
    """
    score = 0
    
    # Time spent (normalized)
    if 'Total Time Spent on Website' in row:
        time_score = min(row['Total Time Spent on Website'] / 1000, 5)
        score += time_score
    
    # Visit frequency
    if 'TotalVisits' in row:
        visit_score = min(row['TotalVisits'] / 5, 3)
        score += visit_score
    
    # Page views
    if 'Page Views Per Visit' in row:
        page_score = min(row['Page Views Per Visit'] / 3, 2)
        score += page_score
    
    return round(score, 2)


def format_dataframe_for_display(df, max_rows=100, max_cols=20):
    """Format DataFrame for display with limited rows/columns"""
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)
    pd.set_option('display.width', 1000)
    return df


def validate_csv_file(filepath, required_columns=None):
    """
    Validate CSV file exists and has required columns
    
    Args:
        filepath: Path to CSV file
        required_columns: List of required column names
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    try:
        df = pd.read_csv(filepath, nrows=1)
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
        
        return True, "Valid CSV file"
    
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"


def memory_usage_report(df):
    """Generate memory usage report for DataFrame"""
    print_section("Memory Usage Report")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Total memory: {memory_mb:.2f} MB")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    
    print("\nTop 10 memory-intensive columns:")
    memory_by_col = df.memory_usage(deep=True).sort_values(ascending=False).head(10)
    for col, mem in memory_by_col.items():
        print(f"  {col:<30}: {mem/1024**2:.2f} MB")


def get_file_size_mb(filepath):
    """Get file size in megabytes"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


if __name__ == "__main__":
    # Test utilities
    print_section("Testing Utilities")
    print(f"Project root: {get_project_root()}")
    print(f"Timestamp: {get_timestamp()}")
    
    # Test segmentation
    test_probs = [0.85, 0.55, 0.25]
    for prob in test_probs:
        segment = classify_segment(prob)
        print(f"Probability {prob:.2f} → Segment: {segment}")
