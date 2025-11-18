"""
Logistic Regression model for lead scoring
"""
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel
from src.utils.config import config


class LogisticModel(BaseModel):
    """Logistic Regression classifier"""
    
    def __init__(self):
        super().__init__("Logistic Regression")
        self.params = config.get('models.logistic_regression', {})
        self.build_model()
    
    def build_model(self):
        """Build logistic regression model"""
        self.model = LogisticRegression(
            max_iter=self.params.get('max_iter', 1000),
            solver=self.params.get('solver', 'lbfgs'),
            C=self.params.get('C', 1.0),
            random_state=42,
            n_jobs=-1
        )
        print(f"✓ {self.model_name} initialized")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Print training info
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.model_name}...")
            print('='*60)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        if verbose:
            print(f"✓ {self.model_name} training complete")
    
    def get_coefficients(self, feature_names):
        """
        Get model coefficients
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame: Feature coefficients
        """
        if not self.is_fitted:
            print(f"⚠ {self.model_name} is not fitted yet")
            return None
        
        import pandas as pd
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        return coef_df


def main():
    """Test Logistic Regression model"""
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load()
    
    if df is not None:
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Train model
        model = LogisticModel()
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Save
        model.save('models/logistic_regression.pkl')


if __name__ == "__main__":
    main()
