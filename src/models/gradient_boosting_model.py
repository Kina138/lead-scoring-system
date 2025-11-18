"""
Gradient Boosting model for lead scoring
"""
from sklearn.ensemble import GradientBoostingClassifier
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel
from src.utils.config import config


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier"""
    
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.params = config.get('models.gradient_boosting', {})
        self.build_model()
    
    def build_model(self):
        """Build gradient boosting model"""
        self.model = GradientBoostingClassifier(
            n_estimators=self.params.get('n_estimators', 100),
            learning_rate=self.params.get('learning_rate', 0.1),
            max_depth=self.params.get('max_depth', 5),
            subsample=self.params.get('subsample', 0.8),
            random_state=42
        )
        print(f"✓ {self.model_name} initialized")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train gradient boosting model
        
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
            print(f"  Number of estimators: {self.model.n_estimators}")
            print(f"  Learning rate: {self.model.learning_rate}")


def main():
    """Test Gradient Boosting model"""
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
        model = GradientBoostingModel()
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Feature importance
        feature_importance = model.get_feature_importance(X_train.columns)
        if feature_importance is not None:
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
        
        # Save
        model.save('models/gradient_boosting.pkl')


if __name__ == "__main__":
    main()
