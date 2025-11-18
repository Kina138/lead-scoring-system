"""
Random Forest model for lead scoring
"""
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel
from src.utils.config import config


class RandomForestModel(BaseModel):
    """Random Forest classifier"""
    
    def __init__(self):
        super().__init__("Random Forest")
        self.params = config.get('models.random_forest', {})
        self.build_model()
    
    def build_model(self):
        """Build random forest model"""
        self.model = RandomForestClassifier(
            n_estimators=self.params.get('n_estimators', 100),
            max_depth=self.params.get('max_depth', 10),
            min_samples_split=self.params.get('min_samples_split', 5),
            min_samples_leaf=self.params.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=self.params.get('n_jobs', -1)
        )
        print(f"✓ {self.model_name} initialized")
    
    def train(self, X_train, y_train, verbose=True):
        """
        Train random forest model
        
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
            print(f"  Number of trees: {self.model.n_estimators}")
            print(f"  Max depth: {self.model.max_depth}")


def main():
    """Test Random Forest model"""
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
        model = RandomForestModel()
        model.train(X_train, y_train)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Feature importance
        feature_importance = model.get_feature_importance(X_train.columns)
        if feature_importance is not None:
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
        
        # Save
        model.save('models/random_forest.pkl')


if __name__ == "__main__":
    main()
