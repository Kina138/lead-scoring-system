"""
Base model class for all ML models
"""
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_metrics, save_model


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix
        
        Returns:
            array: Predicted classes (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            array: Probability of positive class
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # For models without predict_proba
            return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, return_dict=True):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            return_dict: If True, return as dictionary
        
        Returns:
            dict or None: Evaluation metrics
        """
        if not self.is_fitted:
            print(f"⚠ {self.model_name} is not fitted yet")
            return None
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Print metrics
        print_metrics(metrics, title=f"{self.model_name} Performance")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                0       1")
        print(f"Actual  0    {cm[0][0]:>5}  {cm[0][1]:>5}")
        print(f"        1    {cm[1][0]:>5}  {cm[1][1]:>5}")
        
        if return_dict:
            return metrics
    
    def save(self, filepath):
        """Save trained model"""
        if not self.is_fitted:
            print(f"⚠ {self.model_name} is not fitted, cannot save")
            return
        
        save_model(self.model, filepath)
        print(f"✓ {self.model_name} saved to {filepath}")
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance (if available)
        
        Args:
            feature_names: List of feature names
        
        Returns:
            DataFrame: Feature importances
        """
        if not hasattr(self.model, 'feature_importances_'):
            print(f"⚠ {self.model_name} does not support feature importance")
            return None
        
        import pandas as pd
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def __repr__(self):
        fitted_status = "Fitted" if self.is_fitted else "Not Fitted"
        return f"{self.model_name} ({fitted_status})"
