"""
SHAP-based explainability module
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.helpers import print_section, ensure_dir


class SHAPExplainer:
    """SHAP explainability for lead scoring models"""
    
    def __init__(self, model, X_background, model_type='tree'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP
            model_type: 'tree' or 'kernel'
        """
        self.model = model
        self.X_background = X_background
        self.model_type = model_type
        
        # Initialize appropriate explainer
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_background, 100)
            )
        
        print(f"✓ SHAP {model_type} explainer initialized")
    
    def explain_predictions(self, X):
        """Calculate SHAP values for predictions"""
        shap_values = self.explainer.shap_values(X)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class
        
        return shap_values
    
    def plot_summary(self, X, feature_names, max_display=20):
        """Create SHAP summary plot"""
        print_section("Generating SHAP Summary Plot")
        
        shap_values = self.explain_predictions(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        ensure_dir('outputs/visualizations')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ SHAP summary plot saved")
    
    def plot_waterfall(self, X_instance, feature_names, instance_idx=0):
        """Create SHAP waterfall plot for single prediction"""
        shap_values = self.explain_predictions(X_instance)
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=self.explainer.expected_value,
            data=X_instance[instance_idx],
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        
        plt.tight_layout()
        plt.savefig(f'outputs/visualizations/shap_waterfall_{instance_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Waterfall plot saved for instance {instance_idx}")
    
    def get_feature_importance(self, X, feature_names):
        """Get feature importance DataFrame"""
        shap_values = self.explain_predictions(X)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df, top_n=15):
        """Plot top N important features"""
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Features by SHAP Importance', 
                 fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/shap_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved")
