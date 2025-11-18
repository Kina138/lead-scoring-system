"""
Unified model trainer - trains and compares all models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.logistic_model import LogisticModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.neural_network_model import NeuralNetworkModel
from src.utils.helpers import print_section, ensure_dir


class ModelTrainer:
    """Train and compare multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def load_data(self):
        """Load and preprocess data"""
        print_section("Loading and Preprocessing Data")
        
        # Load raw data
        loader = DataLoader()
        df = loader.load()
        
        if df is None:
            raise ValueError("Failed to load data")
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Create validation set for Neural Network
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # Save preprocessor
        ensure_dir('models')
        preprocessor.save('models/preprocessor.pkl')
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_nn': X_train_nn,
            'X_val': X_val,
            'y_train_nn': y_train_nn,
            'y_val': y_val,
            'feature_names': X_train.columns.tolist(),
            'preprocessor': preprocessor
        }
    
    def initialize_models(self, input_dim):
        """Initialize all models"""
        print_section("Initializing Models")
        
        self.models = {
            'Logistic Regression': LogisticModel(),
            'Random Forest': RandomForestModel(),
            'Gradient Boosting': GradientBoostingModel(),
            'Neural Network': NeuralNetworkModel(input_dim=input_dim)
        }
        
        print(f"✓ Initialized {len(self.models)} models")
    
    def train_all(self, data):
        """Train all models"""
        print_section("Training All Models")
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print('='*60)
            
            try:
                if name == 'Neural Network':
                    # Use validation set for NN
                    model.train(
                        data['X_train_nn'], 
                        data['y_train_nn'],
                        data['X_val'],
                        data['y_val'],
                        verbose=True
                    )
                else:
                    model.train(data['X_train'], data['y_train'], verbose=True)
                
                # Save model
                model_path = f"models/{name.lower().replace(' ', '_')}.pkl"
                if name == 'Neural Network':
                    model_path = "models/neural_network.h5"
                model.save(model_path)
                
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
    
    def evaluate_all(self, data):
        """Evaluate all trained models"""
        print_section("Evaluating All Models")
        
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print('='*60)
            
            try:
                metrics = model.evaluate(data['X_test'], data['y_test'])
                self.results[name] = metrics
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
                self.results[name] = None
    
    def compare_models(self):
        """Compare all model performances"""
        print_section("Model Comparison")
        
        # Filter out None results
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            print("⚠ No valid results to compare")
            return
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(valid_results).T
        results_df = results_df.sort_values('auc', ascending=False)
        
        print("\nPerformance Summary (sorted by AUC):")
        print("="*80)
        print(results_df.to_string())
        print("="*80)
        
        # Identify best model
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   AUC: {results_df.loc[self.best_model_name, 'auc']:.4f}")
        print(f"   Accuracy: {results_df.loc[self.best_model_name, 'accuracy']:.4f}")
        
        # Save comparison
        ensure_dir('outputs/reports')
        results_df.to_csv('outputs/reports/model_comparison.csv')
        print("\n✓ Comparison saved to outputs/reports/model_comparison.csv")
        
        return results_df
    
    def plot_comparison(self, results_df):
        """Plot model comparison"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'auc']
        titles = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            data = results_df[metric].sort_values(ascending=True)
            colors = ['#2ecc71' if x == data.max() else '#3498db' for x in data]
            
            data.plot(kind='barh', ax=ax, color=colors, alpha=0.8)
            ax.set_xlabel(title)
            ax.set_title(f'{title} Comparison', fontweight='bold', fontsize=12)
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, v in enumerate(data):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        ensure_dir('outputs/visualizations')
        plt.savefig('outputs/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Comparison plot saved to outputs/visualizations/model_comparison.png")
        
        plt.close()
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("\n" + "="*60)
        print("AI LEAD SCORING - MODEL TRAINING PIPELINE".center(60))
        print("="*60 + "\n")
        
        # 1. Load data
        data = self.load_data()
        
        # 2. Initialize models
        self.initialize_models(input_dim=data['X_train'].shape[1])
        
        # 3. Train all models
        self.train_all(data)
        
        # 4. Evaluate all models
        self.evaluate_all(data)
        
        # 5. Compare results
        results_df = self.compare_models()
        
        # 6. Plot comparison
        if results_df is not None:
            self.plot_comparison(results_df)
        
        print_section("Pipeline Complete!")
        print(f"✓ All models trained and saved in 'models/' directory")
        print(f"✓ Results saved in 'outputs/' directory")
        print(f"\n🏆 Best performing model: {self.best_model_name}")


def main():
    """Main execution"""
    trainer = ModelTrainer()
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
