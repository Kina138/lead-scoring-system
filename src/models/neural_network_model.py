"""
Neural Network (MLP) model for lead scoring
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import BaseModel
from src.utils.config import config
from src.utils.helpers import print_section


class NeuralNetworkModel(BaseModel):
    """Neural Network (Multi-Layer Perceptron) classifier"""
    
    def __init__(self, input_dim=None):
        super().__init__("Neural Network")
        self.params = config.get('models.neural_network', {})
        self.input_dim = input_dim
        self.history = None
        
        if input_dim is not None:
            self.build_model()
    
    def build_model(self):
        """Build neural network architecture"""
        if self.input_dim is None:
            raise ValueError("input_dim must be specified")
        
        # Get architecture params
        layer_sizes = self.params.get('layers', [128, 64, 32])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        learning_rate = self.params.get('learning_rate', 0.001)
        
        # Build model
        self.model = keras.Sequential()
        
        # Input layer + first hidden layer
        self.model.add(layers.Dense(
            layer_sizes[0], 
            activation='relu', 
            input_dim=self.input_dim
        ))
        self.model.add(layers.Dropout(dropout_rate))
        
        # Additional hidden layers
        for size in layer_sizes[1:]:
            self.model.add(layers.Dense(size, activation='relu'))
            self.model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        self.model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ {self.model_name} initialized")
        print(f"  Architecture: {layer_sizes}")
        print(f"  Total parameters: {self.model.count_params():,}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training info
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.model_name}...")
            print('='*60)
        
        # Get training params
        epochs = self.params.get('epochs', 50)
        batch_size = self.params.get('batch_size', 32)
        patience = self.params.get('early_stopping_patience', 5)
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1 if verbose else 0
        )
        
        self.is_fitted = True
        
        if verbose:
            print(f"✓ {self.model_name} training complete")
            final_loss = self.history.history['loss'][-1]
            final_acc = self.history.history['accuracy'][-1]
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Final accuracy: {final_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted yet")
        
        probas = self.model.predict(X, verbose=0).flatten()
        return (probas >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} is not fitted yet")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, filepath):
        """Save trained model"""
        if not self.is_fitted:
            print(f"⚠ {self.model_name} is not fitted, cannot save")
            return
        
        # Save as .h5 or SavedModel format
        if filepath.endswith('.h5'):
            self.model.save(filepath)
        else:
            self.model.save(filepath + '.h5')
        
        print(f"✓ {self.model_name} saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("⚠ No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/nn_training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history plot saved")


def main():
    """Test Neural Network model"""
    from src.data.data_loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from sklearn.model_selection import train_test_split
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load()
    
    if df is not None:
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
        
        # Create validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # Train model
        model = NeuralNetworkModel(input_dim=X_train.shape[1])
        model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        
        # Plot training history
        model.plot_training_history()
        
        # Save
        model.save('models/neural_network')


if __name__ == "__main__":
    main()
