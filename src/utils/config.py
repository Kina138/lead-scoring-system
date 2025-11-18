"""
Configuration loader utility
"""
import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path='configs/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file"""
        # Get project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_file = project_root / self.config_path
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠ Config file not found: {config_file}")
            return self._default_config()
        except Exception as e:
            print(f"⚠ Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self):
        """Return default configuration if file not found"""
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
            },
            'models': {
                'random_forest': {'n_estimators': 100},
                'gradient_boosting': {'n_estimators': 100},
            },
            'segmentation': {
                'high_threshold': 0.7,
                'medium_threshold': 0.3,
            }
        }
    
    def get(self, key_path, default=None):
        """
        Get configuration value by dot-notation path
        Example: config.get('models.random_forest.n_estimators')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __repr__(self):
        return f"Config(path='{self.config_path}')"


# Global config instance
config = Config()
