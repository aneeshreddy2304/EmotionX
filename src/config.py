"""
Configuration Module
====================
This module defines all configuration settings and hyperparameters
for the EmotionX-Hybrid model.
"""

import os
import json
from dataclasses import dataclass, asdict


# =============================================================================
# Directory Paths
# =============================================================================

# Get the project root directory (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model directories
PRETRAINED_DIR = os.path.join(BASE_DIR, "models", "pretrained")  # Pre-trained model
TRAINED_DIR = os.path.join(BASE_DIR, "models", "trained")        # Newly trained model

# Output directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")  # Evaluation outputs
DATA_DIR = os.path.join(BASE_DIR, "data")        # Data cache


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """
    Training configuration dataclass.
    """
    
    # Model architecture
    backbone_name: str = "microsoft/deberta-v3-large"    # Transformer backbone
    max_length: int = 96                    # Max sequence length
    lstm_hidden_size: int = 256             # BiLSTM hidden size
    lstm_num_layers: int = 1                # Number of LSTM layers
    attention_heads: int = 4                # Multi-head attention heads
    dropout: float = 0.2                    # Dropout rate
    projection_dim: int = 128               # Projection dimension
    
    # Training parameters
    batch_size: int = 32                   # Batch size
    num_epochs: int = 5                    # Training epochs
    learning_rate: float = 2e-5             # Learning rate
    weight_decay: float = 0.01              # L2 regularization
    warmup_ratio: float = 0.1               # LR warmup ratio
    early_stopping_patience: int = 2        # Early stopping patience
    
    # Loss function parameters
    focal_gamma: float = 2.0                # Focal loss gamma
    focal_alpha: float = 0.25               # Focal loss alpha
    
    # Contrastive learning (disabled by default)
    use_contrastive: bool = False
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls(**json.load(f))
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
