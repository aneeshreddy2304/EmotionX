# -*- coding: utf-8 -*-
"""
Utility Functions Module
========================

This module provides utility functions for:
- Model saving and loading
- Reproducibility (random seed)
- Device management
- Inference/prediction
"""

import os
import json
import random
import numpy as np
import torch
from typing import Tuple, Dict, List

from transformers import AutoTokenizer

from .config import TrainConfig
from .model import EmotionXHybridModel


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Get the best available compute device.
   
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def check_model_exists(model_dir: str) -> bool:
    """
    Check if model artifacts exist in the specified directory.
    
   
    """
    model_state_path = os.path.join(model_dir, "model_state.pt")
    config_path = os.path.join(model_dir, "train_config.json")
    return os.path.exists(model_state_path) and os.path.exists(config_path)


def save_model_artifacts(
    model: EmotionXHybridModel,
    tokenizer: AutoTokenizer,
    token_to_idf: Dict[str, float],
    label_names: List[str],
    config: TrainConfig,
    model_dir: str
):
    """
    Save all model artifacts to a directory.
   
    """
    os.makedirs(model_dir, exist_ok=True)
    print(f"Saving model artifacts to: {model_dir}")
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(model_dir, "model_state.pt"))
    print("  ✓ model_state.pt")
    
    # Save tokenizer (multiple files)
    tokenizer.save_pretrained(model_dir)
    print("  ✓ tokenizer files")
    
    # Save configuration
    config.save(os.path.join(model_dir, "train_config.json"))
    print("  ✓ train_config.json")
    
    # Save label names
    with open(os.path.join(model_dir, "labels.json"), "w") as f:
        json.dump(label_names, f, indent=2)
    print("  ✓ labels.json")
    
    # Save TF-IDF vocabulary
    with open(os.path.join(model_dir, "token_to_idf.json"), "w") as f:
        json.dump(token_to_idf, f)
    print("  ✓ token_to_idf.json")
    
    # Save manifest (summary of files)
    manifest = {
        "model_state": "model_state.pt",
        "config": "train_config.json",
        "labels": "labels.json",
        "token_to_idf": "token_to_idf.json",
        "backbone_name": config.backbone_name,
        "num_labels": len(label_names)
    }
    with open(os.path.join(model_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("  ✓ manifest.json")
    
    print("Model saved successfully!")


def load_model_artifacts(
    model_dir: str,
    device=None
) -> Tuple[EmotionXHybridModel, AutoTokenizer, Dict, List[str], TrainConfig]:
    """
    Load all model artifacts from a directory.
    
   
    """
    if device is None:
        device = get_device()
    
    print(f"Loading model from: {model_dir}")
    
    # Load configuration
    config = TrainConfig.load(os.path.join(model_dir, "train_config.json"))
    print(f"  Backbone: {config.backbone_name}")
    
    # Load label names
    with open(os.path.join(model_dir, "labels.json"), "r") as f:
        label_names = json.load(f)
    print(f"  Labels: {len(label_names)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Load TF-IDF vocabulary
    with open(os.path.join(model_dir, "token_to_idf.json"), "r") as f:
        token_to_idf = json.load(f)
    
    # Create model architecture
    model = EmotionXHybridModel(
        config=config,
        num_labels=len(label_names),
        token_to_idf=token_to_idf,
        tokenizer=tokenizer
    )
    
    # Load model weights
    state_dict = torch.load(
        os.path.join(model_dir, "model_state.pt"),
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    
    # Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    return model, tokenizer, token_to_idf, label_names, config


def predict_emotions(
    texts: List[str],
    model: EmotionXHybridModel,
    tokenizer: AutoTokenizer,
    label_names: List[str],
    max_length: int,
    device: torch.device
) -> List[Dict]:
    """
    Predict emotions for a list of texts.
    
    
    """
    model.eval()
    
    # Tokenize input texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Run inference
    with torch.no_grad():
        logits = model(
            encodings["input_ids"].to(device),
            encodings["attention_mask"].to(device)
        )
    
    # Convert logits to probabilities
    probs = torch.sigmoid(logits).cpu().numpy()
    
    # Format results
    results = []
    for i, text in enumerate(texts):
        prob_row = probs[i]
        
        # Get labels with probability >= 0.5
        predicted_indices = np.where(prob_row >= 0.5)[0].tolist()
        predicted_labels = [label_names[j] for j in predicted_indices]
        
        # Get top 3 predictions
        top3_idx = np.argsort(prob_row)[::-1][:3]
        top3 = [
            {"label": label_names[j], "score": float(prob_row[j])} 
            for j in top3_idx
        ]
        
        results.append({
            "text": text,
            "probabilities": prob_row.tolist(),
            "predicted_labels": predicted_labels,
            "top3": top3
        })
    
    return results
