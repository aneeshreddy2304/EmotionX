
"""
Data Loading Module
===================

This module handles data loading, preprocessing, and PyTorch Dataset
creation for the GoEmotions dataset.

The GoEmotions dataset contains ~58K Reddit comments labeled with
27 emotion categories (plus neutral, which we exclude).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_go_emotions():
    """
    Load the GoEmotions dataset from HuggingFace Datasets.
    
    The dataset is automatically downloaded and cached locally.
    Reference:
        Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020
    """
    print("Loading GoEmotions dataset from HuggingFace...")
    return load_dataset("go_emotions")


def prepare_splits(raw_dataset) -> Tuple:
    """
    Prepare train/validation/test splits with multi-hot label encoding.
    """
    train_split = raw_dataset["train"]
    test_split = raw_dataset["test"]
    
    # Get all label names from dataset metadata
    all_labels = train_split.features["labels"].feature.names
    
    # Remove 'neutral' label - we only predict emotional states
    label_names = [lbl for lbl in all_labels if lbl != "neutral"]
    num_labels = len(label_names)  # 27 labels
    
    # Create mapping from label name to index
    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}
    
    def labels_to_multihot(example_labels):
        """Convert list of label indices to multi-hot vector."""
        vec = np.zeros(num_labels, dtype=np.float32)
        for idx in example_labels:
            lbl = all_labels[idx]
            if lbl != "neutral":
                vec[label_to_idx[lbl]] = 1.0
        return vec
    
    print("Processing training data...")
    
    # Process training data
    train_texts_full = []
    train_labels_full = []
    for i in range(len(train_split)):
        ex = train_split[int(i)]
        train_texts_full.append(ex["text"])
        train_labels_full.append(labels_to_multihot(ex["labels"]))
    
    # Process test data
    test_texts = []
    test_labels = []
    for i in range(len(test_split)):
        ex = test_split[int(i)]
        test_texts.append(ex["text"])
        test_labels.append(labels_to_multihot(ex["labels"]))
    
    # Split training into train/validation (90/10)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts_full, train_labels_full,
        test_size=0.1,      # 10% for validation
        random_state=42,    # For reproducibility
        shuffle=True
    )
    
    # Convert to numpy arrays
    train_labels = np.stack(train_labels)
    val_labels = np.stack(val_labels)
    test_labels = np.stack(test_labels)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_texts):,} examples")
    print(f"  Val:   {len(val_texts):,} examples")
    print(f"  Test:  {len(test_texts):,} examples")
    print(f"  Labels: {num_labels}")
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_names


def build_tokenizer_and_tfidf(backbone_name: str, train_texts: List[str]):
    """
    Build HuggingFace tokenizer and compute TF-IDF vocabulary.
    
    """
    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    
    print("Fitting TF-IDF vectorizer on training corpus...")
    
    # Create TF-IDF vectorizer using the same tokenizer
    tfidf = TfidfVectorizer(
        tokenizer=tokenizer.tokenize,  # Use HuggingFace tokenizer
        lowercase=False,               # Don't lowercase (tokenizer handles this)
        token_pattern=None,            # Disable default pattern
        max_features=30000             # Limit vocabulary size
    )
    
    # Fit on training texts (we only need the IDF values)
    _ = tfidf.fit_transform(train_texts)
    
    # Extract token-to-IDF mapping
    token_to_idf = {
        token: float(tfidf.idf_[idx]) 
        for token, idx in tfidf.vocabulary_.items()
    }
    
    print(f"TF-IDF vocabulary size: {len(token_to_idf):,} tokens")
    
    return tokenizer, token_to_idf


class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset for GoEmotions.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def create_collate_fn(tokenizer, max_length: int):
    """
    Create a collate function for DataLoader.
   
    """
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        
        # Tokenize with padding to max length in batch
        encodings = tokenizer(
            texts,
            padding=True,           # Pad to longest in batch
            truncation=True,        # Truncate if > max_length
            max_length=max_length,
            return_tensors="pt"     # Return PyTorch tensors
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }
    
    return collate_fn
