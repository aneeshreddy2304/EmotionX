
"""
Visualization Module
====================

This module provides functions for creating evaluation visualizations:
- Confusion matrices
- Precision-recall curves
- Per-label F1 score charts
- Label distribution plots
- Training loss curves

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
from typing import List

# Try to import seaborn for nicer plots
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    save_dir: str,
    top_k: int = 6
):
    """
    Plot confusion matrices for the top-k most frequent emotions.
   
    """
    print("Generating confusion matrices...")
    
    # Get top-k most frequent labels in ground truth
    label_freqs = y_true.sum(axis=0)
    top_indices = np.argsort(label_freqs)[::-1][:top_k]
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_indices):
        # Compute confusion matrix for this label
        cm = confusion_matrix(y_true[:, idx], y_pred[:, idx])
        
        # Plot using seaborn if available, else matplotlib
        if SEABORN_AVAILABLE:
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes']
            )
        else:
            im = axes[i].imshow(cm, cmap='Blues')
            # Add text annotations
            for (j, k), val in np.ndenumerate(cm):
                axes[i].text(k, j, str(val), ha='center', va='center', fontsize=12)
            axes[i].set_xticks([0, 1])
            axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['No', 'Yes'])
            axes[i].set_yticklabels(['No', 'Yes'])
        
        axes[i].set_title(label_names[idx], fontsize=12)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices (Top 6 Emotions)', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: confusion_matrices.png")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str],
    save_dir: str,
    top_k: int = 5
):
    """
    Plot precision-recall curves for the top-k most frequent emotions.
    
    """
    print("Generating precision-recall curves...")
    
    # Get top-k most frequent labels
    label_freqs = y_true.sum(axis=0)
    top_indices = np.argsort(label_freqs)[::-1][:top_k]
    
    plt.figure(figsize=(10, 8))
    
    for idx in top_indices:
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(
            y_true[:, idx], 
            y_probs[:, idx]
        )
        plt.plot(recall, precision, label=label_names[idx], linewidth=2)
    
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves (Top 5 Emotions)", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "precision_recall_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: precision_recall_curves.png")


def plot_per_label_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    save_dir: str
):
    """
    Plot F1 scores for all emotion labels as a horizontal bar chart.
    
    """
    print("Generating per-label F1 chart...")
    
    # Calculate F1 for each label
    f1_scores = []
    for i in range(len(label_names)):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1_scores.append(f1)
    
    # Sort by F1 score (descending)
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_labels = [label_names[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    
    # Color bars by F1 score (red to green colormap)
    colors = plt.cm.RdYlGn(np.array(sorted_f1))
    
    plt.barh(range(len(sorted_labels)), sorted_f1, color=colors)
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlabel('F1 Score', fontsize=12)
    plt.title('Per-Label F1 Scores', fontsize=14)
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # Highest F1 at top
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "per_label_f1_scores.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: per_label_f1_scores.png")


def plot_label_distribution(
    y_true: np.ndarray,
    label_names: List[str],
    save_dir: str
):
    """
    Plot the distribution of labels in the dataset.
    
    """
    print("Generating label distribution chart...")
    
    # Count occurrences of each label
    label_counts = y_true.sum(axis=0)
    
    # Sort by count (descending)
    sorted_indices = np.argsort(label_counts)[::-1]
    sorted_labels = [label_names[i] for i in sorted_indices]
    sorted_counts = [label_counts[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_labels)), sorted_counts, color='steelblue')
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.xlabel('Count', fontsize=12)
    plt.title('Label Distribution in Test Set', fontsize=14)
    plt.gca().invert_yaxis()  # Most frequent at top
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "label_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: label_distribution.png")


def plot_training_loss(
    losses: List[float],
    save_dir: str
):
    """
    Plot training loss over epochs.
   
    """
    print("Generating training loss plot...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(losses) + 1), 
        losses, 
        'b-o',          # Blue line with circle markers
        linewidth=2, 
        markersize=8
    )
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: training_loss.png")


def visualize_attention_for_text(
    text: str,
    model,
    tokenizer,
    max_length: int,
    save_dir: str = None
):
    """
    Visualize attention weights for a single text.
   
    """
    import torch
    
    device = next(model.parameters()).device
    model.eval()
    
    # Tokenize
    encodings = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Get attention weights
    with torch.no_grad():
        _, attn_weights = model(input_ids, attention_mask, return_attn=True)
    
    attn = attn_weights[0].cpu().numpy()
    if len(attn.shape) == 3:
        attn_avg = attn.mean(axis=0)
    elif len(attn.shape) == 2:
        attn_avg = attn
    else:
        print(f"   Unexpected attention shape: {attn.shape}, skipping")
        return
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    
    # Remove padding
    seq_len = attention_mask[0].sum().item()
    tokens = tokens[:seq_len]
    attn_avg = attn_avg[:seq_len, :seq_len]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    if SEABORN_AVAILABLE:
        sns.heatmap(attn_avg, xticklabels=tokens, yticklabels=tokens, cmap='Blues')
    else:
        plt.imshow(attn_avg, cmap='Blues')
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)
    
    plt.title(f'Attention Weights\n"{text[:50]}..."' if len(text) > 50 else f'Attention Weights\n"{text}"')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "attention_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: attention_visualization.png")
    else:
        plt.show()
