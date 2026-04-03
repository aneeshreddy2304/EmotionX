
"""
Evaluation Module
=================

This module implements the evaluation pipeline for EmotionX-Hybrid.

"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Local imports
from .config import PRETRAINED_DIR, TRAINED_DIR, RESULTS_DIR
from .data import load_go_emotions, prepare_splits, GoEmotionsDataset, create_collate_fn
from .utils import get_device, load_model_artifacts, check_model_exists
from .visualization import (
    plot_confusion_matrices, 
    plot_precision_recall_curves,
    plot_per_label_f1, 
    plot_label_distribution,
    visualize_attention_for_text
)


def evaluate_model(
    model,
    data_loader: DataLoader,
    device: torch.device
) -> tuple:
    """
    Run evaluation on a data loader.
    """
    model.eval()
    model.to(device)
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=True, ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_np = batch["labels"].cpu().numpy()
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels_np)
    
    # Concatenate all batches
    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    
    # Convert logits to probabilities and predictions
    probs = 1.0 / (1.0 + np.exp(-logits_np))  # Sigmoid
    preds = (probs >= 0.5).astype(int)
    
    # Compute metrics
    micro_f1 = f1_score(labels_np, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    
    print(f"Evaluation micro-F1: {micro_f1:.4f}, macro-F1: {macro_f1:.4f}")
    
    return labels_np, probs, preds, micro_f1, macro_f1


def run_evaluation(use_trained: bool = False, quick_test: bool = False):
    """
    Run full evaluation pipeline.
    """
    device = get_device()
    model_dir = TRAINED_DIR if use_trained else PRETRAINED_DIR
    model_type = "Newly Trained" if use_trained else "Pre-trained"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Path: {model_dir}")
    if quick_test:
        print("Mode: Quick Test (100 samples)")
    print("=" * 60)
    
    # =================================================================
    # Check Model Exists
    # =================================================================
    if not check_model_exists(model_dir):
        print(f"\n ERROR: Model not found at {model_dir}")
        if use_trained:
            print("   Run training first: python main.py --mode train")
        else:
            print("   Copy pre-trained model files to: models/pretrained/")
        sys.exit(1)
    
    # =================================================================
    # Load Model
    # =================================================================
    model, tokenizer, _, label_names, config = load_model_artifacts(model_dir, device)
    
    # =================================================================
    # Load Test Data
    # =================================================================
    print("\nLoading test data...")
    raw_dataset = load_go_emotions()
    _, _, (test_texts, test_labels), _ = prepare_splits(raw_dataset)
    
    # Quick test mode: use only 100 samples
    if quick_test:
        test_texts = test_texts[:100]
        test_labels = test_labels[:100]
        print(f"Quick test: using {len(test_texts)} samples")
    
    # Create dataset and loader
    test_dataset = GoEmotionsDataset(test_texts, test_labels, tokenizer, config.max_length)
    collate_fn = create_collate_fn(tokenizer, config.max_length)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # =================================================================
    # Run Evaluation
    # =================================================================
    y_true, y_probs, y_pred, micro_f1, macro_f1 = evaluate_model(model, test_loader, device)
    
    # =================================================================
    # Print Results
    # =================================================================
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=label_names, 
        zero_division=0
    )
    print("\nClassification Report:")
    print(report)
    
    # =================================================================
    # Save Classification Report
    # =================================================================
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("EmotionX-Hybrid: Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Test Samples: {len(test_texts)}\n")
        f.write(f"Micro-F1: {micro_f1:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n\n")
        f.write(report)
    print(f"✓ Saved: classification_report.txt")
    
    # =================================================================
    # Generate Visualizations
    # =================================================================
    print("\n" + "-" * 60)
    print("GENERATING VISUALIZATIONS")
    print("-" * 60)
    
    plot_confusion_matrices(y_true, y_pred, label_names, RESULTS_DIR)
    plot_precision_recall_curves(y_true, y_probs, label_names, RESULTS_DIR)
    plot_per_label_f1(y_true, y_pred, label_names, RESULTS_DIR)
    plot_label_distribution(y_true, label_names, RESULTS_DIR)
    
    # Generate attention visualization for example sentence
    example_text = "I am very happy with how things turned out today."
    print(f"\nGenerating attention visualization for: \"{example_text}\"")
    visualize_attention_for_text(
        example_text,
        model=model,
        tokenizer=tokenizer,
        max_length=config.max_length,
        save_dir=RESULTS_DIR
    )
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("✓ EVALUATION COMPLETE!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate EmotionX-Hybrid model")
    parser.add_argument('--use-trained', action='store_true', help='Use newly trained model')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with 100 samples')
    args = parser.parse_args()
    
    run_evaluation(use_trained=args.use_trained, quick_test=args.quick_test)
