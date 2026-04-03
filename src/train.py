
"""
Training Module
===============

This module implements the training pipeline for EmotionX-Hybrid.
Matches the original Colab notebook training logic exactly.

"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

# Local imports
from .config import TrainConfig, TRAINED_DIR, RESULTS_DIR
from .model import EmotionXHybridModel, focal_loss, supervised_contrastive_loss
from .data import (
    load_go_emotions, 
    prepare_splits, 
    build_tokenizer_and_tfidf,
    GoEmotionsDataset, 
    create_collate_fn
)
from .utils import set_seed, get_device, save_model_artifacts
from .visualization import plot_training_loss


def pick_backbone() -> str:
    """
    Select the best available transformer backbone.
    """
    candidates = [
        "microsoft/deberta-v3-large",
        "roberta-large",
        "roberta-base"
    ]
    
    for name in candidates:
        try:
            _ = AutoTokenizer.from_pretrained(name)
            print(f"Using backbone: {name}")
            return name
        except Exception:
            continue
    
    return "roberta-base"


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute F1 metrics from logits and labels.
    Matches original notebook function exactly.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def train_model(
    model: EmotionXHybridModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device
) -> EmotionXHybridModel:
    """
    Train the EmotionX-Hybrid model.
    Matches original notebook training loop exactly.
    """
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    num_training_steps = config.num_epochs * len(train_loader)
    num_warmup_steps = int(config.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_micro_f1 = -1.0
    best_state_dict = None
    epochs_no_improve = 0
    all_train_losses = []

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses = []

        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", leave=True, ncols=100)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_cls = focal_loss(
                    logits,
                    labels,
                    alpha=config.focal_alpha,
                    gamma=config.focal_gamma,
                    reduction="mean",
                )

                if config.use_contrastive:
                    features = model.get_projection(input_ids=input_ids, attention_mask=attention_mask)
                    loss_con = supervised_contrastive_loss(features, labels, temperature=config.contrastive_temperature)
                    loss = loss_cls + config.contrastive_weight * loss_con
                else:
                    loss = loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        all_train_losses.append(avg_train_loss)

        model.eval()
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False, ncols=100):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_np = batch["labels"].cpu().numpy()

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)

                val_logits_list.append(logits.detach().cpu().numpy())
                val_labels_list.append(labels_np)

        if val_logits_list:
            val_logits = np.concatenate(val_logits_list, axis=0)
            val_labels_np = np.concatenate(val_labels_list, axis=0)
            metrics = compute_metrics_from_logits(val_logits, val_labels_np)
            val_micro_f1 = metrics["micro_f1"]
            val_macro_f1 = metrics["macro_f1"]
        else:
            val_micro_f1 = 0.0
            val_macro_f1 = 0.0

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.4f}, "
            f"val_micro_f1={val_micro_f1:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_micro_f1 > best_micro_f1:
            best_micro_f1 = val_micro_f1
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save training loss plot
    if all_train_losses:
        plot_training_loss(all_train_losses, RESULTS_DIR)

    return model


def run_training():
    """
    Main training function.
    """
    set_seed(42)
    device = get_device()
    os.makedirs(TRAINED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("TRAINING NEW MODEL")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {TRAINED_DIR}")
    print("=" * 60)
    
    # Select backbone
    backbone_name = pick_backbone()
    config = TrainConfig(backbone_name=backbone_name)
    
    # Load data
    raw_dataset = load_go_emotions()
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels), label_names = prepare_splits(raw_dataset)
    
    # Build tokenizer and TF-IDF
    tokenizer, token_to_idf = build_tokenizer_and_tfidf(backbone_name, train_texts)
    
    # Create datasets
    train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    # Create data loaders
    collate_fn = create_collate_fn(tokenizer, config.max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Create model
    model = EmotionXHybridModel(config, len(label_names), token_to_idf, tokenizer)
    
    # Print model info
    print(f"\nModel Architecture:")
    print(f"  Backbone: {config.backbone_name}")
    print(f"  LSTM hidden: {config.lstm_hidden_size}")
    print(f"  Attention heads: {config.attention_heads}")
    print(f"  Labels: {len(label_names)}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print()
    
    # Train
    model = train_model(model, train_loader, val_loader, config, device)
    
    # Save
    save_model_artifacts(model, tokenizer, token_to_idf, label_names, config, TRAINED_DIR)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Model saved to: {TRAINED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
