
"""
Model Architecture Module
=========================

This module defines the EmotionX-Hybrid neural network architecture
for multi-label emotion classification.
"""

import numpy as np
import torch
from torch import nn
from transformers import AutoModel


class EmotionXHybridModel(nn.Module):
    """
    EmotionX-Hybrid Model for Multi-Label Emotion Classification.
   
    """
    
    def __init__(self, config, num_labels: int, token_to_idf: dict, tokenizer):
        super().__init__()
        
        self.config = config
        self.num_labels = num_labels
        self.tokenizer = tokenizer
        
        # =================================================================
        # Layer 1: Transformer Backbone
        # =================================================================
        # Load pre-trained transformer model (DeBERTa or RoBERTa)
        # This provides contextual embeddings for each input token
        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        hidden_size = self.backbone.config.hidden_size  # e.g., 1024 for large models
        
        # =================================================================
        # Layer 2: Bidirectional LSTM
        # =================================================================
        # Captures sequential patterns in the gated embeddings
        # Bidirectional: processes text both forward and backward
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=config.lstm_hidden_size,  # 256
            num_layers=config.lstm_num_layers,    # 1
            batch_first=True,
            bidirectional=True  # Output size = lstm_hidden_size * 2 = 512
        )
        
        # =================================================================
        # Layer 3: Multi-Head Self-Attention
        # =================================================================
        # Allows model to attend to different positions in the sequence
        # 4 heads can capture 4 different attention patterns
        self.attention = nn.MultiheadAttention(
            embed_dim=config.lstm_hidden_size * 2,  # 512 (bidirectional)
            num_heads=config.attention_heads,       # 4 heads
            batch_first=True
        )
        
        # =================================================================
        # Regularization: Dropout
        # =================================================================
        self.dropout = nn.Dropout(config.dropout)  # 0.2 dropout rate
        
        # =================================================================
        # Layer 4: Feature Fusion and Classification
        # =================================================================
        # Fused dimension = CLS (1024) + LSTM max-pool (512) + Attn mean-pool (512)
        fused_dim = hidden_size + config.lstm_hidden_size * 2 + config.lstm_hidden_size * 2
        
        # Projection head (for potential contrastive learning)
        self.projection = nn.Linear(fused_dim, config.projection_dim)
        
        # Final classifier: outputs logits for each of 27 emotions
        self.classifier = nn.Linear(fused_dim, num_labels)
        
        # =================================================================
        # TF-IDF Gating Setup
        # =================================================================
        # Store token-to-IDF mapping for gating computation
        self.token_to_idf = token_to_idf if token_to_idf else {}
        
        # Compute IDF statistics for normalization
        if self.token_to_idf:
            idf_vals = np.array(list(self.token_to_idf.values()))
            self.idf_min = float(idf_vals.min())
            self.idf_max = float(idf_vals.max())
            self.idf_default = float(idf_vals.mean())  # For unknown tokens
        else:
            self.idf_min, self.idf_max, self.idf_default = 0.0, 1.0, 0.5
    
    def compute_tfidf_gates(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute TF-IDF based gating weights for each token.
        """
        device = input_ids.device
        
        # Convert token IDs back to token strings
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(seq.tolist()) 
            for seq in input_ids.cpu()
        ]
        
        gates = []
        for tokens in batch_tokens:
            row = []
            for tok in tokens:
                # Look up IDF value (use default for unknown tokens)
                idf = self.token_to_idf.get(tok, self.idf_default)
                
                # Normalize IDF to [0, 1] range
                if self.idf_max > self.idf_min:
                    norm = (idf - self.idf_min) / (self.idf_max - self.idf_min)
                else:
                    norm = 0.5
                
                # Scale to [1.0, 2.0] - minimum weight is 1.0 (no reduction)
                row.append(1.0 + norm)
            gates.append(row)
        
        # Convert to tensor [batch_size, seq_len, 1]
        return torch.tensor(gates, dtype=torch.float32, device=device).unsqueeze(-1)
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Encode input through all layers except final classifier.
       
        """
        # -----------------------------------------------------------------
        # Step 1: Transformer backbone
        # -----------------------------------------------------------------
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        cls_pooled = last_hidden[:, 0, :]        # [batch, hidden_size] - CLS token
        
        # -----------------------------------------------------------------
        # Step 2: Apply TF-IDF gating
        # -----------------------------------------------------------------
        # Weight each token embedding by its TF-IDF importance
        tfidf_gates = self.compute_tfidf_gates(input_ids)
        gated_hidden = last_hidden * tfidf_gates  # Element-wise multiplication
        
        # -----------------------------------------------------------------
        # Step 3: Bidirectional LSTM
        # -----------------------------------------------------------------
        lstm_out, _ = self.lstm(gated_hidden)  # [batch, seq_len, lstm_hidden*2]
        
        # -----------------------------------------------------------------
        # Step 4: Multi-head self-attention
        # -----------------------------------------------------------------
        # Create key padding mask (True for padding positions)
        key_padding_mask = (attention_mask == 0)
        
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        
        # -----------------------------------------------------------------
        # Step 5: Pooling strategies
        # -----------------------------------------------------------------
        # Max pooling over LSTM outputs
        lstm_pooled, _ = torch.max(lstm_out, dim=1)  # [batch, lstm_hidden*2]
        
        # Mean pooling over attention outputs
        attn_pooled = attn_out.mean(dim=1)  # [batch, lstm_hidden*2]
        
        # -----------------------------------------------------------------
        # Step 6: Feature fusion
        # -----------------------------------------------------------------
        # Concatenate all representations
        fused = torch.cat([cls_pooled, lstm_pooled, attn_pooled], dim=-1)
        fused = self.dropout(fused)
        
        return fused, attn_weights
    
    def forward(self, input_ids, attention_mask, return_attn=False):
        """
        Forward pass through the model.
       
        """
        fused, attn_weights = self.encode(input_ids, attention_mask)
        logits = self.classifier(fused)
        
        if return_attn:
            return logits, attn_weights
        return logits
    
    def get_projection(self, input_ids, attention_mask):
        """
        Get projection embeddings (for contrastive learning).
      
        """
        fused, _ = self.encode(input_ids, attention_mask)
        proj = self.projection(fused)
        return proj


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss for multi-label classification.
 
    """
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)
    
    # Normalize features
    features = nn.functional.normalize(features, dim=-1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create positive mask: samples sharing at least one label
    with torch.no_grad():
        label_matrix = (labels @ labels.T) > 0
        mask = label_matrix.fill_diagonal_(False)
    
    # Compute exponential similarities (excluding self)
    exp_sim = torch.exp(similarity_matrix)
    exp_sim = exp_sim * (~torch.eye(features.size(0), dtype=torch.bool, device=features.device))
    
    # Denominator: sum of all negative similarities
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
    
    # Log probability
    log_prob = similarity_matrix - torch.log(denom)
    log_prob = log_prob.masked_fill(~mask, 0.0)
    
    # Average over positives
    num_positives = mask.sum(dim=1)
    num_positives = torch.clamp(num_positives, min=1)
    
    loss = -(log_prob.sum(dim=1) / num_positives)
    return loss.mean()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Focal Loss for multi-label classification.
    """
    # Compute binary cross-entropy loss (unreduced)
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    
    # Compute probabilities
    probs = torch.sigmoid(logits)
    
    # p_t = p if y=1, else (1-p)
    pt = probs * targets + (1 - probs) * (1 - targets)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - pt) ** gamma
    
    # Apply focal weight and alpha
    loss = alpha * focal_weight * bce_loss
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
