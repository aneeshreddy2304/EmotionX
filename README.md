# EmotionX-Hybrid: Multi-Label Emotion Classification

## Project Overview

This project implements **EmotionX-Hybrid**, a multi-label emotion classification system that predicts 27 emotions from text using a hybrid deep learning architecture combining Transformer encoders with BiLSTM and attention mechanisms.

**Key Features:**
- Transformer backbone (DeBERTa-v3-large)
- Bidirectional LSTM for sequential modeling
- TF-IDF gating for token importance weighting
- Multi-head self-attention for context aggregation
- Feature fusion for final classification
- Focal loss for handling class imbalance

**Dataset:** GoEmotions (Google) - 58K Reddit comments labeled with 27 emotions

---

## Project Structure

```
EmotionX_FinalProject/
│
├── main.py                      # Main entry point for all operations
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation file
│
├── src/                         # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and hyperparameters
│   ├── model.py                 # EmotionXHybrid model architecture
│   ├── data.py                  # Data loading and preprocessing
│   ├── utils.py                 # Utility functions (save/load/predict)
│   ├── visualization.py         # Plotting and visualization functions
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation pipeline
│   └── demo.py                  # Interactive Gradio demo
│
├── models/                      # Model storage
│   ├── pretrained/              # Pre-trained model files (provided)
│   │   ├── model_state.pt       # Model weights
│   │   ├── train_config.json    # Training configuration
│   │   ├── labels.json          # Emotion label names
│   │   ├── token_to_idf.json    # TF-IDF vocabulary
│   │   └── tokenizer files...   # HuggingFace tokenizer
│   │
│   └── trained/                 # Newly trained models (auto-generated)
│
├── data/                        # Data directory
│   └── README.txt               # Data loading instructions
│
└── results/                     # Output directory (auto-generated)
    ├── classification_report.txt
    ├── confusion_matrices.png
    ├── precision_recall_curves.png
    ├── per_label_f1_scores.png
    ├── label_distribution.png
    └── training_loss.png
```

---

## Environment Setup

### Step 1: Create Virtual Environment

Unzip the code 
**Windows (PowerShell):**
```powershell
cd EmotionX_FinalProject
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
cd EmotionX_FinalProject
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
cd EmotionX_FinalProject
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Quick Test (Recommended First)

Test with a small subset (100 samples) to verify everything works:

```bash
python main.py --mode evaluate --quick-test
```

### Full Evaluation

Evaluate the pre-trained model on the complete test set:

```bash
python main.py --mode evaluate
```
**Evaluation Outputs (saved to `results/`):**

| File | Description |
|------|-------------|
| `classification_report.txt` | Per-label precision, recall, F1 scores |
| `confusion_matrices.png` | Confusion matrices for top 6 emotions |
| `precision_recall_curves.png` | PR curves for top 5 emotions |
| `per_label_f1_scores.png` | F1 bar chart for all 27 labels |
| `label_distribution.png` | Label frequency in test set |
| `training_loss.png` | Training loss over epochs (after training) |


### Interactive Demo

Launch a web interface for testing predictions:
```bash
python main.py --mode demo
```
Browser will be opening up with UI if it won't open, try to open below url 
http://127.0.0.1:7860

### If you want to Train New Model (Takes 15-40 hrs on CPU and 1hr on GPU NVIDIA A100 High RAM )

Train a new model from scratch (saves to `models/trained/`):

```bash
python main.py --mode train
```

**Note:** Training takes ~1 hour on GPU, 15-40 hours on CPU.

### Use Newly Trained Model

After training, use `--use-trained` flag:

```bash
python main.py --mode evaluate --use-trained
python main.py --mode demo --use-trained
```

---

## Input and Output Formats

### Input

**Text Input:** Plain text strings (English)
- Example: `"I am very happy with how things turned out today!"`

**Data Format:** The GoEmotions dataset is automatically downloaded from HuggingFace Datasets.

### Output

**Prediction Output:**
```python
{
    "text": "I am very happy!",
    "predicted_labels": ["joy", "optimism"],      # Labels above 0.5 threshold
    "probabilities": [0.12, 0.89, ...],           # All 27 probabilities
    "top3": [
        {"label": "joy", "score": 0.89},
        {"label": "optimism", "score": 0.72},
        {"label": "excitement", "score": 0.45}
    ]
}
```


---

## Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=1.10.0 | Deep learning framework |
| transformers | >=4.20.0 | Pre-trained Transformer models |
| datasets | >=2.0.0 | GoEmotions dataset loading |
| scikit-learn | >=1.0.0 | Metrics and evaluation |
| matplotlib | >=3.5.0 | Visualization |
| seaborn | >=0.11.0 | Enhanced plots |
| tqdm | >=4.60.0 | Progress bars |
| gradio | >=3.0.0 | Web demo interface |
| numpy | >=1.20.0 | Numerical operations |

---

## Correspondence to Final Report

| Report Section | Code Location |
|----------------|---------------|
| Model Architecture | `src/model.py` - `EmotionXHybridModel` class |
| Training Procedure | `src/train.py` - `train_model()` function |
| Focal Loss | `src/model.py` - `focal_loss()` function |
| TF-IDF Gating | `src/model.py` - `compute_tfidf_gates()` method |
| Data Preprocessing | `src/data.py` - `prepare_splits()` function |
| Evaluation Metrics | `src/evaluate.py` - F1 scores, classification report |
| Results Figures | `src/visualization.py` - All plotting functions |
| Hyperparameters | `src/config.py` - `TrainConfig` dataclass |

---

## Model Architecture

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│  DeBERTa-v3-large                   │  ← Transformer Backbone
│  (Pre-trained, fine-tuned)          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         TF-IDF Gating               │  ← Token importance weighting
│  (Learned from training corpus)     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│    Bidirectional LSTM (256 units)   │  ← Sequential modeling
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Multi-Head Attention (4 heads)    │  ← Context aggregation
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Feature Fusion                    │  ← Concatenate: CLS + LSTM + Attn
│   (CLS pooling + Max pool + Mean)   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Classifier (27 sigmoid outputs)   │  ← Multi-label prediction
└─────────────────────────────────────┘
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Backbone | DeBERTa-v3-large | Pre-trained transformer |
| Max Length | 96 | Maximum sequence length |
| Batch Size | 32 | Training batch size |
| Learning Rate | 2e-5 | AdamW optimizer |
| Epochs | 5 | Training epochs |
| LSTM Hidden | 256 | BiLSTM hidden size |
| Attention Heads | 4 | Multi-head attention |
| Dropout | 0.2 | Regularization |
| Focal Loss α | 0.25 | Class balance weight |
| Focal Loss γ | 2.0 | Focus on hard examples |

---

## Results

| Metric | Score |
|--------|-------|
| Micro-F1 | 0.5785 |
| Macro-F1 | 0.5216 |

**Training Time:** ~27 minutes/epoch on NVIDIA T4 GPU

---

## Emotion Labels (27)

admiration, amusement, anger, annoyance, approval, caring, confusion,
curiosity, desire, disappointment, disapproval, disgust, embarrassment,
excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
pride, realization, relief, remorse, sadness, surprise


