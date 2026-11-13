<p align="center"><img src="docs/assets/banner.svg" alt="EmotionX — Read the emotional texture inside everyday language." width="100%"></p>

<p align="center"><img src="docs/assets/badges/python.svg" alt="python" height="28"> <img src="docs/assets/badges/pytorch.svg" alt="pytorch" height="28"> <img src="docs/assets/badges/transformers.svg" alt="transformers" height="28"> <img src="docs/assets/badges/scikit-learn.svg" alt="scikit-learn" height="28"> <img src="docs/assets/badges/gradio.svg" alt="gradio" height="28"></p>

<p align="center"><strong>A multi-label emotion-classification system that fuses transformer context, IDF weighting, BiLSTM sequence features, and attention.</strong></p>

## Overview

A multi-label emotion-classification system that fuses transformer context, IDF weighting, BiLSTM sequence features, and attention.

### Why it matters

EmotionX explores richer representations for language that can express several emotions at once, with separate modules for training, evaluation, visualization, and demo use.

## Key features

- Transformer-based text representations
- IDF-gated token features
- Bidirectional LSTM and attention fusion
- Gradio interface for interactive inference

## Architecture

```mermaid
flowchart LR
  A[Text] --> B[Tokenizer + transformer]
  B --> C[IDF gating]
  C --> D[BiLSTM]
  D --> E[Attention fusion]
  E --> F[Multi-label emotions]
  F --> G[Gradio demo]
```

## Technology stack

**python** · **pytorch** · **transformers** · **scikit learn** · **gradio**

## Getting started

Create a Python environment from `requirements.txt`, obtain the expected dataset and pretrained model assets, then use `main.py` to access the project modes. Keep data and model licensing in scope when sharing a demo.
