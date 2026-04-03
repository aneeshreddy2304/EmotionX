
"""
Demo Module
===========

This module provides an interactive Gradio web interface for
testing the EmotionX-Hybrid model.

"""

import os
import sys
import webbrowser
import threading

# Local imports
from .config import PRETRAINED_DIR, TRAINED_DIR
from .utils import get_device, load_model_artifacts, check_model_exists, predict_emotions


def run_demo(use_trained: bool = False):
    """
    Launch Gradio web interface for emotion prediction.
    
    Args:
        use_trained: If True, use newly trained model (models/trained/)
                     If False, use pre-trained model (models/pretrained/)
    
    The interface:
    - Provides a text input box
    - Shows prediction probabilities for all 27 emotions
    - Includes example sentences for testing
    - Opens automatically in default browser
    """
    # Check if gradio is installed
    try:
        import gradio as gr
    except ImportError:
        print("\n❌ ERROR: Gradio not installed.")
        print("   Install with: pip install gradio")
        sys.exit(1)
    
    device = get_device()
    model_dir = TRAINED_DIR if use_trained else PRETRAINED_DIR
    model_type = "Newly Trained" if use_trained else "Pre-trained"
    
    print("\n" + "=" * 60)
    print("EMOTIONX-HYBRID DEMO")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print("=" * 60)
    
    # =================================================================
    # Check Model Exists
    # =================================================================
    if not check_model_exists(model_dir):
        print(f"\n❌ ERROR: Model not found at {model_dir}")
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
    # Create Prediction Function
    # =================================================================
    def gradio_predict(text: str) -> dict:
        """
        Prediction function for Gradio interface.
        
        Args:
            text: Input text from user
        
        Returns:
            Dictionary mapping top 5 emotion labels to probabilities
        """
        if not text.strip():
            return {}
        
        # Run prediction
        results = predict_emotions(
            [text], model, tokenizer, label_names, 
            config.max_length, device
        )
        
        # Get all probabilities
        probs = results[0]["probabilities"]
        
        # Create label -> probability mapping and sort by probability
        all_emotions = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
        
        # Get top 5 emotions
        top5 = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return top5
    
    # =================================================================
    # Create Gradio Interface
    # =================================================================
    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(
            lines=3, 
            label="Enter Text", 
            placeholder="Type a sentence to analyze emotions..."
        ),
        outputs=gr.Label(
            label="Top 5 Predicted Emotions"
        ),
        title="EmotionX-Hybrid: Multi-Label Emotion Classification",
        description=(
            "Enter text to predict emotions. The model outputs the top 5 "
            "emotion predictions from 27 labels trained on GoEmotions dataset."
        )
    )
    
    # =================================================================
    # Auto-Open Browser
    # =================================================================
    def open_browser():
        """Open browser to Gradio interface after short delay."""
        webbrowser.open("http://127.0.0.1:7860")
    
    # Start browser opener in background thread
    threading.Timer(2.0, open_browser).start()
    
    print("\n🌐 Opening browser: http://127.0.0.1:7860")
    print("   Press Ctrl+C to stop the server.\n")
    
    # Launch the demo
    demo.launch()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EmotionX-Hybrid Demo")
    parser.add_argument('--use-trained', action='store_true', help='Use newly trained model')
    args = parser.parse_args()
    
    run_demo(use_trained=args.use_trained)
