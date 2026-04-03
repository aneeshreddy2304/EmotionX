
"""
EmotionX-Hybrid: Multi-Label Emotion Classification
====================================================

Main entry point for training, evaluation, and demo.

This script provides a unified interface for all operations:
- Training new models
- Evaluating pre-trained or newly trained models
- Running interactive demos

Usage Examples:
    # Evaluate pre-trained model
    python main.py --mode evaluate
    
    # Quick test (100 samples, for testing)
    python main.py --mode evaluate --quick-test
    
    # Run interactive demo
    python main.py --mode demo
    
    # Train new model (saves to models/trained/)
    python main.py --mode train
    
    # Evaluate newly trained model
    python main.py --mode evaluate --use-trained
    
    # Demo with newly trained model
    python main.py --mode demo --use-trained
    
    # Full pipeline: train + evaluate + demo
    python main.py --mode all

"""

import argparse
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """
    Parse command line arguments and dispatch to appropriate module.
    """
    parser = argparse.ArgumentParser(
        description='EmotionX-Hybrid: Multi-Label Emotion Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode evaluate                 Evaluate pre-trained model
  python main.py --mode evaluate --quick-test    Quick test (100 samples)
  python main.py --mode evaluate --use-trained   Evaluate newly trained model
  python main.py --mode demo                     Run interactive demo
  python main.py --mode demo --use-trained       Demo with newly trained model
  python main.py --mode train                    Train new model
  python main.py --mode all                      Train + Evaluate + Demo

For more details, see README.md
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'demo', 'all'],
        help='Operation mode: train, evaluate, demo, or all'
    )
    
    parser.add_argument(
        '--use-trained',
        action='store_true',
        help='Use newly trained model (from models/trained/) instead of pre-trained'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode: evaluate on only 100 samples (for testing)'
    )
    
    args = parser.parse_args()
    
    # =================================================================
    # Print Header
    # =================================================================
    print("=" * 60)
    print("EmotionX-Hybrid: Multi-Label Emotion Classification")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    
    if args.mode in ['evaluate', 'demo']:
        print(f"Model: {'Newly Trained' if args.use_trained else 'Pre-trained'}")
    
    if args.quick_test and args.mode == 'evaluate':
        print("Quick Test: Yes (100 samples)")
    
    print("=" * 60)
    
    # =================================================================
    # Dispatch to Appropriate Module
    # =================================================================
    if args.mode == 'train':
        from src.train import run_training
        run_training()
        
    elif args.mode == 'evaluate':
        from src.evaluate import run_evaluation
        run_evaluation(use_trained=args.use_trained, quick_test=args.quick_test)
        
    elif args.mode == 'demo':
        from src.demo import run_demo
        run_demo(use_trained=args.use_trained)
        
    elif args.mode == 'all':
        # Full pipeline: train + evaluate + demo
        from src.train import run_training
        from src.evaluate import run_evaluation
        from src.demo import run_demo
        
        print("\n" + "=" * 60)
        print("[1/3] TRAINING")
        print("=" * 60)
        run_training()
        
        print("\n" + "=" * 60)
        print("[2/3] EVALUATION")
        print("=" * 60)
        run_evaluation(use_trained=True)  # Use newly trained model
        
        print("\n" + "=" * 60)
        print("[3/3] DEMO")
        print("=" * 60)
        run_demo(use_trained=True)


if __name__ == "__main__":
    main()
