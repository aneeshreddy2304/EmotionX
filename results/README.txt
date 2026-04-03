# Results Directory
# ==================

Evaluation outputs are automatically saved here when you run:

    python main.py --mode evaluate

Generated Files:
----------------
- classification_report.txt     Per-label precision, recall, F1 scores
- confusion_matrices.png        Confusion matrices for top 6 emotions
- precision_recall_curves.png   PR curves for top 5 emotions
- per_label_f1_scores.png       F1 bar chart for all 27 labels
- label_distribution.png        Label frequency distribution
- training_loss.png             Training loss over epochs (after training)

These visualizations correspond to the results reported in the final report.
