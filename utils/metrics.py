"""
Metrics computation for ECG classification
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, accuracy_score
)
from sklearn.preprocessing import label_binarize
import config


class MetricsComputer:
    """Compute evaluation metrics for multi-class classification"""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        self.num_classes = num_classes
        self.class_names = {v: k for k, v in config.CLASSES.items()}
    
    def compute_metrics(self, y_true, y_pred_proba, y_pred_labels=None, threshold_strategy="max_prob"):
        """
        Compute all metrics
        
        Args:
            y_true: Ground truth labels (batch_size,)
            y_pred_proba: Predicted probabilities (batch_size, num_classes)
            y_pred_labels: Predicted labels (batch_size,) - if None, derived from y_pred_proba
            threshold_strategy: "max_prob" or "fixed"
        
        Returns:
            dict: Dictionary of computed metrics
        """
        
        # Get predicted labels if not provided
        if y_pred_labels is None:
            if threshold_strategy == "max_prob":
                y_pred_labels = np.argmax(y_pred_proba, axis=1)
            elif threshold_strategy == "fixed":
                # For multi-class, use max probability but with threshold
                max_probs = np.max(y_pred_proba, axis=1)
                y_pred_labels = np.argmax(y_pred_proba, axis=1)
                # Mark as uncertain if below threshold (keep prediction but track confidence)
                y_pred_labels = np.where(max_probs >= config.THRESHOLD_FIXED, y_pred_labels, -1)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred_labels)
        
        # Per-class metrics
        for class_idx in range(self.num_classes):
            class_name = self.class_names.get(class_idx, f"Class_{class_idx}")
            
            # Binarize for one-vs-rest
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_proba_class = y_pred_proba[:, class_idx]
            
            # AUC
            if len(np.unique(y_true_binary)) > 1:  # Only if both classes present
                try:
                    metrics[f'auc_{class_name}'] = roc_auc_score(y_true_binary, y_pred_proba_class)
                except:
                    metrics[f'auc_{class_name}'] = 0.0
            
            # F1 Score
            y_pred_binary = (y_pred_labels == class_idx).astype(int)
            metrics[f'f1_{class_name}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Precision
            metrics[f'precision_{class_name}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Recall (Sensitivity)
            metrics[f'recall_{class_name}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Specificity
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics[f'specificity_{class_name}'] = specificity
        
        # Macro-averaged metrics
        metrics['macro_f1'] = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
        metrics['macro_precision'] = precision_score(y_true, y_pred_labels, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred_labels, average='macro', zero_division=0)
        
        # Weighted-averaged metrics
        metrics['weighted_f1'] = f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
        metrics['weighted_auc'] = roc_auc_score(label_binarize(y_true, classes=range(self.num_classes)),
                                                y_pred_proba, average='weighted', multi_class='ovr')
        
        return metrics
    
    def format_metrics(self, metrics: dict) -> str:
        """Format metrics dictionary for printing"""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{key:25s}: {value:.4f}")
            else:
                lines.append(f"{key:25s}: {value}")
        return "\n".join(lines)
