import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, Union, List, Tuple


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    y_score: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Predicted scores (probabilities or logits)
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Convert tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_score, torch.Tensor) and y_score is not None:
        y_score = y_score.cpu().numpy()
    
    # Compute metrics
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # F1 score
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    
    # ROC AUC score (for binary classification or multilabel)
    if y_score is not None:
        num_classes = y_score.shape[1] if len(y_score.shape) > 1 else 2
        
        if num_classes == 2:
            # Binary classification
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score[:, 1] if len(y_score.shape) > 1 else y_score)
            except ValueError:
                # If there's only one class in y_true, ROC AUC is undefined
                metrics["roc_auc"] = float("nan")
        else:
            # Multi-class classification
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true=np.eye(num_classes)[y_true],
                    y_score=y_score,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                # If there's an issue with the computation
                metrics["roc_auc"] = float("nan")
    
    return metrics 