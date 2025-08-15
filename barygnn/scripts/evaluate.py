import torch
import numpy as np
from barygnn.utils import compute_metrics
import logging
import torch
import torch.nn.functional as F

# Set up logger
logger = logging.getLogger(__name__)

def evaluate_model(model, loader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            if True: #try:
                logits = model(batch)
                # Sanitize targets to be 1D Long indices
                targets = batch.y
                if targets is None:
                    raise ValueError("Batch has no targets 'y'")
                if targets.dim() > 1 and targets.size(-1) == 1:
                    targets = targets.squeeze(-1)
                if targets.dtype != torch.long:
                    targets = targets.long()
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                all_preds.extend(pred.cpu().detach().numpy())
                all_labels.extend(targets.cpu().detach().numpy()) 
                all_logits.extend(F.softmax(logits, dim=1).cpu().detach().numpy())
                
                
            # except Exception as e:
            #     logger.error(f"Error in evaluation batch: {str(e)}")
            #     continue
    
    if len(all_preds) == 0:
        return 0, {"accuracy": 0, "macro_f1": 0, "roc_auc": 0}
    
    avg_loss = total_loss / len(loader)
    
    # Convert logits to numpy array for metrics computation
    all_logits = np.array(all_logits)
    metrics = compute_metrics(all_labels, all_preds, all_logits)
    
    return avg_loss, metrics