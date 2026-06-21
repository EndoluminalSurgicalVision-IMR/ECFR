import torch
import torch.nn.functional as F
import numpy as np

def compute_entropy(probs):
    """
    Computes Shannon Entropy for probability distributions.
    Utilizes torch.clamp to ensure numerical stability against log(0).
    """
    probs = torch.clamp(probs, min=1e-8)
    return -torch.sum(probs * torch.log(probs), dim=1)

def compute_js_divergence(p1, p2):
    """
    Computes Jensen-Shannon Divergence.
    Strictly adheres to PyTorch F.kl_div specification: 
    input=log(Q), target=P to accurately compute KL(P||Q).
    """
    m = 0.5 * (p1 + p2)
    m_log = torch.log(torch.clamp(m, min=1e-8))
    
    # KL(p1 || m)
    kl1 = F.kl_div(m_log, p1, reduction='none').sum(dim=1)
    # KL(p2 || m)
    kl2 = F.kl_div(m_log, p2, reduction='none').sum(dim=1)
    
    return 0.5 * (kl1 + kl2)

def compute_accuracy(probs, targets):
    """
    Calculates top-1 classification accuracy.
    """
    preds = torch.argmax(probs, dim=1)
    return (preds == targets).float().mean().item()

def compute_brier_score(probs, targets, num_classes):
    """
    Calculates the Brier Score for calibration evaluation.
    """
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1)).item()

def compute_ece(probs, targets, n_bins=20):
    """
    Computes the Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(targets)
    
    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()