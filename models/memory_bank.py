import torch
import numpy as np
from collections import deque


def _to_float_list(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).numpy().astype(float).tolist()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).astype(float).tolist()
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


class DynamicMemoryBank:
    """
    Dynamic Memory Bank for real-time threshold calibration and cross-control in ECFR.
    Addresses threshold drift and provides optimization coefficients based on historical quantiles.
    """
    def __init__(self, capacity=128, quantile=0.5, warmup_size=5):
        """
        Args:
            capacity (int): Maximum size of the memory queue. Default: 128.
            quantile (float): The quantile used for dynamic thresholding. 
                              - Use 0.5 for ECFR-0.5 (Target-agnostic).
                              - Use the numerical prior target accuracy for ECFR-Acc
                                (Prior-guided), e.g., 0.83 rather than the string "Acc".
            warmup_size (int): Minimum memory length required before activating the Max penalty.
        """
        self.capacity = capacity
        self.quantile = quantile
        self.warmup_size = warmup_size
        self.ent_queue = deque(maxlen=capacity)
        self.cons_queue = deque(maxlen=capacity)

    def update(self, entropies, consistencies):
        """
        Updates the memory bank with streaming batch statistics.
        """
        self.ent_queue.extend(_to_float_list(entropies))
        self.cons_queue.extend(_to_float_list(consistencies))

    def _get_hard_coefficient(self, history_data, current_vals):
        """
        Determines the optimization direction (1.0 for Minimization, -1.0 for Maximization).
        """
        # Warm-up phase: Default to Minimization (1.0) to stabilize initial representations
        if len(history_data) < self.warmup_size:
            return torch.ones_like(current_vals)
            
        threshold = np.quantile(history_data, self.quantile)
        
        # Hard Switch Logic:
        # If current_val < threshold (High Confidence/Stability) -> Minimize (1.0)
        # If current_val >= threshold (High Risk/Uncertainty) -> Maximize (-1.0)
        coefs = torch.where(current_vals < threshold, 
                            torch.tensor(1.0, device=current_vals.device), 
                            torch.tensor(-1.0, device=current_vals.device))
        return coefs

    def get_coefficients(self, curr_ent, curr_cons):
        """
        Implements the Bidirectional Cross-Reference Mechanism:
        - Consistency coefficient relies on Entropy history.
        - Entropy coefficient relies on Consistency history.
        """
        coef_cons = self._get_hard_coefficient(self.ent_queue, curr_ent)
        coef_ent = self._get_hard_coefficient(self.cons_queue, curr_cons)
        return coef_cons, coef_ent

    def get_thresholds(self):
        """
        Returns the current numerical thresholds, typically used for tracking and visualization.
        """
        if len(self.ent_queue) == 0:
            return 0.3, 0.3 
        ent_thresh = np.quantile(self.ent_queue, self.quantile)
        cons_thresh = np.quantile(self.cons_queue, self.quantile)
        return ent_thresh, cons_thresh
