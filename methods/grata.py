import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_js_divergence, compute_entropy

class GradientAlignmentAdaptation(nn.Module):
    """
    GraTa (Gradient Alignment-based Test-Time Adaptation) adapted for classification.
    Originally proposed for medical image segmentation (AAAI 2025).
    This module dynamically scales the unsupervised entropy loss based on its 
    gradient alignment (cosine similarity) with the consistency loss, acting as a 
    passive avoidance strategy against model collapse.
    """
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward_and_adapt(self, x_weak, x_strong):
        self.optimizer.zero_grad()
        
        # 1. Forward propagation through distinct augmentation views
        logits_weak = self.model(x_weak)
        logits_strong = self.model(x_strong)
        
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)
        
        # 2. Metric Computation
        # Optimize student entropy and consistency between teacher and student
        loss_ent = torch.mean(compute_entropy(probs_strong))
        loss_cons = torch.mean(compute_js_divergence(probs_weak, probs_strong))
        
        # ---------------------------------------------------------------------
        # Gradient Alignment Mechanism (GraTa Core)
        # ---------------------------------------------------------------------
        # Step A: Extract gradients for Entropy Loss
        loss_ent.backward(retain_graph=True)
        grads_ent = []
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads_ent.append(param.grad.clone().view(-1))
                else:
                    grads_ent.append(torch.zeros_like(param).view(-1))
        self.optimizer.zero_grad()
        
        # Step B: Extract gradients for Consistency Loss
        loss_cons.backward(retain_graph=True) 
        grads_cons = []
        for param in self.model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads_cons.append(param.grad.clone().view(-1))
                else:
                    grads_cons.append(torch.zeros_like(param).view(-1))
        self.optimizer.zero_grad()
        
        # Step C: Compute Cosine Similarity between the flattened gradient vectors
        if len(grads_ent) > 0 and len(grads_cons) > 0:
            flat_grad_ent = torch.cat(grads_ent)
            flat_grad_cons = torch.cat(grads_cons)
            
            cos_sim = F.cosine_similarity(flat_grad_ent.unsqueeze(0), flat_grad_cons.unsqueeze(0))
            
            # Step D: Dynamic Weight Calculation
            # If gradients agree (cos_sim > 0) -> alpha = 1.0
            # If gradients conflict (cos_sim <= 0) -> passively decay weight: max(0, 1 + cos_sim)
            alpha = torch.where(cos_sim > 0, 
                                torch.tensor(1.0, device=x_weak.device), 
                                torch.clamp(1.0 + cos_sim, min=0.0))
        else:
            alpha = torch.tensor(1.0, device=x_weak.device)
            
        # 3. Apply Aligned Loss and Final Optimization
        total_loss = loss_cons + alpha * loss_ent
        total_loss.backward()
        self.optimizer.step()
            
        # Return weak logits as the standard evaluation output during TTA
        return logits_weak
