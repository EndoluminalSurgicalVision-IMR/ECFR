import torch
import torch.nn as nn
import copy
from utils.metrics import compute_js_divergence, compute_entropy


def configure_ln_parameters(model, mode="layernorm"):
    """
    Select affine normalization parameters for lightweight test-time adaptation.

    The MICCAI experiments update LayerNorm affine parameters. The broader
    "norm" mode is kept only for demo backbones that do not contain LayerNorm.
    """
    if mode not in {"layernorm", "norm"}:
        raise ValueError("mode must be either 'layernorm' or 'norm'")

    model.requires_grad_(False)
    opt_params = []
    norm_types = (nn.LayerNorm,) if mode == "layernorm" else (
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
    )

    for module in model.modules():
        if isinstance(module, norm_types):
            for param in (getattr(module, "weight", None), getattr(module, "bias", None)):
                if param is not None:
                    param.requires_grad = True
                    opt_params.append(param)
    if not opt_params:
        raise ValueError("No normalization affine parameters found for adaptation.")
    return opt_params

class ECFRAdaptation(nn.Module):
    """
    ECFR: Entropy-Consistency Flow Rectification
    Integrates a Teacher-Student EMA architecture, Stochastic Parameter Restoration, 
    and a Bidirectional Cross-Control Mechanism guided by Memory Bank coefficients.
    """
    def __init__(self, student_model, optimizer, ema_alpha=0.999, p_restore=0.01):
        super().__init__()
        self.student = student_model
        # Initialize an independent Teacher model via deepcopy and freeze its gradients
        self.teacher = copy.deepcopy(student_model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        
        self.optimizer = optimizer
        self.ema_alpha = ema_alpha
        self.p_restore = p_restore

        # Archive initial optimized parameters for Stochastic Restoration
        self.initial_params = {
            name: param.clone().detach()
            for name, param in self.student.named_parameters()
            if param.requires_grad
        }

    def _stochastic_restore(self):
        """Reset a small random subset of trainable parameters to the source state."""
        if self.p_restore <= 0:
            return
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if param.requires_grad:
                    init_param = self.initial_params[name]
                    mask = torch.rand_like(param) < self.p_restore
                    param.data.copy_(torch.where(mask, init_param, param.data))

    def forward_and_adapt(self, x_clean, x_weak, x_strong, memory_bank):
        # ---------------------------------------------------------------------
        # 1. Stochastic Parameter Restoration (Mitigates Catastrophic Forgetting)
        # ---------------------------------------------------------------------
        self._stochastic_restore()

        self.optimizer.zero_grad()
        
        # ---------------------------------------------------------------------
        # 2. Forward Propagation (Teacher-Student Paradigm)
        # ---------------------------------------------------------------------
        # Teacher generates stable pseudo-targets using weakly augmented views
        with torch.no_grad():
            logits_weak = self.teacher(x_weak)
            logits_clean = self.teacher(x_clean) # Evaluated strictly for final metrics
            
        # Student learns from strongly augmented views
        logits_strong = self.student(x_strong)
        
        probs_weak = torch.softmax(logits_weak, dim=1)
        probs_strong = torch.softmax(logits_strong, dim=1)
        
        # ---------------------------------------------------------------------
        # 3. Memory Bank Dynamics & Coefficient Retrieval
        # ---------------------------------------------------------------------
        # Teacher entropy and teacher-student JS divergence diagnose sample state.
        ent_val = compute_entropy(probs_weak)
        cons_val = compute_js_divergence(probs_weak, probs_strong.detach())
        
        memory_bank.update(ent_val, cons_val)

        # Retrieve optimization directions (-1.0 for Max, 1.0 for Min)
        coef_cons, coef_ent = memory_bank.get_coefficients(ent_val, cons_val)
        
        # ---------------------------------------------------------------------
        # 4. Bidirectional Cross-Control Optimization
        # ---------------------------------------------------------------------
        # Optimize student entropy and teacher-student consistency with signed switches.
        loss_ent = torch.mean(coef_ent * compute_entropy(probs_strong))
        loss_cons = torch.mean(coef_cons * compute_js_divergence(probs_weak, probs_strong))
        
        total_loss = loss_ent + loss_cons
        
        total_loss.backward()
        self.optimizer.step()
        
        # ---------------------------------------------------------------------
        # 5. Exponential Moving Average (EMA) Update
        # ---------------------------------------------------------------------
        with torch.no_grad():
            for name, param_s in self.student.named_parameters():
                if param_s.requires_grad:
                    param_t = dict(self.teacher.named_parameters())[name]
                    param_t.data.mul_(self.ema_alpha).add_(param_s.data, alpha=1 - self.ema_alpha)
            
        return logits_clean, ent_val.detach(), cons_val.detach()


# Backward-compatible alias for earlier review-stage code.
DualMinMaxAdaptation = ECFRAdaptation
