# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_bert import FlexBertConfig

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

LOSS2CLS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss,
    "mean_squared_error": nn.MSELoss,
}

if CrossEntropyLoss is not None:
    LOSS2CLS["fa_cross_entropy"] = CrossEntropyLoss


class MoELoadBalancingLoss(nn.Module):
    """Computes Switch Transformer auxiliary loss for load balancing.
    
    Reference: https://arxiv.org/abs/2101.03961 (equations 4-6, page 7)
    
    This loss encourages balanced token allocation across experts to avoid
    scenarios where some experts are overloaded while others are underutilized.
    """
    
    def __init__(self, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            router_logits: Router logits [batch_size, seq_len, num_experts]
            expert_indices: Top-k expert indices [batch_size, seq_len, top_k]
        
        Returns:
            load_balance_loss: Scalar loss value
        """
        # Compute expert probabilities
        expert_probs = F.softmax(router_logits, dim=-1)  # [B, C, n_exp]
        
        # Equation (5): compute ratio of tokens allocated to each expert
        with torch.no_grad():
            one_hot_indices = F.one_hot(expert_indices, num_classes=self.num_experts)  # [B, C, K, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, C, n_exp]
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))  # [n_exp]
        
        # Equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))  # [n_exp]
        
        # Equation (4): scaled dot product between prob / token allocation vectors
        load_balance_loss = self.num_experts * torch.sum(prob_per_expert * tokens_per_expert)
        
        return load_balance_loss


class MoERouterZLoss(nn.Module):
    """Computes router z-loss for MoE models.
    
    Reference: https://arxiv.org/abs/2202.08906 (equation 5, page 7)
    
    This loss constrains the size of router logits to avoid numerical instability
    during training. Large logits can lead to round-off errors in the softmax computation,
    even in float32 precision.
    """
    
    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            router_logits: Router logits [batch_size, seq_len, num_experts]
        
        Returns:
            router_z_loss: Scalar loss value
        """
        # Numerically stable computation: logsumexp is equivalent to log(sum(exp(x)))
        # This avoids overflow issues from directly exponentiating large logits
        router_z_loss = torch.logsumexp(router_logits, dim=-1) ** 2.0  # [B, C]
        
        # Average over all tokens
        router_z_loss = torch.mean(router_z_loss)
        
        return router_z_loss


def get_loss_fn(config: FlexBertConfig) -> nn.Module:
    try:
        loss_class = LOSS2CLS[config.loss_function]
        signature = inspect.signature(loss_class)
        loss_kwargs = {k: v for k, v in config.loss_kwargs.items() if k in signature.parameters}
        return loss_class(**loss_kwargs)
    except KeyError:
        raise ValueError(f"Invalid loss function type: {config.loss_function}, must be one of {LOSS2CLS.keys()}.")
