# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2023, Tri Dao.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_bert import FlexBertConfig
from .activation import get_act_fn
from .normalization import get_norm_layer
from .initialization import ModuleType, init_weights
from .loss import MoELoadBalancingLoss, MoERouterZLoss


class BertResidualGLU(nn.Module):
    """Applies the FFN at the end of each Mosaic BERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: Mosaic BERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a Mosaic BERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, Mosaic BERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.act = get_act_fn(config.hidden_act)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = get_norm_layer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, : self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size :]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class FlexBertMLPBase(nn.Module):
    """A FlexBERT MLP base class for type hints."""

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

    def _init_weights(self, reset_params: bool = False):
        raise NotImplementedError("This is a base class and should not be used directly.")

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is a base class and should not be used directly.")


class FlexBertMLP(FlexBertMLPBase):
    """Applies the MLP at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        return self.Wo(self.drop(self.act(self.Wi(hidden_states))))


class FlexBertGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_in_bias)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wi,
            layer_dim=self.config.hidden_size,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class FlexBertParallelGLU(FlexBertMLPBase):
    """Applies the GLU at the end of each FlexBERT layer using intermediate_ff computed in parallel of the attention.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        self.act = get_act_fn(config.hidden_act)
        self.drop = nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias)

    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.Wo,
            layer_dim=self.config.intermediate_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def forward(self, intermediate_ff: torch.Tensor) -> torch.Tensor:
        input, gate = intermediate_ff.chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class Router(nn.Module):
    """Top-K router for selecting experts."""
    
    def __init__(
        self,
        d: int,
        n_exp: int,
        top_k: int = 2,
        use_noisy_top_k: bool = True,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d = d
        self.n_exp = n_exp
        self.top_k = top_k
        self.use_noisy_top_k = use_noisy_top_k
        self.capacity_factor = capacity_factor
        
        # Router weights to compute logits for each expert
        self.gate = nn.Linear(d, n_exp, bias=False)
        
        # Noise parameters for noisy top-k gating
        if use_noisy_top_k:
            self.w_noise = nn.Linear(d, n_exp, bias=False)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch_size, seq_len, d]
        Returns:
            exp_weight: Expert weights [batch_size * seq_len, top_k]
            exp_mask: Expert mask [batch_size * seq_len, n_exp, exp_capacity]
            exp_batches: Token assignments [n_exp, exp_capacity, d]
        """
        B, C, d = x.size()
        num_tokens = B * C
        x_flat = x.view(num_tokens, d)
        
        # Compute router logits
        logits = self.gate(x_flat)  # [num_tokens, n_exp]
        
        # Add noise for exploration (optional)
        if self.use_noisy_top_k and self.training:
            noise_stddev = F.softplus(self.w_noise(x_flat))
            noise = torch.randn_like(logits) * noise_stddev
            logits = logits + noise
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [num_tokens, top_k]
        
        # Compute expert capacity
        exp_capacity = int((num_tokens * self.top_k * self.capacity_factor) / self.n_exp)
        
        # Create expert assignment mask and batches
        exp_mask = torch.zeros(num_tokens, self.n_exp, exp_capacity, device=x.device)
        exp_batches = torch.zeros(self.n_exp, exp_capacity, d, device=x.device)
        
        # Count tokens assigned to each expert
        expert_counts = torch.zeros(self.n_exp, dtype=torch.long, device=x.device)
        
        # Assign tokens to experts
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[token_idx, k_idx]
                if expert_counts[expert_idx] < exp_capacity:
                    pos = expert_counts[expert_idx]
                    exp_mask[token_idx, expert_idx, pos] = top_k_gates[token_idx, k_idx]
                    exp_batches[expert_idx, pos] = x_flat[token_idx]
                    expert_counts[expert_idx] += 1
        
        return top_k_gates, exp_mask, exp_batches





class FlexBertGLUMoE(FlexBertMLPBase):
    """Mixture of Experts with GLU activation for FlexBERT."""
    
    def __init__(self, config: FlexBertConfig, layer_id: Optional[int] = None):
        super().__init__(config=config, layer_id=layer_id)
        
        self.n_exp = getattr(config, 'moe_num_experts', 4)
        self.top_k = getattr(config, 'moe_top_k', 2)
        self.use_noisy_top_k = getattr(config, 'moe_use_noisy_top_k', True)
        self.capacity_factor = getattr(config, 'moe_capacity_factor', 1.25)
        self.compute_aux_loss = getattr(config, 'moe_compute_aux_loss', True)
        self.load_balance_loss_weight = getattr(config, 'moe_load_balance_loss_weight', 0.01)
        self.router_z_loss_weight = getattr(config, 'moe_router_z_loss_weight', 0.001)
        
        self.router = Router(
            d=config.hidden_size,
            n_exp=self.n_exp,
            top_k=self.top_k,
            use_noisy_top_k=self.use_noisy_top_k,
            capacity_factor=self.capacity_factor,
        )
        
        # GLU experts (each projects to 2x intermediate size)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_in_bias),
                nn.Identity(),  # Placeholder for chunking + activation
                nn.Dropout(config.mlp_dropout_prob) if config.mlp_dropout_prob > 0.0 else nn.Identity(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_out_bias),
            )
            for _ in range(self.n_exp)
        ])
        self.act = get_act_fn(config.hidden_act)
        
        # Initialize auxiliary loss modules
        if self.compute_aux_loss:
            self.load_balance_loss = MoELoadBalancingLoss(num_experts=self.n_exp, top_k=self.top_k)
            self.router_z_loss = MoERouterZLoss()
    
    def _init_weights(self, reset_params: bool = False):
        init_weights(
            self.config,
            self.router.gate,
            layer_dim=self.config.hidden_size,
            layer_id=self.layer_id,
            type_of_module=ModuleType.in_module,
        )
        
        for expert in self.experts:
            for i, module in enumerate(expert):
                if isinstance(module, nn.Linear):
                    init_weights(
                        self.config,
                        module,
                        layer_dim=self.config.hidden_size if i == 0 else self.config.intermediate_size,
                        layer_id=self.layer_id,
                        type_of_module=ModuleType.in_module if i == 0 else ModuleType.out_module,
                    )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        B, C, d = hidden_states.size()
        num_tokens = B * C
        x_flat = hidden_states.view(num_tokens, d)
        
        # Compute router logits for auxiliary loss calculation
        router_logits = self.router.gate(x_flat)  # [num_tokens, n_exp]
        
        exp_weight, exp_mask, exp_batches = self.router(hidden_states)
        
        # Extract top-k indices from router for load balancing loss
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [num_tokens, top_k]
        
        # Apply GLU experts
        exp_out = torch.zeros_like(exp_batches)
        for i, expert in enumerate(self.experts):
            x = expert[0](exp_batches[i])  # Linear projection
            input, gate = x.chunk(2, dim=-1)  # Split for GLU
            x = self.act(input) * gate  # GLU activation
            x = expert[2](x)  # Dropout
            exp_out[i] = expert[3](x)  # Output projection
        
        exp_weight_flat = exp_mask.view(num_tokens, -1)
        exp_out_flat = exp_out.view(-1, d)
        output = torch.matmul(exp_weight_flat, exp_out_flat)
        
        # Compute auxiliary losses
        self.aux_loss = None
        if self.compute_aux_loss:
            # Reshape for loss computation
            router_logits_reshaped = router_logits.view(B, C, -1)
            top_k_indices_reshaped = top_k_indices.view(B, C, -1)
            
            # Compute load balancing loss
            lb_loss = self.load_balance_loss(router_logits_reshaped, top_k_indices_reshaped)
            
            # Compute router z-loss
            z_loss = self.router_z_loss(router_logits_reshaped)
            
            # Combine auxiliary losses with weights
            self.aux_loss = self.load_balance_loss_weight * lb_loss + self.router_z_loss_weight * z_loss
        
        return output.view(*original_shape)


# Update the MLP registry
MLP2CLS = {
    "mlp": FlexBertMLP,
    "glu": FlexBertGLU,
    "parallel_glu": FlexBertParallelGLU,
    "glu_moe": FlexBertGLUMoE,
}


def get_mlp_layer(config: FlexBertConfig, layer_id: Optional[int] = None) -> FlexBertMLPBase:
    try:
        mlp_layer = (
            config.initial_mlp_layer
            if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None
            else config.mlp_layer
        )
        return MLP2CLS[mlp_layer](config, layer_id=layer_id)
    except KeyError as e:
        if layer_id < config.num_initial_layers and getattr(config, "initial_mlp_layer", None) is not None:
            raise ValueError(
                f"Invalid MLP layer type: {config.initial_mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}"
            )
        else:
            raise ValueError(f"Invalid MLP layer type: {config.mlp_layer=}, must be one of {MLP2CLS.keys()}. {e}")


