"""Mixture of Experts (MoE) implementation for GPT-OSS-20B."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import compact
from typing import Any, Tuple, Optional
import einops


class Router(nn.Module):
    """Router for selecting experts in MoE layer.
    
    The router determines which experts to activate for each token.
    """
    num_experts: int = 32
    num_experts_per_tok: int = 4
    hidden_size: int = 2880
    dtype: Any = jnp.bfloat16
    
    @compact
    def __call__(self, hidden_states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            - router_logits: Scores for each expert [batch_size, seq_len, num_experts]
            - selected_experts: Indices of selected experts [batch_size, seq_len, num_experts_per_tok]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear projection to get router logits
        router_weights = self.param(
            'router_weights',
            nn.initializers.normal(stddev=0.02),
            (hidden_size, self.num_experts),
            self.dtype
        )
        
        # Compute router logits
        router_logits = jnp.dot(hidden_states, router_weights)
        
        # Select top-k experts
        # Get top-k values and indices
        top_k_logits, selected_experts = jax.lax.top_k(
            router_logits, 
            k=self.num_experts_per_tok
        )
        
        # Compute softmax probabilities for selected experts
        router_probs = jax.nn.softmax(top_k_logits, axis=-1)
        
        return router_logits, selected_experts, router_probs


class Expert(nn.Module):
    """Single expert in the MoE layer.
    
    Each expert is a standard MLP with SwiGLU activation.
    """
    intermediate_size: int = 2880
    hidden_size: int = 2880
    swiglu_limit: float = 7.0
    dtype: Any = jnp.bfloat16
    
    @compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through expert MLP."""
        # Gate projection
        gate_proj = nn.Dense(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=False,
            name="gate_proj"
        )
        
        # Up projection
        up_proj = nn.Dense(
            self.intermediate_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=False,
            name="up_proj"
        )
        
        # Down projection
        down_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=False,
            name="down_proj"
        )
        
        # SwiGLU activation with clipping
        gate_output = gate_proj(hidden_states)
        gate_output = nn.silu(gate_output)
        
        # Apply swiglu limit if specified
        if self.swiglu_limit > 0:
            gate_output = jnp.clip(gate_output, -self.swiglu_limit, self.swiglu_limit)
        
        up_output = up_proj(hidden_states)
        
        # Element-wise multiplication and down projection
        hidden_states = gate_output * up_output
        hidden_states = down_proj(hidden_states)
        
        return hidden_states


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer for GPT-OSS-20B.
    
    This implements a top-k routing MoE where each token is processed
    by k experts out of n total experts.
    """
    config: Any  # GPTOSSConfig
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        """Initialize MoE components."""
        self.router = Router(
            num_experts=self.config.num_local_experts,
            num_experts_per_tok=self.config.num_experts_per_tok,
            hidden_size=self.config.hidden_size,
            dtype=self.dtype
        )
        
        # Create all experts
        self.experts = [
            Expert(
                intermediate_size=self.config.intermediate_size,
                hidden_size=self.config.hidden_size,
                swiglu_limit=self.config.swiglu_limit,
                dtype=self.dtype,
                name=f"expert_{i}"
            )
            for i in range(self.config.num_local_experts)
        ]
    
    def __call__(self, 
                 hidden_states: jnp.ndarray,
                 deterministic: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            - output: Processed tensor [batch_size, seq_len, hidden_size]
            - router_logits: Router scores (optional, for aux loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens to experts
        router_logits, selected_experts, router_probs = self.router(hidden_states)
        
        # Initialize output tensor
        final_output = jnp.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx in range(self.config.num_local_experts):
            # Create expert
            expert = self.experts[expert_idx]
            
            # Find tokens assigned to this expert
            # Shape: [batch_size, seq_len, num_experts_per_tok]
            expert_mask = (selected_experts == expert_idx).any(axis=-1)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = hidden_states * expert_mask[:, :, None]
                
                # Process through expert
                expert_output = expert(expert_input)
                
                # Get routing weights for this expert
                # Find positions where this expert was selected
                expert_positions = (selected_experts == expert_idx)
                expert_weights = jnp.where(
                    expert_positions,
                    router_probs,
                    0.0
                ).sum(axis=-1, keepdims=True)
                
                # Accumulate weighted output
                final_output += expert_output * expert_weights
        
        # Return output and router logits (for auxiliary loss if needed)
        if self.config.output_router_logits:
            return final_output, router_logits
        else:
            return final_output, None


class MoEBlock(nn.Module):
    """Complete MoE block with layer normalization."""
    config: Any
    dtype: Any = jnp.bfloat16
    
    @compact
    def __call__(self, 
                 hidden_states: jnp.ndarray,
                 deterministic: bool = True) -> jnp.ndarray:
        """Forward pass through MoE block with residual connection."""
        residual = hidden_states
        
        # Apply RMSNorm
        hidden_states = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=self.dtype
        )(hidden_states)
        
        # Apply MoE
        hidden_states, router_logits = MixtureOfExperts(
            config=self.config,
            dtype=self.dtype
        )(hidden_states, deterministic=deterministic)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states