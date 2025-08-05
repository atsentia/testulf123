"""GPT-OSS-20B model implementation using standard Flax components."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import compact
from typing import Any, Optional, Dict, Tuple
import numpy as np

from .config import GPTOSSConfig
from .moe import MixtureOfExperts
from .attention import GroupedQueryAttention


class GPTOSSBlock(nn.Module):
    """Single transformer block for GPT-OSS-20B using standard Flax components."""
    config: GPTOSSConfig
    layer_idx: int
    dtype: Any = jnp.bfloat16
    
    @compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through transformer block."""
        
        # Input layer norm (using standard Flax RMSNorm)
        residual = hidden_states
        hidden_states = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="input_layernorm"
        )(hidden_states)
        
        # Self-attention
        hidden_states = GroupedQueryAttention(
            config=self.config,
            layer_idx=self.layer_idx,
            dtype=self.dtype,
            name="self_attn"
        )(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Post-attention layer norm
        residual = hidden_states
        hidden_states = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="post_attention_layernorm"
        )(hidden_states)
        
        # MoE/MLP layer
        hidden_states, router_logits = MixtureOfExperts(
            config=self.config,
            dtype=self.dtype,
            name="mlp"
        )(hidden_states, deterministic=deterministic)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states


class GPTOSS(nn.Module):
    """GPT-OSS-20B model using standard Flax components."""
    config: GPTOSSConfig
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        """Initialize model components using standard Flax layers."""
        # Token embeddings using standard Flax Embed
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name="embed_tokens"
        )
        
        # Transformer blocks
        self.layers = [
            GPTOSSBlock(
                config=self.config,
                layer_idx=i,
                dtype=self.dtype,
                name=f"layers_{i}"
            )
            for i in range(self.config.num_hidden_layers)
        ]
        
        # Final layer norm using standard Flax RMSNorm
        self.norm = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=self.dtype,
            name="norm"
        )
        
        # Language modeling head using standard Dense layer
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Dense(
                self.config.vocab_size,
                dtype=self.dtype,
                use_bias=False,
                kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
                name="lm_head"
            )
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_dict: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            deterministic: Whether to use deterministic mode
            return_dict: Whether to return a dictionary
            
        Returns:
            Dictionary with 'logits' and optionally 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position IDs
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))
        
        # Process attention mask
        if attention_mask is not None:
            # Convert to proper format for attention layers
            # Shape: [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Apply transformer blocks
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic
            )
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        if self.config.tie_word_embeddings:
            # Reuse embedding weights (standard Flax pattern)
            embedding_weight = self.variables['params']['embed_tokens']['embedding']
            logits = jnp.dot(hidden_states, embedding_weight.T)
        else:
            logits = self.lm_head(hidden_states)
        
        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
            }
        return logits
    
    # Helper methods for testing
    def embed_tokens_only(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        """Get only token embeddings (for testing)."""
        return self.embed_tokens(input_ids)
    
    def apply_attention(
        self,
        hidden_states: jnp.ndarray,
        layer_idx: int,
        layer_type: str,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Apply attention for a specific layer (for testing)."""
        layer = self.layers[layer_idx]
        return layer.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=True
        )
    
    def apply_moe(
        self,
        hidden_states: jnp.ndarray,
        layer_idx: int,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Apply MoE for a specific layer (for testing)."""
        layer = self.layers[layer_idx]
        return layer.mlp(hidden_states, deterministic=True)
    
    def apply_rms_norm(
        self,
        hidden_states: jnp.ndarray,
        layer_idx: int,
        norm_type: str = "input"
    ) -> jnp.ndarray:
        """Apply RMSNorm for a specific layer (for testing)."""
        layer = self.layers[layer_idx]
        if norm_type == "input":
            return layer.input_layernorm(hidden_states)
        else:
            return layer.post_attention_layernorm(hidden_states)
    
    def apply_final_norm(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Apply final layer norm (for testing)."""
        return self.norm(hidden_states)
    
    def compute_logits(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Compute output logits from hidden states (for testing)."""
        if self.config.tie_word_embeddings:
            embedding_weight = self.variables['params']['embed_tokens']['embedding']
            return jnp.dot(hidden_states, embedding_weight.T)
        else:
            return self.lm_head(hidden_states)