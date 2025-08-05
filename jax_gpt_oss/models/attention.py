"""Attention mechanisms for GPT-OSS-20B including sliding window and full attention."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import compact
from typing import Any, Optional, Tuple
import numpy as np
import math


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    dtype: Any = jnp.float32
) -> jnp.ndarray:
    """Create a sliding window attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Size of the sliding window
        dtype: Data type for the mask
        
    Returns:
        Mask of shape [1, 1, seq_len, seq_len]
    """
    # Create indices
    row_idx = jnp.arange(seq_len)[:, None]
    col_idx = jnp.arange(seq_len)[None, :]
    
    # Check if positions are within window
    within_window = jnp.abs(row_idx - col_idx) <= window_size
    
    # Also apply causal mask (can't attend to future)
    causal = row_idx >= col_idx
    
    # Combine masks
    mask = within_window & causal
    
    # Convert to attention mask format (0 = masked, 1 = allowed)
    mask = mask.astype(dtype)
    
    # Add batch and head dimensions
    return mask[None, None, :, :]


def apply_yarn_rope(
    query: jnp.ndarray,
    key: jnp.ndarray,
    position_ids: jnp.ndarray,
    rope_theta: float = 150000.0,
    scaling_factor: float = 32.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply YARN (Yet Another RoPE Extension) scaled rotary position embeddings.
    
    YARN provides better extrapolation for long contexts.
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # Compute frequency bands
    dim_half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim_half, 2, dtype=jnp.float32) / dim_half))
    
    # Apply YARN scaling
    # Compute wavelengths
    wavelengths = 2 * jnp.pi / inv_freq
    ratio = original_max_position / wavelengths
    
    # Smooth interpolation between linear and NTK scaling
    alpha = jnp.clip(
        (scaling_factor * ratio - beta_fast) / (beta_slow - beta_fast),
        0.0,
        1.0
    )
    
    # Interpolate frequencies
    inv_freq_scaled = inv_freq / (scaling_factor ** alpha)
    
    # Compute sin and cos
    position_ids_expanded = position_ids[:, :, None]  # [batch, seq_len, 1]
    freqs = position_ids_expanded * inv_freq_scaled[None, None, :]  # [batch, seq_len, dim_half//2]
    
    # Duplicate for sin and cos
    freqs = jnp.repeat(freqs, 2, axis=-1)  # [batch, seq_len, dim_half]
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    
    # Apply rotary embeddings
    def rotate_half(x):
        x1 = x[..., :dim_half]
        x2 = x[..., dim_half:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    # Reshape cos and sin for broadcasting
    cos = cos[:, :, None, :]  # [batch, seq_len, 1, dim_half]
    sin = sin[:, :, None, :]  # [batch, seq_len, 1, dim_half]
    
    # Apply rotation to first half of dimensions
    q_rot = query[..., :dim_half]
    k_rot = key[..., :dim_half]
    
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    
    # Concatenate with unrotated second half
    query = jnp.concatenate([q_rot, query[..., dim_half:]], axis=-1)
    key = jnp.concatenate([k_rot, key[..., dim_half:]], axis=-1)
    
    return query, key


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) for GPT-OSS-20B.
    
    Uses fewer key-value heads than query heads for efficiency.
    """
    config: Any  # GPTOSSConfig
    layer_idx: int
    dtype: Any = jnp.bfloat16
    
    def setup(self):
        """Initialize attention components."""
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.hidden_size = self.config.hidden_size
        
        # Number of query heads per KV head
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Determine if this layer uses sliding window
        self.layer_type = self.config.layer_types[self.layer_idx]
        self.use_sliding_window = (self.layer_type == "sliding_attention")
        
    @compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through attention layer."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Query, Key, Value projections
        q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            dtype=self.dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name="q_proj"
        )
        
        k_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            dtype=self.dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name="k_proj"
        )
        
        v_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            dtype=self.dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name="v_proj"
        )
        
        o_proj = nn.Dense(
            self.hidden_size,
            dtype=self.dtype,
            use_bias=self.config.attention_bias,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            name="o_proj"
        )
        
        # Project to Q, K, V
        query = q_proj(hidden_states)
        key = k_proj(hidden_states)
        value = v_proj(hidden_states)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply position embeddings
        if position_ids is None:
            position_ids = jnp.arange(seq_len)[None, :]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))
        
        query, key = apply_yarn_rope(
            query, key, position_ids,
            rope_theta=self.config.rope_theta,
            scaling_factor=self.config.rope_scaling['factor'],
            original_max_position=self.config.rope_scaling['original_max_position_embeddings'],
            beta_fast=self.config.rope_scaling['beta_fast'],
            beta_slow=self.config.rope_scaling['beta_slow'],
        )
        
        # Repeat KV heads for GQA
        if self.num_groups > 1:
            key = jnp.repeat(key, self.num_groups, axis=2)
            value = jnp.repeat(value, self.num_groups, axis=2)
        
        # Transpose for attention computation
        # [batch, num_heads, seq_len, head_dim]
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = jnp.matmul(query, key.transpose(0, 1, 3, 2)) * scale
        
        # Apply attention mask
        if self.use_sliding_window:
            # Create sliding window mask
            window_mask = create_sliding_window_mask(
                seq_len,
                self.config.sliding_window,
                dtype=self.dtype
            )
            scores = scores + (1.0 - window_mask) * -1e9
        else:
            # Apply causal mask for full attention
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
            scores = scores + (1.0 - causal_mask) * -1e9
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = nn.softmax(scores, axis=-1)
        
        # Apply dropout if not deterministic
        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = nn.Dropout(
                rate=self.config.attention_dropout,
                deterministic=deterministic
            )(attn_weights)
        
        # Compute attention output
        attn_output = jnp.matmul(attn_weights, value)
        
        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        attn_output = o_proj(attn_output)
        
        return attn_output