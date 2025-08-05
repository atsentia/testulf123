"""Inference and generation utilities for GPT-OSS-20B."""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional, List, Union
import numpy as np

from .models.gpt_oss import GPTOSS
from .models.config import GPTOSSConfig


@jax.jit
def forward_pass(
    model: GPTOSS,
    params: Dict[str, Any],
    input_ids: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray] = None,
    position_ids: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """
    JIT-compiled forward pass through the model.
    
    Args:
        model: Model instance
        params: Model parameters
        input_ids: Input token IDs
        attention_mask: Attention mask
        position_ids: Position IDs
        
    Returns:
        Dictionary with logits and hidden states
    """
    return model.apply(
        params,
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        deterministic=True,
        return_dict=True
    )


def generate(
    model: GPTOSS,
    params: Dict[str, Any],
    prompt: Union[str, jnp.ndarray],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    tokenizer: Optional[Any] = None,
    rng: Optional[jax.random.PRNGKey] = None,
) -> Union[str, jnp.ndarray]:
    """
    Generate text from a prompt using autoregressive sampling.
    
    Args:
        model: Model instance
        params: Model parameters
        prompt: Text prompt or token IDs
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        tokenizer: Optional tokenizer for text conversion
        rng: Random key for sampling
        
    Returns:
        Generated text or token IDs
    """
    if rng is None:
        rng = random.PRNGKey(42)
    
    # Convert prompt to token IDs if needed
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("Tokenizer required for text prompt")
        input_ids = tokenizer.encode(prompt, return_tensors="jax")
    else:
        input_ids = prompt
    
    # Ensure proper shape
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    
    batch_size, seq_len = input_ids.shape
    generated = input_ids
    
    for _ in range(max_tokens):
        # Get model predictions
        outputs = forward_pass(model, params, generated)
        logits = outputs['logits']
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k=min(top_k, next_token_logits.shape[-1]))
            next_token_logits = jnp.full_like(next_token_logits, -float('inf'))
            next_token_logits = next_token_logits.at[jnp.arange(batch_size)[:, None], top_k_indices].set(top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits = jnp.sort(next_token_logits, axis=-1)[:, ::-1]
            sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
            cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
            
            # Find cutoff position
            cutoff_mask = cumsum_probs > top_p
            cutoff_mask = cutoff_mask.at[:, 0].set(False)  # Keep at least one token
            
            # Apply cutoff
            min_logits = jnp.where(cutoff_mask, -float('inf'), sorted_logits)
            next_token_logits = jnp.maximum(next_token_logits, min_logits[:, -1:])
        
        # Sample next token
        probs = jax.nn.softmax(next_token_logits, axis=-1)
        rng, sample_rng = random.split(rng)
        next_token = random.categorical(sample_rng, jnp.log(probs), axis=-1)
        
        # Append to generated sequence
        next_token = next_token[:, None]
        generated = jnp.concatenate([generated, next_token], axis=1)
        
        # Check for EOS token
        if model.config.eos_token_id is not None:
            if jnp.any(next_token == model.config.eos_token_id):
                break
    
    # Convert back to text if tokenizer provided
    if tokenizer is not None and isinstance(prompt, str):
        generated_tokens = generated[0, len(input_ids[0]):]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated


def generate_batch(
    model: GPTOSS,
    params: Dict[str, Any],
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    tokenizer: Any = None,
    rng: Optional[jax.random.PRNGKey] = None,
    pad_token_id: Optional[int] = None,
) -> List[str]:
    """
    Generate text for multiple prompts in a batch.
    
    Args:
        model: Model instance
        params: Model parameters
        prompts: List of text prompts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        tokenizer: Tokenizer for text conversion
        rng: Random key for sampling
        pad_token_id: Padding token ID
        
    Returns:
        List of generated texts
    """
    if tokenizer is None:
        raise ValueError("Tokenizer required for batch generation")
    
    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id
    
    # Tokenize all prompts
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="jax"
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # Generate for batch
    generated = generate(
        model=model,
        params=params,
        prompt=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tokenizer=None,  # Already tokenized
        rng=rng
    )
    
    # Decode each sequence
    results = []
    for i, prompt in enumerate(prompts):
        # Get generated tokens for this sample
        prompt_len = len(tokenizer.encode(prompt))
        generated_tokens = generated[i, prompt_len:]
        
        # Remove padding
        if pad_token_id is not None:
            pad_mask = generated_tokens != pad_token_id
            generated_tokens = generated_tokens[pad_mask]
        
        # Decode
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        results.append(text)
    
    return results