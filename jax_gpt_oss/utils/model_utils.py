"""Model loading and initialization utilities for GPT-OSS-20B."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import orbax.checkpoint as ocp

from ..models.gpt_oss import GPTOSS
from ..models.config import GPTOSSConfig


def initialize_model(
    config: GPTOSSConfig,
    rng: jax.random.PRNGKey,
    dtype: Any = None,
) -> Tuple[GPTOSS, Dict[str, Any]]:
    """
    Initialize a GPT-OSS-20B model with random parameters.
    
    Args:
        config: Model configuration
        rng: JAX random key
        dtype: Data type for parameters (defaults to config.dtype)
        
    Returns:
        Tuple of (model, params)
    """
    if dtype is None:
        dtype = getattr(jnp, config.dtype) if isinstance(config.dtype, str) else config.dtype
    
    model = GPTOSS(config=config, dtype=dtype)
    
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
    
    # Initialize parameters
    params = model.init(rng, dummy_input, deterministic=True)
    
    return model, params


def load_model(
    model_path: str,
    dtype: Any = jnp.bfloat16,
    device: Optional[str] = None,
) -> Tuple[GPTOSS, Dict[str, Any]]:
    """
    Load a GPT-OSS-20B model from disk.
    
    Args:
        model_path: Path to model directory
        dtype: Data type for model parameters
        device: Device to load model on (optional)
        
    Returns:
        Tuple of (model, params)
    """
    model_path = Path(model_path)
    
    # Load configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = GPTOSSConfig(**config_dict)
    
    # Initialize model structure
    rng = jax.random.PRNGKey(0)
    model, dummy_params = initialize_model(config, rng, dtype)
    
    # Check for JAX checkpoint
    jax_checkpoint_path = model_path / "jax_params"
    
    if jax_checkpoint_path.exists():
        # Load JAX checkpoint
        print(f"Loading JAX checkpoint from {jax_checkpoint_path}")
        ckptr = ocp.PyTreeCheckpointer()
        params = ckptr.restore(jax_checkpoint_path)
    else:
        # Try to load and convert from PyTorch weights
        print(f"JAX checkpoint not found, attempting to convert from PyTorch weights...")
        params = convert_from_pytorch(model_path, config, dtype)
    
    # Convert to specified dtype if needed
    if dtype != jnp.float32:
        params = jax.tree_map(lambda x: x.astype(dtype), params)
    
    return model, params


def save_model(
    model: GPTOSS,
    params: Dict[str, Any],
    save_path: str,
    config: GPTOSSConfig,
):
    """
    Save a GPT-OSS-20B model to disk.
    
    Args:
        model: Model instance
        params: Model parameters
        save_path: Path to save directory
        config: Model configuration
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.to_json(save_path / "config.json")
    
    # Save JAX parameters
    jax_checkpoint_path = save_path / "jax_params"
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(jax_checkpoint_path, params)
    
    print(f"Model saved to {save_path}")


def convert_from_pytorch(
    model_path: Path,
    config: GPTOSSConfig,
    dtype: Any = jnp.bfloat16
) -> Dict[str, Any]:
    """
    Convert PyTorch weights to JAX format.
    
    This is a placeholder - actual implementation would load
    safetensors files and convert layer by layer.
    """
    try:
        import safetensors
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch and safetensors required for weight conversion. "
            "Install with: pip install torch safetensors"
        )
    
    # Load safetensors files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")
    
    # Load index file
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
            weight_map = index.get("weight_map", {})
    else:
        weight_map = {}
    
    # This would be implemented with actual conversion logic
    # For now, initialize with random weights as placeholder
    print("Warning: PyTorch to JAX conversion not fully implemented yet")
    print("Initializing with random weights instead")
    
    rng = jax.random.PRNGKey(42)
    model, params = initialize_model(config, rng, dtype)
    
    return params