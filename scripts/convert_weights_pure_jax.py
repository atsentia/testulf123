#!/usr/bin/env python3
"""
Convert GPT-OSS-20B weights to JAX format using pure JAX (minimal NumPy dependency).

This version uses jax.numpy throughout and minimizes NumPy usage.
Only uses NumPy where absolutely necessary (safetensors loading).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm

# Set CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import JAX - this is all we really need!
import jax
import jax.numpy as jnp
from jax import Array
from flax.core import freeze

logger.info(f"JAX version: {jax.__version__}")
logger.info(f"JAX backend: {jax.default_backend()}")

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from jax_gpt_oss.models.config import GPTOSSConfig
except ImportError:
    logger.warning("Could not import GPTOSSConfig, using dict config instead")
    class GPTOSSConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def to_json(self, path):
            with open(path, 'w') as f:
                json.dump({k: v for k, v in self.__dict__.items() 
                          if not k.startswith('_')}, f, indent=2)


def load_safetensors_weights_minimal(model_path: Path) -> Dict[str, Array]:
    """
    Load weights from safetensors with minimal NumPy usage.
    Immediately converts everything to JAX arrays.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # We need numpy just for safetensors loading, but we'll immediately convert to JAX
    import numpy as np
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        sys.exit(1)
        
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Load weights and immediately convert to JAX
    weights = {}
    loaded_files = set()
    
    logger.info(f"Loading {len(weight_map)} weights from safetensors...")
    
    for weight_name, file_name in tqdm(weight_map.items(), desc="Loading weights"):
        if file_name not in loaded_files:
            file_path = model_path / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue
            
            # Try loading with PyTorch for better bfloat16 handling
            try:
                import torch
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in weight_map and weight_map[key] == file_name:
                            tensor = f.get_tensor(key)
                            # Convert PyTorch -> NumPy -> JAX immediately
                            if tensor.dtype == torch.bfloat16:
                                # Convert bfloat16 to float32 first
                                np_array = tensor.to(torch.float32).numpy()
                            else:
                                np_array = tensor.numpy()
                            # Immediately convert to JAX array
                            weights[key] = jnp.array(np_array, dtype=jnp.float32)
            except ImportError:
                # Fallback to numpy loading
                logger.info("PyTorch not available, using numpy loading (bfloat16 weights may not load)")
                with safe_open(file_path, framework="np") as f:
                    for key in f.keys():
                        if key in weight_map and weight_map[key] == file_name:
                            try:
                                np_array = f.get_tensor(key)
                                # Immediately convert to JAX
                                weights[key] = jnp.array(np_array, dtype=jnp.float32)
                            except Exception as e:
                                logger.warning(f"Could not load {key}: {e}")
            
            loaded_files.add(file_name)
    
    logger.info(f"Loaded {len(weights)} weight tensors as JAX arrays")
    return weights


def dequantize_mxfp4(blocks: Array, scales: Array) -> Array:
    """
    Dequantize MXFP4 weights using pure JAX operations.
    """
    # Ensure inputs are JAX arrays
    if not isinstance(blocks, Array):
        blocks = jnp.array(blocks)
    if not isinstance(scales, Array):
        scales = jnp.array(scales)
    
    if blocks.dtype == jnp.uint8:
        # Unpack 4-bit values from uint8 using JAX operations
        low_bits = blocks & 0x0F
        high_bits = (blocks >> 4) & 0x0F
        
        # Convert to signed int4 (-8 to 7)
        low_vals = jnp.where(low_bits > 7, low_bits - 16, low_bits).astype(jnp.float32)
        high_vals = jnp.where(high_bits > 7, high_bits - 16, high_bits).astype(jnp.float32)
        
        # Create result array
        result_shape = list(blocks.shape)
        result_shape[-1] *= 2
        
        # Stack and reshape to interleave values
        scaled_low = low_vals * scales[..., None]
        scaled_high = high_vals * scales[..., None]
        
        # Interleave using stack and reshape
        stacked = jnp.stack([scaled_low, scaled_high], axis=-1)
        result = stacked.reshape(result_shape)
        
        return result
    else:
        # Already dequantized or different format
        return blocks.astype(jnp.float32) * scales


def convert_attention_weights(
    pt_weights: Dict[str, Array],
    layer_idx: int,
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert attention layer weights using JAX arrays."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    jax_weights = {}
    
    # Q, K, V, O projections with biases
    for proj in ["q", "k", "v", "o"]:
        weight_key = f"{prefix}.{proj}_proj.weight"
        bias_key = f"{prefix}.{proj}_proj.bias"
        
        if weight_key in pt_weights:
            # Transpose for JAX convention using JAX operations
            weight = pt_weights[weight_key]
            kernel = jnp.transpose(weight).astype(jnp.float32)
            
            jax_weights[f"{proj}_proj"] = {"kernel": kernel}
            
            if bias_key in pt_weights:
                bias = pt_weights[bias_key].astype(jnp.float32)
                jax_weights[f"{proj}_proj"]["bias"] = bias
    
    # Handle attention sinks if present
    sink_key = f"{prefix}.sinks"
    if sink_key in pt_weights:
        jax_weights["sinks"] = pt_weights[sink_key].astype(jnp.float32)
    
    return jax_weights


def convert_moe_weights(
    pt_weights: Dict[str, Array],
    layer_idx: int,
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert MoE layer weights using JAX arrays."""
    prefix = f"model.layers.{layer_idx}.mlp"
    jax_weights = {}
    
    # Router weights
    router_weight = f"{prefix}.router.weight"
    if router_weight in pt_weights:
        weight = pt_weights[router_weight]
        kernel = jnp.transpose(weight).astype(jnp.float32)
        jax_weights["router"] = {"router_weights": kernel}
    
    # Expert weights (MXFP4 quantized)
    experts = {}
    
    # Gate and up projections
    gate_up_blocks = f"{prefix}.experts.gate_up_proj_blocks"
    gate_up_scales = f"{prefix}.experts.gate_up_proj_scales"
    gate_up_bias = f"{prefix}.experts.gate_up_proj_bias"
    
    if gate_up_blocks in pt_weights and gate_up_scales in pt_weights:
        # Dequantize using JAX operations
        gate_up_weights = dequantize_mxfp4(
            pt_weights[gate_up_blocks],
            pt_weights[gate_up_scales]
        )
        
        # Split for each expert
        num_experts = config.num_local_experts
        expert_size = gate_up_weights.shape[0] // num_experts
        
        for i in range(num_experts):
            start_idx = i * expert_size
            end_idx = (i + 1) * expert_size
            expert_weights = jax.lax.dynamic_slice(
                gate_up_weights, 
                (start_idx, 0), 
                (expert_size, gate_up_weights.shape[1])
            )
            
            # Split gate and up projections
            half_size = expert_weights.shape[-1] // 2
            gate_weights = expert_weights[:, :half_size]
            up_weights = expert_weights[:, half_size:]
            
            experts[f"expert_{i}"] = {
                "gate_proj": {"kernel": jnp.transpose(gate_weights)},
                "up_proj": {"kernel": jnp.transpose(up_weights)}
            }
            
            # Add biases if present
            if gate_up_bias in pt_weights:
                bias = pt_weights[gate_up_bias].astype(jnp.float32)
                expert_bias = jax.lax.dynamic_slice(
                    bias, (start_idx,), (expert_size,)
                )
                experts[f"expert_{i}"]["gate_proj"]["bias"] = expert_bias[:half_size]
                experts[f"expert_{i}"]["up_proj"]["bias"] = expert_bias[half_size:]
    
    # Down projections
    down_blocks = f"{prefix}.experts.down_proj_blocks"
    down_scales = f"{prefix}.experts.down_proj_scales"
    down_bias = f"{prefix}.experts.down_proj_bias"
    
    if down_blocks in pt_weights and down_scales in pt_weights:
        down_weights = dequantize_mxfp4(
            pt_weights[down_blocks],
            pt_weights[down_scales]
        )
        
        num_experts = config.num_local_experts
        expert_size = down_weights.shape[0] // num_experts
        
        for i in range(num_experts):
            start_idx = i * expert_size
            end_idx = (i + 1) * expert_size
            expert_down = jax.lax.dynamic_slice(
                down_weights,
                (start_idx, 0),
                (expert_size, down_weights.shape[1])
            )
            
            if f"expert_{i}" not in experts:
                experts[f"expert_{i}"] = {}
            
            experts[f"expert_{i}"]["down_proj"] = {
                "kernel": jnp.transpose(expert_down)
            }
            
            if down_bias in pt_weights:
                bias = pt_weights[down_bias].astype(jnp.float32)
                expert_bias = jax.lax.dynamic_slice(
                    bias, (start_idx,), (expert_size,)
                )
                experts[f"expert_{i}"]["down_proj"]["bias"] = expert_bias
    
    # Organize into final structure
    jax_weights["router"] = jax_weights.get("router", {})
    for i in range(config.num_local_experts):
        jax_weights[f"expert_{i}"] = experts.get(f"expert_{i}", {})
    
    return jax_weights


def convert_weights_to_jax(
    pt_weights: Dict[str, Array],
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert all weights to JAX format using pure JAX operations."""
    jax_params = {"params": {}}
    
    logger.info("Converting embeddings...")
    # Token embeddings
    if "model.embed_tokens.weight" in pt_weights:
        embedding = pt_weights["model.embed_tokens.weight"].astype(jnp.float32)
        jax_params["params"]["embed_tokens"] = {"embedding": embedding}
    
    # Convert transformer layers
    for layer_idx in tqdm(range(config.num_hidden_layers), desc="Converting layers"):
        layer_params = {}
        
        # Layer norms
        input_ln = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attn_ln = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        if input_ln in pt_weights:
            scale = pt_weights[input_ln].astype(jnp.float32)
            layer_params["input_layernorm"] = {"scale": scale}
        
        if post_attn_ln in pt_weights:
            scale = pt_weights[post_attn_ln].astype(jnp.float32)
            layer_params["post_attention_layernorm"] = {"scale": scale}
        
        # Attention weights
        layer_params["self_attn"] = convert_attention_weights(
            pt_weights, layer_idx, config
        )
        
        # MoE weights
        layer_params["mlp"] = convert_moe_weights(
            pt_weights, layer_idx, config
        )
        
        jax_params["params"][f"layers_{layer_idx}"] = layer_params
    
    # Final layer norm
    if "model.norm.weight" in pt_weights:
        scale = pt_weights["model.norm.weight"].astype(jnp.float32)
        jax_params["params"]["norm"] = {"scale": scale}
    
    # Language modeling head
    if "lm_head.weight" in pt_weights:
        weight = pt_weights["lm_head.weight"]
        kernel = jnp.transpose(weight).astype(jnp.float32)
        jax_params["params"]["lm_head"] = {"kernel": kernel}
    
    return freeze(jax_params)


def save_jax_checkpoint(params: Dict[str, Any], output_path: Path) -> None:
    """Save JAX parameters using Orbax checkpoint."""
    try:
        import orbax.checkpoint as ocp
        
        ckpt_path = output_path / "jax_params"
        
        # Try new API first
        try:
            from orbax.checkpoint import PyTreeCheckpointer
            ckptr = PyTreeCheckpointer()
        except ImportError:
            # Fall back to older API
            ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        
        ckptr.save(ckpt_path, params)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        
    except ImportError:
        logger.error("orbax-checkpoint not installed. Please install: pip install orbax-checkpoint")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-OSS-20B weights to JAX using pure JAX operations"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gpt-oss-20b",
        help="Path to model weights"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/gpt-oss-20b-jax",
        help="Output path for JAX weights"
    )
    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test loading after conversion"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist")
        sys.exit(1)
    
    # Load config
    logger.info("Loading configuration...")
    config_file = model_path / "config.json"
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
        
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    
    config = GPTOSSConfig(**config_dict)
    logger.info(f"Model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    
    # Load weights - minimal NumPy usage, immediate JAX conversion
    logger.info("Loading weights from safetensors...")
    pt_weights = load_safetensors_weights_minimal(model_path)
    
    # Convert using pure JAX operations
    logger.info("\nConverting to JAX format...")
    jax_params = convert_weights_to_jax(pt_weights, config)
    
    # Save outputs
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.to_json(output_path / "config.json")
    logger.info(f"Saved config to {output_path / 'config.json'}")
    
    # Save JAX checkpoint
    logger.info(f"\nSaving JAX weights to {output_path}...")
    save_jax_checkpoint(jax_params, output_path)
    
    logger.info(f"\n✓ Conversion complete! Saved to {output_path}")
    
    # Test loading if requested
    if args.test_loading:
        logger.info("\nTesting model loading...")
        try:
            from jax_gpt_oss.utils.model_utils import load_model
            
            model, params = load_model(str(output_path))
            logger.info("✓ Model loaded successfully!")
            
            # Test forward pass
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            output = model.apply(params, dummy_input, deterministic=True)
            logger.info(f"✓ Forward pass successful! Output shape: {output['logits'].shape}")
            
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()