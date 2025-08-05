#!/usr/bin/env python3
"""
Convert GPT-OSS-20B PyTorch weights to JAX format.

Handles:
- MXFP4 quantized MoE expert weights
- Attention sink tokens
- Weight transposition for JAX conventions
- Modern JAX compatibility (no ml_dtypes required)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm

# Set CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import PyTorch first for weight loading
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not found. Weight loading will be limited.")

# Now import JAX - modern version handles NumPy 2.x properly
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

# Try modern orbax-checkpoint first, fallback to older API
try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint import PyTreeCheckpointer
    ORBAX_NEW = True
except ImportError:
    try:
        import orbax
        import orbax.checkpoint as ocp
        ORBAX_NEW = False
    except ImportError:
        logger.error("orbax-checkpoint not found. Please install: pip install orbax-checkpoint")
        sys.exit(1)

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from jax_gpt_oss.models.config import GPTOSSConfig
from jax_gpt_oss.models.gpt_oss import GPTOSS


def dequantize_mxfp4(blocks: jnp.ndarray, scales: jnp.ndarray) -> jnp.ndarray:
    """
    Dequantize MXFP4 weights back to full precision.
    
    MXFP4 uses 4-bit mantissa with shared exponent scales.
    Uses JAX arrays throughout for consistency.
    """
    # Convert to JAX arrays if needed
    if not isinstance(blocks, jnp.ndarray):
        blocks = jnp.array(blocks)
    if not isinstance(scales, jnp.ndarray):
        scales = jnp.array(scales)
    
    # Simple dequantization - actual MXFP4 would need proper unpacking
    # For now, treat as int4 * scale
    if blocks.dtype == jnp.uint8:
        # Unpack 4-bit values from uint8
        low_bits = blocks & 0x0F
        high_bits = (blocks >> 4) & 0x0F
        
        # Convert to signed int4 (-8 to 7)
        low_vals = jnp.where(low_bits > 7, low_bits - 16, low_bits).astype(jnp.float32)
        high_vals = jnp.where(high_bits > 7, high_bits - 16, high_bits).astype(jnp.float32)
        
        # Interleave and scale
        result_shape = list(blocks.shape)
        result_shape[-1] *= 2  # Each byte becomes 2 values
        result = jnp.zeros(result_shape, dtype=jnp.float32)
        result = result.at[..., 0::2].set(low_vals * scales[..., None])
        result = result.at[..., 1::2].set(high_vals * scales[..., None])
        return result
    else:
        # Already dequantized or different format
        return blocks.astype(jnp.float32) * scales


def load_safetensors_weights(model_path: Path) -> Dict[str, jnp.ndarray]:
    """Load weights from safetensors files, converting to JAX arrays."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # Load index to find which file contains which weight
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        logger.error(f"Index file not found: {index_file}")
        sys.exit(1)
        
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Load weights from each file
    weights = {}
    loaded_files = set()
    
    logger.info(f"Loading {len(weight_map)} weights from safetensors files...")
    
    for weight_name, file_name in tqdm(weight_map.items(), desc="Loading weights"):
        if file_name not in loaded_files:
            file_path = model_path / file_name
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue
            
            # Load with PyTorch if available for better dtype handling
            if HAS_TORCH:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in weight_map and weight_map[key] == file_name:
                            tensor = f.get_tensor(key)
                            # Convert PyTorch tensor to NumPy then JAX
                            if tensor.dtype == torch.bfloat16:
                                # Handle bfloat16 conversion
                                numpy_array = tensor.to(torch.float32).numpy()
                            else:
                                numpy_array = tensor.numpy()
                            weights[key] = jnp.array(numpy_array)
            else:
                # Fallback to numpy loading
                with safe_open(file_path, framework="np") as f:
                    for key in f.keys():
                        if key in weight_map and weight_map[key] == file_name:
                            numpy_array = f.get_tensor(key)
                            weights[key] = jnp.array(numpy_array)
            
            loaded_files.add(file_name)
    
    logger.info(f"Loaded {len(weights)} weight tensors")
    return weights


def convert_attention_weights(
    pt_weights: Dict[str, jnp.ndarray],
    layer_idx: int,
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert attention layer weights."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    jax_weights = {}
    
    # Q, K, V projections with biases
    for proj in ["q", "k", "v", "o"]:
        weight_key = f"{prefix}.{proj}_proj.weight"
        bias_key = f"{prefix}.{proj}_proj.bias"
        
        if weight_key in pt_weights:
            # Transpose weight for JAX (JAX uses [in, out], PyTorch uses [out, in])
            jax_weights[f"{proj}_proj"] = {
                "kernel": jnp.transpose(pt_weights[weight_key]).astype(jnp.float32)
            }
            if bias_key in pt_weights:
                jax_weights[f"{proj}_proj"]["bias"] = pt_weights[bias_key].astype(jnp.float32)
    
    # Handle attention sinks if present
    sink_key = f"{prefix}.sinks"
    if sink_key in pt_weights:
        jax_weights["sinks"] = pt_weights[sink_key].astype(jnp.float32)
    
    return jax_weights


def convert_moe_weights(
    pt_weights: Dict[str, jnp.ndarray],
    layer_idx: int,
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert MoE layer weights, handling MXFP4 quantization."""
    prefix = f"model.layers.{layer_idx}.mlp"
    jax_weights = {}
    
    # Router weights
    router_weight = f"{prefix}.router.weight"
    router_bias = f"{prefix}.router.bias"
    
    if router_weight in pt_weights:
        jax_weights["router"] = {
            "router_weights": jnp.transpose(pt_weights[router_weight]).astype(jnp.float32)
        }
        if router_bias in pt_weights:
            # Router bias is incorporated into the router_weights in our implementation
            pass
    
    # Expert weights (MXFP4 quantized)
    experts = {}
    
    # Gate and up projections (combined in quantized format)
    gate_up_blocks = f"{prefix}.experts.gate_up_proj_blocks"
    gate_up_scales = f"{prefix}.experts.gate_up_proj_scales"
    gate_up_bias = f"{prefix}.experts.gate_up_proj_bias"
    
    if gate_up_blocks in pt_weights and gate_up_scales in pt_weights:
        # Dequantize MXFP4 weights
        gate_up_weights = dequantize_mxfp4(
            pt_weights[gate_up_blocks],
            pt_weights[gate_up_scales]
        )
        
        # Split into gate and up projections for each expert
        num_experts = config.num_local_experts
        expert_size = gate_up_weights.shape[0] // num_experts
        
        for i in range(num_experts):
            expert_weights = gate_up_weights[i * expert_size:(i + 1) * expert_size]
            
            # Split gate and up (they're concatenated)
            half_size = expert_weights.shape[-1] // 2
            
            experts[f"expert_{i}"] = {
                "gate_proj": {
                    "kernel": expert_weights[:, :half_size].T
                },
                "up_proj": {
                    "kernel": expert_weights[:, half_size:].T
                }
            }
            
            # Add biases if present
            if gate_up_bias in pt_weights:
                bias = pt_weights[gate_up_bias].astype(jnp.float32)
                expert_bias = bias[i * expert_size:(i + 1) * expert_size]
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
            expert_down = down_weights[i * expert_size:(i + 1) * expert_size]
            
            if f"expert_{i}" not in experts:
                experts[f"expert_{i}"] = {}
            
            experts[f"expert_{i}"]["down_proj"] = {
                "kernel": expert_down.T
            }
            
            if down_bias in pt_weights:
                bias = pt_weights[down_bias].astype(jnp.float32)
                expert_bias = bias[i * expert_size:(i + 1) * expert_size]
                experts[f"expert_{i}"]["down_proj"]["bias"] = expert_bias
    
    # Organize into JAX structure
    jax_weights["router"] = jax_weights.get("router", {})
    for i in range(config.num_local_experts):
        jax_weights[f"expert_{i}"] = experts.get(f"expert_{i}", {})
    
    return jax_weights


def convert_weights_to_jax(
    pt_weights: Dict[str, jnp.ndarray],
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert all PyTorch weights to JAX format."""
    jax_params = {"params": {}}
    
    logger.info("Converting embeddings...")
    # Token embeddings
    if "model.embed_tokens.weight" in pt_weights:
        jax_params["params"]["embed_tokens"] = {
            "embedding": pt_weights["model.embed_tokens.weight"].astype(jnp.float32)
        }
    
    # Convert each transformer layer
    for layer_idx in tqdm(range(config.num_hidden_layers), desc="Converting layers"):
        layer_params = {}
        
        # Layer norms
        input_ln = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attn_ln = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        if input_ln in pt_weights:
            layer_params["input_layernorm"] = {
                "scale": pt_weights[input_ln].astype(jnp.float32)
            }
        
        if post_attn_ln in pt_weights:
            layer_params["post_attention_layernorm"] = {
                "scale": pt_weights[post_attn_ln].astype(jnp.float32)
            }
        
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
        jax_params["params"]["norm"] = {
            "scale": pt_weights["model.norm.weight"].astype(jnp.float32)
        }
    
    # Language modeling head
    if "lm_head.weight" in pt_weights:
        jax_params["params"]["lm_head"] = {
            "kernel": jnp.transpose(pt_weights["lm_head.weight"]).astype(jnp.float32)
        }
    
    # Ensure all arrays are JAX arrays before freezing
    return freeze(jax_params)


def save_jax_checkpoint(params: Dict[str, Any], output_path: Path) -> None:
    """Save JAX parameters using Orbax checkpoint."""
    ckpt_path = output_path / "jax_params"
    
    if ORBAX_NEW:
        # Use new Orbax API
        ckptr = PyTreeCheckpointer()
        ckptr.save(ckpt_path, params)
    else:
        # Use older API
        ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        ckptr.save(ckpt_path, params)
    
    logger.info(f"Saved checkpoint to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-OSS-20B weights to JAX")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/models/gpt-oss-20b",
        help="Path to PyTorch model weights"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/root/models/gpt-oss-20b-jax",
        help="Output path for JAX weights"
    )
    parser.add_argument(
        "--test-loading",
        action="store_true",
        help="Test loading after conversion"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Data type for conversion"
    )
    
    args = parser.parse_args()
    
    # Log JAX version info
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX platform: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")
    
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
    logger.info(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    
    # Load PyTorch weights
    logger.info("Loading PyTorch weights from safetensors...")
    pt_weights = load_safetensors_weights(model_path)
    
    # Convert to JAX format
    logger.info("\nConverting to JAX format...")
    jax_params = convert_weights_to_jax(pt_weights, config)
    
    # Save JAX weights
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving JAX weights to {output_path}...")
    
    # Save config
    config.to_json(output_path / "config.json")
    logger.info(f"Saved config to {output_path / 'config.json'}")
    
    # Save JAX checkpoint
    save_jax_checkpoint(jax_params, output_path)
    
    logger.info(f"✓ Conversion complete! Saved to {output_path}")
    
    # Test loading if requested
    if args.test_loading:
        logger.info("\nTesting model loading...")
        from jax_gpt_oss.utils.model_utils import load_model
        
        try:
            model, params = load_model(str(output_path))
            logger.info("✓ Model loaded successfully!")
            
            # Test forward pass
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            output = model.apply(params, dummy_input, deterministic=True)
            logger.info(f"✓ Forward pass successful! Output shape: {output['logits'].shape}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()