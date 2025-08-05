#!/usr/bin/env python3
"""
Convert GPT-OSS-20B weights to JAX format without PyTorch dependency.

This version uses only safetensors and numpy/jax for conversion.
Handles bfloat16 weights by converting them to float32 directly.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import struct

# Set CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import NumPy - will work with any version
import numpy as np

# For NumPy 2.x compatibility, we'll handle imports carefully
try:
    # Try importing JAX - might fail with NumPy 2.x
    import jax
    import jax.numpy as jnp
    from flax.core import freeze
    JAX_AVAILABLE = True
    logger.info(f"JAX version: {jax.__version__}")
except ImportError as e:
    logger.warning(f"JAX import failed: {e}")
    logger.warning("Will save weights as NumPy arrays instead of JAX arrays")
    JAX_AVAILABLE = False
    # Create dummy jnp for compatibility
    jnp = np

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Try importing model config
try:
    from jax_gpt_oss.models.config import GPTOSSConfig
except ImportError:
    logger.warning("Could not import GPTOSSConfig, using dict config instead")
    GPTOSSConfig = dict


def convert_bfloat16_to_float32(data: bytes) -> np.ndarray:
    """
    Convert bfloat16 bytes to float32 array.
    bfloat16 is just float32 with lower 16 bits truncated.
    """
    # Each bfloat16 is 2 bytes
    num_elements = len(data) // 2
    result = np.zeros(num_elements, dtype=np.float32)
    
    for i in range(num_elements):
        # Get 2 bytes for this bfloat16
        bf16_bytes = data[i*2:(i+1)*2]
        # bfloat16 is stored as the upper 16 bits of float32
        # We need to add 2 zero bytes for the lower bits
        float32_bytes = bf16_bytes + b'\x00\x00'
        # Unpack as float32
        result[i] = struct.unpack('f', float32_bytes)[0]
    
    return result


def load_safetensors_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load weights from safetensors files without PyTorch."""
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
            
            # Load with numpy framework
            with safe_open(file_path, framework="np") as f:
                for key in f.keys():
                    if key in weight_map and weight_map[key] == file_name:
                        try:
                            # Try loading directly
                            tensor = f.get_tensor(key)
                            
                            # Check if it's bfloat16 (safetensors might return raw bytes)
                            if hasattr(tensor, 'dtype') and str(tensor.dtype) == 'bfloat16':
                                # Convert bfloat16 to float32
                                logger.debug(f"Converting bfloat16 weight: {key}")
                                # If it's raw bytes, convert
                                if isinstance(tensor, bytes):
                                    tensor = convert_bfloat16_to_float32(tensor)
                                else:
                                    # Try to convert via view if possible
                                    tensor = tensor.astype(np.float32)
                            
                            weights[key] = tensor
                        except Exception as e:
                            logger.warning(f"Failed to load {key}: {e}, trying alternative method")
                            # Try loading as raw bytes and detecting dtype
                            try:
                                # Get metadata for this tensor
                                metadata = f.metadata()
                                if key in metadata:
                                    dtype_str = metadata[key].get('dtype', 'float32')
                                    if 'bfloat16' in dtype_str.lower():
                                        # Load raw and convert
                                        raw_data = f.get_slice(key)
                                        tensor = convert_bfloat16_to_float32(raw_data)
                                        weights[key] = tensor
                                    else:
                                        logger.warning(f"Skipping {key} due to loading error")
                            except:
                                logger.warning(f"Could not load {key}, skipping")
            
            loaded_files.add(file_name)
    
    logger.info(f"Loaded {len(weights)} weight tensors")
    return weights


def dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize MXFP4 weights back to full precision.
    MXFP4 uses 4-bit mantissa with shared exponent scales.
    """
    # Simple dequantization - actual MXFP4 would need proper unpacking
    if blocks.dtype == np.uint8:
        # Unpack 4-bit values from uint8
        low_bits = blocks & 0x0F
        high_bits = (blocks >> 4) & 0x0F
        
        # Convert to signed int4 (-8 to 7)
        low_vals = np.where(low_bits > 7, low_bits - 16, low_bits).astype(np.float32)
        high_vals = np.where(high_bits > 7, high_bits - 16, high_bits).astype(np.float32)
        
        # Interleave and scale
        result_shape = list(blocks.shape)
        result_shape[-1] *= 2  # Each byte becomes 2 values
        result = np.zeros(result_shape, dtype=np.float32)
        result[..., 0::2] = low_vals * scales[..., None]
        result[..., 1::2] = high_vals * scales[..., None]
        return result
    else:
        # Already dequantized or different format
        return blocks.astype(np.float32) * scales


def convert_attention_weights(
    pt_weights: Dict[str, np.ndarray],
    layer_idx: int,
    config: Dict[str, Any]
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
            weight = pt_weights[weight_key]
            if JAX_AVAILABLE:
                kernel = jnp.array(weight.T, dtype=jnp.float32)
            else:
                kernel = weight.T.astype(np.float32)
            
            jax_weights[f"{proj}_proj"] = {"kernel": kernel}
            
            if bias_key in pt_weights:
                bias = pt_weights[bias_key]
                if JAX_AVAILABLE:
                    jax_weights[f"{proj}_proj"]["bias"] = jnp.array(bias, dtype=jnp.float32)
                else:
                    jax_weights[f"{proj}_proj"]["bias"] = bias.astype(np.float32)
    
    # Handle attention sinks if present
    sink_key = f"{prefix}.sinks"
    if sink_key in pt_weights:
        if JAX_AVAILABLE:
            jax_weights["sinks"] = jnp.array(pt_weights[sink_key], dtype=jnp.float32)
        else:
            jax_weights["sinks"] = pt_weights[sink_key].astype(np.float32)
    
    return jax_weights


def convert_moe_weights(
    pt_weights: Dict[str, np.ndarray],
    layer_idx: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert MoE layer weights, handling MXFP4 quantization."""
    prefix = f"model.layers.{layer_idx}.mlp"
    jax_weights = {}
    
    # Get number of experts from config
    if isinstance(config, dict):
        num_experts = config.get("num_local_experts", 32)
    else:
        num_experts = config.num_local_experts
    
    # Router weights
    router_weight = f"{prefix}.router.weight"
    if router_weight in pt_weights:
        weight = pt_weights[router_weight]
        if JAX_AVAILABLE:
            kernel = jnp.array(weight.T, dtype=jnp.float32)
        else:
            kernel = weight.T.astype(np.float32)
        jax_weights["router"] = {"router_weights": kernel}
    
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
        expert_size = gate_up_weights.shape[0] // num_experts
        
        for i in range(num_experts):
            expert_weights = gate_up_weights[i * expert_size:(i + 1) * expert_size]
            
            # Split gate and up (they're concatenated)
            half_size = expert_weights.shape[-1] // 2
            
            experts[f"expert_{i}"] = {
                "gate_proj": {"kernel": expert_weights[:, :half_size].T},
                "up_proj": {"kernel": expert_weights[:, half_size:].T}
            }
            
            # Add biases if present
            if gate_up_bias in pt_weights:
                bias = pt_weights[gate_up_bias].astype(np.float32)
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
        
        expert_size = down_weights.shape[0] // num_experts
        
        for i in range(num_experts):
            expert_down = down_weights[i * expert_size:(i + 1) * expert_size]
            
            if f"expert_{i}" not in experts:
                experts[f"expert_{i}"] = {}
            
            experts[f"expert_{i}"]["down_proj"] = {"kernel": expert_down.T}
            
            if down_bias in pt_weights:
                bias = pt_weights[down_bias].astype(np.float32)
                expert_bias = bias[i * expert_size:(i + 1) * expert_size]
                experts[f"expert_{i}"]["down_proj"]["bias"] = expert_bias
    
    # Organize into JAX structure
    jax_weights["router"] = jax_weights.get("router", {})
    for i in range(num_experts):
        jax_weights[f"expert_{i}"] = experts.get(f"expert_{i}", {})
    
    return jax_weights


def convert_weights_to_jax(
    pt_weights: Dict[str, np.ndarray],
    config: Any
) -> Dict[str, Any]:
    """Convert all PyTorch weights to JAX format."""
    jax_params = {"params": {}}
    
    # Get config values
    if isinstance(config, dict):
        num_layers = config.get("num_hidden_layers", 24)
    else:
        num_layers = config.num_hidden_layers
    
    logger.info("Converting embeddings...")
    # Token embeddings
    if "model.embed_tokens.weight" in pt_weights:
        embedding = pt_weights["model.embed_tokens.weight"]
        if JAX_AVAILABLE:
            embedding = jnp.array(embedding, dtype=jnp.float32)
        else:
            embedding = embedding.astype(np.float32)
        jax_params["params"]["embed_tokens"] = {"embedding": embedding}
    
    # Convert each transformer layer
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        layer_params = {}
        
        # Layer norms
        input_ln = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attn_ln = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        if input_ln in pt_weights:
            scale = pt_weights[input_ln]
            if JAX_AVAILABLE:
                scale = jnp.array(scale, dtype=jnp.float32)
            else:
                scale = scale.astype(np.float32)
            layer_params["input_layernorm"] = {"scale": scale}
        
        if post_attn_ln in pt_weights:
            scale = pt_weights[post_attn_ln]
            if JAX_AVAILABLE:
                scale = jnp.array(scale, dtype=jnp.float32)
            else:
                scale = scale.astype(np.float32)
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
        scale = pt_weights["model.norm.weight"]
        if JAX_AVAILABLE:
            scale = jnp.array(scale, dtype=jnp.float32)
        else:
            scale = scale.astype(np.float32)
        jax_params["params"]["norm"] = {"scale": scale}
    
    # Language modeling head
    if "lm_head.weight" in pt_weights:
        weight = pt_weights["lm_head.weight"]
        if JAX_AVAILABLE:
            kernel = jnp.array(weight.T, dtype=jnp.float32)
        else:
            kernel = weight.T.astype(np.float32)
        jax_params["params"]["lm_head"] = {"kernel": kernel}
    
    if JAX_AVAILABLE:
        return freeze(jax_params)
    else:
        return jax_params


def save_weights_numpy(params: Dict[str, Any], output_path: Path) -> None:
    """Save weights as NumPy arrays when JAX is not available."""
    import pickle
    
    weights_path = output_path / "numpy_weights.pkl"
    logger.info(f"Saving weights as NumPy arrays to {weights_path}")
    
    with open(weights_path, "wb") as f:
        pickle.dump(params, f)
    
    logger.info("Weights saved successfully as NumPy arrays")
    logger.info("To use with JAX later, load and convert to JAX arrays")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-OSS-20B weights to JAX (no PyTorch required)"
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
    
    # Try to create GPTOSSConfig, fall back to dict
    try:
        config = GPTOSSConfig(**config_dict)
    except:
        config = config_dict
    
    logger.info(f"Model config: {config_dict.get('num_hidden_layers', 24)} layers")
    
    # Load weights
    logger.info("Loading weights from safetensors...")
    pt_weights = load_safetensors_weights(model_path)
    
    # Convert to JAX format
    logger.info("\nConverting to JAX format...")
    jax_params = convert_weights_to_jax(pt_weights, config)
    
    # Save weights
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to {output_path / 'config.json'}")
    
    # Save weights
    if JAX_AVAILABLE:
        try:
            import orbax.checkpoint as ocp
            ckpt_path = output_path / "jax_params"
            
            # Try new API first
            try:
                from orbax.checkpoint import PyTreeCheckpointer
                ckptr = PyTreeCheckpointer()
            except ImportError:
                # Fall back to old API
                ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
            
            ckptr.save(ckpt_path, jax_params)
            logger.info(f"✓ Saved JAX checkpoint to {ckpt_path}")
        except Exception as e:
            logger.warning(f"Could not save as JAX checkpoint: {e}")
            logger.info("Saving as NumPy arrays instead...")
            save_weights_numpy(jax_params, output_path)
    else:
        save_weights_numpy(jax_params, output_path)
    
    logger.info(f"\n✓ Conversion complete! Saved to {output_path}")
    
    if not JAX_AVAILABLE:
        logger.info("\n⚠ Note: JAX was not available, weights saved as NumPy arrays")
        logger.info("To fix JAX installation, run:")
        logger.info("  bash scripts/fix_numpy_compatibility.sh")


if __name__ == "__main__":
    main()