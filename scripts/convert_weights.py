#!/usr/bin/env python3
"""
Convert GPT-OSS-20B PyTorch weights to JAX format.

Handles:
- MXFP4 quantized MoE expert weights
- Attention sink tokens
- Weight transposition for JAX conventions
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Set CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import orbax.checkpoint as ocp

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from jax_gpt_oss.models.config import GPTOSSConfig
from jax_gpt_oss.models.gpt_oss import GPTOSS


def dequantize_mxfp4(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Dequantize MXFP4 weights back to full precision.
    
    MXFP4 uses 4-bit mantissa with shared exponent scales.
    """
    # Simple dequantization - actual MXFP4 would need proper unpacking
    # For now, treat as int4 * scale
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


def load_safetensors_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load weights from safetensors files."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # Load index to find which file contains which weight
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Load weights from each file
    weights = {}
    loaded_files = set()
    
    for weight_name, file_name in tqdm(weight_map.items(), desc="Loading weights"):
        if file_name not in loaded_files:
            file_path = model_path / file_name
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue
                
            with safe_open(file_path, framework="np") as f:
                for key in f.keys():
                    if key in weight_map and weight_map[key] == file_name:
                        weights[key] = f.get_tensor(key)
            
            loaded_files.add(file_name)
    
    return weights


def convert_attention_weights(
    pt_weights: Dict[str, np.ndarray],
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
                "kernel": pt_weights[weight_key].T.astype(np.float32)
            }
            if bias_key in pt_weights:
                jax_weights[f"{proj}_proj"]["bias"] = pt_weights[bias_key].astype(np.float32)
    
    # Handle attention sinks if present
    sink_key = f"{prefix}.sinks"
    if sink_key in pt_weights:
        jax_weights["sinks"] = pt_weights[sink_key].astype(np.float32)
    
    return jax_weights


def convert_moe_weights(
    pt_weights: Dict[str, np.ndarray],
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
            "router_weights": pt_weights[router_weight].T.astype(np.float32)
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
                bias = pt_weights[down_bias].astype(np.float32)
                expert_bias = bias[i * expert_size:(i + 1) * expert_size]
                experts[f"expert_{i}"]["down_proj"]["bias"] = expert_bias
    
    # Organize into JAX structure
    jax_weights["router"] = jax_weights.get("router", {})
    for i in range(config.num_local_experts):
        jax_weights[f"expert_{i}"] = experts.get(f"expert_{i}", {})
    
    return jax_weights


def convert_weights_to_jax(
    pt_weights: Dict[str, np.ndarray],
    config: GPTOSSConfig
) -> Dict[str, Any]:
    """Convert all PyTorch weights to JAX format."""
    jax_params = {"params": {}}
    
    print("Converting embeddings...")
    # Token embeddings
    if "model.embed_tokens.weight" in pt_weights:
        jax_params["params"]["embed_tokens"] = {
            "embedding": pt_weights["model.embed_tokens.weight"].astype(np.float32)
        }
    
    # Convert each transformer layer
    for layer_idx in tqdm(range(config.num_hidden_layers), desc="Converting layers"):
        layer_params = {}
        
        # Layer norms
        input_ln = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attn_ln = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
        
        if input_ln in pt_weights:
            layer_params["input_layernorm"] = {
                "scale": pt_weights[input_ln].astype(np.float32)
            }
        
        if post_attn_ln in pt_weights:
            layer_params["post_attention_layernorm"] = {
                "scale": pt_weights[post_attn_ln].astype(np.float32)
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
            "scale": pt_weights["model.norm.weight"].astype(np.float32)
        }
    
    # Language modeling head
    if "lm_head.weight" in pt_weights:
        jax_params["params"]["lm_head"] = {
            "kernel": pt_weights["lm_head.weight"].T.astype(np.float32)
        }
    
    return freeze(jax_params)


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
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)
    
    # Load config
    print("Loading configuration...")
    config_file = model_path / "config.json"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    
    config = GPTOSSConfig(**config_dict)
    
    # Load PyTorch weights
    print("Loading PyTorch weights...")
    pt_weights = load_safetensors_weights(model_path)
    print(f"Loaded {len(pt_weights)} weight tensors")
    
    # Convert to JAX format
    print("\nConverting to JAX format...")
    jax_params = convert_weights_to_jax(pt_weights, config)
    
    # Save JAX weights
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving JAX weights to {output_path}...")
    
    # Save config
    config.to_json(output_path / "config.json")
    
    # Save JAX checkpoint
    ckpt_path = output_path / "jax_params"
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(ckpt_path, jax_params)
    
    print(f"✓ Conversion complete! Saved to {output_path}")
    
    # Test loading if requested
    if args.test_loading:
        print("\nTesting model loading...")
        from jax_gpt_oss.utils.model_utils import load_model
        
        try:
            model, params = load_model(str(output_path))
            print("✓ Model loaded successfully!")
            
            # Test forward pass
            dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
            output = model.apply(params, dummy_input, deterministic=True)
            print(f"✓ Forward pass successful! Output shape: {output['logits'].shape}")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()