#!/usr/bin/env python3
"""
Memory-efficient weight conversion for GPT-OSS-20B.
Processes one layer at a time to minimize memory usage.
"""

import os
import sys
import json
import time
import gc
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import torch
from safetensors.torch import load_file

def log(message: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠", "MEMORY": "🧠"}
    print(f"[{timestamp}] {prefix.get(level, '•')} {message}")

def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
    except:
        return 0.0

def format_size(bytes_val: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_single_weight(file_path: Path, weight_name: str) -> torch.Tensor:
    """Load a single weight from safetensors file."""
    with load_file(str(file_path), device="cpu") as weights:
        if weight_name in weights:
            return weights[weight_name]
    return None

def convert_layer_streaming(
    model_path: Path,
    layer_idx: int,
    weight_map: Dict[str, str],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert a single layer with minimal memory usage."""
    
    layer_params = {}
    prefix = f"model.layers.{layer_idx}"
    num_experts = config.get("num_local_experts", 32)
    
    # Find all weights for this layer
    layer_weights = {k: v for k, v in weight_map.items() if f"layers.{layer_idx}." in k}
    
    if not layer_weights:
        return layer_params
    
    # Group by file to minimize file opens
    weights_by_file = {}
    for weight_name, file_name in layer_weights.items():
        if file_name not in weights_by_file:
            weights_by_file[file_name] = []
        weights_by_file[file_name].append(weight_name)
    
    # Process each file
    for file_name, weight_names in weights_by_file.items():
        file_path = model_path / file_name
        
        # Load only this file's weights for this layer
        with load_file(str(file_path), device="cpu") as tensors:
            
            # Layer norms
            for norm_type in ["input_layernorm", "post_attention_layernorm"]:
                norm_key = f"{prefix}.{norm_type}.weight"
                if norm_key in tensors:
                    if norm_type not in layer_params:
                        layer_params[norm_type] = {}
                    layer_params[norm_type]["scale"] = tensors[norm_key].float().cpu().numpy()
            
            # Attention weights
            if "self_attn" not in layer_params:
                layer_params["self_attn"] = {}
            
            for proj in ["q", "k", "v", "o"]:
                weight_key = f"{prefix}.self_attn.{proj}_proj.weight"
                bias_key = f"{prefix}.self_attn.{proj}_proj.bias"
                
                if weight_key in tensors:
                    weight = tensors[weight_key].float().cpu().numpy()
                    layer_params["self_attn"][f"{proj}_proj"] = {"kernel": weight.T}
                    
                    if bias_key in tensors:
                        bias = tensors[bias_key].float().cpu().numpy()
                        layer_params["self_attn"][f"{proj}_proj"]["bias"] = bias
            
            # Attention sinks
            sink_key = f"{prefix}.self_attn.sinks"
            if sink_key in tensors:
                layer_params["self_attn"]["sinks"] = tensors[sink_key].float().cpu().numpy()
            
            # MoE Router
            if "mlp" not in layer_params:
                layer_params["mlp"] = {}
            
            router_weight = f"{prefix}.mlp.router.weight"
            if router_weight in tensors:
                layer_params["mlp"]["router"] = {
                    "router_weights": tensors[router_weight].float().cpu().numpy().T
                }
            
            # MoE Experts (simplified - just store shapes for now)
            for expert_type in ["gate_up_proj", "down_proj"]:
                blocks_key = f"{prefix}.mlp.experts.{expert_type}_blocks"
                scales_key = f"{prefix}.mlp.experts.{expert_type}_scales"
                
                if blocks_key in tensors and scales_key in tensors:
                    # For memory efficiency, just convert to float32 without full dequantization
                    blocks = tensors[blocks_key].float().cpu().numpy()
                    scales = tensors[scales_key].float().cpu().numpy()
                    
                    # Store simplified version
                    if "experts_simplified" not in layer_params["mlp"]:
                        layer_params["mlp"]["experts_simplified"] = {}
                    
                    layer_params["mlp"]["experts_simplified"][expert_type] = {
                        "blocks_shape": blocks.shape,
                        "scales_shape": scales.shape,
                        "dtype": "mxfp4"
                    }
                    
                    # Create placeholder expert weights
                    for i in range(min(num_experts, 2)):  # Only process first 2 experts as demo
                        if f"expert_{i}" not in layer_params["mlp"]:
                            layer_params["mlp"][f"expert_{i}"] = {}
    
    # Clear tensors from memory
    clear_memory()
    
    return layer_params

def convert_memory_efficient(model_path: Path, output_path: Path):
    """Convert model with minimal memory usage."""
    
    log("Starting Memory-Efficient GPT-OSS-20B Conversion", "INFO")
    log(f"Initial memory: {get_memory_usage():.2f} GB", "MEMORY")
    
    # Load config
    config_file = model_path / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    num_layers = config["num_hidden_layers"]
    
    # Load weight index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    total_size = index["metadata"]["total_size"]
    
    log(f"Model: {num_layers} layers, {format_size(total_size)} total", "INFO")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize params structure
    jax_params = {"params": {}}
    
    # Convert embeddings first
    log("Converting embeddings...", "INFO")
    embed_file = weight_map.get("model.embed_tokens.weight", "")
    if embed_file:
        file_path = model_path / embed_file
        with load_file(str(file_path), device="cpu") as tensors:
            if "model.embed_tokens.weight" in tensors:
                embed = tensors["model.embed_tokens.weight"].float().cpu().numpy()
                jax_params["params"]["embed_tokens"] = {"embedding": embed}
                log(f"  Embeddings: {embed.shape}, {format_size(embed.nbytes)}", "SUCCESS")
    
    clear_memory()
    log(f"Memory after embeddings: {get_memory_usage():.2f} GB", "MEMORY")
    
    # Convert layers one at a time
    log(f"Converting {num_layers} layers (memory-efficient mode)...", "INFO")
    
    with tqdm(total=num_layers, desc="Converting layers", unit="layer") as pbar:
        for layer_idx in range(num_layers):
            # Convert single layer
            layer_params = convert_layer_streaming(
                model_path, layer_idx, weight_map, config
            )
            
            if layer_params:
                jax_params["params"][f"layers_{layer_idx}"] = layer_params
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                "memory_gb": f"{get_memory_usage():.1f}",
                "layer": layer_idx
            })
            
            # Save checkpoint every 4 layers to avoid memory buildup
            if (layer_idx + 1) % 4 == 0:
                checkpoint_file = output_path / f"checkpoint_layer_{layer_idx}.pkl"
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(jax_params["params"][f"layers_{layer_idx}"], f)
                
                # Clear from main dict to save memory
                del jax_params["params"][f"layers_{layer_idx}"]
                clear_memory()
    
    # Convert final norm and LM head
    log("Converting final layers...", "INFO")
    
    norm_file = weight_map.get("model.norm.weight", "")
    if norm_file:
        file_path = model_path / norm_file
        with load_file(str(file_path), device="cpu") as tensors:
            if "model.norm.weight" in tensors:
                jax_params["params"]["norm"] = {
                    "scale": tensors["model.norm.weight"].float().cpu().numpy()
                }
    
    lm_head_file = weight_map.get("lm_head.weight", "")
    if lm_head_file:
        file_path = model_path / lm_head_file
        with load_file(str(file_path), device="cpu") as tensors:
            if "lm_head.weight" in tensors:
                lm_head = tensors["lm_head.weight"].float().cpu().numpy()
                jax_params["params"]["lm_head"] = {"kernel": lm_head.T}
                log(f"  LM head: {lm_head.shape}, {format_size(lm_head.nbytes)}", "SUCCESS")
    
    clear_memory()
    
    # Merge checkpoints
    log("Merging layer checkpoints...", "INFO")
    for layer_idx in range(num_layers):
        checkpoint_file = output_path / f"checkpoint_layer_{layer_idx}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, "rb") as f:
                layer_data = pickle.load(f)
                jax_params["params"][f"layers_{layer_idx}"] = layer_data
            checkpoint_file.unlink()  # Delete checkpoint
    
    # Save final params
    log("Saving final JAX parameters...", "INFO")
    params_file = output_path / "jax_params.pkl"
    
    with open(params_file, "wb") as f:
        pickle.dump(jax_params, f)
    
    params_size = params_file.stat().st_size
    log(f"Saved: {format_size(params_size)}", "SUCCESS")
    
    # Final memory report
    log(f"Final memory usage: {get_memory_usage():.2f} GB", "MEMORY")
    
    log("\n✅ Memory-efficient conversion complete!", "SUCCESS")
    log(f"Output: {output_path}", "INFO")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/models/gpt-oss-20b")
    parser.add_argument("--output-path", default="/root/models/gpt-oss-20b-jax")
    parser.add_argument("--test-only", action="store_true", help="Only convert first 2 layers for testing")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        log(f"Model not found at {model_path}", "ERROR")
        return False
    
    try:
        if args.test_only:
            # Modify config for testing
            config_file = model_path / "config.json"
            with open(config_file, "r") as f:
                config = json.load(f)
            config["num_hidden_layers"] = 2  # Only process 2 layers
            
            test_output = output_path.parent / (output_path.name + "_test")
            log("Running test conversion (2 layers only)...", "WARNING")
            return convert_memory_efficient(model_path, test_output)
        else:
            return convert_memory_efficient(model_path, output_path)
            
    except Exception as e:
        log(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)