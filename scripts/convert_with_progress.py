#!/usr/bin/env python3
"""
Convert GPT-OSS-20B weights with detailed progress tracking.
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Force CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import torch
from safetensors.torch import load_file

def log(message: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠", "STEP": "▶"}
    print(f"\n[{timestamp}] {prefix.get(level, '•')} {message}")

def format_size(bytes_val: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"

def dequantize_mxfp4_torch(blocks: torch.Tensor, scales: torch.Tensor) -> np.ndarray:
    """Dequantize MXFP4 weights to float32."""
    # Simple dequantization - treating as scaled int4
    # Convert to float and scale
    result = blocks.float() * scales.unsqueeze(-1)
    return result.cpu().numpy()

def convert_attention_layer(tensors: Dict[str, torch.Tensor], layer_idx: int) -> Dict[str, Any]:
    """Convert attention layer weights."""
    prefix = f"model.layers.{layer_idx}.self_attn"
    result = {}
    
    for proj in ["q", "k", "v", "o"]:
        weight_key = f"{prefix}.{proj}_proj.weight"
        bias_key = f"{prefix}.{proj}_proj.bias"
        
        if weight_key in tensors:
            # Transpose for JAX convention
            weight = tensors[weight_key].float().cpu().numpy()
            result[f"{proj}_proj"] = {"kernel": weight.T}
            
            if bias_key in tensors:
                bias = tensors[bias_key].float().cpu().numpy()
                result[f"{proj}_proj"]["bias"] = bias
    
    # Handle attention sinks
    sink_key = f"{prefix}.sinks"
    if sink_key in tensors:
        result["sinks"] = tensors[sink_key].float().cpu().numpy()
    
    return result

def convert_moe_layer(tensors: Dict[str, torch.Tensor], layer_idx: int, num_experts: int = 32) -> Dict[str, Any]:
    """Convert MoE layer with MXFP4 dequantization."""
    prefix = f"model.layers.{layer_idx}.mlp"
    result = {}
    
    # Router
    router_weight = f"{prefix}.router.weight"
    if router_weight in tensors:
        result["router"] = {
            "router_weights": tensors[router_weight].float().cpu().numpy().T
        }
    
    # Expert weights (MXFP4)
    gate_up_blocks = f"{prefix}.experts.gate_up_proj_blocks"
    gate_up_scales = f"{prefix}.experts.gate_up_proj_scales"
    down_blocks = f"{prefix}.experts.down_proj_blocks"
    down_scales = f"{prefix}.experts.down_proj_scales"
    
    # Process gate_up projections
    if gate_up_blocks in tensors and gate_up_scales in tensors:
        blocks = tensors[gate_up_blocks]
        scales = tensors[gate_up_scales]
        
        # Dequantize
        weights = dequantize_mxfp4_torch(blocks, scales)
        
        # Split by expert
        expert_size = weights.shape[0] // num_experts
        
        for i in range(num_experts):
            expert_weights = weights[i * expert_size:(i + 1) * expert_size]
            half_size = expert_weights.shape[-1] // 2
            
            result[f"expert_{i}"] = {
                "gate_proj": {"kernel": expert_weights[:, :half_size].T},
                "up_proj": {"kernel": expert_weights[:, half_size:].T}
            }
    
    # Process down projections
    if down_blocks in tensors and down_scales in tensors:
        blocks = tensors[down_blocks]
        scales = tensors[down_scales]
        
        # Dequantize
        weights = dequantize_mxfp4_torch(blocks, scales)
        
        # Split by expert
        expert_size = weights.shape[0] // num_experts
        
        for i in range(num_experts):
            expert_weights = weights[i * expert_size:(i + 1) * expert_size]
            
            if f"expert_{i}" not in result:
                result[f"expert_{i}"] = {}
            
            result[f"expert_{i}"]["down_proj"] = {"kernel": expert_weights.T}
    
    return result

def convert_weights_with_progress(model_path: Path, output_path: Path):
    """Convert weights with detailed progress tracking."""
    
    log("Starting GPT-OSS-20B Weight Conversion", "STEP")
    log(f"Input: {model_path}", "INFO")
    log(f"Output: {output_path}", "INFO")
    
    # Load config
    config_file = model_path / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_local_experts"]
    
    log(f"Model configuration: {num_layers} layers, {num_experts} experts per layer", "INFO")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Group by file
    files_to_load = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in files_to_load:
            files_to_load[file_name] = []
        files_to_load[file_name].append(weight_name)
    
    log(f"Will process {len(files_to_load)} safetensor files", "INFO")
    
    # Initialize JAX parameters structure
    jax_params = {"params": {}}
    
    # Process each file
    total_converted = 0
    start_time = time.time()
    
    for file_idx, (file_name, weight_names) in enumerate(files_to_load.items(), 1):
        file_path = model_path / file_name
        file_size = file_path.stat().st_size
        
        log(f"Processing file {file_idx}/{len(files_to_load)}: {file_name} ({format_size(file_size)})", "STEP")
        
        # Load file
        log("Loading tensors...", "INFO")
        tensors = load_file(str(file_path), device="cpu")
        
        # Convert embeddings
        if "model.embed_tokens.weight" in tensors:
            log("Converting embeddings...", "INFO")
            embed_weight = tensors["model.embed_tokens.weight"].float().cpu().numpy()
            jax_params["params"]["embed_tokens"] = {"embedding": embed_weight}
            total_converted += 1
        
        # Convert layers with progress bar
        layers_in_file = set()
        for name in weight_names:
            if "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                layers_in_file.add(layer_num)
        
        if layers_in_file:
            with tqdm(total=len(layers_in_file), desc="Converting layers", unit="layer") as pbar:
                for layer_idx in sorted(layers_in_file):
                    layer_params = {}
                    
                    # Layer norms
                    for norm_type in ["input_layernorm", "post_attention_layernorm"]:
                        norm_key = f"model.layers.{layer_idx}.{norm_type}.weight"
                        if norm_key in tensors:
                            layer_params[norm_type] = {
                                "scale": tensors[norm_key].float().cpu().numpy()
                            }
                    
                    # Attention
                    layer_params["self_attn"] = convert_attention_layer(tensors, layer_idx)
                    
                    # MoE
                    layer_params["mlp"] = convert_moe_layer(tensors, layer_idx, num_experts)
                    
                    jax_params["params"][f"layers_{layer_idx}"] = layer_params
                    total_converted += len(layer_params)
                    
                    pbar.update(1)
                    pbar.set_postfix({"converted": total_converted, "MB": f"{format_size(gc.get_stats()[0]['collected'])}"})
        
        # Convert final layers
        if "model.norm.weight" in tensors:
            log("Converting final norm...", "INFO")
            jax_params["params"]["norm"] = {
                "scale": tensors["model.norm.weight"].float().cpu().numpy()
            }
            total_converted += 1
        
        if "lm_head.weight" in tensors:
            log("Converting LM head...", "INFO")
            jax_params["params"]["lm_head"] = {
                "kernel": tensors["lm_head.weight"].float().cpu().numpy().T
            }
            total_converted += 1
        
        # Clear memory
        del tensors
        gc.collect()
        
        log(f"Completed file {file_idx}/{len(files_to_load)}, total converted: {total_converted}", "SUCCESS")
    
    conversion_time = time.time() - start_time
    log(f"Conversion completed in {conversion_time:.1f} seconds", "SUCCESS")
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    
    log(f"Saving to {output_path}...", "STEP")
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save JAX params (simplified for testing)
    import pickle
    params_file = output_path / "jax_params.pkl"
    
    with open(params_file, "wb") as f:
        pickle.dump(jax_params, f)
    
    params_size = params_file.stat().st_size
    log(f"Saved JAX parameters: {format_size(params_size)}", "SUCCESS")
    
    log(f"\n✅ Conversion complete!", "SUCCESS")
    log(f"  Total tensors converted: {total_converted}", "INFO")
    log(f"  Output location: {output_path}", "INFO")
    log(f"  Time taken: {conversion_time:.1f} seconds", "INFO")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/models/gpt-oss-20b")
    parser.add_argument("--output-path", default="/root/models/gpt-oss-20b-jax")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        log(f"Model path {model_path} not found!", "ERROR")
        return False
    
    try:
        convert_weights_with_progress(model_path, output_path)
        return True
    except Exception as e:
        log(f"Conversion failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)