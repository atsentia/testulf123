#!/usr/bin/env python3
"""
Minimal test of weight conversion - just convert embeddings and first layer.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import torch
from safetensors.torch import load_file

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

def test_minimal_conversion():
    """Test converting just embeddings and first layer."""
    
    log("=== Minimal Conversion Test ===")
    
    model_path = Path("/root/models/gpt-oss-20b")
    output_path = Path("/root/models/gpt-oss-20b-jax-test")
    
    if not model_path.exists():
        log("ERROR: Model not found")
        return False
    
    # Load config
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
    
    log(f"Model: {config['num_hidden_layers']} layers, {config['vocab_size']} vocab")
    
    # Load index
    with open(model_path / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find embeddings file
    embed_weight_name = "model.embed_tokens.weight"
    embed_file = weight_map.get(embed_weight_name)
    
    if not embed_file:
        log("ERROR: Embeddings not found")
        return False
    
    log(f"Loading embeddings from {embed_file}...")
    
    # Load embeddings
    file_path = model_path / embed_file
    start_time = time.time()
    
    tensors = load_file(str(file_path), device="cpu")
    
    if embed_weight_name in tensors:
        embed = tensors[embed_weight_name]
        log(f"  Shape: {embed.shape}")
        log(f"  Dtype: {embed.dtype}")
        log(f"  Size: {embed.element_size() * embed.nelement() / (1024**2):.1f} MB")
        
        # Convert to numpy
        embed_np = embed.float().cpu().numpy()
        log(f"  Converted to numpy: {embed_np.shape}, {embed_np.dtype}")
    else:
        log("ERROR: Embedding tensor not found")
        return False
    
    load_time = time.time() - start_time
    log(f"  Load time: {load_time:.2f}s")
    
    # Find first layer weights
    log("\nLooking for layer 0 weights...")
    layer_0_weights = [k for k in weight_map.keys() if "layers.0." in k]
    log(f"  Found {len(layer_0_weights)} layer 0 tensors")
    
    # Show some examples
    for weight_name in layer_0_weights[:5]:
        file_name = weight_map[weight_name]
        log(f"    {weight_name} -> {file_name}")
    
    # Load one attention weight as test
    test_weight = "model.layers.0.self_attn.q_proj.weight"
    if test_weight in weight_map:
        file_name = weight_map[test_weight]
        file_path = model_path / file_name
        
        log(f"\nLoading {test_weight}...")
        tensors = load_file(str(file_path), device="cpu")
        
        if test_weight in tensors:
            weight = tensors[test_weight]
            log(f"  Shape: {weight.shape}")
            log(f"  Dtype: {weight.dtype}")
            
            # Convert
            weight_np = weight.float().cpu().numpy()
            log(f"  Converted: {weight_np.shape}, {weight_np.dtype}")
            
            # JAX format (transpose)
            weight_jax = weight_np.T
            log(f"  JAX format: {weight_jax.shape}")
    
    # Create output
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save test params
    import pickle
    test_params = {
        "params": {
            "embed_tokens": {"embedding": embed_np},
            "test_weight": weight_jax if 'weight_jax' in locals() else None
        }
    }
    
    with open(output_path / "test_params.pkl", "wb") as f:
        pickle.dump(test_params, f)
    
    log(f"\n✅ Test conversion successful!")
    log(f"Output saved to {output_path}")
    
    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / (1024**3)
        log(f"Memory used: {mem_gb:.2f} GB")
    except:
        pass
    
    return True

if __name__ == "__main__":
    success = test_minimal_conversion()
    sys.exit(0 if success else 1)