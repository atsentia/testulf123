#!/usr/bin/env python3
"""
Test weight loading with proper bfloat16 handling.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
from datetime import datetime

def log(message: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠", "PROGRESS": "⏳"}
    print(f"[{timestamp}] {prefix.get(level, '•')} {message}")

def load_weights_info(model_path: Path) -> Dict:
    """Load weight info without loading actual tensors."""
    try:
        from safetensors import safe_open
    except ImportError:
        log("safetensors not found, installing...", "WARNING")
        os.system("pip3 install --break-system-packages safetensors")
        from safetensors import safe_open
    
    log(f"Analyzing weights at {model_path}", "INFO")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    total_size = index["metadata"]["total_size"]
    
    log(f"Total model size: {total_size / (1024**3):.2f} GB", "INFO")
    log(f"Number of tensors: {len(weight_map)}", "INFO")
    
    # Analyze weight structure
    layer_counts = {}
    weight_types = {}
    
    for weight_name in weight_map.keys():
        # Count layer types
        if "layers" in weight_name:
            layer_num = weight_name.split(".layers.")[1].split(".")[0]
            layer_counts[layer_num] = layer_counts.get(layer_num, 0) + 1
        
        # Categorize weight types
        if "mlp.experts" in weight_name:
            if "_blocks" in weight_name:
                weight_types["mxfp4_blocks"] = weight_types.get("mxfp4_blocks", 0) + 1
            elif "_scales" in weight_name:
                weight_types["mxfp4_scales"] = weight_types.get("mxfp4_scales", 0) + 1
            else:
                weight_types["mlp_expert"] = weight_types.get("mlp_expert", 0) + 1
        elif "self_attn" in weight_name:
            if "sinks" in weight_name:
                weight_types["attention_sinks"] = weight_types.get("attention_sinks", 0) + 1
            else:
                weight_types["attention"] = weight_types.get("attention", 0) + 1
        elif "router" in weight_name:
            weight_types["router"] = weight_types.get("router", 0) + 1
        elif "layernorm" in weight_name:
            weight_types["layernorm"] = weight_types.get("layernorm", 0) + 1
        elif "embed" in weight_name:
            weight_types["embedding"] = weight_types.get("embedding", 0) + 1
        elif "lm_head" in weight_name:
            weight_types["lm_head"] = weight_types.get("lm_head", 0) + 1
        else:
            weight_types["other"] = weight_types.get("other", 0) + 1
    
    log(f"\nModel has {len(layer_counts)} transformer layers", "SUCCESS")
    
    log("\nWeight type distribution:", "INFO")
    for wtype, count in sorted(weight_types.items()):
        log(f"  {wtype}: {count} tensors", "INFO")
    
    # Check first file to understand data types
    first_file = list(set(weight_map.values()))[0]
    file_path = model_path / first_file
    
    log(f"\nAnalyzing data types from {first_file}...", "INFO")
    
    with safe_open(file_path, framework="pt") as f:  # Use PyTorch framework for bfloat16
        # Get metadata for first few tensors
        sample_count = 0
        for key in f.keys():
            if sample_count >= 5:
                break
            
            # Get tensor metadata without loading
            tensor_info = f.get_slice(key)
            log(f"  {key}:", "INFO")
            log(f"    Shape: {tensor_info.get_shape()}", "INFO")
            log(f"    Dtype: {tensor_info.get_dtype()}", "INFO")
            sample_count += 1
    
    return {
        "weight_map": weight_map,
        "total_size": total_size,
        "layer_count": len(layer_counts),
        "weight_types": weight_types
    }

def test_single_weight_load(model_path: Path):
    """Test loading a single weight tensor."""
    log("\nTesting single weight load...", "PROGRESS")
    
    try:
        # Try with PyTorch backend for bfloat16 support
        log("Attempting to load with PyTorch backend...", "INFO")
        
        import torch
        from safetensors.torch import load_file
        
        first_file = model_path / "model-00000-of-00002.safetensors"
        log(f"Loading first tensor from {first_file.name}...", "PROGRESS")
        
        start_time = time.time()
        tensors = load_file(str(first_file), device="cpu")
        load_time = time.time() - start_time
        
        # Get first tensor
        first_key = list(tensors.keys())[0]
        first_tensor = tensors[first_key]
        
        log(f"✓ Loaded {first_key}", "SUCCESS")
        log(f"  Shape: {first_tensor.shape}", "INFO")
        log(f"  Dtype: {first_tensor.dtype}", "INFO")
        log(f"  Memory: {first_tensor.element_size() * first_tensor.nelement() / (1024**2):.2f} MB", "INFO")
        log(f"  Load time: {load_time:.2f}s", "INFO")
        
        # Convert to numpy
        log("Converting to numpy...", "PROGRESS")
        np_tensor = first_tensor.float().numpy()  # Convert bfloat16 to float32 for numpy
        log(f"  Numpy shape: {np_tensor.shape}, dtype: {np_tensor.dtype}", "SUCCESS")
        
        return True
        
    except ImportError:
        log("PyTorch not available, installing...", "WARNING")
        log("This may take a while for CPU-only version...", "INFO")
        os.system("pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu")
        return False
    except Exception as e:
        log(f"Error loading weight: {e}", "ERROR")
        return False

def main():
    log("="*60, "INFO")
    log("GPT-OSS-20B Weight Analysis", "INFO")
    log("="*60, "INFO")
    
    model_path = Path("/root/models/gpt-oss-20b")
    
    if not model_path.exists():
        log(f"Model not found at {model_path}", "ERROR")
        return False
    
    # Analyze weight structure
    info = load_weights_info(model_path)
    
    # Test loading a single weight
    success = test_single_weight_load(model_path)
    
    if success:
        log("\n✅ Weight loading test successful!", "SUCCESS")
        log("The model uses bfloat16 format and requires PyTorch for conversion", "INFO")
    else:
        log("\n⚠ Weight loading needs PyTorch for bfloat16 support", "WARNING")
        log("Run again after PyTorch installation completes", "INFO")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)