#!/usr/bin/env python3
"""
Verbose weight conversion with detailed progress tracking.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"

def log(message: str, level: str = "INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠"}
    print(f"[{timestamp}] {prefix.get(level, '•')} {message}")

def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0

def load_safetensors_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load weights with detailed progress."""
    try:
        from safetensors import safe_open
    except ImportError:
        log("Installing safetensors...", "WARNING")
        os.system("pip3 install --break-system-packages safetensors")
        from safetensors import safe_open
    
    log(f"Loading weights from {model_path}", "INFO")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    log(f"Reading index from {index_file}", "INFO")
    
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    total_size_bytes = index["metadata"]["total_size"]
    total_size_gb = total_size_bytes / (1024**3)
    
    log(f"Total model size: {total_size_gb:.2f} GB", "INFO")
    log(f"Number of weight tensors: {len(weight_map)}", "INFO")
    
    # Group weights by file
    files_to_load = {}
    for weight_name, file_name in weight_map.items():
        if file_name not in files_to_load:
            files_to_load[file_name] = []
        files_to_load[file_name].append(weight_name)
    
    log(f"Loading from {len(files_to_load)} safetensor files", "INFO")
    
    weights = {}
    loaded_bytes = 0
    
    for file_idx, (file_name, weight_names) in enumerate(files_to_load.items(), 1):
        file_path = model_path / file_name
        file_size = get_file_size_mb(file_path)
        
        log(f"[{file_idx}/{len(files_to_load)}] Loading {file_name} ({file_size:.1f} MB)", "INFO")
        
        if not file_path.exists():
            log(f"File not found: {file_path}", "ERROR")
            continue
        
        start_time = time.time()
        
        with safe_open(file_path, framework="np") as f:
            with tqdm(total=len(weight_names), desc=f"  Extracting weights", unit="tensor") as pbar:
                for weight_name in weight_names:
                    tensor = f.get_tensor(weight_name)
                    weights[weight_name] = tensor
                    loaded_bytes += tensor.nbytes
                    pbar.update(1)
                    pbar.set_postfix({"MB": f"{loaded_bytes/(1024**2):.1f}"})
        
        load_time = time.time() - start_time
        speed_mbps = file_size / load_time if load_time > 0 else 0
        log(f"  Loaded in {load_time:.1f}s ({speed_mbps:.1f} MB/s)", "SUCCESS")
    
    loaded_gb = loaded_bytes / (1024**3)
    log(f"Total loaded: {loaded_gb:.2f} GB", "SUCCESS")
    
    return weights

def test_basic_conversion():
    """Quick test of weight loading and basic conversion."""
    log("="*60, "INFO")
    log("GPT-OSS-20B Weight Conversion Test", "INFO")
    log("="*60, "INFO")
    
    model_path = Path("/root/models/gpt-oss-20b")
    
    if not model_path.exists():
        log(f"Model not found at {model_path}", "ERROR")
        log("Please download first with: bash scripts/download_model.sh", "INFO")
        return False
    
    # Check files
    log("Checking model files...", "INFO")
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors"
    ]
    
    missing = []
    for file in required_files:
        file_path = model_path / file
        if file_path.exists():
            size_mb = get_file_size_mb(file_path)
            log(f"  ✓ {file} ({size_mb:.1f} MB)", "SUCCESS")
        else:
            log(f"  ✗ {file} (missing)", "ERROR")
            missing.append(file)
    
    if missing:
        log(f"Missing {len(missing)} required files", "ERROR")
        return False
    
    # Load a few weights to test
    log("\nTesting weight loading...", "INFO")
    weights = load_safetensors_weights(model_path)
    
    # Check some key weights
    log("\nChecking weight shapes...", "INFO")
    sample_weights = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.router.weight",
        "lm_head.weight"
    ]
    
    for weight_name in sample_weights:
        if weight_name in weights:
            shape = weights[weight_name].shape
            dtype = weights[weight_name].dtype
            size_mb = weights[weight_name].nbytes / (1024**2)
            log(f"  {weight_name}: shape={shape}, dtype={dtype}, size={size_mb:.2f}MB", "SUCCESS")
        else:
            log(f"  {weight_name}: NOT FOUND", "WARNING")
    
    # Check for MXFP4 quantized weights
    log("\nChecking for MXFP4 quantized weights...", "INFO")
    mxfp4_patterns = ["_blocks", "_scales"]
    quantized_weights = [k for k in weights.keys() if any(p in k for p in mxfp4_patterns)]
    
    if quantized_weights:
        log(f"  Found {len(quantized_weights)} quantized weight tensors", "SUCCESS")
        # Show a few examples
        for weight in quantized_weights[:3]:
            log(f"    - {weight}", "INFO")
    else:
        log("  No MXFP4 quantized weights found", "WARNING")
    
    # Memory usage
    total_memory_mb = sum(w.nbytes for w in weights.values()) / (1024**2)
    log(f"\nTotal memory used: {total_memory_mb:.1f} MB", "INFO")
    
    log("\n✅ Basic weight loading test passed!", "SUCCESS")
    return True

if __name__ == "__main__":
    success = test_basic_conversion()
    sys.exit(0 if success else 1)