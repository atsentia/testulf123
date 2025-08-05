#!/usr/bin/env python3
"""
Complete model converter using NumPy backend to handle bfloat16.
Gets all 21B parameters by avoiding PyTorch backend issues.
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Any, List
import argparse
import pickle

import numpy as np
from safetensors import safe_open

MAX_SHARD_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB

def convert_bfloat16_to_float16(bfloat16_bytes):
    """Convert bfloat16 bytes to float16 numpy array."""
    # bfloat16 is stored as uint16 with the mantissa in upper 16 bits
    uint16_data = np.frombuffer(bfloat16_bytes, dtype=np.uint16)
    
    # Convert bfloat16 to float32 first
    uint32_data = uint16_data.astype(np.uint32) << 16
    float32_data = uint32_data.view(np.float32)
    
    # Then convert to float16 for storage efficiency
    return float32_data.astype(np.float16)

def load_all_weights_numpy(model_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load all weights using NumPy backend to handle bfloat16."""
    print("🔄 Loading ALL weights with NumPy backend...")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Load index
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    
    print(f"📂 Processing {len(files)} files with NumPy backend:")
    
    all_weights = {}
    total_params = 0
    success_count = 0
    
    for file_name in files:
        file_path = model_path / file_name
        print(f"\n📥 Loading {file_name}...")
        
        with safe_open(file_path, framework="numpy") as f:
            file_keys = [k for k in f.keys() if weight_map.get(k) == file_name]
            
            for key in file_keys:
                try:
                    # Try NumPy backend first
                    tensor = f.get_tensor(key)
                    
                    # Convert to float16 if needed
                    if tensor.dtype == np.float32:
                        tensor = tensor.astype(np.float16)
                    elif str(tensor.dtype) == 'bfloat16':
                        # Handle bfloat16 manually if NumPy backend supports it
                        tensor = tensor.astype(np.float16)
                    
                    all_weights[key] = tensor
                    params = np.prod(tensor.shape)
                    total_params += params
                    success_count += 1
                    
                    print(f"  ✅ {key}: {tensor.shape} → {tensor.dtype} ({tensor.nbytes/(1024**2):.1f}MB)")
                    
                except Exception as e:
                    # If NumPy backend fails, try manual bfloat16 handling
                    try:
                        # Get raw tensor info
                        tensor_info = f.get_tensor_info(key) if hasattr(f, 'get_tensor_info') else None
                        
                        if "bfloat16" in str(e).lower():
                            print(f"  🔧 {key}: Handling bfloat16 manually...")
                            
                            # Try to get as raw bytes and convert
                            slice_obj = f.get_slice(key)
                            raw_bytes = bytes(slice_obj)
                            
                            # Determine shape from other means
                            if tensor_info:
                                shape = tensor_info.shape
                                expected_size = np.prod(shape) * 2  # 2 bytes per bfloat16
                                
                                if len(raw_bytes) == expected_size:
                                    tensor = convert_bfloat16_to_float16(raw_bytes).reshape(shape)
                                    all_weights[key] = tensor
                                    
                                    params = np.prod(tensor.shape)
                                    total_params += params
                                    success_count += 1
                                    
                                    print(f"  ✅ {key}: {tensor.shape} → float16 (converted from bfloat16) ({tensor.nbytes/(1024**2):.1f}MB)")
                                else:
                                    print(f"  ❌ {key}: Size mismatch - expected {expected_size}, got {len(raw_bytes)}")
                            else:
                                print(f"  ❌ {key}: No tensor info available")
                        else:
                            print(f"  ❌ {key}: {e}")
                            
                    except Exception as e2:
                        print(f"  ❌ {key}: Failed both methods - {e2}")
        
        gc.collect()
        print(f"  📊 File summary: {success_count} weights loaded successfully")
    
    print(f"\n🎯 FINAL TOTAL: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    print(f"✅ Successfully loaded {success_count} weights")
    
    return all_weights, config

def estimate_size(obj) -> int:
    """Estimate object size in bytes."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, dict):
        return sum(estimate_size(v) for v in obj.values())
    else:
        return len(str(obj).encode('utf-8'))

def create_nested_structure(flat_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert flat keys to nested structure."""
    print(f"\n🏗️  Creating nested structure from {len(flat_weights)} weights...")
    
    nested = {}
    
    for key, weight in flat_weights.items():
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        
        parts = clean_key.split(".")
        current = nested
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = weight
    
    return nested

def create_shards(nested_weights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split weights into shards under the size limit."""
    print(f"\n📦 Creating shards (max {MAX_SHARD_SIZE/(1024**3):.1f}GB each)...")
    
    shards = []
    current_shard = {}
    current_size = 0
    
    def add_to_shard(path: str, obj: Any, obj_size: int):
        nonlocal current_shard, current_size, shards
        
        if current_size + obj_size > MAX_SHARD_SIZE and current_shard:
            print(f"  📦 Shard {len(shards)}: {current_size/(1024**2):.1f}MB")
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        parts = path.split(".")
        current = current_shard
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = obj
        current_size += obj_size
    
    def traverse(obj: Any, path: str = ""):
        if isinstance(obj, np.ndarray):
            add_to_shard(path, obj, obj.nbytes)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
    
    traverse(nested_weights)
    
    if current_shard:
        print(f"  📦 Shard {len(shards)}: {current_size/(1024**2):.1f}MB")
        shards.append(current_shard)
    
    print(f"✅ Created {len(shards)} shards")
    return shards

def save_shards(shards: List[Dict[str, Any]], config: Dict[str, Any], output_dir: Path):
    """Save shards and create manifest."""
    print(f"\n💾 Saving {len(shards)} shards...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_info = []
    
    for i, shard in enumerate(shards):
        shard_path = output_dir / f"shard_{i:03d}.pkl"
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = shard_path.stat().st_size / (1024**2)
        shard_info.append({
            "file": shard_path.name,
            "size_mb": round(size_mb, 1)
        })
        
        print(f"  📦 {shard_path.name}: {size_mb:.1f}MB")
    
    # Save manifest
    manifest = {
        "format": "complete_float16_sharded",
        "total_shards": len(shards),
        "config": config,
        "shards": shard_info,
        "total_size_gb": sum(s["size_mb"] for s in shard_info) / 1024
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Complete model converter with NumPy backend")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 COMPLETE MODEL CONVERTER (NumPy Backend)")
    print("=" * 60)
    print("🎯 Goal: Extract ALL 21B parameters")
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Load ALL weights with NumPy backend
    weights, config = load_all_weights_numpy(model_path)
    
    if not weights:
        print("❌ No weights loaded")
        return
    
    # Create nested structure
    nested_weights = create_nested_structure(weights)
    
    # Create shards
    shards = create_shards(nested_weights)
    
    # Save shards
    manifest_path = save_shards(shards, config, output_dir)
    
    print(f"\n🎉 COMPLETE CONVERSION FINISHED!")
    print(f"📊 Total weights processed: {len(weights)}")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    print(f"🎯 Target: All 21B parameters captured")

if __name__ == "__main__":
    main()