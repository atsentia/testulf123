#!/usr/bin/env python3
"""
Hybrid converter that handles bfloat16 by direct file parsing.
Uses PyTorch for working weights, manual parsing for bfloat16.
"""

import os
import json
import gc
import struct
from pathlib import Path
from typing import Dict, Any, List
import argparse
import pickle

import numpy as np
from safetensors import safe_open

try:
    import torch
except ImportError:
    print("❌ PyTorch required for hybrid approach")
    exit(1)

MAX_SHARD_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB

def parse_safetensors_header(file_path: Path) -> Dict[str, Any]:
    """Parse safetensors file to get tensor metadata."""
    with open(file_path, 'rb') as f:
        # Read header length (first 8 bytes)
        header_size = struct.unpack('<Q', f.read(8))[0]
        
        # Read and parse JSON header
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
        # Return metadata and data offset
        return header, 8 + header_size

def read_bfloat16_tensor(file_path: Path, tensor_info: Dict[str, Any], data_offset: int) -> np.ndarray:
    """Read bfloat16 tensor directly from file bytes."""
    shape = tensor_info['shape']
    data_start = data_offset + tensor_info['data_offsets'][0]
    data_end = data_offset + tensor_info['data_offsets'][1]
    
    with open(file_path, 'rb') as f:
        f.seek(data_start)
        raw_bytes = f.read(data_end - data_start)
    
    # Convert bfloat16 bytes to float32
    # bfloat16 is 16-bit with same exponent as float32
    uint16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
    
    # Expand bfloat16 to float32 by shifting to upper 16 bits
    uint32_data = np.zeros(len(uint16_data), dtype=np.uint32)
    uint32_data = uint16_data.astype(np.uint32) << 16
    
    float32_data = uint32_data.view(np.float32)
    
    # Convert to float16 for storage efficiency
    float16_data = float32_data.astype(np.float16)
    
    return float16_data.reshape(shape)

def load_all_weights_hybrid(model_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load weights using hybrid approach: PyTorch + manual bfloat16."""
    print("🔄 Loading with HYBRID approach (PyTorch + manual bfloat16)...")
    
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
    
    print(f"📂 Processing {len(files)} files with hybrid approach:")
    
    all_weights = {}
    total_params = 0
    success_count = 0
    bfloat16_count = 0
    
    for file_name in files:
        file_path = model_path / file_name
        print(f"\n📥 Loading {file_name}...")
        
        # Parse safetensors header for manual access
        try:
            header, data_offset = parse_safetensors_header(file_path)
        except Exception as e:
            print(f"  ❌ Failed to parse header: {e}")
            continue
        
        # First, try PyTorch backend for compatible tensors
        with safe_open(file_path, framework="pt") as f:
            file_keys = [k for k in f.keys() if weight_map.get(k) == file_name]
            
            for key in file_keys:
                try:
                    # Try PyTorch backend first
                    tensor = f.get_tensor(key)
                    
                    # Convert to float16 if needed
                    if hasattr(tensor, 'numpy'):
                        if tensor.dtype.name == 'bfloat16':
                            np_tensor = tensor.to(torch.float16).numpy()
                        else:
                            np_tensor = tensor.numpy()
                            if np_tensor.dtype == np.float32:
                                np_tensor = np_tensor.astype(np.float16)
                    else:
                        np_tensor = tensor
                        if np_tensor.dtype == np.float32:
                            np_tensor = np_tensor.astype(np.float16)
                    
                    all_weights[key] = np_tensor
                    params = np.prod(np_tensor.shape)
                    total_params += params
                    success_count += 1
                    
                    print(f"  ✅ {key}: {np_tensor.shape} → {np_tensor.dtype} ({np_tensor.nbytes/(1024**2):.1f}MB)")
                    
                except Exception as e:
                    # If PyTorch fails, try manual bfloat16 parsing
                    if "bfloat16" in str(e).lower() or "BFloat16" in str(e):
                        print(f"  🔧 {key}: Manual bfloat16 parsing...")
                        
                        try:
                            # Get tensor info from header
                            if key in header:
                                tensor_info = header[key]
                                
                                # Read bfloat16 tensor manually
                                np_tensor = read_bfloat16_tensor(file_path, tensor_info, data_offset)
                                
                                all_weights[key] = np_tensor
                                params = np.prod(np_tensor.shape)
                                total_params += params
                                success_count += 1
                                bfloat16_count += 1
                                
                                print(f"  ✅ {key}: {np_tensor.shape} → float16 (from bfloat16) ({np_tensor.nbytes/(1024**2):.1f}MB)")
                            else:
                                print(f"  ❌ {key}: Not found in header")
                                
                        except Exception as e2:
                            print(f"  ❌ {key}: Manual parsing failed - {e2}")
                    else:
                        print(f"  ❌ {key}: {e}")
        
        gc.collect()
        print(f"  📊 File: {success_count} total, {bfloat16_count} from manual bfloat16")
    
    print(f"\n🎯 FINAL TOTAL: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    print(f"✅ Successfully loaded {success_count} weights")
    print(f"🔧 Manual bfloat16 conversions: {bfloat16_count}")
    
    return all_weights, config

# Use the same sharding functions from previous converters
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
    total_size = 0
    
    for i, shard in enumerate(shards):
        shard_path = output_dir / f"shard_{i:03d}.pkl"
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = shard_path.stat().st_size / (1024**2)
        total_size += size_mb
        shard_info.append({
            "file": shard_path.name,
            "size_mb": round(size_mb, 1)
        })
        
        print(f"  📦 {shard_path.name}: {size_mb:.1f}MB")
    
    # Save manifest
    manifest = {
        "format": "hybrid_float16_complete",
        "total_shards": len(shards),
        "total_size_gb": round(total_size / 1024, 2),
        "config": config,
        "shards": shard_info,
        "conversion_notes": "Hybrid approach: PyTorch backend + manual bfloat16 parsing"
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Hybrid converter with manual bfloat16 handling")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 HYBRID CONVERTER (PyTorch + Manual bfloat16)")
    print("=" * 65)
    print("🎯 Goal: Extract ALL parameters via hybrid approach")
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Load ALL weights with hybrid method
    weights, config = load_all_weights_hybrid(model_path)
    
    if not weights:
        print("❌ No weights loaded")
        return
    
    # Create nested structure
    nested_weights = create_nested_structure(weights)
    
    # Create shards
    shards = create_shards(nested_weights)
    
    # Save shards
    manifest_path = save_shards(shards, config, output_dir)
    
    print(f"\n🎉 HYBRID CONVERSION COMPLETE!")
    print(f"📊 Total weights: {len(weights)}")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    print(f"🎯 Hybrid approach successfully handled bfloat16!")

if __name__ == "__main__":
    main()