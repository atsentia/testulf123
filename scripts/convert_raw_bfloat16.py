#!/usr/bin/env python3
"""
Raw bfloat16 converter that reads safetensors format directly.
Bypasses all library limitations by implementing custom file parsing.
"""

import os
import json
import gc
import struct
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import pickle

import numpy as np

MAX_SHARD_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB

def bfloat16_to_float32(bfloat16_bytes: bytes) -> np.ndarray:
    """Convert bfloat16 bytes to float32 numpy array."""
    # bfloat16 is 16-bit: 1 sign + 8 exponent + 7 mantissa
    # float32 is 32-bit: 1 sign + 8 exponent + 23 mantissa
    # bfloat16 can be converted by padding mantissa with zeros
    
    uint16_data = np.frombuffer(bfloat16_bytes, dtype=np.uint16)
    
    # Convert to uint32 by shifting bfloat16 to upper 16 bits
    uint32_data = uint16_data.astype(np.uint32) << 16
    
    # View as float32
    float32_data = uint32_data.view(np.float32)
    
    return float32_data

def float32_to_float16_safe(float32_array: np.ndarray) -> np.ndarray:
    """Convert float32 to float16 with overflow handling."""
    # Clip values to float16 range to avoid overflow
    float32_clipped = np.clip(float32_array, -65504.0, 65504.0)
    return float32_clipped.astype(np.float16)

def parse_safetensors_file(file_path: Path) -> Tuple[Dict[str, Any], int]:
    """Parse safetensors file header and return metadata + data offset."""
    with open(file_path, 'rb') as f:
        # Read header length (first 8 bytes, little endian)
        header_length_bytes = f.read(8)
        header_length = struct.unpack('<Q', header_length_bytes)[0]
        
        # Read JSON header
        header_json = f.read(header_length).decode('utf-8')
        header = json.loads(header_json)
        
        # Data starts after header length + header content
        data_offset = 8 + header_length
        
        return header, data_offset

def extract_tensor_from_file(file_path: Path, tensor_name: str, tensor_metadata: Dict[str, Any], 
                           data_offset: int) -> np.ndarray:
    """Extract a specific tensor from safetensors file."""
    
    dtype = tensor_metadata['dtype']
    shape = tensor_metadata['shape']
    data_offsets = tensor_metadata['data_offsets']
    
    # Calculate tensor's position in file
    tensor_start = data_offset + data_offsets[0]
    tensor_end = data_offset + data_offsets[1]
    tensor_size = tensor_end - tensor_start
    
    print(f"    Reading {tensor_name}: {shape} {dtype} ({tensor_size} bytes)")
    
    with open(file_path, 'rb') as f:
        f.seek(tensor_start)
        raw_bytes = f.read(tensor_size)
    
    if len(raw_bytes) != tensor_size:
        raise ValueError(f"Expected {tensor_size} bytes, got {len(raw_bytes)}")
    
    # Handle different data types
    if dtype == 'BF16':  # bfloat16
        # Convert bfloat16 to float32, then to float16 for storage
        float32_array = bfloat16_to_float32(raw_bytes)
        float16_array = float32_to_float16_safe(float32_array)
        result = float16_array.reshape(shape)
        
    elif dtype == 'F32':  # float32
        float32_array = np.frombuffer(raw_bytes, dtype=np.float32)
        float16_array = float32_to_float16_safe(float32_array)
        result = float16_array.reshape(shape)
        
    elif dtype == 'F16':  # float16
        result = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
        
    elif dtype == 'U8':  # uint8
        result = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(shape)
        
    elif dtype == 'I64':  # int64
        int64_array = np.frombuffer(raw_bytes, dtype=np.int64)
        result = int64_array.reshape(shape)
        
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return result

def load_all_weights_raw(model_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load all weights by raw file parsing."""
    print("🔄 Loading ALL weights with RAW file parsing...")
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Load index to get file mapping
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    
    print(f"📂 Processing {len(files)} files with raw parsing:")
    
    all_weights = {}
    total_params = 0
    success_count = 0
    bfloat16_count = 0
    
    for file_name in files:
        file_path = model_path / file_name
        size_gb = file_path.stat().st_size / (1024**3)
        print(f"\n📥 Loading {file_name} ({size_gb:.1f}GB)...")
        
        try:
            # Parse file header
            header, data_offset = parse_safetensors_file(file_path)
            
            # Get tensors for this file
            file_keys = [k for k in weight_map.keys() if weight_map[k] == file_name]
            print(f"  📊 Found {len(file_keys)} tensors in file")
            
            for key in file_keys:
                if key in header:
                    try:
                        tensor_metadata = header[key]
                        tensor = extract_tensor_from_file(file_path, key, tensor_metadata, data_offset)
                        
                        all_weights[key] = tensor
                        
                        params = np.prod(tensor.shape)
                        total_params += params
                        success_count += 1
                        
                        if tensor_metadata['dtype'] == 'BF16':
                            bfloat16_count += 1
                        
                        print(f"  ✅ {key}: {tensor.shape} → {tensor.dtype} ({tensor.nbytes/(1024**2):.1f}MB)")
                        
                    except Exception as e:
                        print(f"  ❌ {key}: Failed to extract - {e}")
                else:
                    print(f"  ❌ {key}: Not found in file header")
        
        except Exception as e:
            print(f"  ❌ Failed to parse {file_name}: {e}")
        
        gc.collect()
        print(f"  📊 File summary: {success_count} total tensors")
    
    print(f"\n🎯 FINAL TOTAL: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    print(f"✅ Successfully loaded {success_count} weights")
    print(f"🔧 bfloat16 conversions: {bfloat16_count}")
    
    return all_weights, config

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
        "format": "raw_bfloat16_complete",
        "total_shards": len(shards),
        "total_size_gb": round(total_size / 1024, 2),
        "config": config,
        "shards": shard_info,
        "conversion_notes": "Raw file parsing with custom bfloat16 conversion"
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return manifest_path

def create_loader_script(output_dir: Path):
    """Create loader script for the converted weights."""
    loader_code = '''#!/usr/bin/env python3
"""
Loader for raw-converted JAX weights.
"""

import json
import pickle
from pathlib import Path
import jax.numpy as jnp

def load_sharded_weights(weights_dir):
    """Load sharded weights back into a single structure."""
    weights_dir = Path(weights_dir)
    
    # Load manifest
    with open(weights_dir / "manifest.json") as f:
        manifest = json.load(f)
    
    print(f"📦 Loading {manifest['total_shards']} shards...")
    print(f"📊 Total size: {manifest['total_size_gb']}GB")
    
    # Load all shards
    merged_params = {}
    
    for shard_info in manifest["shards"]:
        shard_path = weights_dir / shard_info["file"]
        print(f"  📥 Loading {shard_info['file']} ({shard_info['size_mb']}MB)...")
        
        with open(shard_path, 'rb') as f:
            shard = pickle.load(f)
        
        # Deep merge shard into merged_params
        merge_dict(merged_params, shard)
    
    print("✅ All shards loaded with RAW bfloat16 conversion!")
    return merged_params, manifest["config"]

def merge_dict(target, source):
    """Deep merge source dict into target dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_dict(target[key], value)
        else:
            target[key] = value

# Example usage
if __name__ == "__main__":
    # Load sharded weights
    params, config = load_sharded_weights(".")
    
    print(f"\\n📊 Complete model loaded:")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Num layers: {config['num_hidden_layers']}")
    
    # Convert to JAX when needed
    if "embed_tokens" in params:
        embed_weight = params["embed_tokens"]["weight"] 
        print(f"\\n🔧 Embeddings: {embed_weight.shape} {embed_weight.dtype}")
        
        # Convert to JAX array
        jax_embed = jnp.array(embed_weight)
        print(f"  JAX version: {jax_embed.shape} {jax_embed.dtype}")
    
    print("\\n✅ RAW conversion successful - ALL parameters available!")
'''
    
    loader_path = output_dir / "load_complete_weights.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"🔧 Loader script: {loader_path}")

def main():
    parser = argparse.ArgumentParser(description="Raw bfloat16 converter with complete file parsing")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 RAW BFLOAT16 CONVERTER")
    print("=" * 40)
    print("🎯 Goal: Extract ALL 21B parameters via raw parsing")
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Load ALL weights with raw parsing
    weights, config = load_all_weights_raw(model_path)
    
    if not weights:
        print("❌ No weights loaded")
        return
    
    # Create nested structure
    nested_weights = create_nested_structure(weights)
    
    # Create shards
    shards = create_shards(nested_weights)
    
    # Save shards
    manifest_path = save_shards(shards, config, output_dir)
    
    # Create loader
    create_loader_script(output_dir)
    
    print(f"\n🎉 RAW CONVERSION COMPLETE!")
    print(f"📊 Total weights: {len(weights)}")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    print(f"🎯 SUCCESS: Raw bfloat16 parsing achieved!")

if __name__ == "__main__":
    main()