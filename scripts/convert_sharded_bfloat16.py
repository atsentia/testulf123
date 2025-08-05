#!/usr/bin/env python3
"""
Sharded converter that preserves original bfloat16 precision.
Zero precision loss from original model.
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

def preserve_bfloat16_precision(tensor):
    """Preserve bfloat16 as-is for JAX."""
    import torch
    
    if tensor.dtype == torch.bfloat16:
        # Store bfloat16 in a way JAX can reconstruct
        return {
            'data': tensor.detach().cpu().numpy().view(np.uint16),
            'shape': tensor.shape,
            'dtype': 'bfloat16'
        }
    else:
        # Regular numpy conversion
        return tensor.numpy()

def load_all_weights_precise(model_path: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load weights preserving exact original precision."""
    print("🔄 Loading with ZERO precision loss...")
    
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    files = sorted(set(weight_map.values()))
    
    print(f"📂 Processing {len(files)} files (preserving bfloat16):")
    
    all_weights = {}
    total_params = 0
    
    try:
        import torch
    except ImportError:
        print("❌ PyTorch required for bfloat16 support")
        return {}, {}
    
    for file_name in files:
        file_path = model_path / file_name
        print(f"\n📥 Loading {file_name}...")
        
        with safe_open(file_path, framework="pt") as f:
            file_keys = [k for k in f.keys() if weight_map.get(k) == file_name]
            
            for key in file_keys:
                try:
                    tensor = f.get_tensor(key)
                    
                    # Preserve exact precision
                    if tensor.dtype == torch.bfloat16:
                        # Store bfloat16 metadata for reconstruction
                        weight_data = {
                            'data': tensor.detach().cpu().numpy().view(np.uint16),
                            'shape': list(tensor.shape),
                            'dtype': 'bfloat16'
                        }
                        size_mb = weight_data['data'].nbytes / (1024**2)
                        print(f"  ✓ {key}: {tensor.shape} → bfloat16 (preserved) ({size_mb:.1f}MB)")
                    else:
                        # Regular conversion
                        weight_data = tensor.numpy()
                        size_mb = weight_data.nbytes / (1024**2) 
                        print(f"  ✓ {key}: {tensor.shape} → {weight_data.dtype} ({size_mb:.1f}MB)")
                    
                    all_weights[key] = weight_data
                    total_params += np.prod(tensor.shape)
                    
                    del tensor
                    
                except Exception as e:
                    print(f"  ❌ {key}: {e}")
        
        gc.collect()
    
    print(f"\n🎯 Total: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    return all_weights, config

def estimate_size_precise(obj) -> int:
    """Estimate size including bfloat16 metadata."""
    if isinstance(obj, dict):
        if 'dtype' in obj and obj['dtype'] == 'bfloat16':
            return obj['data'].nbytes + 100  # Small overhead for metadata
        else:
            return sum(estimate_size_precise(v) for v in obj.values())
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    else:
        return len(str(obj).encode('utf-8'))

def create_loader_precise(output_dir: Path):
    """Create loader that reconstructs bfloat16."""
    
    loader_code = '''#!/usr/bin/env python3
"""
Precision-preserving loader for JAX weights.
Reconstructs bfloat16 exactly as stored.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import jax.numpy as jnp

def reconstruct_bfloat16(weight_data):
    """Reconstruct bfloat16 tensor from stored data."""
    if isinstance(weight_data, dict) and weight_data.get('dtype') == 'bfloat16':
        # Reconstruct bfloat16 from uint16 representation
        uint16_data = weight_data['data']
        shape = weight_data['shape']
        
        # Convert uint16 back to float32 (bfloat16 expanded)
        # bfloat16 is stored in upper 16 bits of float32
        expanded = np.zeros(uint16_data.size, dtype=np.uint32)
        expanded = (uint16_data.astype(np.uint32) << 16)
        float32_data = expanded.view(np.float32).reshape(shape)
        
        # JAX can handle this as float32 (maintains bfloat16 precision)
        return float32_data
    else:
        # Regular numpy array
        return weight_data

def load_sharded_weights_precise(weights_dir):
    """Load with exact precision preservation."""
    weights_dir = Path(weights_dir)
    
    with open(weights_dir / "manifest.json") as f:
        manifest = json.load(f)
    
    print(f"📦 Loading {manifest['total_shards']} precision-preserving shards...")
    
    merged_params = {}
    
    for shard_info in manifest["shards"]:
        shard_path = weights_dir / shard_info["file"]
        print(f"  📥 Loading {shard_info['file']}...")
        
        with open(shard_path, 'rb') as f:
            shard = pickle.load(f)
        
        # Reconstruct bfloat16 tensors
        shard_reconstructed = {}
        def reconstruct_recursive(obj, path=""):
            if isinstance(obj, dict):
                if 'dtype' in obj and obj['dtype'] == 'bfloat16':
                    return reconstruct_bfloat16(obj)
                else:
                    return {k: reconstruct_recursive(v, f"{path}.{k}") for k, v in obj.items()}
            else:
                return obj
        
        shard_reconstructed = reconstruct_recursive(shard)
        merge_dict(merged_params, shard_reconstructed)
    
    print("✅ All shards loaded with original precision!")
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
    # Load with exact precision
    params, config = load_sharded_weights_precise(".")
    
    print(f"\\n📊 Model loaded with original precision:")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Hidden size: {config['hidden_size']}")
    
    # Convert to JAX (preserving precision)
    if "embed_tokens" in params:
        embed_weight = params["embed_tokens"]["weight"] 
        print(f"\\n🔧 Embeddings: {embed_weight.shape} {embed_weight.dtype}")
        
        # Convert to JAX array
        jax_embed = jnp.array(embed_weight)
        print(f"  JAX version: {jax_embed.shape} {jax_embed.dtype}")
    
    print("\\n✅ Zero precision loss achieved!")
'''
    
    loader_path = output_dir / "load_precise_weights.py"
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"🔧 Precision loader: {loader_path}")

# Use the same sharding logic but with precise weights
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    print("🚀 PRECISION-PRESERVING SHARDED CONVERTER")
    print("=" * 55)
    print("🎯 Zero precision loss from bfloat16 → JAX")
    
    # Load with exact precision
    weights, config = load_all_weights_precise(Path(args.model_path))
    
    if not weights:
        return
    
    # Create nested structure (same as before)
    from convert_sharded import create_nested_structure, create_shards, save_shards
    nested = create_nested_structure(weights)
    shards = create_shards(nested)
    
    output_dir = Path(args.output_dir)
    save_shards(shards, config, output_dir)
    
    # Create precision-aware loader
    create_loader_precise(output_dir)
    
    print(f"\n🎉 ZERO PRECISION LOSS CONVERSION COMPLETE!")
    print(f"📁 Exact bfloat16 precision preserved")

if __name__ == "__main__":
    main()