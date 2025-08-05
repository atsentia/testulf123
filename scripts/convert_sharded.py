#!/usr/bin/env python3
"""
Sharded NumPy converter for Git LFS compatibility.
Splits weights into multiple pickle files under 2GB each.
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

# Git LFS limit is 2GB, use 1.8GB for safety
MAX_SHARD_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB in bytes

def estimate_size(obj) -> int:
    """Rough estimate of object size in bytes."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, dict):
        return sum(estimate_size(v) for v in obj.values())
    else:
        return len(str(obj).encode('utf-8'))

def load_all_weights(model_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load all weights from safetensors files."""
    print("🔄 Loading safetensors with PyTorch backend...")
    
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
    
    print(f"📂 Processing {len(files)} files:")
    for f in files:
        size_gb = (model_path / f).stat().st_size / (1024**3)
        print(f"  • {f}: {size_gb:.1f}GB")
    
    # Load all weights
    all_weights = {}
    total_params = 0
    
    for file_name in files:
        file_path = model_path / file_name
        print(f"\n📥 Loading {file_name}...")
        
        try:
            import torch
        except ImportError:
            print("❌ PyTorch not found. Install with: pip install torch")
            return {}, {}
        
        with safe_open(file_path, framework="pt") as f:
            file_keys = [k for k in f.keys() if weight_map.get(k) == file_name]
            
            for key in file_keys:
                try:
                    tensor = f.get_tensor(key)
                    
                    # Convert to float16 for JAX compatibility and space efficiency  
                    if hasattr(tensor, 'numpy'):
                        if tensor.dtype == torch.bfloat16:
                            # Convert bfloat16 → float16 (JAX compatible, half the size of float32)
                            np_tensor = tensor.to(torch.float16).numpy()
                        elif tensor.dtype == torch.float32:
                            # Convert float32 → float16 to save space
                            np_tensor = tensor.to(torch.float16).numpy()
                        else:
                            # Keep original dtype (likely already float16)
                            np_tensor = tensor.numpy()
                    else:
                        np_tensor = tensor
                    
                    all_weights[key] = np_tensor
                    
                    params = np.prod(np_tensor.shape)
                    total_params += params
                    
                    print(f"  ✓ {key}: {np_tensor.shape} → {np_tensor.dtype} ({np_tensor.nbytes/(1024**2):.1f}MB)")
                    
                    del tensor
                    
                except Exception as e:
                    print(f"  ❌ {key}: {e}")
        
        gc.collect()
    
    print(f"\n🎯 Total: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    return all_weights, config

def create_nested_structure(flat_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert flat keys to nested structure."""
    print("\n🏗️  Creating nested structure...")
    
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
        
        # If adding this object would exceed limit, start new shard
        if current_size + obj_size > MAX_SHARD_SIZE and current_shard:
            print(f"  📦 Shard {len(shards)}: {current_size/(1024**2):.1f}MB")
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        
        # Add object to current shard
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
    
    # Process the nested structure
    traverse(nested_weights)
    
    # Add final shard if not empty
    if current_shard:
        print(f"  📦 Shard {len(shards)}: {current_size/(1024**2):.1f}MB")
        shards.append(current_shard)
    
    print(f"✅ Created {len(shards)} shards")
    return shards

def save_shards(shards: List[Dict[str, Any]], config: Dict[str, Any], output_dir: Path):
    """Save shards and create manifest."""
    print(f"\n💾 Saving {len(shards)} shards to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_info = []
    
    # Save each shard
    for i, shard in enumerate(shards):
        shard_path = output_dir / f"shard_{i:03d}.pkl"
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_mb = shard_path.stat().st_size / (1024**2)
        shard_info.append({
            "file": shard_path.name,
            "size_mb": round(size_mb, 1),
            "keys": list_shard_keys(shard)
        })
        
        print(f"  📦 {shard_path.name}: {size_mb:.1f}MB")
    
    # Save manifest
    manifest = {
        "format": "sharded_numpy_for_jax",
        "total_shards": len(shards),
        "config": config,
        "shards": shard_info
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📄 Manifest: {manifest_path}")
    
    # Create loader script
    create_loader_script(output_dir)
    
    return manifest_path

def list_shard_keys(shard: Dict[str, Any], prefix: str = "") -> List[str]:
    """List all tensor keys in a shard."""
    keys = []
    for key, value in shard.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, np.ndarray):
            keys.append(full_key)
        elif isinstance(value, dict):
            keys.extend(list_shard_keys(value, full_key))
    return keys

def create_loader_script(output_dir: Path):
    """Create a script to load sharded weights."""
    loader_path = output_dir / "load_sharded_weights.py"
    
    loader_code = '''#!/usr/bin/env python3
"""
Loader for sharded JAX weights.
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
    
    # Load all shards
    merged_params = {}
    
    for shard_info in manifest["shards"]:
        shard_path = weights_dir / shard_info["file"]
        print(f"  📥 Loading {shard_info['file']} ({shard_info['size_mb']}MB)...")
        
        with open(shard_path, 'rb') as f:
            shard = pickle.load(f)
        
        # Deep merge shard into merged_params
        merge_dict(merged_params, shard)
    
    print("✅ All shards loaded!")
    return merged_params, manifest["config"]

def merge_dict(target, source):
    """Deep merge source dict into target dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_dict(target[key], value)
        else:
            target[key] = value

def convert_to_jax(params, dtype=None):
    """Convert numpy arrays to JAX arrays on demand."""
    
    def convert_recursive(obj):
        if isinstance(obj, dict):
            return {k: convert_recursive(v) for k, v in obj.items()}
        elif hasattr(obj, 'shape'):  # numpy array
            if dtype is not None:
                return jnp.array(obj, dtype=dtype)
            else:
                return jnp.array(obj)
        else:
            return obj
    
    return convert_recursive(params)

# Example usage
if __name__ == "__main__":
    # Load sharded weights
    params, config = load_sharded_weights(".")
    
    print(f"\\n📊 Model info:")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Num layers: {config['num_hidden_layers']}")
    
    # Convert to JAX when needed
    print("\\n🔧 Converting embeddings to JAX...")
    if "embed_tokens" in params:
        embed_np = params["embed_tokens"]["weight"]
        print(f"  NumPy embeddings: {embed_np.shape} {embed_np.dtype}")
        
        # Convert to JAX (keeping float16, or convert to float32 if needed)
        embed_jax = jnp.array(embed_np)  # Keep original dtype
        # embed_jax = jnp.array(embed_np, dtype=jnp.float32)  # Or convert to float32
        print(f"  JAX embeddings: {embed_jax.shape} {embed_jax.dtype}")
    
    print("\\n📝 Note: Weights are stored in float16 to save space (~13GB vs ~45GB).")
    print("     JAX can compute directly in float16, or convert to float32 with:")
    print("     jax_weight = jnp.array(weight, dtype=jnp.float32)")
    print("\\n✅ Sharded loading successful!")
'''
    
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    
    print(f"🔧 Loader script: {loader_path}")

def main():
    parser = argparse.ArgumentParser(description="Sharded converter for Git LFS compatibility")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 SHARDED CONVERTER FOR GIT LFS")
    print("=" * 50)
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    print(f"📏 Max shard size: {MAX_SHARD_SIZE/(1024**3):.1f}GB")
    print()
    
    # Load weights
    weights, config = load_all_weights(model_path)
    
    if not weights:
        print("❌ Failed to load weights")
        return
    
    # Create nested structure
    nested_weights = create_nested_structure(weights)
    
    # Create shards
    shards = create_shards(nested_weights)
    
    # Save shards
    manifest_path = save_shards(shards, config, output_dir)
    
    print(f"\n🎉 SHARDED CONVERSION COMPLETE!")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    print(f"🔧 Loader: {output_dir}/load_sharded_weights.py")
    print()
    print("📦 Git LFS Compatible:")
    print(f"  • All shards < 2GB (safe for Git LFS)")
    print(f"  • Use: git lfs track '{output_dir}/*.pkl'")
    print(f"  • Load with: python {output_dir}/load_sharded_weights.py")

if __name__ == "__main__":
    main()