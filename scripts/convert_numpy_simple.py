#!/usr/bin/env python3
"""
Simple NumPy-based converter that avoids JAX segfaults.
Converts safetensors to pickle format that JAX can load.
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Any
import argparse
import pickle

import numpy as np
from safetensors import safe_open

def load_all_weights(model_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load all weights from safetensors files using PyTorch backend."""
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
        
        # Import torch here to avoid loading if not needed
        try:
            import torch
        except ImportError:
            print("❌ PyTorch not found. Install with: pip install torch")
            return {}, {}
        
        with safe_open(file_path, framework="pt") as f:
            file_keys = [k for k in f.keys() if weight_map.get(k) == file_name]
            
            for key in file_keys:
                try:
                    # Load as PyTorch tensor
                    tensor = f.get_tensor(key)
                    
                    # Convert to NumPy float32
                    if hasattr(tensor, 'numpy'):
                        np_tensor = tensor.float().numpy()
                    else:
                        np_tensor = tensor.astype(np.float32)
                    
                    all_weights[key] = np_tensor
                    
                    params = np.prod(np_tensor.shape)
                    total_params += params
                    
                    print(f"  ✓ {key}: {np_tensor.shape} → {np_tensor.dtype}")
                    
                    # Clean up PyTorch tensor
                    del tensor
                    
                except Exception as e:
                    print(f"  ❌ {key}: {e}")
        
        # Force garbage collection
        gc.collect()
        print(f"  💾 Memory cleanup completed")
    
    print(f"\n🎯 Total: {total_params:,} parameters ({total_params/1e9:.2f}B)")
    return all_weights, config

def create_jax_structure(flat_weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Convert flat PyTorch keys to nested JAX structure."""
    print("\n🏗️  Creating nested JAX structure...")
    
    nested = {}
    
    for key, weight in flat_weights.items():
        # Remove "model." prefix
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        
        # Split into parts and create nested structure
        parts = clean_key.split(".")
        current = nested
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = weight
        print(f"  📝 {clean_key}")
    
    return nested

def save_for_jax(data: Dict[str, Any], output_path: Path):
    """Save data in format that JAX can easily load."""
    print(f"\n💾 Saving JAX-compatible pickle to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = output_path.stat().st_size / (1024**2)
    print(f"✅ Saved! Size: {size_mb:.1f}MB")

def create_loading_example(output_path: Path):
    """Create an example script showing how to load in JAX."""
    example_path = output_path.parent / "load_example.py"
    
    example_code = f'''#!/usr/bin/env python3
"""
Example: Loading converted weights in JAX
"""

import pickle
import jax.numpy as jnp

# Load the converted weights
with open("{output_path}", "rb") as f:
    data = pickle.load(f)

params = data["params"]
config = data["config"]

print(f"Model config: vocab_size={{config['vocab_size']}}")
print(f"Hidden size: {{config['hidden_size']}}")

# Convert any weight to JAX array when needed
# Example: embeddings
if "embed_tokens" in params and "weight" in params["embed_tokens"]:
    embed_weight = jnp.array(params["embed_tokens"]["weight"])
    print(f"Embeddings: {{embed_weight.shape}} {{embed_weight.dtype}}")

# Example: first layer attention
if "layers" in params and "0" in params["layers"]:
    layer_0 = params["layers"]["0"]
    if "self_attn" in layer_0 and "q_proj" in layer_0["self_attn"]:
        q_weight = jnp.array(layer_0["self_attn"]["q_proj"]["weight"])
        print(f"Layer 0 Q projection: {{q_weight.shape}} {{q_weight.dtype}}")

print("✅ JAX loading successful!")
'''
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"📝 Created loading example: {example_path}")

def main():
    parser = argparse.ArgumentParser(description="Simple NumPy-based safetensors to JAX converter")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output", required=True, help="Output pickle file")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 SIMPLE NUMPY → JAX CONVERTER")
    print("=" * 45)
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_path}")
    print()
    
    # Load all weights with NumPy
    weights, config = load_all_weights(model_path)
    
    if not weights:
        print("❌ Failed to load weights")
        return
    
    # Convert to JAX structure
    jax_params = create_jax_structure(weights)
    
    # Package data
    data = {
        "params": jax_params,
        "config": config,
        "format": "numpy_arrays_for_jax"
    }
    
    # Save
    save_for_jax(data, output_path)
    
    # Create loading example
    create_loading_example(output_path)
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📄 Weights: {output_path}")
    print(f"📖 Loading example: {output_path.parent}/load_example.py")
    print()
    print("🔧 JAX Usage:")
    print("   import pickle, jax.numpy as jnp")
    print(f"   with open('{output_path}', 'rb') as f:")
    print("       data = pickle.load(f)")
    print("   params = data['params']")
    print("   # Convert any weight: jnp.array(params['embed_tokens']['weight'])")

if __name__ == "__main__":
    main()