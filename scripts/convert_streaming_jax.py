#!/usr/bin/env python3
"""
Memory-efficient streaming JAX converter.
Processes weights in chunks and saves incrementally.
"""

import os
import json
import gc
from pathlib import Path
from typing import Dict, Any
import argparse
import pickle

# JAX setup
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

import jax
import jax.numpy as jnp
from safetensors import safe_open

def load_config(model_path: Path) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Load model config and weight mapping."""
    
    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Load index
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    return config, index["weight_map"]

def convert_file_streaming(model_path: Path, file_name: str, weight_map: Dict[str, str], 
                          output_dir: Path, batch_size: int = 10):
    """Convert one safetensors file in streaming fashion."""
    
    file_path = model_path / file_name
    size_gb = file_path.stat().st_size / (1024**3)
    
    print(f"\n📥 Processing {file_name} ({size_gb:.1f}GB)...")
    
    # Get keys for this file
    file_keys = [k for k in weight_map.keys() if weight_map[k] == file_name]
    print(f"  📊 {len(file_keys)} tensors to convert")
    
    # Process in batches to control memory
    jax_weights = {}
    processed = 0
    
    for i in range(0, len(file_keys), batch_size):
        batch_keys = file_keys[i:i+batch_size]
        print(f"  🔄 Batch {i//batch_size + 1}: processing {len(batch_keys)} tensors...")
        
        with safe_open(file_path, framework="pt") as f:
            for key in batch_keys:
                try:
                    # Load PyTorch tensor
                    tensor = f.get_tensor(key)
                    
                    # Convert to float32 numpy
                    if hasattr(tensor, 'numpy'):
                        np_tensor = tensor.float().numpy()
                    else:
                        np_tensor = tensor.astype('float32')
                    
                    # Convert to JAX immediately
                    jax_tensor = jnp.array(np_tensor)
                    jax_weights[key] = jax_tensor
                    
                    processed += 1
                    print(f"    ✓ {key}: {jax_tensor.shape}")
                    
                    # Clear PyTorch tensor
                    del tensor, np_tensor
                    
                except Exception as e:
                    print(f"    ❌ {key}: {e}")
        
        # Force garbage collection after each batch
        gc.collect()
        
        # Save intermediate results if batch is large enough
        if len(jax_weights) > 50:
            chunk_path = output_dir / f"chunk_{file_name}_{i//batch_size}.pkl"
            save_chunk(jax_weights, chunk_path)
            jax_weights.clear()
            gc.collect()
    
    # Save remaining weights
    if jax_weights:
        chunk_path = output_dir / f"chunk_{file_name}_final.pkl"
        save_chunk(jax_weights, chunk_path)
    
    print(f"  ✅ Processed {processed} tensors from {file_name}")
    return processed

def save_chunk(weights: Dict[str, Any], chunk_path: Path):
    """Save a chunk of weights."""
    print(f"    💾 Saving chunk: {chunk_path.name}")
    
    with open(chunk_path, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = chunk_path.stat().st_size / (1024**2)
    print(f"    📦 Chunk saved: {size_mb:.1f}MB")

def merge_chunks(output_dir: Path, output_file: Path):
    """Merge all chunks into final weights file."""
    print(f"\n🔗 Merging chunks...")
    
    # Find all chunk files
    chunk_files = list(output_dir.glob("chunk_*.pkl"))
    chunk_files.sort()
    
    print(f"  📂 Found {len(chunk_files)} chunk files")
    
    all_weights = {}
    
    for chunk_file in chunk_files:
        print(f"  📥 Loading {chunk_file.name}...")
        
        with open(chunk_file, 'rb') as f:
            chunk_weights = pickle.load(f)
        
        all_weights.update(chunk_weights)
        
        # Clean up chunk file
        chunk_file.unlink()
        
        gc.collect()
    
    # Create nested structure
    print(f"  🏗️  Creating nested structure...")
    nested_weights = create_nested_structure(all_weights)
    
    # Save final file
    print(f"  💾 Saving final weights...")
    with open(output_file, 'wb') as f:
        pickle.dump(nested_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    final_size_mb = output_file.stat().st_size / (1024**2)
    print(f"  ✅ Final file: {final_size_mb:.1f}MB")

def create_nested_structure(flat_weights: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat keys to nested structure."""
    nested = {}
    
    for key, value in flat_weights.items():
        # Remove "model." prefix
        clean_key = key
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        
        # Split and nest
        parts = clean_key.split(".")
        current = nested
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return nested

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--batch-size", type=int, default=5, help="Tensors per batch")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output)
    output_dir = output_path.parent
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 STREAMING JAX CONVERTER")
    print("=" * 40)
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_path}")
    print(f"📦 Batch size: {args.batch_size}")
    print()
    
    # Load config and mapping
    config, weight_map = load_config(model_path)
    
    # Get unique files
    files = sorted(set(weight_map.values()))
    print(f"📂 Files to process: {len(files)}")
    
    total_processed = 0
    
    # Process each file
    for file_name in files:
        processed = convert_file_streaming(
            model_path, file_name, weight_map, output_dir, args.batch_size
        )
        total_processed += processed
    
    # Merge all chunks
    merge_chunks(output_dir, output_path)
    
    # Save config
    config_path = output_path.parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📊 Total tensors: {total_processed}")
    print(f"📄 Weights: {output_path}")
    print(f"⚙️  Config: {config_path}")

if __name__ == "__main__":
    main()