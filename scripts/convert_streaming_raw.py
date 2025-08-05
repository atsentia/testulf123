#!/usr/bin/env python3
"""
Streaming raw bfloat16 converter that saves immediately to avoid memory buildup.
"""

import os
import json
import gc
import struct
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import tempfile

import numpy as np

MAX_SHARD_SIZE = 1.8 * 1024 * 1024 * 1024  # 1.8GB

def bfloat16_to_float32(bfloat16_bytes: bytes) -> np.ndarray:
    """Convert bfloat16 bytes to float32 numpy array."""
    uint16_data = np.frombuffer(bfloat16_bytes, dtype=np.uint16)
    uint32_data = uint16_data.astype(np.uint32) << 16
    float32_data = uint32_data.view(np.float32)
    return float32_data

def float32_to_float16_safe(float32_array: np.ndarray) -> np.ndarray:
    """Convert float32 to float16 with overflow handling."""
    float32_clipped = np.clip(float32_array, -65504.0, 65504.0)
    return float32_clipped.astype(np.float16)

def parse_safetensors_file(file_path: Path) -> Tuple[Dict[str, Any], int]:
    """Parse safetensors file header and return metadata + data offset."""
    with open(file_path, 'rb') as f:
        header_length_bytes = f.read(8)
        header_length = struct.unpack('<Q', header_length_bytes)[0]
        header_json = f.read(header_length).decode('utf-8')
        header = json.loads(header_json)
        data_offset = 8 + header_length
        return header, data_offset

def extract_tensor_from_file(file_path: Path, tensor_name: str, tensor_metadata: Dict[str, Any], 
                           data_offset: int) -> np.ndarray:
    """Extract a specific tensor from safetensors file."""
    
    dtype = tensor_metadata['dtype']
    shape = tensor_metadata['shape']
    data_offsets = tensor_metadata['data_offsets']
    
    tensor_start = data_offset + data_offsets[0]
    tensor_end = data_offset + data_offsets[1]
    tensor_size = tensor_end - tensor_start
    
    with open(file_path, 'rb') as f:
        f.seek(tensor_start)
        raw_bytes = f.read(tensor_size)
    
    if len(raw_bytes) != tensor_size:
        raise ValueError(f"Expected {tensor_size} bytes, got {len(raw_bytes)}")
    
    # Handle different data types
    if dtype == 'BF16':  # bfloat16
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

class StreamingShardWriter:
    """Writes tensors to shards as we go, managing memory automatically."""
    
    def __init__(self, output_dir: Path, max_shard_size: int = MAX_SHARD_SIZE):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_shard = {}
        self.current_size = 0
        self.shard_count = 0
        self.shard_info = []
        self.total_params = 0
        
    def add_tensor(self, key: str, tensor: np.ndarray):
        """Add tensor to current shard, finalizing if needed."""
        tensor_size = tensor.nbytes
        self.total_params += np.prod(tensor.shape)
        
        # Check if we need to finalize current shard
        if self.current_size + tensor_size > self.max_shard_size and self.current_shard:
            self._finalize_current_shard()
        
        # Add tensor to nested structure
        parts = key.split(".")
        if parts[0] == "model":
            parts = parts[1:]  # Remove "model." prefix
            
        current = self.current_shard
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = tensor
        self.current_size += tensor_size
        
        print(f"  📦 Added {key}: {tensor.shape} → {tensor.dtype} ({tensor_size/(1024**2):.1f}MB)")
        
        # Force garbage collection after each tensor
        gc.collect()
    
    def _finalize_current_shard(self):
        """Save current shard to disk and reset."""
        if not self.current_shard:
            return
            
        shard_path = self.output_dir / f"shard_{self.shard_count:03d}.pkl"
        
        # Save using a temporary file for safety
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=self.output_dir) as tmp_f:
            import pickle
            pickle.dump(self.current_shard, tmp_f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path = tmp_f.name
        
        # Atomic move
        os.rename(tmp_path, shard_path)
        
        size_mb = shard_path.stat().st_size / (1024**2)
        self.shard_info.append({
            "file": shard_path.name,
            "size_mb": round(size_mb, 1)
        })
        
        print(f"🗂️  Finalized {shard_path.name}: {size_mb:.1f}MB (estimated {self.current_size/(1024**2):.1f}MB)")
        
        # Reset for next shard
        self.current_shard = {}
        self.current_size = 0
        self.shard_count += 1
        
        # Force garbage collection
        gc.collect()
    
    def finalize_all(self, config: Dict[str, Any]) -> Tuple[Path, int]:
        """Finalize all shards and create manifest."""
        # Finalize any remaining shard
        if self.current_shard:
            self._finalize_current_shard()
        
        # Calculate total size
        total_size_mb = sum(info["size_mb"] for info in self.shard_info)
        
        # Create manifest
        manifest = {
            "format": "streaming_raw_bfloat16_complete",
            "total_shards": len(self.shard_info),
            "total_size_gb": round(total_size_mb / 1024, 2),
            "total_parameters": int(self.total_params),  # Convert to standard int
            "config": config,
            "shards": self.shard_info,
            "conversion_notes": "Streaming raw file parsing with custom bfloat16 conversion"
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path, self.total_params

def load_all_weights_streaming(model_path: Path, output_dir: Path) -> Tuple[Path, int]:
    """Load all weights using streaming approach."""
    print("🔄 Loading ALL weights with STREAMING raw parsing...")
    
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
    
    print(f"📂 Processing {len(files)} files with streaming approach:")
    
    # Initialize streaming writer
    writer = StreamingShardWriter(output_dir)
    
    success_count = 0
    bfloat16_count = 0
    
    for file_name in files:
        file_path = model_path / file_name
        size_gb = file_path.stat().st_size / (1024**3)
        print(f"\n📥 Processing {file_name} ({size_gb:.1f}GB)...")
        
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
                        print(f"    Reading {key}: {tensor_metadata['shape']} {tensor_metadata['dtype']}")
                        
                        # Extract tensor 
                        tensor = extract_tensor_from_file(file_path, key, tensor_metadata, data_offset)
                        
                        # Immediately add to streaming writer
                        writer.add_tensor(key, tensor)
                        
                        success_count += 1
                        if tensor_metadata['dtype'] == 'BF16':
                            bfloat16_count += 1
                        
                        # Delete tensor reference to free memory
                        del tensor
                        
                    except Exception as e:
                        print(f"  ❌ {key}: Failed to extract - {e}")
                else:
                    print(f"  ❌ {key}: Not found in file header")
        
        except Exception as e:
            print(f"  ❌ Failed to parse {file_name}: {e}")
        
        # Force cleanup after each file
        gc.collect()
        print(f"  📊 File completed. Total tensors so far: {success_count}")
    
    # Finalize all shards
    manifest_path, total_params = writer.finalize_all(config)
    
    print(f"\n🎯 STREAMING CONVERSION COMPLETE!")
    print(f"📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"✅ Successfully loaded {success_count} weights")
    print(f"🔧 bfloat16 conversions: {bfloat16_count}")
    print(f"📁 Output: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    
    return manifest_path, total_params

def main():
    parser = argparse.ArgumentParser(description="Streaming raw bfloat16 converter")
    parser.add_argument("--model-path", required=True, help="Path to safetensors model")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("🚀 STREAMING RAW BFLOAT16 CONVERTER")
    print("=" * 45)
    print("🎯 Goal: Extract ALL 21B parameters via streaming")
    print(f"📂 Model: {model_path}")
    print(f"💾 Output: {output_dir}")
    print()
    
    # Load weights with streaming approach
    manifest_path, total_params = load_all_weights_streaming(model_path, output_dir)
    
    if total_params == 0:
        print("❌ No parameters extracted")
        return
    
    print(f"\n🎉 SUCCESS: {total_params:,} parameters extracted!")
    print(f"🎯 Streaming raw bfloat16 conversion completed!")

if __name__ == "__main__":
    main()