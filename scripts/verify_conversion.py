#!/usr/bin/env python3
"""
Quick verification script to confirm parameter count from converted shards.
"""

import json
import pickle
from pathlib import Path
import argparse
import numpy as np

def count_parameters_in_shard(shard_path: Path) -> int:
    """Count parameters in a single shard."""
    with open(shard_path, 'rb') as f:
        shard = pickle.load(f)
    
    total_params = 0
    
    def count_nested(obj, path=""):
        nonlocal total_params
        if isinstance(obj, np.ndarray):
            params = np.prod(obj.shape) 
            total_params += params
            print(f"  {path}: {obj.shape} → {params:,} params")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                count_nested(value, new_path)
    
    count_nested(shard)
    return total_params

def verify_conversion(model_dir: Path):
    """Verify the complete conversion."""
    print("🔍 CONVERSION VERIFICATION")
    print("=" * 40)
    
    # Load manifest
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"📄 Manifest: {manifest['format']}")
    print(f"📊 Claimed parameters: {manifest['total_parameters']:,}")
    print(f"🗂️  Total shards: {manifest['total_shards']}")
    print(f"💾 Total size: {manifest['total_size_gb']}GB")
    print()
    
    # Verify each shard
    total_verified_params = 0
    
    for shard_info in manifest["shards"]:
        shard_path = model_dir / shard_info["file"]
        
        if not shard_path.exists():
            print(f"❌ Shard missing: {shard_path}")
            continue
            
        print(f"🔍 Verifying {shard_info['file']}...")
        shard_params = count_parameters_in_shard(shard_path)
        total_verified_params += shard_params
        
        actual_size_mb = shard_path.stat().st_size / (1024**2)
        claimed_size_mb = shard_info["size_mb"]
        
        print(f"  ✅ Parameters: {shard_params:,}")
        print(f"  📦 Size: {actual_size_mb:.1f}MB (claimed: {claimed_size_mb}MB)")
        print()
    
    # Final verification
    claimed_params = manifest['total_parameters']
    print("🎯 FINAL VERIFICATION:")
    print(f"  Claimed: {claimed_params:,} parameters ({claimed_params/1e9:.2f}B)")
    print(f"  Verified: {total_verified_params:,} parameters ({total_verified_params/1e9:.2f}B)")
    
    if total_verified_params == claimed_params:
        print("  ✅ PERFECT MATCH!")
    elif abs(total_verified_params - claimed_params) < 1000:
        print("  ✅ VERY CLOSE MATCH (within 1K params)")
    else:
        diff = abs(total_verified_params - claimed_params)
        print(f"  ⚠️  MISMATCH: {diff:,} parameter difference")
    
    # Check if we hit the 21B target
    target_21b = 21e9
    if total_verified_params >= target_21b:
        print(f"  🎉 SUCCESS: Extracted {total_verified_params/1e9:.2f}B ≥ 21B target!")
    else:
        missing = target_21b - total_verified_params  
        print(f"  ❌ MISSING: {missing/1e9:.2f}B parameters short of 21B target")

def main():
    parser = argparse.ArgumentParser(description="Verify converted model parameters")
    parser.add_argument("--model-dir", required=True, help="Path to converted model directory")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    verify_conversion(model_dir)

if __name__ == "__main__":
    main()