#!/usr/bin/env python3
"""Test loading GPT-OSS-20B model on CPU."""

import os
import sys
import time
import argparse
from pathlib import Path
import psutil
import numpy as np

# Force CPU mode
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "0"  # Use 32-bit for lower memory

import jax
import jax.numpy as jnp
from jax import random

sys.path.append(str(Path(__file__).parent.parent))
from jax_gpt_oss import load_model, GPTOSSConfig


def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def test_model_loading(model_path: str, max_seq_len: int = 32):
    """Test model loading and basic inference on CPU."""
    
    print("="*60)
    print("GPT-OSS-20B JAX Model Loading Test")
    print("="*60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")
    
    # Memory before loading
    mem_before = get_memory_usage()
    print(f"\nMemory before loading: {mem_before:.2f} GB")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    start_time = time.time()
    
    try:
        model, params = load_model(model_path, dtype=jnp.float32)
        load_time = time.time() - start_time
        
        mem_after_load = get_memory_usage()
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        print(f"Memory after loading: {mem_after_load:.2f} GB")
        print(f"Memory used: {mem_after_load - mem_before:.2f} GB")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Count parameters
    print("\n" + "-"*40)
    print("Model Statistics:")
    param_count = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {param_count:,}")
    print(f"Parameter size (float32): {param_count * 4 / 1024**3:.2f} GB")
    
    # Test forward pass
    print("\n" + "-"*40)
    print(f"Testing forward pass (sequence length: {max_seq_len})...")
    
    # Create dummy input
    batch_size = 1
    input_ids = jnp.ones((batch_size, max_seq_len), dtype=jnp.int32)
    
    # Compile forward pass
    @jax.jit
    def forward(params, input_ids):
        return model.apply(params, input_ids, deterministic=True)
    
    # Warmup
    print("Compiling...")
    compile_start = time.time()
    _ = forward(params, input_ids)
    compile_time = time.time() - compile_start
    print(f"✓ Compiled in {compile_time:.2f} seconds")
    
    # Timed forward pass
    print("\nRunning forward pass...")
    forward_start = time.time()
    output = forward(params, input_ids)
    output['logits'].block_until_ready()
    forward_time = time.time() - forward_start
    
    print(f"✓ Forward pass completed in {forward_time:.2f} seconds")
    print(f"Output shape: {output['logits'].shape}")
    print(f"Tokens/second: {max_seq_len / forward_time:.2f}")
    
    # Memory after forward pass
    mem_after_forward = get_memory_usage()
    print(f"\nMemory after forward pass: {mem_after_forward:.2f} GB")
    print(f"Peak memory usage: {mem_after_forward:.2f} GB")
    
    # Test generation (very short)
    print("\n" + "-"*40)
    print("Testing generation (5 tokens)...")
    
    from jax_gpt_oss import generate
    
    rng = random.PRNGKey(42)
    prompt_ids = jnp.array([[1, 2, 3, 4, 5]])  # Dummy prompt
    
    gen_start = time.time()
    generated = generate(
        model=model,
        params=params,
        prompt=prompt_ids,
        max_tokens=5,
        temperature=1.0,
        rng=rng
    )
    gen_time = time.time() - gen_start
    
    print(f"✓ Generated {generated.shape[1] - 5} tokens in {gen_time:.2f} seconds")
    print(f"Generated shape: {generated.shape}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✓ Model loading: PASSED ({load_time:.2f}s)")
    print(f"✓ Forward pass: PASSED ({forward_time:.2f}s)")
    print(f"✓ Generation: PASSED ({gen_time:.2f}s)")
    print(f"✓ Memory usage: {mem_after_forward:.2f} GB")
    print("\n✅ All tests passed!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test GPT-OSS-20B model loading")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/models/gpt-oss-20b-jax",
        help="Path to JAX model"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32,
        help="Maximum sequence length for testing"
    )
    
    args = parser.parse_args()
    
    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not installed, memory tracking disabled")
        print("Install with: pip install psutil")
    
    success = test_model_loading(args.model_path, args.max_seq_len)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()