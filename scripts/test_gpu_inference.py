#!/usr/bin/env python3
"""
Test GPT-OSS-20B inference on GPU after weight conversion.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import logging

# Configure JAX for GPU
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # Force GPU usage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")
    
    # Check if we have GPU
    if jax.default_backend() != 'gpu':
        logger.warning("JAX is not using GPU! Check CUDA installation.")
    else:
        logger.info("✓ JAX is using GPU")
        
except ImportError as e:
    logger.error(f"JAX not available: {e}")
    sys.exit(1)

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from jax_gpt_oss.utils.model_utils import load_model
    from jax_gpt_oss.inference import generate_text
except ImportError as e:
    logger.error(f"Could not import model utilities: {e}")
    logger.info("Make sure you're in the correct directory and have installed the package")
    sys.exit(1)


def benchmark_inference(model, params, seq_lengths=[32, 64, 128], num_runs=3):
    """Benchmark inference speed on different sequence lengths."""
    logger.info("Running inference benchmarks...")
    
    results = {}
    
    for seq_len in seq_lengths:
        logger.info(f"\nBenchmarking sequence length: {seq_len}")
        
        # Create dummy input
        dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(2):
            _ = model.apply(params, dummy_input, deterministic=True)
        
        # Benchmark
        times = []
        for run in range(num_runs):
            start_time = time.time()
            output = model.apply(params, dummy_input, deterministic=True)
            jax.block_until_ready(output['logits'])  # Ensure computation is complete
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            logger.info(f"  Run {run+1}: {elapsed:.4f}s")
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = seq_len / avg_time
        
        results[seq_len] = {
            'avg_time': avg_time,
            'tokens_per_sec': tokens_per_sec,
            'times': times
        }
        
        logger.info(f"  Average: {avg_time:.4f}s ({tokens_per_sec:.2f} tokens/sec)")
    
    return results


def test_generation(model, params, tokenizer=None, prompt="Hello, I am"):
    """Test text generation if tokenizer is available."""
    logger.info(f"\nTesting generation with prompt: '{prompt}'")
    
    try:
        # Try to use the generation utility
        if tokenizer is None:
            logger.info("No tokenizer provided, using dummy tokens")
            # Create dummy input (assuming common tokens)
            input_ids = jnp.array([[15496, 11, 314, 716]], dtype=jnp.int32)  # "Hello, I am"
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="jax")
        
        # Generate
        start_time = time.time()
        output = generate_text(
            model=model,
            params=params,
            input_ids=input_ids,
            max_length=input_ids.shape[1] + 20,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        end_time = time.time()
        
        generated_tokens = output.shape[1] - input_ids.shape[1]
        generation_time = end_time - start_time
        tokens_per_sec = generated_tokens / generation_time
        
        logger.info(f"Generated {generated_tokens} tokens in {generation_time:.2f}s")
        logger.info(f"Generation speed: {tokens_per_sec:.2f} tokens/sec")
        
        if tokenizer:
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated text: '{generated_text}'")
        else:
            logger.info(f"Generated token IDs: {output[0].tolist()}")
            
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        logger.info("This is normal if generation utilities aren't implemented yet")


def check_memory_usage():
    """Check GPU memory usage."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = map(int, line.split(', '))
                logger.info(f"GPU {i}: {used}MB / {total}MB ({used/total*100:.1f}% used)")
        else:
            logger.info("Could not get GPU memory info")
    except Exception as e:
        logger.info(f"Could not check GPU memory: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test GPT-OSS-20B inference on GPU")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gpt-oss-20b-jax",
        help="Path to converted JAX model"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmarks"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true", 
        help="Skip text generation test"
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[32, 64, 128],
        help="Sequence lengths for benchmarking"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist")
        logger.info("Make sure you've downloaded the converted weights with Git LFS")
        sys.exit(1)
    
    # Check GPU memory before loading
    logger.info("GPU memory before loading:")
    check_memory_usage()
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    try:
        start_time = time.time()
        model, params = load_model(str(model_path))
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check memory after loading
    logger.info("\nGPU memory after loading:")
    check_memory_usage()
    
    # Test basic forward pass
    logger.info("\nTesting basic forward pass...")
    try:
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        start_time = time.time()
        output = model.apply(params, dummy_input, deterministic=True)
        jax.block_until_ready(output['logits'])
        forward_time = time.time() - start_time
        
        logger.info(f"✓ Forward pass successful in {forward_time:.4f}s")
        logger.info(f"  Output shape: {output['logits'].shape}")
        logger.info(f"  Output dtype: {output['logits'].dtype}")
        
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run benchmarks if requested
    if args.benchmark:
        benchmark_results = benchmark_inference(model, params, args.seq_lengths)
        
        logger.info("\n=== Benchmark Summary ===")
        for seq_len, results in benchmark_results.items():
            logger.info(f"Seq {seq_len:3d}: {results['avg_time']:.4f}s avg, {results['tokens_per_sec']:6.2f} tok/s")
    
    # Test generation if requested
    if not args.no_generation:
        test_generation(model, params)
    
    # Final memory check
    logger.info("\nFinal GPU memory usage:")
    check_memory_usage()
    
    logger.info("\n✅ GPU inference test complete!")


if __name__ == "__main__":
    main()