#!/usr/bin/env python3
"""
Modal.com GPU benchmark for GPT-OSS-20B JAX implementation.

Tests performance on NVIDIA H100, A100, and B200 GPUs.

Usage:
    modal run scripts/modal_gpu_benchmark.py
"""

import modal
import time
from typing import Dict, List, Any
import json
from datetime import datetime

# Create Modal stub
stub = modal.Stub("gpt-oss-20b-jax-benchmark")

# Define image with JAX and dependencies
gpu_image = (
    modal.Image.debian_slim()
    .pip_install(
        "jax[cuda12_pip]",  # JAX with CUDA 12 support
        "flax",
        "optax",
        "numpy",
        "safetensors",
        "transformers",  # For tokenizer
        "torch",  # For weight conversion
        "einops",
        "tqdm",
        "pandas",
    )
    .run_commands(
        "apt-get update && apt-get install -y git wget",
    )
)

# Volume for model storage (shared across runs)
model_volume = modal.NetworkFileSystem.new().persisted("gpt-oss-20b-model")


@stub.function(
    image=gpu_image,
    gpu="H100",  # Default to H100
    timeout=1800,  # 30 minutes timeout
    network_file_systems={"/models": model_volume},
)
def benchmark_h100(
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 10,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Benchmark on NVIDIA H100 (80GB)."""
    return run_benchmark("H100", batch_size, seq_length, num_iterations, use_bf16)


@stub.function(
    image=gpu_image,
    gpu="A100",  # A100 GPU
    timeout=1800,
    network_file_systems={"/models": model_volume},
)
def benchmark_a100(
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 10,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Benchmark on NVIDIA A100 (40GB/80GB)."""
    return run_benchmark("A100", batch_size, seq_length, num_iterations, use_bf16)


@stub.function(
    image=gpu_image,
    gpu="A100:8",  # 8x A100 for B200 simulation (B200 not yet available)
    timeout=1800,
    network_file_systems={"/models": model_volume},
    memory=131072,  # 128GB RAM
)
def benchmark_b200_sim(
    batch_size: int = 1,
    seq_length: int = 512,
    num_iterations: int = 10,
    use_bf16: bool = True,
) -> Dict[str, Any]:
    """Benchmark on simulated B200 (using 8xA100 as proxy)."""
    # Note: B200 is not yet available on Modal, using 8xA100 as a high-end proxy
    return run_benchmark("B200_sim", batch_size, seq_length, num_iterations, use_bf16)


def run_benchmark(
    gpu_type: str,
    batch_size: int,
    seq_length: int,
    num_iterations: int,
    use_bf16: bool,
) -> Dict[str, Any]:
    """Run the actual benchmark."""
    import jax
    import jax.numpy as jnp
    from jax import random
    import numpy as np
    
    # Import our model (this would be installed in the container)
    import sys
    sys.path.append('/app')
    from jax_gpt_oss import GPTOSS, GPTOSSConfig, load_model
    
    print(f"=== GPT-OSS-20B JAX Benchmark on {gpu_type} ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    
    # Check GPU memory
    for device in jax.devices():
        if hasattr(device, 'memory_stats'):
            stats = device.memory_stats()
            if stats:
                total_memory = stats.get('bytes_limit', 0) / 1e9
                print(f"GPU Memory: {total_memory:.1f} GB")
    
    # Model configuration
    config = GPTOSSConfig()
    dtype = jnp.bfloat16 if use_bf16 else jnp.float32
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Data type: {dtype}")
    print(f"  Model parameters: 21B (3.6B active)")
    
    # Check if model weights exist
    model_path = "/models/gpt-oss-20b"
    import os
    if not os.path.exists(model_path):
        print(f"Downloading model weights to {model_path}...")
        download_model_weights(model_path)
    
    # Load model
    print("\nLoading model...")
    load_start = time.time()
    try:
        model, params = load_model(model_path, dtype=dtype)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Use dummy model for testing
        print("Using dummy model for benchmark...")
        model = GPTOSS(config, dtype=dtype)
        rng = random.PRNGKey(42)
        dummy_input = jnp.ones((1, 128), dtype=jnp.int32)
        params = model.init(rng, dummy_input)
        load_time = 0.0
    
    # Prepare input
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.float32)
    
    # JIT compile the forward pass
    print("\nCompiling model...")
    
    @jax.jit
    def forward_pass(params, input_ids, attention_mask):
        return model.apply(
            params,
            input_ids,
            attention_mask=attention_mask,
            deterministic=True
        )
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        output = forward_pass(params, input_ids, attention_mask)
        output['logits'].block_until_ready()
    
    # Benchmark
    print(f"\nRunning {num_iterations} iterations...")
    latencies = []
    throughputs = []
    
    for i in range(num_iterations):
        start_time = time.time()
        
        output = forward_pass(params, input_ids, attention_mask)
        output['logits'].block_until_ready()
        
        end_time = time.time()
        iteration_time = end_time - start_time
        
        latencies.append(iteration_time)
        tokens_per_second = (batch_size * seq_length) / iteration_time
        throughputs.append(tokens_per_second)
        
        if i % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {iteration_time:.3f}s "
                  f"({tokens_per_second:.1f} tokens/s)")
    
    # Calculate statistics
    latencies = np.array(latencies)
    throughputs = np.array(throughputs)
    
    # Memory usage
    memory_stats = {}
    for device in jax.devices():
        if hasattr(device, 'memory_stats'):
            stats = device.memory_stats()
            if stats:
                memory_stats = {
                    'bytes_in_use_gb': stats.get('bytes_in_use', 0) / 1e9,
                    'peak_bytes_gb': stats.get('peak_bytes_in_use', 0) / 1e9,
                    'bytes_limit_gb': stats.get('bytes_limit', 0) / 1e9,
                }
                break
    
    results = {
        'gpu_type': gpu_type,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'dtype': str(dtype),
            'num_iterations': num_iterations,
        },
        'performance': {
            'mean_latency_s': float(np.mean(latencies)),
            'std_latency_s': float(np.std(latencies)),
            'min_latency_s': float(np.min(latencies)),
            'max_latency_s': float(np.max(latencies)),
            'p50_latency_s': float(np.percentile(latencies, 50)),
            'p90_latency_s': float(np.percentile(latencies, 90)),
            'p99_latency_s': float(np.percentile(latencies, 99)),
            'mean_throughput_tokens_s': float(np.mean(throughputs)),
            'std_throughput_tokens_s': float(np.std(throughputs)),
            'model_load_time_s': load_time,
        },
        'memory': memory_stats,
        'device_info': {
            'device_count': jax.device_count(),
            'devices': str(jax.devices()),
        }
    }
    
    print("\n=== Results Summary ===")
    print(f"Mean latency: {results['performance']['mean_latency_s']:.3f}s")
    print(f"Mean throughput: {results['performance']['mean_throughput_tokens_s']:.1f} tokens/s")
    print(f"P90 latency: {results['performance']['p90_latency_s']:.3f}s")
    if memory_stats:
        print(f"Memory used: {memory_stats['bytes_in_use_gb']:.1f}GB / "
              f"{memory_stats['bytes_limit_gb']:.1f}GB")
    
    return results


def download_model_weights(model_path: str):
    """Download model weights if not present."""
    import os
    import subprocess
    
    os.makedirs(model_path, exist_ok=True)
    
    # Download essential files
    base_url = "https://huggingface.co/openai/gpt-oss-20b/resolve/main"
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00000-of-00002.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    
    for file in files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            url = f"{base_url}/{file}"
            subprocess.run(["wget", "-q", "-O", file_path, url], check=True)


@stub.local_entrypoint()
def main(
    gpu_types: List[str] = ["H100", "A100"],  # B200 not yet available
    batch_sizes: List[int] = [1, 4, 8],
    seq_lengths: List[int] = [512, 1024, 2048],
    output_file: str = "benchmark_results.json",
):
    """Run benchmarks across different configurations."""
    print("Starting GPT-OSS-20B JAX GPU Benchmarks")
    print("="*60)
    
    all_results = []
    
    for gpu_type in gpu_types:
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                print(f"\nBenchmarking {gpu_type} with batch={batch_size}, seq_len={seq_length}")
                print("-"*40)
                
                try:
                    if gpu_type == "H100":
                        results = benchmark_h100.remote(
                            batch_size=batch_size,
                            seq_length=seq_length,
                            num_iterations=10,
                            use_bf16=True
                        )
                    elif gpu_type == "A100":
                        results = benchmark_a100.remote(
                            batch_size=batch_size,
                            seq_length=seq_length,
                            num_iterations=10,
                            use_bf16=True
                        )
                    elif gpu_type == "B200":
                        results = benchmark_b200_sim.remote(
                            batch_size=batch_size,
                            seq_length=seq_length,
                            num_iterations=10,
                            use_bf16=True
                        )
                    else:
                        print(f"Unknown GPU type: {gpu_type}")
                        continue
                    
                    all_results.append(results)
                    
                    # Print summary
                    print(f"✓ {gpu_type}: {results['performance']['mean_throughput_tokens_s']:.1f} tokens/s")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    all_results.append({
                        'gpu_type': gpu_type,
                        'config': {
                            'batch_size': batch_size,
                            'seq_length': seq_length,
                        },
                        'error': str(e)
                    })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    
    # Print comparison table
    print("\n=== Performance Comparison ===")
    print(f"{'GPU':<10} {'Batch':<8} {'Seq Len':<10} {'Throughput (tokens/s)':<20} {'Latency P90 (s)':<15}")
    print("-"*70)
    
    for result in all_results:
        if 'error' not in result:
            gpu = result['gpu_type']
            batch = result['config']['batch_size']
            seq_len = result['config']['seq_length']
            throughput = result['performance']['mean_throughput_tokens_s']
            p90 = result['performance']['p90_latency_s']
            print(f"{gpu:<10} {batch:<8} {seq_len:<10} {throughput:<20.1f} {p90:<15.3f}")


if __name__ == "__main__":
    main()