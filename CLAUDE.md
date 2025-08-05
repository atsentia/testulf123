# CLAUDE.md - Development Guide for GPT-OSS-20B JAX Implementation

This file provides guidance for Claude Code when working with this repository.

## Project Overview

JAX/Flax implementation of OpenAI's GPT-OSS-20B model - a 21B parameter model with 3.6B active parameters using Mixture of Experts (MoE) architecture.

## Current Status (August 5, 2025)

### ✅ Completed
- Core model architecture with MoE and sliding window attention
- Weight conversion script with MXFP4 dequantization
- Inference pipeline with temperature/top-k/top-p sampling
- Comprehensive testing framework
- Model weights downloaded (13.5GB)

### 🚧 In Progress
- Weight conversion testing on CPU
- Git LFS setup for weight storage

### 📋 TODO
- Complete numerical parity testing with HuggingFace
- Optimize for TPU/GPU deployment
- Add streaming generation support

## Architecture Details

### Unique Features
1. **MXFP4 Quantization**: MoE experts use 4-bit quantization with shared scales
2. **Attention Sinks**: Special tokens for sliding window attention stability
3. **Alternating Attention**: Layers alternate between sliding (128 tokens) and full attention
4. **Biased Attention**: All attention layers include bias terms (unusual for modern transformers)

### Model Configuration
- Vocabulary: 201,088 tokens
- Hidden size: 2,880
- Layers: 24 transformer blocks
- Attention heads: 64 (8 KV heads with GQA)
- MoE: 32 experts, 4 active per token
- Context length: 131K tokens (YARN RoPE scaling)

## File Structure
```
jax-for-gpt-oss-20b/
├── jax_gpt_oss/
│   ├── models/
│   │   ├── config.py       # Configuration
│   │   ├── gpt_oss.py      # Main model (uses standard Flax components)
│   │   ├── moe.py          # MoE implementation
│   │   └── attention.py    # Sliding/full attention
│   ├── utils/
│   │   └── model_utils.py  # Loading/saving utilities
│   └── inference.py        # Generation pipeline
├── scripts/
│   ├── convert_weights.py  # PyTorch → JAX conversion
│   ├── test_loading.py     # CPU loading test
│   └── modal_gpu_benchmark.py # GPU benchmarking
├── tests/
│   ├── test_numerical_parity.py # Layer-by-layer comparison
│   └── test_components.py       # Unit tests
└── requirements.txt
```

## Weight Management

### Downloaded Weights
- Location: `/root/models/gpt-oss-20b/`
- Size: 13.5GB (3 safetensor files)
- Format: MXFP4 quantized for MoE layers

### Converted Weights
- Location: `/root/models/gpt-oss-20b-jax/`
- Format: Orbax checkpoint (JAX native)

### Git LFS (for private repo)
```bash
# Initialize Git LFS
git lfs track "*.safetensors"
git lfs track "jax_params/*"
git add .gitattributes

# Add weights
git add models/
git commit -m "Add model weights via Git LFS"
git push
```

## Commands

### Convert Weights
```bash
python scripts/convert_weights.py \
  --model-path /root/models/gpt-oss-20b \
  --output-path /root/models/gpt-oss-20b-jax \
  --test-loading
```

### Test Loading (CPU)
```bash
# Force CPU mode and test
JAX_PLATFORM_NAME=cpu python scripts/test_loading.py \
  --model-path /root/models/gpt-oss-20b-jax \
  --max-seq-len 32
```

### Run Tests
```bash
# Unit tests
python tests/test_components.py

# Numerical comparison (requires PyTorch)
python tests/test_numerical_parity.py \
  --model-path /root/models/gpt-oss-20b \
  --tolerance 1e-3
```

### Modal GPU Benchmark
```bash
modal run scripts/modal_gpu_benchmark.py
```

## Performance Considerations

### CPU Mode
- Very slow but functional for testing
- Use small sequence lengths (32-64 tokens)
- Set `JAX_ENABLE_X64=0` to reduce memory usage
- Expect ~1-2 tokens/second on CPU

### Memory Requirements
- Model weights: ~13.5GB (MXFP4) or ~54GB (float32)
- Inference: Add ~4-8GB for activations
- CPU mode: ~20GB RAM minimum
- GPU: 40GB+ VRAM recommended

## Important Notes

1. **MXFP4 Dequantization**: The conversion script includes basic MXFP4 → float32 conversion. The actual MXFP4 format may need refinement based on OpenAI's specific implementation.

2. **Attention Sinks**: The model includes special "sink" tokens in attention layers. These are preserved during conversion but their exact usage may need investigation.

3. **Expert Routing**: The MoE implementation uses top-k routing. Load balancing and auxiliary losses are not yet implemented.

4. **CPU Testing**: Always set `JAX_PLATFORM_NAME=cpu` when testing on CPU to avoid JAX trying to use unavailable accelerators.

## Development Workflow

1. Make changes to model architecture if needed
2. Run unit tests: `python tests/test_components.py`
3. Convert weights if structure changed
4. Test loading on CPU with minimal sequence length
5. Commit and push changes
6. Run Modal benchmarks for GPU testing

## Troubleshooting

### Out of Memory on CPU
- Reduce sequence length in test_loading.py
- Use `JAX_ENABLE_X64=0` for 32-bit mode
- Consider using dtype=bfloat16 instead of float32

### Slow Conversion
- Weight conversion is CPU-bound and slow
- Expect 5-10 minutes for full conversion
- Use `--test-loading` flag to verify immediately

### Git LFS Issues
- Ensure Git LFS is installed: `git lfs install`
- Check LFS status: `git lfs status`
- For large files: `git lfs migrate import --include="*.safetensors"`

## References
- Model: https://huggingface.co/openai/gpt-oss-20b
- MXFP4: Microscale data formats for deep learning
- YARN: Yet Another RoPE Extension for context length
- Attention Sinks: Stabilizing attention in sliding window models