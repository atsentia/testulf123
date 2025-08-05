# JAX Implementation for GPT-OSS-20B

A high-performance JAX/Flax implementation of OpenAI's GPT-OSS-20B model, featuring Mixture of Experts (MoE) architecture and optimized for TPU/GPU inference.

## Model Architecture

GPT-OSS-20B is a 21B parameter model with 3.6B active parameters, featuring:
- **Mixture of Experts (MoE)**: 32 experts with 4 active per token
- **Hybrid Attention**: Alternating sliding window (128 tokens) and full attention layers
- **YARN RoPE Scaling**: Extended context length up to 131K tokens
- **Vocabulary Size**: 201,088 tokens
- **Hidden Size**: 2,880
- **Layers**: 24 transformer blocks
- **Attention Heads**: 64 (8 KV heads with GQA)

## Installation

```bash
# Clone the repository
git clone https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b

# Install dependencies
pip install -r requirements.txt

# Download model weights (13.5GB)
bash scripts/download_model.sh
```

## Quick Start

```python
from jax_gpt_oss import load_model, generate

# Load model
model, params = load_model("./models/gpt-oss-20b")

# Generate text
prompt = "The future of AI is"
output = generate(model, params, prompt, max_tokens=100)
print(output)
```

## Project Structure

```
jax-for-gpt-oss-20b/
├── jax_gpt_oss/
│   ├── models/          # Model architecture
│   │   ├── gpt_oss.py   # Main GPT-OSS model
│   │   ├── moe.py       # Mixture of Experts layer
│   │   └── config.py    # Configuration
│   ├── utils/           # Utilities
│   └── inference.py     # Inference engine
├── scripts/             # Helper scripts
├── tests/              # Test suite
└── examples/           # Usage examples
```

## Features

- ✅ Pure JAX/Flax implementation
- ✅ MoE (Mixture of Experts) support
- ✅ Sliding window attention
- ✅ YARN RoPE position encoding
- ✅ TPU and GPU optimized
- ✅ Batch inference support
- ✅ INT8/FP16/BF16 quantization

## Performance

Performance varies based on hardware, batch size, and sequence length. The JAX implementation is optimized for:
- TPU v3/v4 pods
- NVIDIA A100/H100 GPUs
- Apple Silicon (via JAX Metal)

## License

Apache 2.0 License (same as the original GPT-OSS-20B model)

## Acknowledgments

- OpenAI for the GPT-OSS-20B model
- JAX and Flax teams for the excellent frameworks
- Inspired by the ml-diffucoder JAX implementation patterns

## Citation

```bibtex
@software{gptoss20b_jax,
  title = {JAX Implementation for GPT-OSS-20B},
  author = {Atsentia},
  year = {2025},
  url = {https://github.com/atsentia/jax-for-gpt-oss-20b}
}
```