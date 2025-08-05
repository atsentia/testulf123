# Weight Conversion Guide

## Quick Start

The streaming converter is the recommended approach - it's memory-efficient and doesn't require PyTorch.

### Prerequisites
```bash
# Create virtual environment
python3 -m venv jax_env
source jax_env/bin/activate

# Install dependencies (no PyTorch needed!)
pip install --upgrade pip
pip install "numpy==1.26.4"  # NumPy 1.x for compatibility
pip install "jax[cpu]" flax orbax-checkpoint safetensors tqdm
```

### Convert Weights
```bash
# Test with first 2 layers (quick validation)
python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b --test-only

# Full conversion (takes 5-10 minutes)
python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b
```

## Available Conversion Scripts

### 1. `convert_weights_streaming.py` (Recommended)
- **Memory efficient**: Processes one layer at a time
- **No PyTorch required**: Uses only safetensors + JAX
- **Safe**: Avoids segfaults from memory issues
- **Works with any RAM**: Tested on low-memory systems

```bash
python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b
```

### 2. `convert_weights_no_torch.py` (Alternative)
- No PyTorch required
- Loads all weights at once (needs more RAM)
- Handles bfloat16 manually

```bash
python scripts/convert_weights_no_torch.py --model-path models/gpt-oss-20b
```

### 3. `convert_weights_pure_jax.py` (Advanced)
- Uses PyTorch for better bfloat16 handling
- Pure JAX operations throughout
- Requires PyTorch + JAX compatibility

```bash
python scripts/convert_weights_pure_jax.py --model-path models/gpt-oss-20b
```

### 4. `convert_weights.py` (Original)
- Full-featured with MXFP4 dequantization
- Requires PyTorch + JAX compatibility
- Use if other scripts don't work

```bash
python scripts/convert_weights.py --model-path models/gpt-oss-20b
```

## Troubleshooting

### NumPy Compatibility Issues
```bash
# If you get NumPy 2.x errors, use NumPy 1.x
pip uninstall -y numpy
pip install "numpy==1.26.4"

# Then reinstall JAX
pip install "jax[cpu]" --force-reinstall
```

### Out of Memory
```bash
# Use streaming converter (processes one layer at a time)
python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b --test-only

# Set memory limits
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
```

### GPU vs CPU
```bash
# Force CPU mode (safer for conversion)
export JAX_PLATFORM_NAME=cpu

# For GPU (if you have enough VRAM)
export JAX_PLATFORM_NAME=gpu
```

## Output Structure

After conversion, you'll have:
```
models/gpt-oss-20b-jax/
├── config.json           # Model configuration
└── jax_params/           # JAX checkpoint directory
    ├── checkpoint
    └── ...
```

## Git LFS Setup (for sharing weights)

```bash
# Initialize Git LFS for model weights
git lfs track "models/**/*.safetensors"
git lfs track "jax_params/**"
git lfs track "*.pkl"

# Add files
git add .gitattributes
git add models/gpt-oss-20b-jax/
git commit -m "Add converted JAX weights"
git push
```

## Memory Requirements

- **Streaming converter**: ~4-8GB RAM minimum
- **Other converters**: ~20-40GB RAM
- **Model weights**: ~13.5GB original, ~15-20GB JAX format
- **GPU conversion**: 16GB+ VRAM recommended

## Verification

After conversion, test loading:
```bash
python -c "
import jax.numpy as jnp
from jax_gpt_oss.utils.model_utils import load_model

try:
    model, params = load_model('models/gpt-oss-20b-jax')
    print('✓ Model loaded successfully!')
    
    # Test forward pass
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    output = model.apply(params, dummy_input, deterministic=True)
    print(f'✓ Forward pass: {output[\"logits\"].shape}')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

## Performance Notes

- **Conversion time**: 5-15 minutes depending on method
- **CPU inference**: ~1-2 tokens/second (testing only)
- **GPU inference**: 10-50+ tokens/second (depending on GPU)
- **Recommended**: Convert on CPU, infer on GPU