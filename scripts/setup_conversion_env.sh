#!/bin/bash
# Quick setup script for weight conversion

echo "=== Setting up JAX environment for weight conversion ==="
echo ""

# Create virtual environment
if [ ! -d "jax_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv jax_env
else
    echo "Virtual environment already exists"
fi

# Activate environment
echo "Activating virtual environment..."
source jax_env/bin/activate

# Upgrade pip
pip install --upgrade pip

echo ""
echo "Installing dependencies..."

# Install NumPy 1.x first (critical for compatibility)
pip install "numpy==1.26.4"

# Install JAX (CPU version - safer for conversion)
pip install "jax[cpu]"

# Install other required packages
pip install flax orbax-checkpoint safetensors tqdm

echo ""
echo "Verifying installation..."
python -c "
import numpy as np
import jax
import jax.numpy as jnp
import safetensors
print(f'✓ NumPy: {np.__version__}')
print(f'✓ JAX: {jax.__version__} (backend: {jax.default_backend()})')
print(f'✓ Safetensors: {safetensors.__version__}')

# Quick JAX test
x = jnp.ones(3)
print(f'✓ JAX test: {x.sum()} = 3.0')
print('')
print('✅ Environment ready for conversion!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "To use this environment:"
    echo "  source jax_env/bin/activate"
    echo ""
    echo "Then run conversion:"
    echo "  python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b --test-only"
    echo "  python scripts/convert_weights_streaming.py --model-path models/gpt-oss-20b"
else
    echo ""
    echo "=== Setup Failed ==="
    echo "Please check error messages above"
    exit 1
fi