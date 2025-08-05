#!/bin/bash
# Fix NumPy compatibility issues for JAX and PyTorch

echo "=== Fixing NumPy Compatibility Issues ==="
echo ""
echo "Current NumPy version issue detected. This script will fix it."
echo ""

# Option 1: Downgrade NumPy (simplest solution)
echo "Option 1: Downgrading to NumPy 1.x (Recommended for compatibility)"
echo "This will ensure all packages work together"
echo ""

read -p "Proceed with NumPy downgrade? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downgrading NumPy to 1.x..."
    
    # Uninstall current NumPy
    pip3 uninstall -y numpy 2>/dev/null || true
    
    # Install NumPy 1.x
    pip3 install 'numpy<2.0' --force-reinstall
    
    # Reinstall JAX with compatible versions
    pip3 uninstall -y jax jaxlib ml_dtypes 2>/dev/null || true
    pip3 install --upgrade "jax[cpu]==0.4.30" "jaxlib==0.4.30"
    
    # Install other dependencies
    pip3 install flax optax orbax-checkpoint safetensors tqdm
    
    echo ""
    echo "Testing installation..."
    python3 -c "
import sys
print('Python:', sys.version)
try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
    assert int(np.__version__.split('.')[0]) < 2, 'NumPy 2.x still present!'
except Exception as e:
    print(f'✗ NumPy Error: {e}')
    sys.exit(1)

try:
    import jax
    print(f'✓ JAX: {jax.__version__}')
    import jax.numpy as jnp
    x = jnp.ones(3)
    print(f'  JAX test: {x.sum()} = 3.0')
except Exception as e:
    print(f'✗ JAX Error: {e}')
    sys.exit(1)

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError:
    print('⚠ PyTorch not installed (optional for conversion)')

print('')
print('✅ All packages installed successfully!')
"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== Fix Complete ==="
        echo "You can now run:"
        echo "  python3 scripts/convert_weights.py --model-path models/gpt-oss-20b"
    else
        echo ""
        echo "=== Fix Failed ==="
        echo "Please try manual installation or use the no-PyTorch conversion script"
    fi
else
    echo ""
    echo "=== Alternative: Use No-PyTorch Conversion ==="
    echo "You can use the alternative conversion script that doesn't require PyTorch:"
    echo "  python3 scripts/convert_weights_no_torch.py --model-path models/gpt-oss-20b"
fi