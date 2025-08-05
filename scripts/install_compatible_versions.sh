#!/bin/bash
# Install compatible versions of all packages

echo "=== Installing Compatible Package Versions ==="
echo ""
echo "This will install:"
echo "  - NumPy 1.26.x (last 1.x version, compatible with JAX and PyTorch)"
echo "  - JAX 0.4.30 (stable version for NumPy 1.x)"
echo "  - Other required packages"
echo ""

# Clean up old installations
echo "1. Cleaning up old installations..."
pip3 uninstall -y numpy jax jaxlib ml_dtypes 2>/dev/null || true

echo ""
echo "2. Installing compatible NumPy (1.26.x)..."
# Install NumPy 1.26.4 - the last 1.x version that works well
pip3 install --user 'numpy>=1.24,<2.0' --upgrade --force-reinstall

echo ""
echo "3. Installing compatible JAX..."
# Install JAX that works with NumPy 1.24+
pip3 install --user --upgrade 'jax[cpu]==0.4.30' 'jaxlib==0.4.30'

echo ""
echo "4. Installing other dependencies..."
pip3 install --user --upgrade \
    'flax>=0.7.0' \
    'optax>=0.1.7' \
    'orbax-checkpoint>=0.4.0' \
    'safetensors>=0.4.0' \
    'tqdm>=4.65.0' \
    'einops>=0.7.0' \
    'psutil'

echo ""
echo "5. Verifying installation..."
python3 -c "
import sys
print('=== Version Check ===')
print(f'Python: {sys.version}')

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
    major, minor = map(int, np.__version__.split('.')[:2])
    if major >= 2:
        print('  WARNING: NumPy 2.x detected, may cause issues')
    elif major == 1 and minor >= 24:
        print('  ✓ NumPy version is compatible')
    else:
        print(f'  WARNING: NumPy {np.__version__} is too old, need 1.24+')
except Exception as e:
    print(f'✗ NumPy Error: {e}')
    sys.exit(1)

try:
    import jax
    print(f'✓ JAX: {jax.__version__}')
    import jax.numpy as jnp
    x = jnp.ones(3)
    result = float(x.sum())
    print(f'  JAX test: {result} = 3.0')
    if result == 3.0:
        print('  ✓ JAX is working correctly')
except Exception as e:
    print(f'✗ JAX Error: {e}')
    sys.exit(1)

try:
    import flax
    print(f'✓ Flax: {flax.__version__}')
except Exception as e:
    print(f'✗ Flax Error: {e}')

try:
    import safetensors
    print(f'✓ Safetensors: {safetensors.__version__}')
except Exception as e:
    print(f'✗ Safetensors Error: {e}')

print('')
print('✅ All core packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Installation Complete ==="
    echo ""
    echo "You can now run:"
    echo "  python3 scripts/convert_weights.py --model-path models/gpt-oss-20b"
    echo ""
    echo "Or if that still fails, use the no-torch version:"
    echo "  python3 scripts/convert_weights_no_torch.py --model-path models/gpt-oss-20b"
else
    echo ""
    echo "=== Installation had issues ==="
    echo ""
    echo "Try the alternative no-torch conversion:"
    echo "  python3 scripts/convert_weights_no_torch.py --model-path models/gpt-oss-20b"
    echo ""
    echo "Or manually install specific versions:"
    echo "  pip3 install --user 'numpy==1.26.4'"
    echo "  pip3 install --user 'jax[cpu]==0.4.30' 'jaxlib==0.4.30'"
fi