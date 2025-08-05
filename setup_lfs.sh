#!/bin/bash
# Setup Git LFS and add model weights

echo "Setting up Git LFS for model weights..."

# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "jax_params/**"
git lfs track "models/**/*.safetensors"
git lfs track "models/**/jax_params/**"

# Create models directory structure
mkdir -p models/gpt-oss-20b

# Copy weights from download location if they exist
if [ -d "/root/models/gpt-oss-20b" ]; then
    echo "Copying model weights..."
    cp -r /root/models/gpt-oss-20b/* models/gpt-oss-20b/
    echo "✓ Weights copied to models/gpt-oss-20b/"
else
    echo "⚠ Weights not found at /root/models/gpt-oss-20b"
    echo "Please download them first using scripts/download_model.sh"
fi

# Add to git
git add .gitattributes
git add models/

echo ""
echo "Git LFS setup complete!"
echo ""
echo "To commit and push the weights:"
echo "  git commit -m 'Add GPT-OSS-20B weights via Git LFS'"
echo "  git push"
echo ""
echo "Note: Pushing 13.5GB may take a while depending on your connection."