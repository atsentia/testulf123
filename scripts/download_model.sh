#!/bin/bash
# Download GPT-OSS-20B model weights

MODEL_DIR="models/gpt-oss-20b"
BASE_URL="https://huggingface.co/openai/gpt-oss-20b/resolve/main"

echo "=== GPT-OSS-20B Model Download ==="
echo "Target: $MODEL_DIR"
echo ""

mkdir -p "$MODEL_DIR"

# Essential files
FILES=(
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "generation_config.json"
    "model.safetensors.index.json"
    "LICENSE"
    "README.md"
)

echo "Downloading configuration files..."
for file in "${FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$file" ]; then
        echo "⬇ Downloading $file..."
        wget -q --show-progress -O "$MODEL_DIR/$file" "$BASE_URL/$file" 2>/dev/null || echo "✗ Failed: $file"
    else
        echo "✓ Already exists: $file"
    fi
done

# Model weights
WEIGHTS=(
    "model-00000-of-00002.safetensors"
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
)

echo ""
echo "Downloading model weights (13.5GB total)..."
for file in "${WEIGHTS[@]}"; do
    if [ ! -f "$MODEL_DIR/$file" ]; then
        echo "⬇ Downloading $file..."
        wget -c -q --show-progress -O "$MODEL_DIR/$file" "$BASE_URL/$file" 2>/dev/null
    else
        echo "✓ Already exists: $file"
    fi
done

echo ""
echo "✅ Download complete!"
echo "Model location: $MODEL_DIR"
echo ""
echo "To convert to JAX format:"
echo "  python scripts/convert_weights.py --model-path $MODEL_DIR"