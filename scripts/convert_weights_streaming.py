#!/usr/bin/env python3
"""
Memory-efficient streaming conversion of GPT-OSS-20B weights to JAX format.
Processes weights one at a time to minimize memory usage.
"""

import os
import sys
import json
import argparse
import gc
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from tqdm import tqdm
import pickle

# Set CPU mode for JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
# Limit JAX memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import NumPy
import numpy as np

# Try importing JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    logger.info(f"JAX version: {jax.__version__}")
    # Limit JAX memory usage
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
except ImportError as e:
    logger.warning(f"JAX not available: {e}")
    JAX_AVAILABLE = False


def convert_layer_streaming(model_path: Path, layer_idx: int, output_dir: Path) -> None:
    """
    Convert a single layer's weights and save immediately.
    This minimizes memory usage by processing one layer at a time.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find weights for this layer
    layer_prefix = f"model.layers.{layer_idx}"
    layer_weights = {}
    
    # Collect all weight names for this layer
    weight_names = [k for k in weight_map.keys() if k.startswith(layer_prefix)]
    
    if not weight_names:
        logger.warning(f"No weights found for layer {layer_idx}")
        return
    
    # Load only this layer's weights
    loaded_files = set()
    for weight_name in weight_names:
        file_name = weight_map[weight_name]
        
        if file_name not in loaded_files:
            file_path = model_path / file_name
            if not file_path.exists():
                continue
            
            with safe_open(file_path, framework="np") as f:
                for key in f.keys():
                    if key.startswith(layer_prefix):
                        try:
                            tensor = f.get_tensor(key)
                            # Convert to float32 if needed
                            if tensor.dtype != np.float32:
                                tensor = tensor.astype(np.float32)
                            layer_weights[key] = tensor
                        except Exception as e:
                            logger.warning(f"Could not load {key}: {e}")
            
            loaded_files.add(file_name)
    
    # Convert layer weights to JAX format
    layer_params = {}
    
    # Layer norms
    input_ln = f"{layer_prefix}.input_layernorm.weight"
    post_attn_ln = f"{layer_prefix}.post_attention_layernorm.weight"
    
    if input_ln in layer_weights:
        if JAX_AVAILABLE:
            scale = jnp.array(layer_weights[input_ln], dtype=jnp.float32)
        else:
            scale = layer_weights[input_ln].astype(np.float32)
        layer_params["input_layernorm"] = {"scale": scale}
    
    if post_attn_ln in layer_weights:
        if JAX_AVAILABLE:
            scale = jnp.array(layer_weights[post_attn_ln], dtype=jnp.float32)
        else:
            scale = layer_weights[post_attn_ln].astype(np.float32)
        layer_params["post_attention_layernorm"] = {"scale": scale}
    
    # Attention weights
    attn_params = {}
    attn_prefix = f"{layer_prefix}.self_attn"
    
    for proj in ["q", "k", "v", "o"]:
        weight_key = f"{attn_prefix}.{proj}_proj.weight"
        bias_key = f"{attn_prefix}.{proj}_proj.bias"
        
        if weight_key in layer_weights:
            weight = layer_weights[weight_key]
            # Transpose for JAX convention
            kernel = weight.T.astype(np.float32)
            if JAX_AVAILABLE:
                kernel = jnp.array(kernel)
            
            attn_params[f"{proj}_proj"] = {"kernel": kernel}
            
            if bias_key in layer_weights:
                bias = layer_weights[bias_key].astype(np.float32)
                if JAX_AVAILABLE:
                    bias = jnp.array(bias)
                attn_params[f"{proj}_proj"]["bias"] = bias
    
    layer_params["self_attn"] = attn_params
    
    # MLP/MoE weights (simplified - skipping MXFP4 for now)
    mlp_params = {}
    mlp_prefix = f"{layer_prefix}.mlp"
    
    # Router
    router_weight = f"{mlp_prefix}.router.weight"
    if router_weight in layer_weights:
        weight = layer_weights[router_weight]
        kernel = weight.T.astype(np.float32)
        if JAX_AVAILABLE:
            kernel = jnp.array(kernel)
        mlp_params["router"] = {"router_weights": kernel}
    
    layer_params["mlp"] = mlp_params
    
    # Save this layer immediately
    layer_file = output_dir / f"layer_{layer_idx}.pkl"
    with open(layer_file, "wb") as f:
        pickle.dump(layer_params, f)
    
    logger.info(f"Saved layer {layer_idx} to {layer_file}")
    
    # Free memory
    del layer_weights
    del layer_params
    gc.collect()


def convert_embeddings_streaming(model_path: Path, output_dir: Path) -> None:
    """Convert and save embeddings separately."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Find embedding weights
    embed_params = {}
    
    if "model.embed_tokens.weight" in weight_map:
        file_name = weight_map["model.embed_tokens.weight"]
        file_path = model_path / file_name
        
        if file_path.exists():
            with safe_open(file_path, framework="np") as f:
                if "model.embed_tokens.weight" in f.keys():
                    try:
                        embedding = f.get_tensor("model.embed_tokens.weight")
                        if embedding.dtype != np.float32:
                            embedding = embedding.astype(np.float32)
                        
                        if JAX_AVAILABLE:
                            embedding = jnp.array(embedding)
                        
                        embed_params["embed_tokens"] = {"embedding": embedding}
                        
                        # Save immediately
                        embed_file = output_dir / "embeddings.pkl"
                        with open(embed_file, "wb") as f:
                            pickle.dump(embed_params, f)
                        
                        logger.info(f"Saved embeddings to {embed_file}")
                        
                        # Free memory
                        del embedding
                        del embed_params
                        gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Could not convert embeddings: {e}")


def convert_final_layers_streaming(model_path: Path, output_dir: Path) -> None:
    """Convert and save final norm and lm_head."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")
    
    # Load index
    index_file = model_path / "model.safetensors.index.json"
    with open(index_file, "r") as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    final_params = {}
    
    # Load final weights
    for weight_name in ["model.norm.weight", "lm_head.weight"]:
        if weight_name in weight_map:
            file_name = weight_map[weight_name]
            file_path = model_path / file_name
            
            if file_path.exists():
                with safe_open(file_path, framework="np") as f:
                    if weight_name in f.keys():
                        try:
                            tensor = f.get_tensor(weight_name)
                            if tensor.dtype != np.float32:
                                tensor = tensor.astype(np.float32)
                            
                            if weight_name == "model.norm.weight":
                                if JAX_AVAILABLE:
                                    tensor = jnp.array(tensor)
                                final_params["norm"] = {"scale": tensor}
                            
                            elif weight_name == "lm_head.weight":
                                # Transpose for JAX
                                kernel = tensor.T
                                if JAX_AVAILABLE:
                                    kernel = jnp.array(kernel)
                                final_params["lm_head"] = {"kernel": kernel}
                            
                        except Exception as e:
                            logger.error(f"Could not convert {weight_name}: {e}")
    
    # Save final layers
    if final_params:
        final_file = output_dir / "final_layers.pkl"
        with open(final_file, "wb") as f:
            pickle.dump(final_params, f)
        logger.info(f"Saved final layers to {final_file}")
        
        # Free memory
        del final_params
        gc.collect()


def combine_checkpoints(output_dir: Path, config: Dict[str, Any]) -> None:
    """Combine all saved layer files into final checkpoint."""
    logger.info("Combining layer checkpoints...")
    
    combined_params = {"params": {}}
    
    # Load embeddings
    embed_file = output_dir / "embeddings.pkl"
    if embed_file.exists():
        with open(embed_file, "rb") as f:
            embed_params = pickle.load(f)
            combined_params["params"].update(embed_params)
    
    # Load each layer
    num_layers = config.get("num_hidden_layers", 24)
    for layer_idx in range(num_layers):
        layer_file = output_dir / f"layer_{layer_idx}.pkl"
        if layer_file.exists():
            with open(layer_file, "rb") as f:
                layer_params = pickle.load(f)
                combined_params["params"][f"layers_{layer_idx}"] = layer_params
    
    # Load final layers
    final_file = output_dir / "final_layers.pkl"
    if final_file.exists():
        with open(final_file, "rb") as f:
            final_params = pickle.load(f)
            combined_params["params"].update(final_params)
    
    # Save combined checkpoint
    if JAX_AVAILABLE:
        try:
            from flax.core import freeze
            import orbax.checkpoint as ocp
            
            frozen_params = freeze(combined_params)
            
            ckpt_path = output_dir / "jax_params"
            try:
                from orbax.checkpoint import PyTreeCheckpointer
                ckptr = PyTreeCheckpointer()
            except ImportError:
                ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
            
            ckptr.save(ckpt_path, frozen_params)
            logger.info(f"Saved JAX checkpoint to {ckpt_path}")
            
        except Exception as e:
            logger.warning(f"Could not save as JAX checkpoint: {e}")
            # Save as pickle instead
            with open(output_dir / "params.pkl", "wb") as f:
                pickle.dump(combined_params, f)
            logger.info("Saved as pickle file")
    else:
        # Save as pickle
        with open(output_dir / "params.pkl", "wb") as f:
            pickle.dump(combined_params, f)
        logger.info("Saved as pickle file (JAX not available)")
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    for layer_idx in range(num_layers):
        layer_file = output_dir / f"layer_{layer_idx}.pkl"
        if layer_file.exists():
            layer_file.unlink()
    
    if embed_file.exists():
        embed_file.unlink()
    if final_file.exists():
        final_file.unlink()
    
    logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient streaming conversion of GPT-OSS-20B weights"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gpt-oss-20b",
        help="Path to model weights"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/gpt-oss-20b-jax",
        help="Output path for JAX weights"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test with first 2 layers only"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    
    if not model_path.exists():
        logger.error(f"Model path {model_path} does not exist")
        sys.exit(1)
    
    # Load config
    logger.info("Loading configuration...")
    config_file = model_path / "config.json"
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
        
    with open(config_file, "r") as f:
        config = json.load(f)
    
    num_layers = config.get("num_hidden_layers", 24)
    if args.test_only:
        num_layers = min(2, num_layers)
        logger.info(f"TEST MODE: Converting only {num_layers} layers")
    else:
        logger.info(f"Converting {num_layers} layers")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Convert embeddings first
    logger.info("Converting embeddings...")
    try:
        convert_embeddings_streaming(model_path, output_path)
    except Exception as e:
        logger.error(f"Failed to convert embeddings: {e}")
        logger.info("Continuing without embeddings...")
    
    # Convert each layer
    for layer_idx in tqdm(range(num_layers), desc="Converting layers"):
        try:
            convert_layer_streaming(model_path, layer_idx, output_path)
        except Exception as e:
            logger.error(f"Failed to convert layer {layer_idx}: {e}")
            continue
    
    # Convert final layers
    logger.info("Converting final layers...")
    try:
        convert_final_layers_streaming(model_path, output_path)
    except Exception as e:
        logger.error(f"Failed to convert final layers: {e}")
        logger.info("Continuing without final layers...")
    
    # Combine all checkpoints
    logger.info("Combining checkpoints...")
    combine_checkpoints(output_path, config)
    
    logger.info(f"\n✓ Streaming conversion complete! Saved to {output_path}")
    
    if args.test_only:
        logger.info("\nTEST MODE: Only converted first 2 layers")
        logger.info("Run without --test-only flag for full conversion")


if __name__ == "__main__":
    main()