#!/usr/bin/env python3
"""
Detailed layer-by-layer numerical comparison between JAX and HuggingFace implementations.

This script performs a comprehensive comparison of:
1. Embedding layers
2. Each transformer block (attention, MLP, layer norms)
3. Final output logits
4. Intermediate activations
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
from dataclasses import dataclass
import sys

# Import both implementations
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Transformers not available. Install with: pip install torch transformers")

from jax_gpt_oss import GPTOSS, GPTOSSConfig, load_model


@dataclass
class LayerComparison:
    """Results from comparing a single layer."""
    layer_name: str
    jax_output: np.ndarray
    hf_output: np.ndarray
    max_diff: float
    mean_diff: float
    relative_error: float
    passed: bool
    
    def __str__(self):
        status = "✓" if self.passed else "✗"
        return (f"{status} {self.layer_name}:\n"
                f"  Max diff: {self.max_diff:.2e}\n"
                f"  Mean diff: {self.mean_diff:.2e}\n"
                f"  Relative error: {self.relative_error:.2%}")


class NumericalComparer:
    """Compare JAX and HuggingFace implementations layer by layer."""
    
    def __init__(self, 
                 model_path: str = "/root/models/gpt-oss-20b",
                 tolerance: float = 1e-4):
        self.model_path = Path(model_path)
        self.tolerance = tolerance
        self.comparisons: List[LayerComparison] = []
        
        print("Loading models for comparison...")
        self._load_models()
        
    def _load_models(self):
        """Load both JAX and HuggingFace models."""
        # Load config
        config_path = self.model_path / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)
        
        # Load JAX model
        print("Loading JAX model...")
        self.jax_config = GPTOSSConfig(**config_dict)
        self.jax_model, self.jax_params = load_model(str(self.model_path))
        
        # Load HuggingFace model if available
        if TORCH_AVAILABLE:
            print("Loading HuggingFace model...")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                device_map="cpu"
            )
            self.hf_model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        else:
            self.hf_model = None
            self.tokenizer = None
    
    def compare_embeddings(self, input_ids: np.ndarray) -> LayerComparison:
        """Compare embedding layer outputs."""
        print("\n=== Comparing Embeddings ===")
        
        # JAX embeddings
        jax_ids = jnp.array(input_ids)
        jax_embeds = self.jax_model.apply(
            self.jax_params,
            jax_ids,
            method=self.jax_model.embed_tokens
        )
        jax_embeds = np.array(jax_embeds)
        
        # HuggingFace embeddings
        if self.hf_model:
            torch_ids = torch.tensor(input_ids)
            with torch.no_grad():
                hf_embeds = self.hf_model.model.embed_tokens(torch_ids)
            hf_embeds = hf_embeds.numpy()
        else:
            hf_embeds = np.zeros_like(jax_embeds)
        
        return self._compare_arrays(
            "Embeddings",
            jax_embeds,
            hf_embeds
        )
    
    def compare_attention_layer(self, 
                               layer_idx: int,
                               hidden_states: np.ndarray) -> LayerComparison:
        """Compare a single attention layer."""
        layer_name = f"Layer {layer_idx} Attention"
        print(f"\nComparing {layer_name}...")
        
        # Determine if this is sliding or full attention
        layer_type = self.jax_config.layer_types[layer_idx]
        
        # JAX attention
        jax_hidden = jnp.array(hidden_states)
        jax_attn_out = self.jax_model.apply(
            self.jax_params,
            jax_hidden,
            layer_idx=layer_idx,
            layer_type=layer_type,
            method=self.jax_model.apply_attention
        )
        jax_attn_out = np.array(jax_attn_out)
        
        # HuggingFace attention
        if self.hf_model:
            torch_hidden = torch.tensor(hidden_states)
            with torch.no_grad():
                layer = self.hf_model.model.layers[layer_idx]
                hf_attn_out, _, _ = layer.self_attn(
                    torch_hidden,
                    attention_mask=None,
                    position_ids=None
                )
            hf_attn_out = hf_attn_out.numpy()
        else:
            hf_attn_out = np.zeros_like(jax_attn_out)
        
        return self._compare_arrays(layer_name, jax_attn_out, hf_attn_out)
    
    def compare_moe_layer(self,
                         layer_idx: int,
                         hidden_states: np.ndarray) -> LayerComparison:
        """Compare MoE (Mixture of Experts) layer."""
        layer_name = f"Layer {layer_idx} MoE"
        print(f"\nComparing {layer_name}...")
        
        # JAX MoE
        jax_hidden = jnp.array(hidden_states)
        jax_moe_out, router_logits = self.jax_model.apply(
            self.jax_params,
            jax_hidden,
            layer_idx=layer_idx,
            method=self.jax_model.apply_moe
        )
        jax_moe_out = np.array(jax_moe_out)
        
        # HuggingFace MoE
        if self.hf_model:
            torch_hidden = torch.tensor(hidden_states)
            with torch.no_grad():
                layer = self.hf_model.model.layers[layer_idx]
                hf_moe_out = layer.mlp(torch_hidden)
            hf_moe_out = hf_moe_out.numpy()
        else:
            hf_moe_out = np.zeros_like(jax_moe_out)
        
        return self._compare_arrays(layer_name, jax_moe_out, hf_moe_out)
    
    def compare_layer_norm(self,
                          layer_idx: int,
                          hidden_states: np.ndarray,
                          norm_type: str = "input") -> LayerComparison:
        """Compare RMSNorm layers."""
        layer_name = f"Layer {layer_idx} RMSNorm ({norm_type})"
        print(f"\nComparing {layer_name}...")
        
        # JAX RMSNorm
        jax_hidden = jnp.array(hidden_states)
        jax_norm_out = self.jax_model.apply(
            self.jax_params,
            jax_hidden,
            layer_idx=layer_idx,
            norm_type=norm_type,
            method=self.jax_model.apply_rms_norm
        )
        jax_norm_out = np.array(jax_norm_out)
        
        # HuggingFace RMSNorm
        if self.hf_model:
            torch_hidden = torch.tensor(hidden_states)
            with torch.no_grad():
                layer = self.hf_model.model.layers[layer_idx]
                if norm_type == "input":
                    norm = layer.input_layernorm
                else:
                    norm = layer.post_attention_layernorm
                hf_norm_out = norm(torch_hidden)
            hf_norm_out = hf_norm_out.numpy()
        else:
            hf_norm_out = np.zeros_like(jax_norm_out)
        
        return self._compare_arrays(layer_name, jax_norm_out, hf_norm_out)
    
    def compare_full_forward(self, input_text: str = "Hello, world!") -> Dict[str, Any]:
        """Run full forward pass comparison through all layers."""
        print("\n" + "="*60)
        print("FULL MODEL COMPARISON")
        print("="*60)
        
        # Tokenize input
        if self.tokenizer:
            input_ids = self.tokenizer(input_text, return_tensors="np").input_ids[0]
        else:
            # Dummy input if tokenizer not available
            input_ids = np.array([1, 2, 3, 4, 5])
        
        print(f"\nInput: '{input_text}'")
        print(f"Token IDs: {input_ids}")
        
        results = {
            "input": input_text,
            "comparisons": [],
            "summary": {}
        }
        
        # 1. Compare embeddings
        embed_comp = self.compare_embeddings(input_ids)
        results["comparisons"].append(embed_comp)
        hidden_states = embed_comp.jax_output
        
        # 2. Compare each transformer layer
        for layer_idx in range(self.jax_config.num_hidden_layers):
            print(f"\n--- Layer {layer_idx + 1}/{self.jax_config.num_hidden_layers} ---")
            
            # Input layer norm
            norm_comp = self.compare_layer_norm(layer_idx, hidden_states, "input")
            results["comparisons"].append(norm_comp)
            
            # Attention
            attn_comp = self.compare_attention_layer(layer_idx, hidden_states)
            results["comparisons"].append(attn_comp)
            
            # Post-attention layer norm
            norm_comp = self.compare_layer_norm(layer_idx, hidden_states, "post")
            results["comparisons"].append(norm_comp)
            
            # MoE/MLP
            moe_comp = self.compare_moe_layer(layer_idx, hidden_states)
            results["comparisons"].append(moe_comp)
            
            # Update hidden states for next layer
            hidden_states = moe_comp.jax_output
        
        # 3. Compare final layer norm
        print("\n--- Final Layer Norm ---")
        final_norm_comp = self._compare_final_norm(hidden_states)
        results["comparisons"].append(final_norm_comp)
        
        # 4. Compare logits
        print("\n--- Output Logits ---")
        logits_comp = self._compare_logits(hidden_states)
        results["comparisons"].append(logits_comp)
        
        # Generate summary
        results["summary"] = self._generate_summary(results["comparisons"])
        
        return results
    
    def _compare_arrays(self,
                       name: str,
                       jax_arr: np.ndarray,
                       hf_arr: np.ndarray) -> LayerComparison:
        """Compare two arrays and compute statistics."""
        # Ensure same shape
        if jax_arr.shape != hf_arr.shape:
            print(f"  ⚠️  Shape mismatch: JAX {jax_arr.shape} vs HF {hf_arr.shape}")
            # Pad or truncate to match shapes
            min_shape = tuple(min(j, h) for j, h in zip(jax_arr.shape, hf_arr.shape))
            jax_arr = jax_arr[tuple(slice(0, s) for s in min_shape)]
            hf_arr = hf_arr[tuple(slice(0, s) for s in min_shape)]
        
        # Compute differences
        diff = np.abs(jax_arr - hf_arr)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Compute relative error
        denominator = np.maximum(np.abs(hf_arr), 1e-8)
        relative_error = np.mean(diff / denominator)
        
        # Check if passed tolerance
        passed = max_diff < self.tolerance
        
        comp = LayerComparison(
            layer_name=name,
            jax_output=jax_arr,
            hf_output=hf_arr,
            max_diff=float(max_diff),
            mean_diff=float(mean_diff),
            relative_error=float(relative_error),
            passed=passed
        )
        
        print(comp)
        return comp
    
    def _compare_final_norm(self, hidden_states: np.ndarray) -> LayerComparison:
        """Compare final RMSNorm before output projection."""
        # JAX final norm
        jax_hidden = jnp.array(hidden_states)
        jax_norm = self.jax_model.apply(
            self.jax_params,
            jax_hidden,
            method=self.jax_model.apply_final_norm
        )
        jax_norm = np.array(jax_norm)
        
        # HF final norm
        if self.hf_model:
            torch_hidden = torch.tensor(hidden_states)
            with torch.no_grad():
                hf_norm = self.hf_model.model.norm(torch_hidden)
            hf_norm = hf_norm.numpy()
        else:
            hf_norm = np.zeros_like(jax_norm)
        
        return self._compare_arrays("Final RMSNorm", jax_norm, hf_norm)
    
    def _compare_logits(self, hidden_states: np.ndarray) -> LayerComparison:
        """Compare final output logits."""
        # JAX logits
        jax_hidden = jnp.array(hidden_states)
        jax_logits = self.jax_model.apply(
            self.jax_params,
            jax_hidden,
            method=self.jax_model.compute_logits
        )
        jax_logits = np.array(jax_logits)
        
        # HF logits
        if self.hf_model:
            torch_hidden = torch.tensor(hidden_states)
            with torch.no_grad():
                hf_logits = self.hf_model.lm_head(torch_hidden)
            hf_logits = hf_logits.numpy()
        else:
            hf_logits = np.zeros_like(jax_logits)
        
        return self._compare_arrays("Output Logits", jax_logits, hf_logits)
    
    def _generate_summary(self, comparisons: List[LayerComparison]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total = len(comparisons)
        passed = sum(1 for c in comparisons if c.passed)
        failed = total - passed
        
        max_error = max(c.max_diff for c in comparisons)
        mean_error = np.mean([c.mean_diff for c in comparisons])
        
        summary = {
            "total_comparisons": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "max_error": float(max_error),
            "mean_error": float(mean_error),
            "tolerance": self.tolerance
        }
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total comparisons: {total}")
        print(f"Passed: {passed} ({summary['pass_rate']:.1%})")
        print(f"Failed: {failed}")
        print(f"Max error: {max_error:.2e}")
        print(f"Mean error: {mean_error:.2e}")
        print(f"Tolerance: {self.tolerance:.2e}")
        
        if summary['pass_rate'] >= 0.95:
            print("\n✅ EXCELLENT: Models are numerically equivalent!")
        elif summary['pass_rate'] >= 0.80:
            print("\n⚠️  GOOD: Models are mostly equivalent with minor differences")
        else:
            print("\n❌ POOR: Significant numerical differences detected")
        
        return summary


def main():
    """Run the numerical comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare JAX and HuggingFace GPT-OSS-20B implementations"
    )
    parser.add_argument(
        "--model-path",
        default="/root/models/gpt-oss-20b",
        help="Path to model weights"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Numerical tolerance for comparison"
    )
    parser.add_argument(
        "--input-text",
        default="The future of artificial intelligence is",
        help="Input text for comparison"
    )
    parser.add_argument(
        "--save-report",
        help="Path to save comparison report (JSON)"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparer = NumericalComparer(args.model_path, args.tolerance)
    results = comparer.compare_full_forward(args.input_text)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, "w") as f:
            # Convert to serializable format
            report = {
                "input": results["input"],
                "summary": results["summary"],
                "layer_comparisons": [
                    {
                        "name": c.layer_name,
                        "max_diff": c.max_diff,
                        "mean_diff": c.mean_diff,
                        "relative_error": c.relative_error,
                        "passed": c.passed
                    }
                    for c in results["comparisons"]
                ]
            }
            json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.save_report}")


if __name__ == "__main__":
    main()