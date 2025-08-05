#!/usr/bin/env python3
"""Comprehensive tests for GPT-OSS-20B JAX components."""

import unittest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_gpt_oss.models.config import GPTOSSConfig
from jax_gpt_oss.models.moe import Router, Expert, MixtureOfExperts
from jax_gpt_oss.models.attention import (
    GroupedQueryAttention,
    create_sliding_window_mask,
    apply_yarn_rope
)


class TestConfig(unittest.TestCase):
    """Test configuration handling."""
    
    def test_config_initialization(self):
        """Test config creation with default values."""
        config = GPTOSSConfig()
        
        self.assertEqual(config.vocab_size, 201088)
        self.assertEqual(config.hidden_size, 2880)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual(config.num_attention_heads, 64)
        self.assertEqual(config.num_key_value_heads, 8)
        self.assertEqual(config.num_local_experts, 32)
        self.assertEqual(config.num_experts_per_tok, 4)
        
    def test_layer_types(self):
        """Test alternating layer type pattern."""
        config = GPTOSSConfig()
        
        # Should alternate between sliding and full attention
        for i in range(config.num_hidden_layers):
            if i % 2 == 0:
                self.assertEqual(config.layer_types[i], "sliding_attention")
            else:
                self.assertEqual(config.layer_types[i], "full_attention")
    
    def test_rope_scaling_config(self):
        """Test YARN RoPE scaling configuration."""
        config = GPTOSSConfig()
        
        self.assertIsNotNone(config.rope_scaling)
        self.assertEqual(config.rope_scaling['factor'], 32.0)
        self.assertEqual(config.rope_scaling['rope_type'], 'yarn')
        self.assertEqual(config.rope_scaling['original_max_position_embeddings'], 4096)


class TestRouter(unittest.TestCase):
    """Test the MoE router component."""
    
    def setUp(self):
        self.rng = random.PRNGKey(42)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 128
        self.num_experts = 8
        self.num_experts_per_tok = 2
        
    def test_router_output_shape(self):
        """Test router output shapes."""
        router = Router(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            dtype=jnp.float32
        )
        
        # Create dummy input
        hidden_states = jnp.ones((self.batch_size, self.seq_len, self.hidden_size))
        
        # Initialize router
        params = router.init(self.rng, hidden_states)
        
        # Forward pass
        router_logits, selected_experts, router_probs = router.apply(
            params, hidden_states
        )
        
        # Check shapes
        self.assertEqual(
            router_logits.shape,
            (self.batch_size, self.seq_len, self.num_experts)
        )
        self.assertEqual(
            selected_experts.shape,
            (self.batch_size, self.seq_len, self.num_experts_per_tok)
        )
        self.assertEqual(
            router_probs.shape,
            (self.batch_size, self.seq_len, self.num_experts_per_tok)
        )
    
    def test_router_selects_top_k(self):
        """Test that router selects exactly k experts."""
        router = Router(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size
        )
        
        hidden_states = random.normal(
            self.rng, (self.batch_size, self.seq_len, self.hidden_size)
        )
        
        params = router.init(self.rng, hidden_states)
        _, selected_experts, _ = router.apply(params, hidden_states)
        
        # Check that we have exactly k unique experts per token
        for b in range(self.batch_size):
            for s in range(self.seq_len):
                experts = selected_experts[b, s]
                # Should have exactly num_experts_per_tok selections
                self.assertEqual(len(experts), self.num_experts_per_tok)
                # All should be valid expert indices
                self.assertTrue(jnp.all(experts >= 0))
                self.assertTrue(jnp.all(experts < self.num_experts))


class TestExpert(unittest.TestCase):
    """Test individual expert MLP."""
    
    def setUp(self):
        self.rng = random.PRNGKey(42)
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 128
        self.intermediate_size = 256
        
    def test_expert_output_shape(self):
        """Test expert MLP output shape."""
        expert = Expert(
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            dtype=jnp.float32
        )
        
        hidden_states = jnp.ones((self.batch_size, self.seq_len, self.hidden_size))
        params = expert.init(self.rng, hidden_states)
        output = expert.apply(params, hidden_states)
        
        # Output should have same shape as input
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_expert_swiglu_activation(self):
        """Test SwiGLU activation with clipping."""
        expert = Expert(
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            swiglu_limit=1.0,  # Low limit for testing
            dtype=jnp.float32
        )
        
        # Create input that would produce large activations
        hidden_states = jnp.ones((1, 1, self.hidden_size)) * 10.0
        params = expert.init(self.rng, hidden_states)
        output = expert.apply(params, hidden_states)
        
        # Check that output is bounded (due to swiglu_limit)
        self.assertTrue(jnp.all(jnp.abs(output) < 100.0))


class TestMixtureOfExperts(unittest.TestCase):
    """Test complete MoE layer."""
    
    def setUp(self):
        self.rng = random.PRNGKey(42)
        self.config = GPTOSSConfig()
        # Use smaller values for testing
        self.config.num_local_experts = 4
        self.config.num_experts_per_tok = 2
        self.config.hidden_size = 128
        self.config.intermediate_size = 256
        
        self.batch_size = 2
        self.seq_len = 10
        
    def test_moe_output_shape(self):
        """Test MoE layer output shape."""
        moe = MixtureOfExperts(
            config=self.config,
            dtype=jnp.float32
        )
        
        hidden_states = jnp.ones((self.batch_size, self.seq_len, self.config.hidden_size))
        params = moe.init(self.rng, hidden_states)
        output, router_logits = moe.apply(params, hidden_states)
        
        # Output shape should match input
        self.assertEqual(output.shape, hidden_states.shape)
        
        # Router logits should be None if not outputting them
        if not self.config.output_router_logits:
            self.assertIsNone(router_logits)
    
    def test_moe_deterministic_mode(self):
        """Test MoE in deterministic mode."""
        moe = MixtureOfExperts(
            config=self.config,
            dtype=jnp.float32
        )
        
        hidden_states = random.normal(
            self.rng, (self.batch_size, self.seq_len, self.config.hidden_size)
        )
        params = moe.init(self.rng, hidden_states)
        
        # Run twice in deterministic mode
        output1, _ = moe.apply(params, hidden_states, deterministic=True)
        output2, _ = moe.apply(params, hidden_states, deterministic=True)
        
        # Outputs should be identical
        np.testing.assert_allclose(output1, output2, rtol=1e-5)


class TestAttention(unittest.TestCase):
    """Test attention mechanisms."""
    
    def setUp(self):
        self.rng = random.PRNGKey(42)
        self.config = GPTOSSConfig()
        self.config.hidden_size = 128
        self.config.num_attention_heads = 8
        self.config.num_key_value_heads = 2
        self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        self.batch_size = 2
        self.seq_len = 16
        
    def test_sliding_window_mask(self):
        """Test sliding window mask creation."""
        seq_len = 10
        window_size = 3
        
        mask = create_sliding_window_mask(seq_len, window_size)
        
        # Check shape
        self.assertEqual(mask.shape, (1, 1, seq_len, seq_len))
        
        # Check that diagonal is all ones (can attend to self)
        diagonal = jnp.diagonal(mask[0, 0])
        np.testing.assert_array_equal(diagonal, jnp.ones(seq_len))
        
        # Check window constraint
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # Future position
                    self.assertEqual(mask[0, 0, i, j], 0)
                elif abs(i - j) <= window_size:  # Within window
                    self.assertEqual(mask[0, 0, i, j], 1)
                else:  # Outside window
                    self.assertEqual(mask[0, 0, i, j], 0)
    
    def test_yarn_rope(self):
        """Test YARN RoPE position encoding."""
        batch_size = 2
        seq_len = 10
        num_heads = 4
        head_dim = 32
        
        query = random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        key = random.normal(self.rng, (batch_size, seq_len, num_heads, head_dim))
        position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        rotated_q, rotated_k = apply_yarn_rope(
            query, key, position_ids,
            rope_theta=10000.0,
            scaling_factor=1.0,  # No scaling for basic test
            original_max_position=seq_len,
            beta_fast=32.0,
            beta_slow=1.0
        )
        
        # Check shapes are preserved
        self.assertEqual(rotated_q.shape, query.shape)
        self.assertEqual(rotated_k.shape, key.shape)
        
        # Check that rotation was applied (values should differ)
        self.assertFalse(jnp.allclose(rotated_q, query))
        self.assertFalse(jnp.allclose(rotated_k, key))
    
    def test_grouped_query_attention(self):
        """Test GQA with different layer types."""
        for layer_idx, layer_type in enumerate(['sliding_attention', 'full_attention']):
            with self.subTest(layer_type=layer_type):
                self.config.layer_types = [layer_type]
                
                attn = GroupedQueryAttention(
                    config=self.config,
                    layer_idx=0,
                    dtype=jnp.float32
                )
                
                hidden_states = random.normal(
                    self.rng,
                    (self.batch_size, self.seq_len, self.config.hidden_size)
                )
                
                params = attn.init(self.rng, hidden_states)
                output = attn.apply(params, hidden_states, deterministic=True)
                
                # Check output shape
                self.assertEqual(output.shape, hidden_states.shape)
                
                # Output should be different from input
                self.assertFalse(jnp.allclose(output, hidden_states))


class TestIntegration(unittest.TestCase):
    """Integration tests for combined components."""
    
    def test_attention_moe_sequence(self):
        """Test attention followed by MoE (typical transformer block)."""
        rng = random.PRNGKey(42)
        config = GPTOSSConfig()
        config.hidden_size = 128
        config.num_attention_heads = 8
        config.num_key_value_heads = 2
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.num_local_experts = 4
        config.num_experts_per_tok = 2
        
        batch_size = 2
        seq_len = 10
        
        # Create components
        attn = GroupedQueryAttention(config=config, layer_idx=0, dtype=jnp.float32)
        moe = MixtureOfExperts(config=config, dtype=jnp.float32)
        
        # Initialize
        hidden_states = random.normal(rng, (batch_size, seq_len, config.hidden_size))
        attn_params = attn.init(rng, hidden_states)
        moe_params = moe.init(rng, hidden_states)
        
        # Forward pass
        attn_output = attn.apply(attn_params, hidden_states, deterministic=True)
        moe_output, _ = moe.apply(moe_params, attn_output, deterministic=True)
        
        # Check final output
        self.assertEqual(moe_output.shape, hidden_states.shape)
        self.assertFalse(jnp.allclose(moe_output, hidden_states))


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestExpert))
    suite.addTests(loader.loadTestsFromTestCase(TestMixtureOfExperts))
    suite.addTests(loader.loadTestsFromTestCase(TestAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)