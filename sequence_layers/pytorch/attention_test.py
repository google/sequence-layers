# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for PyTorch attention layers."""

import fractions
import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import attention


class DotProductSelfAttentionTest(SequenceLayerTest):
    """Test DotProductSelfAttention layer."""
    
    def test_basic_self_attention(self):
        """Test basic self-attention functionality."""
        x = self.random_sequence(2, 8, 16)
        
        # Test basic self-attention
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        self.assertEqual(layer.input_latency, 0)
    
    def test_self_attention_with_future_horizon(self):
        """Test self-attention with future horizon."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=3,
            max_future_horizon=2,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertTrue(layer.supports_step())
        self.assertEqual(layer.input_latency, 2)
    
    def test_self_attention_causal(self):
        """Test causal self-attention."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=-1,
            max_future_horizon=0,
        )
        
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertFalse(layer.supports_step())  # Not steppable with infinite past
    
    def test_self_attention_unmasked(self):
        """Test self-attention with unmasked sequences."""
        x = self.random_sequence(2, 8, 16)
        
        # Test with unlimited horizons
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=-1,
            max_future_horizon=-1,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertFalse(layer.supports_step())  # No step support for unlimited horizons
        self.assertEqual(layer.input_latency, 0)
    
    def test_self_attention_with_dropout(self):
        """Test self-attention with dropout."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
            attention_probabilities_dropout_rate=0.1,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertTrue(layer.supports_step())
        self.assertEqual(layer.input_latency, 0)
    
    def test_self_attention_with_per_dim_scale(self):
        """Test self-attention with per-dimension scaling."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
            per_dim_scale=True,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertTrue(layer.supports_step())
        self.assertEqual(layer.input_latency, 0)
    
    def test_self_attention_with_soft_cap(self):
        """Test self-attention with soft cap."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
            attention_logits_soft_cap=30.0,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertTrue(layer.supports_step())
        self.assertEqual(layer.input_latency, 0)
    
    def test_self_attention_with_gqa(self):
        """Test self-attention with grouped query attention."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=8,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
            num_kv_heads=2,  # 8 query heads, 2 kv heads
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (8, 8))
        self.assertTrue(layer.supports_step())
    
    def test_self_attention_projection_configs(self):
        """Test different projection configurations."""
        x = self.random_sequence(2, 8, 16)
        
        # Test each projection configuration
        projection_configs = [
            attention.CombinedQueryKeyValueProjection(),
            attention.CombinedQueryKeyValueProjection(share_kv_projection=True),
            attention.SeparateQueryKeyValueProjection(),
            attention.QueryAndKeyValueProjection(),
            attention.QueryAndSharedKeyValueProjection(),
        ]
        
        for config in projection_configs:
            with self.subTest(config=config):
                layer = attention.DotProductSelfAttention(
                    input_size=16,
                    num_heads=4,
                    units_per_head=8,
                    max_past_horizon=7,
                    max_future_horizon=0,
                    input_projection=config,
                )
                
                y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
                
                self.assertEqual(y.channel_shape, (4, 8))
                self.assertTrue(layer.supports_step())
    
    def test_self_attention_emits(self):
        """Test self-attention emits."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
        )
        
        # Test layer with emits
        y, emits = layer.layer_with_emits(x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertIsInstance(emits, attention.SelfAttentionEmits)
        self.assertEqual(emits.probabilities.shape, (2, 8, 4, 8))
    
    def test_self_attention_state_management(self):
        """Test self-attention state management."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=4,
            max_future_horizon=2,
        )
        
        # Test initial state
        state = layer.get_initial_state(2, x.channel_spec, training=False)
        
        self.assertIsInstance(state, dict)
        self.assertIn('kv_buffer_keys', state)
        self.assertIn('kv_buffer_values', state)
        self.assertIn('kv_buffer_mask', state)
        self.assertIn('query_delay_buffer', state)
        self.assertIn('time_step', state)
        
        # Check buffer shapes - they should start empty and grow as needed
        self.assertEqual(state['kv_buffer_keys'].shape, (2, 0, 4, 8))  # Empty initially
        self.assertEqual(state['kv_buffer_values'].shape, (2, 0, 4, 8))  # Empty initially
        self.assertEqual(state['kv_buffer_mask'].shape, (2, 0))  # Empty initially
        
        # Check query delay buffer - should be pre-allocated for future horizon
        self.assertIsNotNone(state['query_delay_buffer'])
        self.assertEqual(state['query_delay_buffer'].shape, (2, 2, 4, 8))  # Pre-allocated for future horizon
    
    def test_self_attention_step_execution(self):
        """Test step-wise execution."""
        x = self.random_sequence(2, 8, 16)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=4,
            max_future_horizon=0,
        )
        
        # Test step-by-step execution
        state = layer.get_initial_state(2, x.channel_spec, training=False)
        
        outputs = []
        for t in range(x.shape[1]):
            x_t = x[:, t:t+1]
            y_t, state, emits = layer.step_with_emits(x_t, state, training=False)
            outputs.append(y_t)
        
        # Concatenate step outputs
        y_step = Sequence.concatenate_sequences(outputs)
        
        # Compare with layer output
        y_layer = layer.layer(x, training=False)
        
        self.assertSequencesClose(y_step, y_layer, atol=1e-6)
    
    def test_self_attention_validation(self):
        """Test parameter validation."""
        x = self.random_sequence(2, 8, 16)
        
        # Test invalid past horizon
        with self.assertRaises(ValueError):
            attention.DotProductSelfAttention(
                input_size=16,
                num_heads=4,
                units_per_head=8,
                max_past_horizon=-2,
                max_future_horizon=0,
            )
        
        # Test invalid future horizon
        with self.assertRaises(ValueError):
            attention.DotProductSelfAttention(
                input_size=16,
                num_heads=4,
                units_per_head=8,
                max_past_horizon=4,
                max_future_horizon=-2,
            )
        
        # Test both horizons zero
        with self.assertRaises(ValueError):
            attention.DotProductSelfAttention(
                input_size=16,
                num_heads=4,
                units_per_head=8,
                max_past_horizon=0,
                max_future_horizon=0,
            )
        
        # Test invalid GQA configuration
        with self.assertRaises(ValueError):
            attention.DotProductSelfAttention(
                input_size=16,
                num_heads=5,
                units_per_head=8,
                num_kv_heads=3,  # 5 not divisible by 3
                max_past_horizon=4,
                max_future_horizon=0,
            )
    
    def test_self_attention_output_shapes(self):
        """Test output shape computation."""
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
        )
        
        # Test valid input shape
        input_shape = (16,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4, 8))
        
        # Test invalid input shape
        with self.assertRaises(ValueError):
            layer.get_output_shape((4, 16))  # Too many dimensions
    
    def test_self_attention_different_lengths(self):
        """Test with different sequence lengths."""
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
        )
        
        # Test with different sequence lengths
        for seq_len in [1, 4, 16, 32]:
            with self.subTest(seq_len=seq_len):
                x = self.random_sequence(2, seq_len, 16)
                y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
                
                self.assertEqual(y.channel_shape, (4, 8))
                self.assertEqual(y.shape[1], seq_len)
    
    def test_self_attention_masking(self):
        """Test with masked inputs."""
        # Create sequence with some invalid timesteps
        values = torch.randn(2, 8, 16)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False  # Last 3 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = attention.DotProductSelfAttention(
            input_size=16,
            num_heads=4,
            units_per_head=8,
            max_past_horizon=7,
            max_future_horizon=0,
        )
        
        y = self.verify_contract(layer, x, training=False, grad_atol=1e-5)
        
        self.assertEqual(y.channel_shape, (4, 8))
        self.assertEqual(y.mask.shape, (2, 8))


class AttentionFunctionsTest(SequenceLayerTest):
    """Test attention utility functions."""
    
    def test_scale_query(self):
        """Test query scaling function."""
        queries = torch.randn(2, 8, 4, 16)
        
        # Test default scaling
        scaled = attention._scale_query(queries)
        expected_scale = 1.0 / math.sqrt(16)
        expected = queries * expected_scale
        self.assertTrue(torch.allclose(scaled, expected))
        
        # Test with per-dim scale
        per_dim_scale = torch.randn(16)
        scaled = attention._scale_query(queries, per_dim_scale=per_dim_scale)
        expected = queries * (1.0 + per_dim_scale) / math.sqrt(16)
        self.assertTrue(torch.allclose(scaled, expected))
        
        # Test with custom query scale
        query_scale = 0.5
        scaled = attention._scale_query(queries, query_scale=query_scale)
        expected = queries * query_scale
        self.assertTrue(torch.allclose(scaled, expected))
    
    def test_causal_mask_creation(self):
        """Test causal mask creation."""
        device = torch.device('cpu')
        
        # Test causal mask (past only)
        mask = attention._self_attention_causal_mask(4, 4, -1, 0, device)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 1, 4, 4))
        
        # Check that it's lower triangular
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        self.assertTrue(torch.equal(mask[0, 0], expected))
        
        # Test finite horizons
        mask = attention._self_attention_causal_mask(4, 4, 2, 1, device)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 1, 4, 4))
        
        # Test unmasked
        mask = attention._self_attention_causal_mask(4, 4, -1, -1, device)
        self.assertIsNone(mask)
    
    def test_soft_cap_logits(self):
        """Test soft cap for attention logits."""
        logits = torch.randn(2, 8, 4, 8) * 100  # Large logits
        soft_cap = 50.0
        
        capped = attention._soft_cap_attention_logits(logits, soft_cap)
        
        # Should be bounded by soft_cap
        self.assertTrue(torch.all(torch.abs(capped) <= soft_cap))
        
        # Test with small logits (should be mostly unchanged)
        small_logits = torch.randn(2, 8, 4, 8) * 0.1
        capped_small = attention._soft_cap_attention_logits(small_logits, soft_cap)
        self.assertTrue(torch.allclose(capped_small, small_logits, atol=1e-3))
    
    def test_dot_product_attention(self):
        """Test core dot product attention function."""
        batch_size, query_time, key_time, num_heads, units_per_head = 2, 6, 8, 4, 16
        
        queries = torch.randn(batch_size, query_time, num_heads, units_per_head)
        keys = torch.randn(batch_size, key_time, num_heads, units_per_head)
        values = torch.randn(batch_size, key_time, num_heads, units_per_head)
        
        context, probs = attention.dot_product_attention(queries, keys, values)
        
        # Check output shapes
        self.assertEqual(context.shape, (batch_size, query_time, num_heads, units_per_head))
        self.assertEqual(probs.shape, (batch_size, query_time, num_heads, key_time))
        
        # Check that probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums)))
        
        # Test with mask
        mask = torch.ones(batch_size, key_time, dtype=torch.bool)
        mask[:, 4:] = False  # Mask last 4 timesteps
        
        context_masked, probs_masked = attention.dot_product_attention(
            queries, keys, values, mask=mask
        )
        
        # Check that masked positions have zero probability
        self.assertTrue(torch.allclose(probs_masked[:, :, :, 4:], torch.zeros_like(probs_masked[:, :, :, 4:])))


class AttentionInputProjectionTest(SequenceLayerTest):
    """Test attention input projection helper."""
    
    def test_combined_projection(self):
        """Test combined QKV projection."""
        x = self.random_sequence(2, 8, 16)
        
        projection = attention.AttentionInputProjection(
            input_size=16,
            num_query_heads=4,
            num_kv_heads=4,
            units_per_head=8,
            projection_config=attention.CombinedQueryKeyValueProjection(),
        )
        
        queries, keys, values = projection.get_qkv(x)
        
        self.assertEqual(queries.shape, (2, 8, 4, 8))
        self.assertEqual(keys.shape, (2, 8, 4, 8))
        self.assertEqual(values.shape, (2, 8, 4, 8))
    
    def test_separate_projection(self):
        """Test separate QKV projections."""
        x = self.random_sequence(2, 8, 16)
        
        projection = attention.AttentionInputProjection(
            input_size=16,
            num_query_heads=4,
            num_kv_heads=4,
            units_per_head=8,
            projection_config=attention.SeparateQueryKeyValueProjection(),
        )
        
        queries, keys, values = projection.get_qkv(x)
        
        self.assertEqual(queries.shape, (2, 8, 4, 8))
        self.assertEqual(keys.shape, (2, 8, 4, 8))
        self.assertEqual(values.shape, (2, 8, 4, 8))
    
    def test_query_and_kv_projection(self):
        """Test query and combined KV projection."""
        x = self.random_sequence(2, 8, 16)
        
        projection = attention.AttentionInputProjection(
            input_size=16,
            num_query_heads=4,
            num_kv_heads=4,
            units_per_head=8,
            projection_config=attention.QueryAndKeyValueProjection(),
        )
        
        queries, keys, values = projection.get_qkv(x)
        
        self.assertEqual(queries.shape, (2, 8, 4, 8))
        self.assertEqual(keys.shape, (2, 8, 4, 8))
        self.assertEqual(values.shape, (2, 8, 4, 8))
    
    def test_shared_kv_projection(self):
        """Test shared KV projection."""
        x = self.random_sequence(2, 8, 16)
        
        projection = attention.AttentionInputProjection(
            input_size=16,
            num_query_heads=4,
            num_kv_heads=4,
            units_per_head=8,
            projection_config=attention.QueryAndSharedKeyValueProjection(),
        )
        
        queries, keys, values = projection.get_qkv(x)
        
        self.assertEqual(queries.shape, (2, 8, 4, 8))
        self.assertEqual(keys.shape, (2, 8, 4, 8))
        self.assertEqual(values.shape, (2, 8, 4, 8))
        
        # Keys and values should be identical
        self.assertTrue(torch.equal(keys.values, values.values))
    
    def test_gqa_projection(self):
        """Test grouped query attention projection."""
        x = self.random_sequence(2, 8, 16)
        
        projection = attention.AttentionInputProjection(
            input_size=16,
            num_query_heads=8,
            num_kv_heads=4,  # Grouped query attention
            units_per_head=8,
            projection_config=attention.SeparateQueryKeyValueProjection(),
        )
        
        queries, keys, values = projection.get_qkv(x)
        
        self.assertEqual(queries.shape, (2, 8, 8, 8))
        self.assertEqual(keys.shape, (2, 8, 4, 8))
        self.assertEqual(values.shape, (2, 8, 4, 8))


if __name__ == '__main__':
    unittest.main() 