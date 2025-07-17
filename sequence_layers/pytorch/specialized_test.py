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
"""Tests for PyTorch specialized layers."""

import fractions
import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence, ChannelSpec
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import position
from sequence_layers.pytorch import time_varying
from sequence_layers.pytorch import conditioning


class AddTimingSignalTest(SequenceLayerTest):
    """Test AddTimingSignal layer."""
    
    def test_add_timing_signal_basic(self):
        """Test basic AddTimingSignal functionality."""
        x = self.random_sequence(2, 8, 16)
        
        layer = position.AddTimingSignal()
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Check that timing signal was added (values should be different)
        self.assertFalse(torch.allclose(x.values, y.values))
    
    def test_add_timing_signal_trainable_scale(self):
        """Test AddTimingSignal with trainable scale."""
        x = self.random_sequence(2, 8, 16)
        
        layer = position.AddTimingSignal(trainable_scale=True)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
        self.assertTrue(hasattr(layer, 'scale'))
        self.assertTrue(isinstance(layer.scale, torch.nn.Parameter))
    
    def test_add_timing_signal_custom_timescales(self):
        """Test AddTimingSignal with custom timescales."""
        x = self.random_sequence(2, 8, 16)
        
        layer = position.AddTimingSignal(min_timescale=0.1, max_timescale=100.0)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
    
    def test_add_timing_signal_multidimensional(self):
        """Test AddTimingSignal with multidimensional channels."""
        x = self.random_sequence(2, 8, 4, 4)
        
        layer = position.AddTimingSignal()
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 4))
    
    def test_add_timing_signal_input_validation(self):
        """Test AddTimingSignal input validation."""
        x = Sequence(torch.randint(0, 10, (2, 8, 16)), torch.ones(2, 8, dtype=torch.bool))
        
        layer = position.AddTimingSignal()
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False)


class ApplyRotaryPositionalEncodingTest(SequenceLayerTest):
    """Test ApplyRotaryPositionalEncoding layer."""
    
    def test_rope_basic(self):
        """Test basic RoPE functionality."""
        x = self.random_sequence(2, 8, 16)
        
        layer = position.ApplyRotaryPositionalEncoding()
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Check that RoPE was applied (values should be different)
        self.assertFalse(torch.allclose(x.values, y.values))
    
    def test_rope_custom_wavelength(self):
        """Test RoPE with custom wavelength."""
        x = self.random_sequence(2, 8, 16)
        
        layer = position.ApplyRotaryPositionalEncoding(max_wavelength=1000.0)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
    
    def test_rope_different_axis(self):
        """Test RoPE with different axis."""
        x = self.random_sequence(2, 8, 4, 8)
        
        layer = position.ApplyRotaryPositionalEncoding(axis=-1)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 8))
    
    def test_rope_input_validation(self):
        """Test RoPE input validation."""
        # Test with odd dimension
        x = self.random_sequence(2, 8, 15)
        
        layer = position.ApplyRotaryPositionalEncoding()
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False)
        
        # Test with integer input
        x = Sequence(torch.randint(0, 10, (2, 8, 16)), torch.ones(2, 8, dtype=torch.bool))
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False)


class SequenceEmbeddingTest(SequenceLayerTest):
    """Test SequenceEmbedding layer."""
    
    def test_sequence_embedding_basic(self):
        """Test basic SequenceEmbedding functionality."""
        # Create integer input
        x = Sequence(
            torch.randint(0, 10, (2, 8)),
            torch.ones(2, 8, dtype=torch.bool)
        )
        
        layer = time_varying.SequenceEmbedding(
            dimension=16,
            num_embeddings=10,
            num_steps=8
        )
        
        # Disable gradient testing for integer inputs
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (16,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Check output dtype
        self.assertEqual(y.values.dtype, torch.float32)
    
    def test_sequence_embedding_step_dependency(self):
        """Test that SequenceEmbedding is step-dependent."""
        # Create input with same values at different time steps
        x = Sequence(
            torch.zeros(2, 8, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.bool)
        )
        
        layer = time_varying.SequenceEmbedding(
            dimension=16,
            num_embeddings=10,
            num_steps=8
        )
        
        y = layer.layer(x, training=False)
        
        # Embeddings should be different at different time steps
        self.assertFalse(torch.allclose(y.values[:, 0], y.values[:, 1]))
    
    def test_sequence_embedding_out_of_bounds(self):
        """Test SequenceEmbedding with out-of-bounds indices."""
        # Create input with values >= num_embeddings
        x = Sequence(
            torch.full((2, 8), 15, dtype=torch.long),
            torch.ones(2, 8, dtype=torch.bool)
        )
        
        layer = time_varying.SequenceEmbedding(
            dimension=16,
            num_embeddings=10,
            num_steps=8
        )
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False)


class SequenceDenseTest(SequenceLayerTest):
    """Test SequenceDense layer."""
    
    def test_sequence_dense_basic(self):
        """Test basic SequenceDense functionality."""
        x = self.random_sequence(2, 8, 16)
        
        layer = time_varying.SequenceDense(
            units=32,
            num_steps=8
        )
        
        # Disable padding test for time-varying layers
        y = self.verify_contract(layer, x, training=False, test_padding_invariance=False)
        
        self.assertEqual(y.channel_shape, (32,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_sequence_dense_with_activation(self):
        """Test SequenceDense with activation."""
        x = self.random_sequence(2, 8, 16)
        
        layer = time_varying.SequenceDense(
            units=32,
            num_steps=8,
            activation='relu'
        )
        
        # Disable padding test for time-varying layers
        y = self.verify_contract(layer, x, training=False, test_padding_invariance=False)
        
        self.assertEqual(y.channel_shape, (32,))
        
        # Check that activation was applied
        self.assertTrue(torch.all(y.values >= 0))
    
    def test_sequence_dense_no_bias(self):
        """Test SequenceDense without bias."""
        x = self.random_sequence(2, 8, 16)
        
        layer = time_varying.SequenceDense(
            units=32,
            num_steps=8,
            use_bias=False
        )
        
        # Disable padding test for time-varying layers
        y = self.verify_contract(layer, x, training=False, test_padding_invariance=False)
        
        self.assertEqual(y.channel_shape, (32,))
        self.assertIsNone(layer.bias)
    
    def test_sequence_dense_step_dependency(self):
        """Test that SequenceDense is step-dependent."""
        # Create input with same values at different time steps
        x = Sequence(
            torch.ones(2, 8, 16),
            torch.ones(2, 8, dtype=torch.bool)
        )
        
        layer = time_varying.SequenceDense(
            units=32,
            num_steps=8
        )
        
        y = layer.layer(x, training=False)
        
        # Outputs should be different at different time steps (different kernels)
        self.assertFalse(torch.allclose(y.values[:, 0], y.values[:, 1]))


class ConditioningTest(SequenceLayerTest):
    """Test Conditioning layer."""
    
    def test_conditioning_identity_add(self):
        """Test Conditioning with identity projection and add combination."""
        x = self.random_sequence(2, 8, 16)
        
        # Create conditioning tensor
        conditioning_tensor = torch.randn(2, 8, 16)
        constants = {'condition': conditioning_tensor}
        
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.IDENTITY,
            combination=conditioning.CombinationMode.ADD
        )
        
        y = layer.layer(x, training=False, constants=constants)
        
        self.assertEqual(y.channel_shape, (16,))
        
        # Check that conditioning was added
        expected = x.values + conditioning_tensor
        self.assertTrue(torch.allclose(y.values, expected))
    
    def test_conditioning_linear_concat(self):
        """Test Conditioning with linear projection and concat combination."""
        x = self.random_sequence(2, 8, 16)
        
        # Create conditioning tensor with different size
        conditioning_tensor = torch.randn(2, 8, 8)
        constants = {'condition': conditioning_tensor}
        
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.LINEAR,
            combination=conditioning.CombinationMode.CONCAT
        )
        
        y = layer.layer(x, training=False, constants=constants)
        
        self.assertEqual(y.channel_shape, (32,))  # 16 + 16 (projected)
    
    def test_conditioning_affine(self):
        """Test Conditioning with affine combination."""
        x = self.random_sequence(2, 8, 16)
        
        # Create conditioning tensor
        conditioning_tensor = torch.randn(2, 8, 8)
        constants = {'condition': conditioning_tensor}
        
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.LINEAR_AFFINE,
            combination=conditioning.CombinationMode.AFFINE
        )
        
        y = layer.layer(x, training=False, constants=constants)
        
        self.assertEqual(y.channel_shape, (16,))
    
    def test_conditioning_validation(self):
        """Test Conditioning parameter validation."""
        # Test invalid combination/projection pairing
        with self.assertRaises(ValueError):
            conditioning.Conditioning(
                conditioning_name='condition',
                projection=conditioning.ProjectionMode.IDENTITY,
                combination=conditioning.CombinationMode.AFFINE
            )
        
        # Test missing conditioning in constants
        x = self.random_sequence(2, 8, 16)
        
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.IDENTITY,
            combination=conditioning.CombinationMode.ADD
        )
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False, constants=None)
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False, constants={})
    
    def test_conditioning_sequence_input(self):
        """Test Conditioning with Sequence as conditioning input."""
        x = self.random_sequence(2, 8, 16)
        
        # Create conditioning as Sequence
        conditioning_seq = self.random_sequence(2, 8, 16)
        constants = {'condition': conditioning_seq}
        
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.IDENTITY,
            combination=conditioning.CombinationMode.ADD
        )
        
        y = layer.layer(x, training=False, constants=constants)
        
        self.assertEqual(y.channel_shape, (16,))
        
        # Check that conditioning was added
        expected = x.values + conditioning_seq.values
        self.assertTrue(torch.allclose(y.values, expected))


class SpecializedLayersPropertiesTest(SequenceLayerTest):
    """Test specialized layer properties."""
    
    def test_position_layers_properties(self):
        """Test position layer properties."""
        # AddTimingSignal
        layer = position.AddTimingSignal()
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # ApplyRotaryPositionalEncoding
        layer = position.ApplyRotaryPositionalEncoding()
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_time_varying_layers_properties(self):
        """Test time-varying layer properties."""
        # SequenceEmbedding
        layer = time_varying.SequenceEmbedding(
            dimension=16,
            num_embeddings=10,
            num_steps=8
        )
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # SequenceDense
        layer = time_varying.SequenceDense(
            units=32,
            num_steps=8
        )
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_conditioning_layers_properties(self):
        """Test conditioning layer properties."""
        layer = conditioning.Conditioning(
            conditioning_name='condition',
            projection=conditioning.ProjectionMode.IDENTITY,
            combination=conditioning.CombinationMode.ADD
        )
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)


class SpecializedLayersIntegrationTest(SequenceLayerTest):
    """Test specialized layer integration."""
    
    def test_position_with_combinators(self):
        """Test position layers with combinators."""
        from sequence_layers.pytorch.combinators import Sequential
        
        x = self.random_sequence(2, 8, 16)
        
        layer = Sequential([
            position.AddTimingSignal(),
            position.ApplyRotaryPositionalEncoding()
        ])
        
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (16,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_time_varying_with_embeddings(self):
        """Test time-varying layers with embeddings."""
        from sequence_layers.pytorch.combinators import Sequential
        
        # Create integer input
        x = Sequence(
            torch.randint(0, 10, (2, 8)),
            torch.ones(2, 8, dtype=torch.bool)
        )
        
        layer = Sequential([
            time_varying.SequenceEmbedding(
                dimension=16,
                num_embeddings=10,
                num_steps=8
            ),
            time_varying.SequenceDense(
                units=32,
                num_steps=8,
                activation='relu'
            )
        ])
        
        y = self.verify_contract(layer, x, training=False, test_gradients=False, test_padding_invariance=False)
        
        self.assertEqual(y.channel_shape, (32,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_conditioning_with_dense(self):
        """Test conditioning layers with dense layers."""
        from sequence_layers.pytorch.combinators import Sequential
        from sequence_layers.pytorch.dense import Dense
        
        x = self.random_sequence(2, 8, 16)
        
        # Create conditioning
        conditioning_tensor = torch.randn(2, 8, 8)
        constants = {'condition': conditioning_tensor}
        
        layer = Sequential([
            conditioning.Conditioning(
                conditioning_name='condition',
                projection=conditioning.ProjectionMode.LINEAR,
                combination=conditioning.CombinationMode.ADD
            ),
            Dense(32)
        ])
        
        y = layer.layer(x, training=False, constants=constants)
        
        self.assertEqual(y.channel_shape, (32,))


if __name__ == '__main__':
    unittest.main() 