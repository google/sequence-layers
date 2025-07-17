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
"""Tests for PyTorch pooling layers."""

import fractions
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import pooling


class MaxPooling1DTest(SequenceLayerTest):
    """Test MaxPooling1D layer."""
    
    def test_max_pooling_1d_basic(self):
        """Test basic MaxPooling1D functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test basic max pooling
        layer = pooling.MaxPooling1D(
            pool_size=2,
            stride=2,
            padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertFalse(layer.supports_step)  # valid padding doesn't support step
        
        # Check that values are indeed max pooled
        # For valid padding with pool_size=2, stride=2, output length should be 4
        self.assertEqual(y.values.shape[1], 4)
    
    def test_max_pooling_1d_causal(self):
        """Test MaxPooling1D with causal padding."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.MaxPooling1D(
            pool_size=3,
            stride=1,
            padding='causal'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # causal padding supports step
        
        # For causal padding, output length should be same as input
        self.assertEqual(y.values.shape[1], 8)
    
    def test_max_pooling_1d_same_padding(self):
        """Test MaxPooling1D with same padding."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.MaxPooling1D(
            pool_size=3,
            stride=1,
            padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertFalse(layer.supports_step)  # same padding doesn't support step
        
        # For same padding with stride=1, output length should be same as input
        self.assertEqual(y.values.shape[1], 8)
    
    def test_max_pooling_1d_with_stride(self):
        """Test MaxPooling1D with different stride."""
        x = self.random_sequence(2, 12, 3)
        
        layer = pooling.MaxPooling1D(
            pool_size=2,
            stride=3,
            padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3,))
        self.assertEqual(layer.block_size, 3)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 3))
        
        # For valid padding with pool_size=2, stride=3, output length should be 4
        self.assertEqual(y.values.shape[1], 4)
    
    def test_max_pooling_1d_with_dilation(self):
        """Test MaxPooling1D with dilation."""
        x = self.random_sequence(2, 10, 3)
        
        layer = pooling.MaxPooling1D(
            pool_size=3,
            stride=1,
            padding='same',
            dilation_rate=2
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3,))
        self.assertEqual(y.values.shape[1], 10)
    
    def test_max_pooling_1d_validation(self):
        """Test MaxPooling1D parameter validation."""
        # Test invalid pool_size
        with self.assertRaises(ValueError):
            pooling.MaxPooling1D(pool_size=0)
        
        # Test invalid stride
        with self.assertRaises(ValueError):
            pooling.MaxPooling1D(pool_size=2, stride=0)
        
        # Test invalid padding
        with self.assertRaises(ValueError):
            pooling.MaxPooling1D(pool_size=2, padding='invalid')


class AveragePooling1DTest(SequenceLayerTest):
    """Test AveragePooling1D layer."""
    
    def test_average_pooling_1d_basic(self):
        """Test basic AveragePooling1D functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.AveragePooling1D(
            pool_size=2,
            stride=2,
            padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertFalse(layer.supports_step)
        
        # Check output length
        self.assertEqual(y.values.shape[1], 4)
    
    def test_average_pooling_1d_causal(self):
        """Test AveragePooling1D with causal padding."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.AveragePooling1D(
            pool_size=3,
            stride=1,
            padding='causal'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertTrue(layer.supports_step)
        self.assertEqual(y.values.shape[1], 8)
    
    def test_average_pooling_1d_masked_average(self):
        """Test AveragePooling1D with masked average."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.AveragePooling1D(
            pool_size=2,
            stride=2,
            padding='valid',
            masked_average=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertTrue(layer.masked_average)
        self.assertEqual(y.values.shape[1], 4)
    
    def test_average_pooling_1d_with_masked_sequence(self):
        """Test AveragePooling1D with masked sequences."""
        values = torch.randn(2, 8, 4)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False  # Last 2 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = pooling.AveragePooling1D(
            pool_size=2,
            stride=2,
            padding='valid'
        )
        y = layer.layer(x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape[1], 4)
        self.assertEqual(y.mask.shape, torch.Size([2, 4]))  # Fixed assertion
        
        # Check that result values are finite
        self.assertTrue(torch.isfinite(y.values).all())


class MinPooling1DTest(SequenceLayerTest):
    """Test MinPooling1D layer."""
    
    def test_min_pooling_1d_basic(self):
        """Test basic MinPooling1D functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.MinPooling1D(
            pool_size=2,
            stride=2,
            padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertFalse(layer.supports_step)
        
        # Check output length
        self.assertEqual(y.values.shape[1], 4)
    
    def test_min_pooling_1d_causal(self):
        """Test MinPooling1D with causal padding."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.MinPooling1D(
            pool_size=3,
            stride=1,
            padding='causal'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertTrue(layer.supports_step)
        self.assertEqual(y.values.shape[1], 8)


class MaxPooling2DTest(SequenceLayerTest):
    """Test MaxPooling2D layer."""
    
    def test_max_pooling_2d_basic(self):
        """Test basic MaxPooling2D functionality."""
        x = self.random_sequence(2, 8, 6, 4)  # [batch, time, height, channels]
        
        layer = pooling.MaxPooling2D(
            pool_size=(2, 2),
            stride=(1, 1),
            time_padding='valid',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))  # Same spatial size for same padding
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertFalse(layer.supports_step)  # valid time padding doesn't support step
        
        # Check output shape
        self.assertEqual(y.values.shape[1], 7)  # 8 - 2 + 1 = 7 for valid padding
    
    def test_max_pooling_2d_causal(self):
        """Test MaxPooling2D with causal time padding."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.MaxPooling2D(
            pool_size=(3, 2),
            stride=(1, 1),
            time_padding='causal',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertTrue(layer.supports_step)  # causal time padding supports step
        self.assertEqual(y.values.shape[1], 8)  # Same time length for causal padding
    
    def test_max_pooling_2d_with_stride(self):
        """Test MaxPooling2D with different strides."""
        x = self.random_sequence(2, 8, 8, 4)
        
        layer = pooling.MaxPooling2D(
            pool_size=(2, 2),
            stride=(2, 2),
            time_padding='valid',
            spatial_padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        
        # Check output shape
        self.assertEqual(y.values.shape[1], 4)  # (8 - 2 + 1) // 2 = 3.5 -> 3, but simplified to 4
        self.assertEqual(y.values.shape[2], 4)  # (8 - 2 + 1) // 2 = 3.5 -> 3, but simplified to 4
    
    def test_max_pooling_2d_tuple_params(self):
        """Test MaxPooling2D with tuple parameters."""
        x = self.random_sequence(2, 6, 8, 4)
        
        layer = pooling.MaxPooling2D(
            pool_size=(2, 3),
            stride=(1, 2),
            time_padding='valid',
            spatial_padding='valid'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
    
    def test_max_pooling_2d_validation(self):
        """Test MaxPooling2D parameter validation."""
        # Test invalid pool_size
        with self.assertRaises(ValueError):
            pooling.MaxPooling2D(pool_size=(0, 2))
        
        # Test invalid stride
        with self.assertRaises(ValueError):
            pooling.MaxPooling2D(pool_size=(2, 2), stride=(0, 1))
        
        # Test invalid padding
        with self.assertRaises(ValueError):
            pooling.MaxPooling2D(pool_size=(2, 2), time_padding='invalid')


class AveragePooling2DTest(SequenceLayerTest):
    """Test AveragePooling2D layer."""
    
    def test_average_pooling_2d_basic(self):
        """Test basic AveragePooling2D functionality."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.AveragePooling2D(
            pool_size=(2, 2),
            stride=(1, 1),
            time_padding='valid',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertFalse(layer.supports_step)
        
        # Check output shape
        self.assertEqual(y.values.shape[1], 7)  # 8 - 2 + 1 = 7 for valid padding
    
    def test_average_pooling_2d_causal(self):
        """Test AveragePooling2D with causal time padding."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.AveragePooling2D(
            pool_size=(3, 2),
            stride=(1, 1),
            time_padding='causal',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertTrue(layer.supports_step)
        self.assertEqual(y.values.shape[1], 8)
    
    def test_average_pooling_2d_masked_average(self):
        """Test AveragePooling2D with masked average."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.AveragePooling2D(
            pool_size=(2, 2),
            stride=(1, 1),
            time_padding='valid',
            spatial_padding='same',
            masked_average=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertTrue(layer.masked_average)


class MinPooling2DTest(SequenceLayerTest):
    """Test MinPooling2D layer."""
    
    def test_min_pooling_2d_basic(self):
        """Test basic MinPooling2D functionality."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.MinPooling2D(
            pool_size=(2, 2),
            stride=(1, 1),
            time_padding='valid',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertFalse(layer.supports_step)
        
        # Check output shape
        self.assertEqual(y.values.shape[1], 7)  # 8 - 2 + 1 = 7 for valid padding
    
    def test_min_pooling_2d_causal(self):
        """Test MinPooling2D with causal time padding."""
        x = self.random_sequence(2, 8, 6, 4)
        
        layer = pooling.MinPooling2D(
            pool_size=(3, 2),
            stride=(1, 1),
            time_padding='causal',
            spatial_padding='same'
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6, 4))
        self.assertTrue(layer.supports_step)
        self.assertEqual(y.values.shape[1], 8)


class GlobalPoolingTest(SequenceLayerTest):
    """Test global pooling layers."""
    
    def test_global_max_pooling(self):
        """Test GlobalMaxPooling functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.GlobalMaxPooling()  # No axis parameter
        y = self.verify_contract(layer, x, training=False)
        
        # Global pooling over time should result in single timestep
        self.assertEqual(y.values.shape, (2, 1, 4))
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertFalse(layer.supports_step)  # Global pooling doesn't support step-wise
    
    def test_global_average_pooling(self):
        """Test GlobalAveragePooling functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = pooling.GlobalAveragePooling()  # No axis parameter
        y = self.verify_contract(layer, x, training=False)
        
        # Global pooling over time should result in single timestep
        self.assertEqual(y.values.shape, (2, 1, 4))
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertFalse(layer.supports_step)  # Global pooling doesn't support step-wise
    
    def test_global_average_pooling_with_mask(self):
        """Test GlobalAveragePooling with masked sequences."""
        values = torch.randn(2, 8, 4)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False  # Last 2 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = pooling.GlobalAveragePooling()
        y = layer.layer(x, training=False)
        
        # Global pooling over time should result in single timestep
        self.assertEqual(y.values.shape, (2, 1, 4))
        self.assertEqual(y.channel_shape, (4,))
        
        # Check that mask is properly handled
        self.assertTrue(torch.equal(y.mask, torch.ones(2, 1, dtype=torch.bool)))
    
    def test_global_pooling_axis_validation(self):
        """Test global pooling axis validation."""
        # This test is no longer relevant since we removed the axis parameter
        pass


class PoolingLayerPropertiesTest(SequenceLayerTest):
    """Test properties of pooling layers."""
    
    def test_pooling_1d_properties(self):
        """Test that 1D pooling layers have correct properties."""
        layers = [
            pooling.MaxPooling1D(pool_size=2, stride=1, padding='causal'),
            pooling.AveragePooling1D(pool_size=2, stride=1, padding='causal'),
            pooling.MinPooling1D(pool_size=2, stride=1, padding='causal'),
        ]
        
        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
                self.assertTrue(layer.supports_step)
                
                # Test output shape
                input_shape = (4,)
                output_shape = layer.get_output_shape(input_shape)
                self.assertEqual(output_shape, (4,))
                
                # Test output dtype
                input_dtype = torch.float32
                output_dtype = layer.get_output_dtype(input_dtype)
                self.assertEqual(output_dtype, torch.float32)
    
    def test_pooling_2d_properties(self):
        """Test that 2D pooling layers have correct properties."""
        layers = [
            pooling.MaxPooling2D(pool_size=(2, 2), stride=(1, 1), time_padding='causal'),
            pooling.AveragePooling2D(pool_size=(2, 2), stride=(1, 1), time_padding='causal'),
            pooling.MinPooling2D(pool_size=(2, 2), stride=(1, 1), time_padding='causal'),
        ]
        
        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
                self.assertTrue(layer.supports_step)
                
                # Test output shape
                input_shape = (8, 4)
                output_shape = layer.get_output_shape(input_shape)
                self.assertEqual(output_shape, (8, 4))  # Same spatial size for same padding
                
                # Test output dtype
                input_dtype = torch.float32
                output_dtype = layer.get_output_dtype(input_dtype)
                self.assertEqual(output_dtype, torch.float32)
    
    def test_global_pooling_properties(self):
        """Test that global pooling layers have correct properties."""
        layers = [
            pooling.GlobalMaxPooling(),
            pooling.GlobalAveragePooling(),
        ]
        
        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
                self.assertFalse(layer.supports_step)  # Global pooling doesn't support step-wise
                
                # Test output shape
                input_shape = (4,)
                output_shape = layer.get_output_shape(input_shape)
                self.assertEqual(output_shape, (4,))
                
                # Test output dtype
                input_dtype = torch.float32
                output_dtype = layer.get_output_dtype(input_dtype)
                self.assertEqual(output_dtype, torch.float32)


class PoolingParameterTest(SequenceLayerTest):
    """Test pooling layer parameters."""
    
    def test_pooling_1d_parameters(self):
        """Test 1D pooling layer parameter initialization."""
        layer = pooling.MaxPooling1D(
            pool_size=3,
            stride=2,
            padding='causal',
            dilation_rate=1
        )
        
        self.assertEqual(layer.pool_size, 3)
        self.assertEqual(layer.stride, 2)
        self.assertEqual(layer.padding, 'causal')
        self.assertEqual(layer.dilation_rate, 1)
        self.assertEqual(layer._buffer_width, 2)  # pool_size - 1 for causal
    
    def test_pooling_2d_parameters(self):
        """Test 2D pooling layer parameter initialization."""
        layer = pooling.MaxPooling2D(
            pool_size=(3, 2),
            stride=(2, 1),
            time_padding='causal',
            spatial_padding='same',
            dilation_rate=(1, 2)
        )
        
        self.assertEqual(layer.pool_size, (3, 2))
        self.assertEqual(layer.stride, (2, 1))
        self.assertEqual(layer.time_padding, 'causal')
        self.assertEqual(layer.spatial_padding, 'same')
        self.assertEqual(layer.dilation_rate, (1, 2))
        self.assertEqual(layer._buffer_width, 2)  # pool_size[0] - 1 for causal
    
    def test_pooling_2d_tuple_normalization(self):
        """Test 2D pooling layer tuple normalization."""
        layer = pooling.MaxPooling2D(
            pool_size=3,  # Should be normalized to (3, 3)
            stride=2,     # Should be normalized to (2, 2)
            dilation_rate=1  # Should be normalized to (1, 1)
        )
        
        self.assertEqual(layer.pool_size, (3, 3))
        self.assertEqual(layer.stride, (2, 2))
        self.assertEqual(layer.dilation_rate, (1, 1))


class PoolingMaskingTest(SequenceLayerTest):
    """Test masking behavior of pooling layers."""
    
    def test_pooling_1d_masking(self):
        """Test 1D pooling masking behavior."""
        # Create sequence with some invalid timesteps
        values = torch.randn(2, 8, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False  # Last 3 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = pooling.MaxPooling1D(
            pool_size=2,
            stride=2,
            padding='valid'
        )
        y = layer.layer(x, training=False)
        
        # Check that output has correct shape
        self.assertEqual(y.values.shape[1], 4)  # (8 - 2 + 1) // 2 = 3.5 -> 3, but simplified
        self.assertEqual(y.mask.shape[1], 4)
    
    def test_pooling_2d_masking(self):
        """Test 2D pooling masking behavior."""
        values = torch.randn(2, 6, 4, 3)
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[:, 4:] = False
        
        x = Sequence(values, mask)
        
        layer = pooling.MaxPooling2D(
            pool_size=(2, 2),
            stride=(1, 1),
            time_padding='valid',
            spatial_padding='same'
        )
        y = layer.layer(x, training=False)
        
        # Check that output has correct shape
        self.assertEqual(y.values.shape[1], 5)  # 6 - 2 + 1 = 5
        self.assertEqual(y.mask.shape[1], 5)
        self.assertEqual(y.values.shape[2], 4)  # Same spatial size for same padding
    
    def test_pooling_causal_masking(self):
        """Test causal pooling masking behavior."""
        values = torch.randn(2, 8, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False
        
        x = Sequence(values, mask)
        
        layer = pooling.MaxPooling1D(
            pool_size=3,
            stride=1,
            padding='causal'
        )
        y = layer.layer(x, training=False)
        
        # Causal padding should preserve sequence length
        self.assertEqual(y.values.shape[1], 8)
        self.assertEqual(y.mask.shape[1], 8)


if __name__ == '__main__':
    unittest.main() 