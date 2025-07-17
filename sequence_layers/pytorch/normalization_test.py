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
"""Tests for PyTorch normalization layers."""

import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import normalization


class LayerNormTest(SequenceLayerTest):
    """Test LayerNorm layer."""
    
    def test_layer_norm_basic(self):
        """Test basic LayerNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        # Test basic layer norm
        layer = normalization.LayerNorm(
            normalized_shape=[6],
            axis=-1,
            epsilon=1e-5,
            elementwise_affine=True,
            bias=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check normalization properties (approximately normalized)
        normalized_values = y.values
        mean = torch.mean(normalized_values, dim=-1, keepdim=True)
        var = torch.var(normalized_values, dim=-1, keepdim=True, unbiased=False)
        
        # Mean should be close to 0, variance close to 1
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-4))
    
    def test_layer_norm_no_affine(self):
        """Test LayerNorm without affine transformation."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.LayerNorm(
            normalized_shape=[4],
            axis=-1,
            elementwise_affine=False
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no parameters were created
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias_param)
    
    def test_layer_norm_multiple_axes(self):
        """Test LayerNorm with multiple normalization axes."""
        x = self.random_sequence(2, 6, 4, 3)  # [batch, time, height, channels]
        
        layer = normalization.LayerNorm(
            normalized_shape=[4, 3],
            axis=[-2, -1],
            epsilon=1e-5
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 3))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_layer_norm_validation(self):
        """Test LayerNorm parameter validation."""
        # Test with invalid axis
        x = self.random_sequence(2, 6, 4)
        layer = normalization.LayerNorm(normalized_shape=[4], axis=10)
        
        with self.assertRaises(ValueError):
            layer.layer(x, training=False)


class RMSNormTest(SequenceLayerTest):
    """Test RMSNorm layer."""
    
    def test_rms_norm_basic(self):
        """Test basic RMSNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.RMSNorm(
            normalized_shape=[6],
            axis=-1,
            epsilon=1e-6,
            elementwise_affine=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check RMS normalization properties
        normalized_values = y.values
        mean_square = torch.mean(normalized_values ** 2, dim=-1, keepdim=True)
        
        # RMS should be close to 1 (since weight initialized to 1)
        self.assertTrue(torch.allclose(mean_square, torch.ones_like(mean_square), atol=1e-4))
    
    def test_rms_norm_no_scale(self):
        """Test RMSNorm without scale parameter."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.RMSNorm(
            normalized_shape=[4],
            axis=-1,
            elementwise_affine=False
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no scale parameter was created
        self.assertIsNone(layer.weight)
    
    def test_rms_norm_different_epsilon(self):
        """Test RMSNorm with different epsilon values."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.RMSNorm(
            normalized_shape=[4],
            axis=-1,
            epsilon=1e-8
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)


class BatchNormTest(SequenceLayerTest):
    """Test BatchNorm layer."""
    
    def test_batch_norm_basic(self):
        """Test basic BatchNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.BatchNorm(
            num_features=6,
            axis=-1,
            epsilon=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )
        
        # First run in training mode to populate running stats
        layer.layer(x, training=True)
        
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that running stats were updated
        self.assertIsNotNone(layer.running_mean)
        self.assertIsNotNone(layer.running_var)
        self.assertTrue(layer.num_batches_tracked > 0)
    
    def test_batch_norm_eval_mode(self):
        """Test BatchNorm in evaluation mode."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.BatchNorm(
            num_features=6,
            axis=-1,
            track_running_stats=True
        )
        
        # First run in training mode to populate running stats
        layer.layer(x, training=True)
        
        # Then test in eval mode
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_batch_norm_no_affine(self):
        """Test BatchNorm without affine transformation."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.BatchNorm(
            num_features=4,
            axis=-1,
            affine=False
        )
        
        # First run in training mode to populate running stats
        layer.layer(x, training=True)
        
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no affine parameters were created
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)
    
    def test_batch_norm_no_tracking(self):
        """Test BatchNorm without tracking running statistics."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.BatchNorm(
            num_features=4,
            axis=-1,
            track_running_stats=False
        )
        
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no running stats were created
        self.assertIsNone(layer.running_mean)
        self.assertIsNone(layer.running_var)
        self.assertIsNone(layer.num_batches_tracked)
    
    def test_batch_norm_with_masked_sequence(self):
        """Test BatchNorm with masked sequences."""
        values = torch.randn(2, 8, 4)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False  # Last 2 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = normalization.BatchNorm(
            num_features=4,
            axis=-1
        )
        y = layer.layer(x, training=True)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        self.assertTrue(torch.equal(y.mask, x.mask))


class GroupNormTest(SequenceLayerTest):
    """Test GroupNorm layer."""
    
    def test_group_norm_basic(self):
        """Test basic GroupNorm functionality."""
        x = self.random_sequence(2, 8, 8)  # 8 channels
        
        layer = normalization.GroupNorm(
            num_groups=4,
            num_channels=8,
            epsilon=1e-5,
            affine=True
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_group_norm_single_group(self):
        """Test GroupNorm with single group (equivalent to LayerNorm)."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.GroupNorm(
            num_groups=1,
            num_channels=4,
            epsilon=1e-5,
            affine=True
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_group_norm_all_groups(self):
        """Test GroupNorm with all groups (equivalent to InstanceNorm)."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.GroupNorm(
            num_groups=4,
            num_channels=4,
            epsilon=1e-5,
            affine=True
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_group_norm_no_affine(self):
        """Test GroupNorm without affine transformation."""
        x = self.random_sequence(2, 6, 8)
        
        layer = normalization.GroupNorm(
            num_groups=2,
            num_channels=8,
            affine=False
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no affine parameters were created
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)
    
    def test_group_norm_validation(self):
        """Test GroupNorm parameter validation."""
        # Test with invalid group configuration
        with self.assertRaises(ValueError):
            normalization.GroupNorm(
                num_groups=3,
                num_channels=8  # 8 is not divisible by 3
            )


class InstanceNormTest(SequenceLayerTest):
    """Test InstanceNorm layer."""
    
    def test_instance_norm_basic(self):
        """Test basic InstanceNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.InstanceNorm(
            num_features=6,
            epsilon=1e-5,
            affine=True
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_instance_norm_no_affine(self):
        """Test InstanceNorm without affine transformation."""
        x = self.random_sequence(2, 6, 4)
        
        layer = normalization.InstanceNorm(
            num_features=4,
            affine=False
        )
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Check that no affine parameters were created
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)
    
    def test_instance_norm_with_masked_sequence(self):
        """Test InstanceNorm with masked sequences."""
        values = torch.randn(2, 8, 4)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False  # Last 2 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = normalization.InstanceNorm(
            num_features=4,
            epsilon=1e-5
        )
        y = layer.layer(x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)
        self.assertTrue(torch.equal(y.mask, x.mask))


class SequenceSpecificNormTest(SequenceLayerTest):
    """Test sequence-specific normalization layers."""
    
    def test_sequence_layer_norm(self):
        """Test SequenceLayerNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.SequenceLayerNorm(
            normalized_shape=[6],
            axis=-1,
            epsilon=1e-5
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_masked_batch_norm(self):
        """Test MaskedBatchNorm functionality."""
        x = self.random_sequence(2, 8, 6)
        
        layer = normalization.MaskedBatchNorm(
            num_features=6,
            axis=-1,
            epsilon=1e-5
        )
        
        # First run in training mode to populate running stats
        layer.layer(x, training=True)
        
        # Test in evaluation mode (step-wise training not supported)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Test with masked input
        values = torch.randn(2, 8, 6)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 6:] = False
        
        x_masked = Sequence(values, mask)
        y_masked = layer.layer(x_masked, training=True)
        
        self.assertEqual(y_masked.channel_shape, (6,))
        self.assertEqual(y_masked.values.shape, x_masked.values.shape)
        self.assertTrue(torch.equal(y_masked.mask, x_masked.mask))


class NormalizationLayerPropertiesTest(SequenceLayerTest):
    """Test properties of normalization layers."""
    
    def test_layer_norm_properties(self):
        """Test LayerNorm layer properties."""
        layer = normalization.LayerNorm(
            normalized_shape=[4],
            axis=-1,
            epsilon=1e-5
        )
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_rms_norm_properties(self):
        """Test RMSNorm layer properties."""
        layer = normalization.RMSNorm(
            normalized_shape=[4],
            axis=-1,
            epsilon=1e-6
        )
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_batch_norm_properties(self):
        """Test BatchNorm layer properties."""
        layer = normalization.BatchNorm(
            num_features=4,
            axis=-1,
            epsilon=1e-5
        )
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_group_norm_properties(self):
        """Test GroupNorm layer properties."""
        layer = normalization.GroupNorm(
            num_groups=2,
            num_channels=8,
            epsilon=1e-5
        )
        
        # Test output shape
        input_shape = (8,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (8,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_instance_norm_properties(self):
        """Test InstanceNorm layer properties."""
        layer = normalization.InstanceNorm(
            num_features=4,
            epsilon=1e-5
        )
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)


class NormalizationParameterTest(SequenceLayerTest):
    """Test normalization layer parameters."""
    
    def test_layer_norm_parameters(self):
        """Test LayerNorm parameter initialization."""
        layer = normalization.LayerNorm(
            normalized_shape=[4],
            elementwise_affine=True,
            bias=True
        )
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (4,))
        self.assertEqual(layer.bias_param.shape, (4,))
        
        # Check parameter initialization
        self.assertTrue(torch.allclose(layer.weight, torch.ones(4)))
        self.assertTrue(torch.allclose(layer.bias_param, torch.zeros(4)))
    
    def test_rms_norm_parameters(self):
        """Test RMSNorm parameter initialization."""
        layer = normalization.RMSNorm(
            normalized_shape=[4],
            elementwise_affine=True
        )
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (4,))
        
        # Check parameter initialization
        self.assertTrue(torch.allclose(layer.weight, torch.ones(4)))
    
    def test_batch_norm_parameters(self):
        """Test BatchNorm parameter initialization."""
        layer = normalization.BatchNorm(
            num_features=4,
            affine=True,
            track_running_stats=True
        )
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (4,))
        self.assertEqual(layer.bias.shape, (4,))
        self.assertEqual(layer.running_mean.shape, (4,))
        self.assertEqual(layer.running_var.shape, (4,))
        
        # Check parameter initialization
        self.assertTrue(torch.allclose(layer.weight, torch.ones(4)))
        self.assertTrue(torch.allclose(layer.bias, torch.zeros(4)))
        self.assertTrue(torch.allclose(layer.running_mean, torch.zeros(4)))
        self.assertTrue(torch.allclose(layer.running_var, torch.ones(4)))
    
    def test_group_norm_parameters(self):
        """Test GroupNorm parameter initialization."""
        layer = normalization.GroupNorm(
            num_groups=2,
            num_channels=8,
            affine=True
        )
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (8,))
        self.assertEqual(layer.bias.shape, (8,))
        
        # Check parameter initialization
        self.assertTrue(torch.allclose(layer.weight, torch.ones(8)))
        self.assertTrue(torch.allclose(layer.bias, torch.zeros(8)))
    
    def test_instance_norm_parameters(self):
        """Test InstanceNorm parameter initialization."""
        layer = normalization.InstanceNorm(
            num_features=4,
            affine=True
        )
        
        # Check parameter shapes
        self.assertEqual(layer.weight.shape, (4,))
        self.assertEqual(layer.bias.shape, (4,))
        
        # Check parameter initialization
        self.assertTrue(torch.allclose(layer.weight, torch.ones(4)))
        self.assertTrue(torch.allclose(layer.bias, torch.zeros(4)))


if __name__ == '__main__':
    unittest.main() 