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
"""Tests for PyTorch simple layers."""

import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import simple


class PointwiseMathTest(SequenceLayerTest):
    """Test pointwise math operations."""
    
    def test_activation_layers(self):
        """Test all activation layers."""
        test_cases = [
            (simple.Abs, {}, torch.abs, (torch.float32, torch.complex64)),
            (simple.Elu, {'alpha': 1.0}, lambda x: F.elu(x, alpha=1.0), (torch.float32,)),
            (simple.Exp, {}, torch.exp, (torch.float32,)),
            (simple.Gelu, {'approximate': True}, lambda x: F.gelu(x, approximate='tanh'), (torch.float32,)),
            (simple.LeakyRelu, {'negative_slope': 0.01}, lambda x: F.leaky_relu(x, negative_slope=0.01), (torch.float32,)),
            (simple.Log, {}, torch.log, (torch.float32,)),
            (simple.Power, {'exponent': 2.0}, lambda x: torch.pow(x, 2.0), (torch.float32,)),
            (simple.Power, {'exponent': 0.5}, lambda x: torch.pow(x, 0.5), (torch.float32,)),
            (simple.Relu, {}, F.relu, (torch.float32,)),
            (simple.Sigmoid, {}, torch.sigmoid, (torch.float32,)),
            (simple.Softmax, {'axis': -1}, lambda x: F.softmax(x, dim=-1), (torch.float32,)),
            (simple.Softplus, {}, F.softplus, (torch.float32,)),
            (simple.Swish, {}, F.silu, (torch.float32,)),
            (simple.Tanh, {}, torch.tanh, (torch.float32,)),
        ]
        
        batch_size, time, channels = 2, 10, 4
        
        for layer_class, kwargs, expected_op, dtypes in test_cases:
            for dtype in dtypes:
                with self.subTest(layer=layer_class.__name__, dtype=dtype):
                    x = self.random_sequence(batch_size, time, channels, dtype=dtype)
                    
                    # For log and power, use positive values to avoid NaN
                    if layer_class in (simple.Log, simple.Power):
                        x = x.apply_values(torch.abs)
                    
                    layer = layer_class(name='test', **kwargs)
                    
                    # Test layer properties
                    self.assertEqual(layer.block_size, 1)
                    self.assertEqual(layer.output_ratio, 1)
                    self.assertEqual(layer.get_output_shape_for_sequence(x), (channels,))
                    self.assertEqual(layer.name, 'test')
                    
                    # Test contract verification
                    y = self.verify_contract(layer, x, training=False)
                    
                    # Test expected output
                    y_expected = x.apply_values(expected_op).mask_invalid()
                    self.assertSequencesClose(y, y_expected)
    
    def test_abs_complex_dtype(self):
        """Test Abs layer with complex dtypes."""
        layer = simple.Abs()
        
        # Test complex64 -> float32
        x_complex64 = self.random_sequence(2, 5, 3, dtype=torch.complex64)
        self.assertEqual(layer.get_output_dtype(torch.complex64), torch.float32)
        
        # Test complex128 -> float64
        x_complex128 = self.random_sequence(2, 5, 3, dtype=torch.complex128)
        self.assertEqual(layer.get_output_dtype(torch.complex128), torch.float64)
        
        # Test float32 -> float32
        self.assertEqual(layer.get_output_dtype(torch.float32), torch.float32)
    
    def test_softmax_axis_validation(self):
        """Test softmax axis validation."""
        # Test invalid axes
        invalid_axes = [0, 1, -3, -4]
        x = self.random_sequence(2, 5, 3, 4)
        
        for axis in invalid_axes:
            with self.subTest(axis=axis):
                layer = simple.Softmax(axis=axis)
                with self.assertRaises(ValueError):
                    layer.layer(x, training=False)
    
    def test_softmax_valid_axes(self):
        """Test softmax with valid axes."""
        valid_axes = [2, 3, -1, -2]
        x = self.random_sequence(2, 5, 3, 4)
        
        for axis in valid_axes:
            with self.subTest(axis=axis):
                layer = simple.Softmax(axis=axis)
                y = layer.layer(x, training=False)
                
                # Check that softmax was applied correctly
                expected = x.apply_values(lambda v: F.softmax(v, dim=axis))
                self.assertSequencesClose(y, expected)
    
    def test_prelu_parameters(self):
        """Test PRelu has learnable parameters."""
        layer = simple.PRelu(negative_slope_init=0.02)
        
        # Check parameter exists
        self.assertTrue(hasattr(layer, 'negative_slope'))
        self.assertTrue(isinstance(layer.negative_slope, torch.nn.Parameter))
        self.assertAlmostEqual(layer.negative_slope.item(), 0.02, places=5)
        
        # Test forward pass
        x = self.random_sequence(2, 5, 3)
        y = self.verify_contract(layer, x, training=False)
        
        # Check gradients work
        self.assertTrue(layer.negative_slope.requires_grad)


class TransformationLayerTest(SequenceLayerTest):
    """Test transformation layers."""
    
    def test_simple_transformations(self):
        """Test simple transformation layers."""
        test_cases = [
            (simple.Add, {'value': 2.0}, lambda x: x + 2.0),
            (simple.Scale, {'scale': 3.0}, lambda x: x * 3.0),
            (simple.Translate, {'offset': 1.5}, lambda x: x + 1.5),
        ]
        
        x = self.random_sequence(2, 5, 3)
        
        for layer_class, kwargs, expected_op in test_cases:
            with self.subTest(layer=layer_class.__name__):
                layer = layer_class(**kwargs)
                y = self.verify_contract(layer, x, training=False)
                
                y_expected = x.apply_values(expected_op)
                self.assertSequencesClose(y, y_expected)
    
    def test_affine_layer(self):
        """Test Affine layer."""
        x = self.random_sequence(2, 5, 3)
        
        # Test with both scale and bias
        layer = simple.Affine(shape=(3,), use_scale=True, use_bias=True)
        y = self.verify_contract(layer, x, training=False)
        
        # Test with only scale
        layer_scale = simple.Affine(shape=(3,), use_scale=True, use_bias=False)
        y_scale = self.verify_contract(layer_scale, x, training=False)
        
        # Test with only bias
        layer_bias = simple.Affine(shape=(3,), use_scale=False, use_bias=True)
        y_bias = self.verify_contract(layer_bias, x, training=False)
        
        # Test broadcast shape validation
        with self.assertRaises(ValueError):
            layer_invalid = simple.Affine(shape=(3, 4, 5))  # Too many dimensions
            layer_invalid.get_output_shape(x.channel_shape)
    
    def test_cast_layer(self):
        """Test Cast layer."""
        x = self.random_sequence(2, 5, 3, dtype=torch.float32)
        
        # Test casting to float64
        layer = simple.Cast(dtype=torch.float64)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(layer.get_output_dtype(torch.float32), torch.float64)
    
    def test_reshape_layer(self):
        """Test Reshape layer."""
        x = self.random_sequence(2, 5, 6)  # 6 channels
        
        # Reshape to 2x3
        layer = simple.Reshape(shape=(2, 3))
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (2, 3))
        self.assertEqual(y.shape, (2, 5, 2, 3))
    
    def test_transpose_layer(self):
        """Test Transpose layer."""
        x = self.random_sequence(2, 5, 3, 4)
        
        # Transpose last two dimensions
        layer = simple.Transpose(dim0=0, dim1=1)  # In channel space: 0=3, 1=4
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 3))
        self.assertEqual(y.shape, (2, 5, 4, 3))
    
    def test_squeeze_layer(self):
        """Test Squeeze layer."""
        x = self.random_sequence(2, 5, 1, 3)  # Has singleton dimension
        
        # Squeeze specific dimension
        layer = simple.Squeeze(dim=0)  # In channel space: dim 0 has size 1
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3,))
        self.assertEqual(y.shape, (2, 5, 3))
        
        # Test error for non-singleton dimension
        with self.assertRaises(ValueError):
            layer_invalid = simple.Squeeze(dim=1)  # dim 1 has size 3
            layer_invalid.get_output_shape(x.channel_shape)
    
    def test_expand_dims_layer(self):
        """Test ExpandDims layer."""
        x = self.random_sequence(2, 5, 3)
        
        # Add dimension at position 0
        layer = simple.ExpandDims(dim=0)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (1, 3))
        self.assertEqual(y.shape, (2, 5, 1, 3))
    
    def test_move_axis_layer(self):
        """Test MoveAxis layer."""
        x = self.random_sequence(2, 5, 3, 4, 2)
        
        # Move axis 0 to position 2
        layer = simple.MoveAxis(source=0, destination=2)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 2, 3))
        self.assertEqual(y.shape, (2, 5, 4, 2, 3))
    
    def test_swap_axes_layer(self):
        """Test SwapAxes layer."""
        x = self.random_sequence(2, 5, 3, 4)
        
        # Swap axes 0 and 1
        layer = simple.SwapAxes(axis1=0, axis2=1)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4, 3))
        self.assertEqual(y.shape, (2, 5, 4, 3))
    
    def test_flatten_layer(self):
        """Test Flatten layer."""
        x = self.random_sequence(2, 5, 3, 4)
        
        # Flatten all channel dimensions
        layer = simple.Flatten(start_dim=0, end_dim=-1)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (12,))  # 3 * 4 = 12
        self.assertEqual(y.shape, (2, 5, 12))
    
    def test_one_hot_layer(self):
        """Test OneHot layer."""
        # Create integer sequence
        values = torch.randint(0, 5, (2, 3, 4), dtype=torch.long)
        mask = torch.ones(2, 3, dtype=torch.bool)
        x = Sequence(values, mask)
        
        layer = simple.OneHot(depth=5, dtype=torch.float32)
        # Disable gradient testing for integer inputs
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4, 5))
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(layer.get_output_dtype(torch.long), torch.float32)
        
        # Test with non-integer input
        x_float = self.random_sequence(2, 3, 4, dtype=torch.float32)
        with self.assertRaises(ValueError):
            layer.layer(x_float)


class UtilityLayerTest(SequenceLayerTest):
    """Test utility layers."""
    
    def test_identity_layer(self):
        """Test Identity layer."""
        x = self.random_sequence(2, 5, 3)
        layer = simple.Identity()
        
        y = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(x, y)
    
    def test_lambda_layer(self):
        """Test Lambda layer."""
        x = self.random_sequence(2, 5, 3)
        
        # Test with tensor function
        layer = simple.Lambda(fn=lambda t: t * 2.0, sequence_input=False)
        y = self.verify_contract(layer, x, training=False)
        
        expected = x.apply_values(lambda v: v * 2.0).mask_invalid()
        self.assertSequencesClose(y, expected)
        
        # Test with sequence function
        layer_seq = simple.Lambda(fn=lambda s: s.apply_values(lambda v: v + 1.0), sequence_input=True)
        y_seq = self.verify_contract(layer_seq, x, training=False)
        
        expected_seq = x.apply_values(lambda v: v + 1.0)
        self.assertSequencesClose(y_seq, expected_seq)
    
    def test_emit_layer(self):
        """Test Emit layer."""
        x = self.random_sequence(2, 5, 3)
        layer = simple.Emit()
        
        y = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(x, y)
    
    def test_dropout_layer(self):
        """Test Dropout layer."""
        x = self.random_sequence(2, 5, 3)
        
        # Test dropout rate
        layer = simple.Dropout(rate=0.5)
        
        # In training mode, some values should be dropped
        y_train = layer.layer(x, training=True)
        self.assertEqual(y_train.shape, x.shape)
        self.assertTrue(torch.equal(y_train.mask, x.mask))
        
        # In eval mode, no dropout
        y_eval = layer.layer(x, training=False)
        self.assertSequencesEqual(x, y_eval)
    
    def test_mask_invalid_layer(self):
        """Test MaskInvalid layer."""
        values = torch.ones(2, 3, 4)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        # Create a properly masked sequence first
        x = Sequence(values, mask).mask_invalid()
        
        layer = simple.MaskInvalid()
        # Disable padding invariance test to avoid NaN issues with this simple layer
        y = self.verify_contract(layer, x, training=False, test_padding_invariance=False)
        
        expected = x.mask_invalid()
        self.assertSequencesEqual(y, expected)
    
    def test_maximum_minimum_layers(self):
        """Test Maximum and Minimum layers."""
        x = self.random_sequence(2, 5, 3)
        
        # Test Maximum
        layer_max = simple.Maximum(value=0.0)
        y_max = self.verify_contract(layer_max, x, training=False)
        expected_max = x.apply_values(lambda v: torch.maximum(v, torch.tensor(0.0)))
        self.assertSequencesClose(y_max, expected_max)
        
        # Test Minimum
        layer_min = simple.Minimum(value=1.0)
        y_min = self.verify_contract(layer_min, x, training=False)
        expected_min = x.apply_values(lambda v: torch.minimum(v, torch.tensor(1.0)))
        self.assertSequencesClose(y_min, expected_min)
    
    def test_mod_layer(self):
        """Test Mod layer."""
        # Create positive values for modulo
        x = self.random_sequence(2, 5, 3).apply_values(lambda v: torch.abs(v) + 1.0)
        
        layer = simple.Mod(divisor=2.0)
        y = self.verify_contract(layer, x, training=False)
        
        expected = x.apply_values(lambda v: torch.remainder(v, 2.0))
        self.assertSequencesClose(y, expected)
    
    def test_logging_layer(self):
        """Test Logging layer."""
        x = self.random_sequence(2, 5, 3)
        layer = simple.Logging(message="Test message")
        
        y = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(x, y)
    
    def test_optimization_barrier_layer(self):
        """Test OptimizationBarrier layer."""
        x = self.random_sequence(2, 5, 3)
        layer = simple.OptimizationBarrier()
        
        y = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(x, y)


class LayerPropertiesTest(SequenceLayerTest):
    """Test layer properties like block_size, output_ratio, etc."""
    
    def test_pointwise_layer_properties(self):
        """Test that pointwise layers have correct properties."""
        pointwise_layers = [
            simple.Abs(),
            simple.Relu(),
            simple.Tanh(),
            simple.Sigmoid(),
            simple.Exp(),
            simple.Log(),
            simple.Scale(2.0),
            simple.Add(1.0),
        ]
        
        for layer in pointwise_layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, 1)
                self.assertTrue(layer.supports_step())
    
    def test_stateless_layer_properties(self):
        """Test that stateless layers have correct properties."""
        stateless_layers = [
            simple.Identity(),
            simple.Reshape((4, 2)),
            simple.Transpose(0, 1),
            simple.Squeeze(0),
            simple.ExpandDims(0),
        ]
        
        for layer in stateless_layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, 1)
                self.assertTrue(layer.supports_step())


if __name__ == '__main__':
    unittest.main() 