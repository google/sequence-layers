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
"""Tests for PyTorch combinator layers."""

import fractions
import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import combinators
from sequence_layers.pytorch import simple
from sequence_layers.pytorch import dense


class SequentialTest(SequenceLayerTest):
    """Test Sequential combinator."""
    
    def test_sequential_basic(self):
        """Test basic Sequential functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Create a sequence of layers
        layers = [
            dense.Dense(8),
            simple.Relu(),
            dense.Dense(4)
        ]
        
        layer = combinators.Sequential(layers)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Test that it produces the same result as applying layers individually
        y_manual = x
        for l in layers:
            y_manual = l.layer(y_manual, training=False)
        
        self.assertSequencesClose(y, y_manual)
    
    def test_sequential_empty(self):
        """Test Sequential with empty layer list."""
        x = self.random_sequence(2, 8, 4)
        
        layer = combinators.Sequential([])
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertSequencesClose(x, y)
    
    def test_sequential_different_ratios(self):
        """Test Sequential with layers having different output ratios."""
        x = self.random_sequence(2, 8, 4)
        
        from sequence_layers.pytorch.pooling import AveragePooling1D
        
        # Mix layers with different output ratios
        layers = [
            dense.Dense(8),
            AveragePooling1D(pool_size=2, stride=2)  # output_ratio = 1/2
        ]
        
        layer = combinators.Sequential(layers)
        y = self.verify_contract(layer, x, training=False)
        
        # Should have combined output ratio
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertEqual(y.shape[1], 4)  # 8 / 2
    
    def test_sequential_validation(self):
        """Test Sequential parameter validation."""
        # Test invalid layer type
        with self.assertRaises(ValueError):
            combinators.Sequential([dense.Dense(8), torch.nn.Linear(4, 4)])


class ParallelTest(SequenceLayerTest):
    """Test Parallel combinator."""
    
    def test_parallel_stack(self):
        """Test Parallel with STACK combination."""
        x = self.random_sequence(2, 8, 4)
        
        layers = [
            dense.Dense(8),
            dense.Dense(8),
            dense.Dense(8)
        ]
        
        layer = combinators.Parallel(layers, combination=combinators.CombinationMode.STACK)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3, 8))  # 3 layers stacked
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_parallel_add(self):
        """Test Parallel with ADD combination."""
        x = self.random_sequence(2, 8, 4)
        
        layers = [
            dense.Dense(8),
            dense.Dense(8),
            dense.Dense(8)
        ]
        
        layer = combinators.Parallel(layers, combination=combinators.CombinationMode.ADD)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_parallel_mean(self):
        """Test Parallel with MEAN combination."""
        x = self.random_sequence(2, 8, 4)
        
        layers = [
            dense.Dense(8),
            dense.Dense(8),
            dense.Dense(8)
        ]
        
        layer = combinators.Parallel(layers, combination=combinators.CombinationMode.MEAN)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_parallel_empty(self):
        """Test Parallel with empty layer list."""
        x = self.random_sequence(2, 8, 4)
        
        layer = combinators.Parallel([])
        y = layer.layer(x, training=False)  # Test layer method directly
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertSequencesClose(x, y)
    
    def test_parallel_single_layer(self):
        """Test Parallel with single layer."""
        x = self.random_sequence(2, 8, 4)
        
        layer = combinators.Parallel([dense.Dense(8)])
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (1, 8))  # Single layer stacked
    
    def test_parallel_validation(self):
        """Test Parallel parameter validation."""
        # Test mismatched output ratios
        from sequence_layers.pytorch.pooling import AveragePooling1D
        
        with self.assertRaises(ValueError):
            combinators.Parallel([
                dense.Dense(8),
                AveragePooling1D(pool_size=2, stride=2)  # Different output ratio
            ])
        
        # Test invalid layer type
        with self.assertRaises(ValueError):
            combinators.Parallel([dense.Dense(8), torch.nn.Linear(4, 4)])


class ResidualTest(SequenceLayerTest):
    """Test Residual combinator."""
    
    def test_residual_single_layer(self):
        """Test Residual with single layer."""
        x = self.random_sequence(2, 8, 4)
        
        layer = combinators.Residual(dense.Dense(4))  # Same size for residual
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Test that it's actually adding the residual
        main_output = layer.main_layers.layer(x, training=False)
        shortcut_output = layer.shortcut.layer(x, training=False)
        expected = Sequence(main_output.values + shortcut_output.values, main_output.mask)
        
        self.assertSequencesClose(y, expected)
    
    def test_residual_sequential_layers(self):
        """Test Residual with sequential layers."""
        x = self.random_sequence(2, 8, 4)
        
        layers = [
            dense.Dense(8),
            simple.Relu(),
            dense.Dense(4)  # Back to original size
        ]
        
        layer = combinators.Residual(layers)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_residual_custom_shortcut(self):
        """Test Residual with custom shortcut."""
        x = self.random_sequence(2, 8, 4)
        
        main_layers = [dense.Dense(8), simple.Relu()]
        shortcut = dense.Dense(8)  # Project to same size
        
        layer = combinators.Residual(main_layers, shortcut)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_residual_validation(self):
        """Test Residual parameter validation."""
        # Test mismatched output ratios
        from sequence_layers.pytorch.pooling import AveragePooling1D
        
        with self.assertRaises(ValueError):
            combinators.Residual(
                dense.Dense(8),
                shortcut=AveragePooling1D(pool_size=2, stride=2)  # Different output ratio
            )


class RepeatTest(SequenceLayerTest):
    """Test Repeat combinator."""
    
    def test_repeat_basic(self):
        """Test basic Repeat functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Create a layer that preserves shape
        child_layer = dense.Dense(4)  # Same input/output size
        
        layer = combinators.Repeat(child_layer, num_repeats=3)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Test that it's equivalent to applying the layer 3 times
        y_manual = x
        for _ in range(3):
            y_manual = child_layer.layer(y_manual, training=False)
        
        self.assertSequencesClose(y, y_manual)
    
    def test_repeat_single(self):
        """Test Repeat with single repetition."""
        x = self.random_sequence(2, 8, 4)
        
        child_layer = dense.Dense(4)
        layer = combinators.Repeat(child_layer, num_repeats=1)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        
        # Should be equivalent to applying the layer once
        y_direct = child_layer.layer(x, training=False)
        self.assertSequencesClose(y, y_direct)
    
    def test_repeat_complex_layer(self):
        """Test Repeat with complex layer."""
        x = self.random_sequence(2, 8, 4)
        
        # Create a more complex layer that preserves shape
        child_layer = combinators.Sequential([
            dense.Dense(8),
            simple.Relu(),
            dense.Dense(4)
        ])
        
        layer = combinators.Repeat(child_layer, num_repeats=2)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_repeat_validation(self):
        """Test Repeat parameter validation."""
        # Test invalid num_repeats
        with self.assertRaises(ValueError):
            combinators.Repeat(dense.Dense(4), num_repeats=0)
        
        with self.assertRaises(ValueError):
            combinators.Repeat(dense.Dense(4), num_repeats=-1)
        
        # Test layer with output_ratio != 1
        from sequence_layers.pytorch.pooling import AveragePooling1D
        
        with self.assertRaises(ValueError):
            combinators.Repeat(AveragePooling1D(pool_size=2, stride=2), num_repeats=2)


class CombinatorPropertiesTest(SequenceLayerTest):
    """Test combinator layer properties."""
    
    def test_sequential_properties(self):
        """Test Sequential layer properties."""
        layers = [
            dense.Dense(8),
            simple.Relu(),
            dense.Dense(4)
        ]
        
        layer = combinators.Sequential(layers)
        
        # Test block size (should be LCM of all layer block sizes)
        self.assertEqual(layer.block_size, 1)
        
        # Test output ratio (should be product of all layer output ratios)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        
        # Test supports_step (should be True if all layers support it)
        self.assertTrue(layer.supports_step)
        
        # Test output shape computation
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_parallel_properties(self):
        """Test Parallel layer properties."""
        layers = [
            dense.Dense(8),
            dense.Dense(8),
            dense.Dense(8)
        ]
        
        layer = combinators.Parallel(layers, combination=combinators.CombinationMode.STACK)
        
        # Test block size
        self.assertEqual(layer.block_size, 1)
        
        # Test output ratio
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        
        # Test supports_step
        self.assertTrue(layer.supports_step)
        
        # Test output shape for STACK
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (3, 8))  # 3 layers stacked
    
    def test_residual_properties(self):
        """Test Residual layer properties."""
        layer = combinators.Residual(dense.Dense(4))
        
        # Test block size
        self.assertEqual(layer.block_size, 1)
        
        # Test output ratio
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        
        # Test supports_step
        self.assertTrue(layer.supports_step)
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
    
    def test_repeat_properties(self):
        """Test Repeat layer properties."""
        child_layer = dense.Dense(4)
        layer = combinators.Repeat(child_layer, num_repeats=5)
        
        # Test that properties are inherited from child layer
        self.assertEqual(layer.block_size, child_layer.block_size)
        self.assertEqual(layer.output_ratio, child_layer.output_ratio)
        self.assertEqual(layer.supports_step, child_layer.supports_step)
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))


class CombinatorParameterTest(SequenceLayerTest):
    """Test combinator layer parameters."""
    
    def test_sequential_parameters(self):
        """Test Sequential layer parameter initialization."""
        layers = [
            dense.Dense(8),
            simple.Relu(),
            dense.Dense(4)
        ]
        
        layer = combinators.Sequential(layers)
        
        # Check that all layers are stored correctly
        self.assertEqual(len(layer.layers), 3)
        self.assertIsInstance(layer.layers[0], dense.Dense)
        self.assertIsInstance(layer.layers[1], simple.Relu)
        self.assertIsInstance(layer.layers[2], dense.Dense)
    
    def test_parallel_parameters(self):
        """Test Parallel layer parameter initialization."""
        layers = [
            dense.Dense(8),
            dense.Dense(8),
            dense.Dense(8)
        ]
        
        layer = combinators.Parallel(layers, combination=combinators.CombinationMode.ADD)
        
        # Check parameters
        self.assertEqual(len(layer.layers), 3)
        self.assertEqual(layer.combination, combinators.CombinationMode.ADD)
        
        for i, l in enumerate(layer.layers):
            self.assertIsInstance(l, dense.Dense)
    
    def test_residual_parameters(self):
        """Test Residual layer parameter initialization."""
        main_layers = [dense.Dense(8), simple.Relu()]
        shortcut = dense.Dense(8)
        
        layer = combinators.Residual(main_layers, shortcut)
        
        # Check parameters
        self.assertIsInstance(layer.main_layers, combinators.Sequential)
        self.assertEqual(layer.shortcut, shortcut)
    
    def test_repeat_parameters(self):
        """Test Repeat layer parameter initialization."""
        child_layer = dense.Dense(4)
        layer = combinators.Repeat(child_layer, num_repeats=3)
        
        # Check parameters
        self.assertEqual(layer.child_layer, child_layer)
        self.assertEqual(layer.num_repeats, 3)


class CombinatorIntegrationTest(SequenceLayerTest):
    """Test combinator integration and complex compositions."""
    
    def test_nested_combinators(self):
        """Test nested combinator structures."""
        x = self.random_sequence(2, 8, 4)
        
        # Create a complex nested structure
        inner_sequential = combinators.Sequential([
            dense.Dense(8),
            simple.Relu()
        ])
        
        inner_parallel = combinators.Parallel([
            dense.Dense(4),
            dense.Dense(4)
        ], combination=combinators.CombinationMode.ADD)
        
        outer_sequential = combinators.Sequential([
            inner_sequential,
            inner_parallel
        ])
        
        layer = combinators.Residual(outer_sequential)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
    
    def test_transformer_block(self):
        """Test a transformer-like block using combinators."""
        x = self.random_sequence(2, 8, 4)
        
        # Create a simple transformer block pattern
        attention_block = combinators.Sequential([
            dense.Dense(4),  # Simplified attention
            simple.Relu()
        ])
        
        ffn_block = combinators.Sequential([
            dense.Dense(16),  # FFN expand
            simple.Relu(),
            dense.Dense(4)   # FFN contract
        ])
        
        # Transformer block: residual around attention + residual around FFN
        transformer_block = combinators.Sequential([
            combinators.Residual(attention_block),
            combinators.Residual(ffn_block)
        ])
        
        layer = combinators.Repeat(transformer_block, num_repeats=2)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)


if __name__ == '__main__':
    unittest.main() 