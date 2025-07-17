# Copyright 2024 Google LLC
#
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
"""Tests for PyTorch sequence types."""

import unittest
import torch
import numpy as np

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceArray,
    SequenceLayer,
    ShapeDType,
    ChannelSpec,
    Stateless,
    StatelessPointwise,
    StatelessPointwiseFunctor,
)
from sequence_layers.pytorch.utils_test import SequenceLayerTest


class SequenceTest(unittest.TestCase):
    """Test the Sequence class."""
    
    def test_basic_creation(self):
        """Test basic sequence creation."""
        values = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)
        
        seq = Sequence(values, mask)
        
        self.assertEqual(seq.shape, torch.Size([2, 3, 4]))
        self.assertEqual(seq.dtype, torch.float32)
        self.assertEqual(seq.channel_shape, torch.Size([4]))
        self.assertTrue(torch.equal(seq.values, values))
        self.assertTrue(torch.equal(seq.mask, mask))
    
    def test_creation_without_mask(self):
        """Test sequence creation without providing mask."""
        values = torch.randn(2, 3, 4)
        
        seq = Sequence(values)
        
        expected_mask = torch.ones(2, 3, dtype=torch.bool)
        self.assertTrue(torch.equal(seq.mask, expected_mask))
    
    def test_mask_invalid(self):
        """Test masking invalid timesteps."""
        values = torch.ones(2, 3, 4)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        
        seq = Sequence(values, mask)
        masked_seq = seq.mask_invalid()
        
        expected_values = torch.tensor([
            [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        ], dtype=torch.float32)
        
        self.assertTrue(torch.equal(masked_seq.values, expected_values))
        self.assertTrue(torch.equal(masked_seq.mask, mask))
    
    def test_concatenate(self):
        """Test sequence concatenation."""
        values1 = torch.randn(2, 3, 4)
        mask1 = torch.ones(2, 3, dtype=torch.bool)
        seq1 = Sequence(values1, mask1)
        
        values2 = torch.randn(2, 2, 4)
        mask2 = torch.ones(2, 2, dtype=torch.bool)
        seq2 = Sequence(values2, mask2)
        
        concatenated = seq1.concatenate(seq2)
        
        self.assertEqual(concatenated.shape, torch.Size([2, 5, 4]))
        self.assertTrue(torch.equal(concatenated.values[:, :3], values1))
        self.assertTrue(torch.equal(concatenated.values[:, 3:], values2))
        self.assertTrue(torch.equal(concatenated.mask[:, :3], mask1))
        self.assertTrue(torch.equal(concatenated.mask[:, 3:], mask2))
    
    def test_pad_time(self):
        """Test time padding."""
        values = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)
        seq = Sequence(values, mask)
        
        padded = seq.pad_time(1, 2, valid=False, pad_value=0.0)
        
        self.assertEqual(padded.shape, torch.Size([2, 6, 4]))
        self.assertTrue(torch.equal(padded.values[:, 1:4], values))
        self.assertTrue(torch.equal(padded.values[:, 0], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(padded.values[:, 4:], torch.zeros(2, 2, 4)))
        
        expected_mask = torch.tensor([[False, True, True, True, False, False],
                                    [False, True, True, True, False, False]])
        self.assertTrue(torch.equal(padded.mask, expected_mask))
    
    def test_slicing(self):
        """Test sequence slicing."""
        values = torch.randn(2, 5, 4)
        mask = torch.ones(2, 5, dtype=torch.bool)
        seq = Sequence(values, mask)
        
        # Test time slicing
        sliced = seq[:, 1:3]
        self.assertEqual(sliced.shape, torch.Size([2, 2, 4]))
        self.assertTrue(torch.equal(sliced.values, values[:, 1:3]))
        self.assertTrue(torch.equal(sliced.mask, mask[:, 1:3]))
        
        # Test batch slicing
        sliced_batch = seq[0]
        self.assertEqual(sliced_batch.shape, torch.Size([1, 5, 4]))  # Batch dimension preserved
        self.assertTrue(torch.equal(sliced_batch.values, values[0:1]))  # Need to include batch dimension
        self.assertTrue(torch.equal(sliced_batch.mask, mask[0:1]))  # Need to include batch dimension


class SequenceArrayTest(unittest.TestCase):
    """Test the SequenceArray class."""
    
    def test_basic_usage(self):
        """Test basic SequenceArray usage."""
        array = SequenceArray.new(torch.float32, size=3)
        
        # Write sequences
        seq1 = Sequence(torch.randn(2, 3, 4))
        seq2 = Sequence(torch.randn(2, 2, 4))
        seq3 = Sequence(torch.randn(2, 1, 4))
        
        array.write(0, seq1)
        array.write(1, seq2)
        array.write(2, seq3)
        
        # Concatenate
        result = array.concat()
        
        self.assertEqual(result.shape, torch.Size([2, 6, 4]))
        self.assertTrue(torch.equal(result.values[:, :3], seq1.values))
        self.assertTrue(torch.equal(result.values[:, 3:5], seq2.values))
        self.assertTrue(torch.equal(result.values[:, 5:], seq3.values))


class IdentityLayer(StatelessPointwise):
    """A simple identity layer for testing."""
    
    def layer(self, x, training=False, initial_state=None, constants=None):
        """Return the input unchanged."""
        return x


class ScaleLayer(StatelessPointwiseFunctor):
    """A simple scale layer for testing."""
    
    def __init__(self, scale=2.0, name=None):
        super().__init__(name=name)
        self.scale = scale
    
    def fn(self, values, mask):
        """Scale the values."""
        return values * self.scale, mask


class LayerContractTest(SequenceLayerTest):
    """Test sequence layer functionality."""
    
    def test_identity_layer(self):
        """Test the identity layer."""
        layer = IdentityLayer()
        x = self.random_sequence(2, 5, 4)
        
        # Test basic functionality
        y = layer.layer(x, training=False)
        self.assertSequencesEqual(x, y)
        
        # Test step functionality
        state = layer.get_initial_state(x.shape[0], x.channel_spec, training=False)
        y_step, new_state = layer.step(x, state, training=False)
        self.assertSequencesEqual(x, y_step)
        self.assertEqual(state, new_state)
        
        # Test contract verification
        y_verified = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(x, y_verified)
    
    def test_scale_layer(self):
        """Test the scale layer."""
        layer = ScaleLayer(scale=2.0)
        x = self.random_sequence(2, 5, 4)
        
        # Test basic functionality
        y = layer.layer(x, training=False)
        expected = x.apply_values(lambda v: v * 2.0)
        self.assertSequencesEqual(y, expected)
        
        # Test contract verification
        y_verified = self.verify_contract(layer, x, training=False)
        self.assertSequencesEqual(y, expected)
    
    def test_layer_properties(self):
        """Test layer properties."""
        layer = IdentityLayer()
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, 1)
        self.assertTrue(layer.supports_step())
        
        # Test output shape
        input_shape = torch.Size([4, 5])
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, input_dtype)


if __name__ == '__main__':
    unittest.main() 