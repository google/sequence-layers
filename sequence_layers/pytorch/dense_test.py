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
"""Tests for PyTorch dense layers."""

import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import dense


class BasicDenseTest(SequenceLayerTest):
    """Test basic dense layers."""
    
    def test_dense_layer(self):
        """Test Dense layer."""
        x = self.random_sequence(2, 5, 8)
        
        # Test without activation
        layer = dense.Dense(features=6, use_bias=True)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.shape, (2, 5, 6))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, 1)
        
        # Test with activation
        layer_with_activation = dense.Dense(features=4, activation=torch.relu)
        y_activated = self.verify_contract(layer_with_activation, x, training=False)
        
        self.assertEqual(y_activated.channel_shape, (4,))
        self.assertEqual(y_activated.shape, (2, 5, 4))
        
        # Test without bias
        layer_no_bias = dense.Dense(features=3, use_bias=False)
        y_no_bias = self.verify_contract(layer_no_bias, x, training=False)
        
        self.assertEqual(y_no_bias.channel_shape, (3,))
        self.assertEqual(y_no_bias.shape, (2, 5, 3))
    
    def test_dense_rank_validation(self):
        """Test Dense layer rank validation."""
        # Test with rank 2 input (should fail)
        x_rank2 = self.random_sequence(2, 5)  # No channel dimension
        layer = dense.Dense(features=4)
        
        with self.assertRaises(ValueError):
            layer.layer(x_rank2, training=False)
    
    def test_dense_shaped_layer(self):
        """Test DenseShaped layer."""
        x = self.random_sequence(2, 5, 3, 4)  # Input shape: (3, 4)
        
        # Test reshape to different output shape
        layer = dense.DenseShaped(output_shape=(2, 3))
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (2, 3))
        self.assertEqual(y.shape, (2, 5, 2, 3))
        
        # Test with activation
        layer_with_activation = dense.DenseShaped(output_shape=(5,), activation=F.relu)
        y_activated = self.verify_contract(layer_with_activation, x, training=False)
        
        self.assertEqual(y_activated.channel_shape, (5,))
        self.assertEqual(y_activated.shape, (2, 5, 5))
        
        # Test without bias
        layer_no_bias = dense.DenseShaped(output_shape=(8,), use_bias=False)
        y_no_bias = self.verify_contract(layer_no_bias, x, training=False)
        
        self.assertEqual(y_no_bias.channel_shape, (8,))
        self.assertEqual(y_no_bias.shape, (2, 5, 8))
    
    def test_einsum_dense_layer(self):
        """Test EinsumDense layer."""
        x = self.random_sequence(2, 5, 4)  # Input shape: (4,)
        
        # Test basic einsum transformation
        layer = dense.EinsumDense(
            equation="...a,ab->...b",
            output_shape=(6,),
            bias_axes="b"
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(y.shape, (2, 5, 6))
        
        # Test without bias
        layer_no_bias = dense.EinsumDense(
            equation="...a,ab->...b",
            output_shape=(3,),
            bias_axes="",
            use_bias=False
        )
        y_no_bias = self.verify_contract(layer_no_bias, x, training=False)
        
        self.assertEqual(y_no_bias.channel_shape, (3,))
        self.assertEqual(y_no_bias.shape, (2, 5, 3))
    
    def test_einsum_dense_validation(self):
        """Test EinsumDense validation."""
        # Test invalid equation format
        with self.assertRaises(ValueError):
            dense.EinsumDense(equation="abc", output_shape=(5,))
        
        # Test equation without ellipses
        with self.assertRaises(ValueError):
            dense.EinsumDense(equation="a,ab->b", output_shape=(5,))


class EmbeddingTest(SequenceLayerTest):
    """Test embedding layers."""
    
    def test_embedding_layer(self):
        """Test Embedding layer."""
        # Create integer sequence
        values = torch.randint(0, 10, (2, 5, 3), dtype=torch.long)
        mask = torch.ones(2, 5, dtype=torch.bool)
        x = Sequence(values, mask)
        
        layer = dense.Embedding(num_embeddings=10, embedding_dim=8)
        # Disable gradient testing for integer inputs
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (3, 8))
        self.assertEqual(y.shape, (2, 5, 3, 8))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, 1)
        
        # Test dtype validation
        self.assertEqual(layer.get_output_dtype(torch.long), layer.embedding.weight.dtype)
        
        # Test with invalid input dtype
        x_float = self.random_sequence(2, 5, 3, dtype=torch.float32)
        with self.assertRaises(ValueError):
            layer.layer(x_float, training=False)
    
    def test_embedding_attend(self):
        """Test Embedding attend method."""
        layer = dense.Embedding(num_embeddings=5, embedding_dim=4)
        
        # Create query with correct embedding dimension
        query = self.random_sequence(2, 3, 4)  # Last dim matches embedding_dim
        
        attended = layer.attend(query)
        
        self.assertEqual(attended.channel_shape, (5,))  # num_embeddings
        self.assertEqual(attended.shape, (2, 3, 5))
        
        # Test with invalid query shape
        query_invalid = self.random_sequence(2, 3)  # No channel dimension
        with self.assertRaises(ValueError):
            layer.attend(query_invalid)
    
    def test_embedding_transpose_layer(self):
        """Test EmbeddingTranspose layer."""
        embedding = dense.Embedding(num_embeddings=10, embedding_dim=6)
        layer = dense.EmbeddingTranspose(embedding=embedding, use_bias=True)
        
        # Create query with correct embedding dimension
        query = self.random_sequence(2, 4, 6)  # Last dim matches embedding_dim
        
        y = self.verify_contract(layer, query, training=False)
        
        self.assertEqual(y.channel_shape, (10,))  # num_embeddings
        self.assertEqual(y.shape, (2, 4, 10))
        
        # Test without bias
        layer_no_bias = dense.EmbeddingTranspose(embedding=embedding, use_bias=False)
        y_no_bias = self.verify_contract(layer_no_bias, query, training=False)
        
        self.assertEqual(y_no_bias.channel_shape, (10,))
        self.assertEqual(y_no_bias.shape, (2, 4, 10))
    
    def test_embedding_transpose_validation(self):
        """Test EmbeddingTranspose validation."""
        embedding = dense.Embedding(num_embeddings=5, embedding_dim=4)
        layer = dense.EmbeddingTranspose(embedding=embedding)
        
        # Test with wrong input dimension
        query_wrong_dim = self.random_sequence(2, 3, 3)  # Wrong last dim
        with self.assertRaises(ValueError):
            layer.get_output_shape(query_wrong_dim.channel_shape)
        
        # Test with no channel dimension
        with self.assertRaises(ValueError):
            layer.get_output_shape(())


class GatedUnitsTest(SequenceLayerTest):
    """Test gated unit layers."""
    
    def test_gated_unit_layer(self):
        """Test GatedUnit layer."""
        x = self.random_sequence(2, 5, 8)  # Even number of channels
        
        # Test basic gated unit
        layer = dense.GatedUnit(feature_activation=None, gate_activation=torch.sigmoid)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))  # Half of input channels
        self.assertEqual(y.shape, (2, 5, 4))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, 1)
        
        # Test with feature activation
        layer_with_feature = dense.GatedUnit(
            feature_activation=torch.tanh,
            gate_activation=torch.sigmoid
        )
        y_with_feature = self.verify_contract(layer_with_feature, x, training=False)
        
        self.assertEqual(y_with_feature.channel_shape, (4,))
        self.assertEqual(y_with_feature.shape, (2, 5, 4))
    
    def test_gated_unit_validation(self):
        """Test GatedUnit validation."""
        # Test with odd number of channels
        x_odd = self.random_sequence(2, 5, 7)  # Odd number of channels
        layer = dense.GatedUnit()
        
        with self.assertRaises(ValueError):
            layer.get_output_shape(x_odd.channel_shape)
        
        # Test with no channels
        with self.assertRaises(ValueError):
            layer.get_output_shape(())
    
    def test_gated_linear_unit(self):
        """Test GatedLinearUnit layer."""
        x = self.random_sequence(2, 5, 6)  # Even number of channels
        
        layer = dense.GatedLinearUnit()
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3,))  # Half of input channels
        self.assertEqual(y.shape, (2, 5, 3))
        
        # Verify that it uses no feature activation and sigmoid gate activation
        self.assertIsNone(layer.feature_activation)
        self.assertEqual(layer.gate_activation, torch.sigmoid)
    
    def test_gated_tanh_unit(self):
        """Test GatedTanhUnit layer."""
        x = self.random_sequence(2, 5, 10)  # Even number of channels
        
        layer = dense.GatedTanhUnit()
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (5,))  # Half of input channels
        self.assertEqual(y.shape, (2, 5, 5))
        
        # Verify that it uses tanh feature activation and sigmoid gate activation
        self.assertEqual(layer.feature_activation, torch.tanh)
        self.assertEqual(layer.gate_activation, torch.sigmoid)


class LayerPropertiesTest(SequenceLayerTest):
    """Test layer properties for dense layers."""
    
    def test_dense_layer_properties(self):
        """Test that dense layers have correct properties."""
        layers = [
            dense.Dense(features=8),
            dense.DenseShaped(output_shape=(4, 2)),
            dense.EinsumDense(equation="...a,ab->...b", output_shape=(5,)),
            dense.Embedding(num_embeddings=10, embedding_dim=6),
            dense.GatedUnit(),
            dense.GatedLinearUnit(),
            dense.GatedTanhUnit(),
        ]
        
        for layer in layers:
            with self.subTest(layer=layer.__class__.__name__):
                self.assertEqual(layer.block_size, 1)
                self.assertEqual(layer.output_ratio, 1)
                self.assertTrue(layer.supports_step())
    
    def test_embedding_transpose_properties(self):
        """Test EmbeddingTranspose properties."""
        embedding = dense.Embedding(num_embeddings=5, embedding_dim=4)
        layer = dense.EmbeddingTranspose(embedding=embedding)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, 1)
        self.assertTrue(layer.supports_step())


if __name__ == '__main__':
    unittest.main() 