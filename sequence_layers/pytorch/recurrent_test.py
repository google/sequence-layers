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
"""Tests for PyTorch recurrent layers."""

import fractions
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import recurrent


class LSTMTest(SequenceLayerTest):
    """Test LSTM layer."""
    
    def test_lstm_basic(self):
        """Test basic LSTM layer functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test basic LSTM
        layer = recurrent.LSTM(
            input_size=4, hidden_size=6, use_bias=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Check parameter shapes
        self.assertEqual(layer.input_weights.shape, (4, 24))  # 4 * 6 = 24
        self.assertEqual(layer.recurrent_weights.shape, (6, 24))
        self.assertEqual(layer.bias.shape, (24,))
        
        # Check forget gate bias is initialized to 1
        forget_bias = layer.bias[6:12]  # Second quarter
        self.assertTrue(torch.allclose(forget_bias, torch.ones(6)))
    
    def test_lstm_no_bias(self):
        """Test LSTM without bias."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.LSTM(
            input_size=3, hidden_size=4, use_bias=False
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertIsNone(layer.bias)
    
    def test_lstm_custom_activations(self):
        """Test LSTM with custom activation functions."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.LSTM(
            input_size=3, hidden_size=4, 
            activation=F.relu,
            recurrent_activation=F.sigmoid
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.activation, F.relu)
        self.assertEqual(layer.recurrent_activation, F.sigmoid)
    
    def test_lstm_state_management(self):
        """Test LSTM state initialization and management."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.LSTM(input_size=3, hidden_size=4)
        
        # Test initial state
        state = layer.get_initial_state(2, x.channel_spec, training=False)
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)  # (c, h)
        
        c, h = state
        self.assertEqual(c.shape, (2, 1, 4))
        self.assertEqual(h.shape, (2, 1, 4))
        self.assertTrue(torch.allclose(c, torch.zeros_like(c)))
        self.assertTrue(torch.allclose(h, torch.zeros_like(h)))
    
    def test_lstm_single_timestep(self):
        """Test LSTM with single timestep input."""
        x = self.random_sequence(2, 1, 3)
        
        layer = recurrent.LSTM(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, (2, 1, 4))
    
    def test_lstm_masking(self):
        """Test LSTM with masked inputs."""
        # Create sequence with some invalid timesteps
        values = torch.randn(2, 8, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False  # Last 3 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = recurrent.LSTM(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.mask.shape, (2, 8))
    
    def test_lstm_validation(self):
        """Test LSTM parameter validation."""
        # Test rank validation
        x_rank2 = self.random_sequence(2, 6)  # No channel dimension
        layer = recurrent.LSTM(input_size=3, hidden_size=4)
        
        with self.assertRaises(ValueError):
            layer.layer(x_rank2, training=False)


class GRUTest(SequenceLayerTest):
    """Test GRU layer."""
    
    def test_gru_basic(self):
        """Test basic GRU layer functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test basic GRU
        layer = recurrent.GRU(
            input_size=4, hidden_size=6, use_bias=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Check parameter shapes
        self.assertEqual(layer.input_weights.shape, (4, 18))  # 3 * 6 = 18
        self.assertEqual(layer.recurrent_weights.shape, (6, 18))
        self.assertEqual(layer.bias.shape, (18,))
    
    def test_gru_no_bias(self):
        """Test GRU without bias."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.GRU(
            input_size=3, hidden_size=4, use_bias=False
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertIsNone(layer.bias)
    
    def test_gru_custom_activations(self):
        """Test GRU with custom activation functions."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.GRU(
            input_size=3, hidden_size=4, 
            activation=F.relu,
            recurrent_activation=F.sigmoid
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.activation, F.relu)
        self.assertEqual(layer.recurrent_activation, F.sigmoid)
    
    def test_gru_state_management(self):
        """Test GRU state initialization and management."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.GRU(input_size=3, hidden_size=4)
        
        # Test initial state
        state = layer.get_initial_state(2, x.channel_spec, training=False)
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape, (2, 1, 4))
        self.assertTrue(torch.allclose(state, torch.zeros_like(state)))
    
    def test_gru_single_timestep(self):
        """Test GRU with single timestep input."""
        x = self.random_sequence(2, 1, 3)
        
        layer = recurrent.GRU(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, (2, 1, 4))
    
    def test_gru_masking(self):
        """Test GRU with masked inputs."""
        # Create sequence with some invalid timesteps
        values = torch.randn(2, 8, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False  # Last 3 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = recurrent.GRU(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.mask.shape, (2, 8))


class VanillaRNNTest(SequenceLayerTest):
    """Test VanillaRNN layer."""
    
    def test_vanilla_rnn_basic(self):
        """Test basic VanillaRNN layer functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test basic VanillaRNN
        layer = recurrent.VanillaRNN(
            input_size=4, hidden_size=6, use_bias=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Check parameter shapes
        self.assertEqual(layer.input_weights.shape, (4, 6))
        self.assertEqual(layer.recurrent_weights.shape, (6, 6))
        self.assertEqual(layer.bias.shape, (6,))
    
    def test_vanilla_rnn_no_bias(self):
        """Test VanillaRNN without bias."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.VanillaRNN(
            input_size=3, hidden_size=4, use_bias=False
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertIsNone(layer.bias)
    
    def test_vanilla_rnn_custom_activation(self):
        """Test VanillaRNN with custom activation function."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.VanillaRNN(
            input_size=3, hidden_size=4, 
            activation=F.relu
        )
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.activation, F.relu)
    
    def test_vanilla_rnn_state_management(self):
        """Test VanillaRNN state initialization and management."""
        x = self.random_sequence(2, 6, 3)
        
        layer = recurrent.VanillaRNN(input_size=3, hidden_size=4)
        
        # Test initial state
        state = layer.get_initial_state(2, x.channel_spec, training=False)
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape, (2, 1, 4))
        self.assertTrue(torch.allclose(state, torch.zeros_like(state)))
    
    def test_vanilla_rnn_single_timestep(self):
        """Test VanillaRNN with single timestep input."""
        x = self.random_sequence(2, 1, 3)
        
        layer = recurrent.VanillaRNN(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, (2, 1, 4))
    
    def test_vanilla_rnn_masking(self):
        """Test VanillaRNN with masked inputs."""
        # Create sequence with some invalid timesteps
        values = torch.randn(2, 8, 3)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 5:] = False  # Last 3 timesteps are invalid
        
        x = Sequence(values, mask)
        
        layer = recurrent.VanillaRNN(input_size=3, hidden_size=4)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.mask.shape, (2, 8))


class RecurrentLayerPropertiesTest(SequenceLayerTest):
    """Test properties of recurrent layers."""
    
    def test_lstm_properties(self):
        """Test LSTM layer properties."""
        layer = recurrent.LSTM(input_size=4, hidden_size=6)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (6,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_gru_properties(self):
        """Test GRU layer properties."""
        layer = recurrent.GRU(input_size=3, hidden_size=5)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Test output shape
        input_shape = (3,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (5,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_vanilla_rnn_properties(self):
        """Test VanillaRNN layer properties."""
        layer = recurrent.VanillaRNN(input_size=5, hidden_size=3)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())
        
        # Test output shape
        input_shape = (5,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (3,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)


class RecurrentLayerInitializationTest(SequenceLayerTest):
    """Test initialization of recurrent layers."""
    
    def test_lstm_initialization(self):
        """Test LSTM weight initialization."""
        layer = recurrent.LSTM(input_size=4, hidden_size=6)
        
        # Check input weights are properly scaled
        input_std = layer.input_weights.std().item()
        expected_std = 1.0 / np.sqrt(4)  # 1 / sqrt(input_size)
        self.assertAlmostEqual(input_std, expected_std, places=1)
        
        # Check recurrent weights are orthogonal (approximately)
        recurrent_weights = layer.recurrent_weights
        # Each gate should be roughly orthogonal
        for i in range(4):
            gate_weights = recurrent_weights[:, i*6:(i+1)*6]
            # For orthogonal matrices, W @ W.T should be close to identity
            product = torch.mm(gate_weights, gate_weights.t())
            identity = torch.eye(6)
            # Allow some tolerance for numerical precision
            self.assertTrue(torch.allclose(product, identity, atol=1e-1))
    
    def test_gru_initialization(self):
        """Test GRU weight initialization."""
        layer = recurrent.GRU(input_size=3, hidden_size=4)
        
        # Check input weights are properly scaled
        input_std = layer.input_weights.std().item()
        expected_std = 1.0 / np.sqrt(3)  # 1 / sqrt(input_size)
        self.assertAlmostEqual(input_std, expected_std, places=1)
        
        # Check recurrent weights are orthogonal (approximately)
        recurrent_weights = layer.recurrent_weights
        # Each gate should be roughly orthogonal
        for i in range(3):
            gate_weights = recurrent_weights[:, i*4:(i+1)*4]
            product = torch.mm(gate_weights, gate_weights.t())
            identity = torch.eye(4)
            self.assertTrue(torch.allclose(product, identity, atol=1e-1))
    
    def test_vanilla_rnn_initialization(self):
        """Test VanillaRNN weight initialization."""
        layer = recurrent.VanillaRNN(input_size=5, hidden_size=3)
        
        # Check input weights are properly scaled
        input_std = layer.input_weights.std().item()
        expected_std = 1.0 / np.sqrt(5)  # 1 / sqrt(input_size)
        self.assertAlmostEqual(input_std, expected_std, places=1)
        
        # Check recurrent weights are orthogonal
        recurrent_weights = layer.recurrent_weights
        product = torch.mm(recurrent_weights, recurrent_weights.t())
        identity = torch.eye(3)
        self.assertTrue(torch.allclose(product, identity, atol=1e-1))


class RecurrentLayerContractTest(SequenceLayerTest):
    """Test contract compliance of recurrent layers."""
    
    def test_lstm_contract(self):
        """Test LSTM contract compliance."""
        x = self.random_sequence(2, 10, 4)
        layer = recurrent.LSTM(input_size=4, hidden_size=6)
        
        # Test with different sequence lengths
        for seq_len in [1, 5, 10]:
            x_short = self.random_sequence(2, seq_len, 4)
            self.verify_contract(layer, x_short, training=False)
    
    def test_gru_contract(self):
        """Test GRU contract compliance."""
        x = self.random_sequence(2, 10, 3)
        layer = recurrent.GRU(input_size=3, hidden_size=5)
        
        # Test with different sequence lengths
        for seq_len in [1, 5, 10]:
            x_short = self.random_sequence(2, seq_len, 3)
            self.verify_contract(layer, x_short, training=False)
    
    def test_vanilla_rnn_contract(self):
        """Test VanillaRNN contract compliance."""
        x = self.random_sequence(2, 10, 5)
        layer = recurrent.VanillaRNN(input_size=5, hidden_size=3)
        
        # Test with different sequence lengths
        for seq_len in [1, 5, 10]:
            x_short = self.random_sequence(2, seq_len, 5)
            self.verify_contract(layer, x_short, training=False)


if __name__ == '__main__':
    unittest.main() 