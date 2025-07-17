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
"""Recurrent layers for PyTorch."""

import fractions
import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    'LSTM',
    'GRU',
    'VanillaRNN',
]


# =============================================================================
# Utility Functions
# =============================================================================

def unit_forget_bias_init(shape: Tuple[int, ...], dtype: torch.dtype) -> Tensor:
    """Initialize LSTM bias with forget gate bias set to 1.
    
    This is recommended in Jozefowicz et al. for better training stability.
    """
    if len(shape) != 1 or shape[0] % 4 != 0:
        raise ValueError(
            f'Expected single dimensional shape divisible by 4, got: {shape}'
        )
    
    units = shape[0] // 4
    bias = torch.zeros(shape, dtype=dtype)
    # Set forget gate bias to 1 (second quarter of the bias vector)
    bias[units:2*units] = 1.0
    return bias


# =============================================================================
# Base RNN Layer
# =============================================================================

class BaseRNN(SequenceLayer):
    """Base class for RNN layers with common functionality."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def supports_step(self) -> bool:
        """RNN layers support step-wise execution."""
        return True
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1, 1)


# =============================================================================
# LSTM Layer
# =============================================================================

class LSTM(BaseRNN):
    """Long Short-Term Memory (LSTM) layer."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 recurrent_activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.activation = activation or torch.tanh
        self.recurrent_activation = recurrent_activation or torch.sigmoid
        
        # Initialize input-to-hidden weights (4 * hidden_size for i, f, g, o gates)
        self.input_weights = nn.Parameter(
            torch.randn(input_size, 4 * hidden_size) / math.sqrt(input_size)
        )
        
        # Initialize hidden-to-hidden weights (recurrent weights)
        self.recurrent_weights = nn.Parameter(
            self._orthogonal_init(hidden_size, 4 * hidden_size)
        )
        
        # Initialize bias with forget gate bias set to 1
        if use_bias:
            self.bias = nn.Parameter(
                unit_forget_bias_init((4 * hidden_size,), torch.float32)
            )
        else:
            self.register_parameter('bias', None)
    
    def _orthogonal_init(self, rows: int, cols: int) -> Tensor:
        """Initialize weights using orthogonal initialization."""
        # Create random matrix
        tensor = torch.randn(rows, cols)
        
        # Apply orthogonal initialization to each gate separately
        gate_size = cols // 4
        for i in range(4):
            start_idx = i * gate_size
            end_idx = start_idx + gate_size
            gate_tensor = tensor[:, start_idx:end_idx]
            
            # Apply orthogonal initialization
            if rows >= gate_size:
                # QR decomposition for orthogonal initialization
                q, r = torch.linalg.qr(gate_tensor)
                # Ensure positive diagonal
                d = torch.diag(r)
                q *= d.sign().unsqueeze(0).expand_as(q)
                tensor[:, start_idx:end_idx] = q
            else:
                # For smaller matrices, use normalized random
                tensor[:, start_idx:end_idx] = F.normalize(gate_tensor, dim=0)
        
        return tensor
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'LSTM requires rank 3 input. Got: {input_shape}')
        return (self.hidden_size,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for LSTM (cell state, hidden state)."""
        device = next(self.parameters()).device
        dtype = torch.float32  # Always use float32 for state
        
        # Initial cell state (c) and hidden state (h)
        c = torch.zeros(batch_size, 1, self.hidden_size, dtype=dtype, device=device)
        h = torch.zeros(batch_size, 1, self.hidden_size, dtype=dtype, device=device)
        
        return (c, h)
    
    def _cell(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """LSTM cell computation for a single timestep."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f'Expected [batch, 1, features] input, got: {x.shape}')
        
        c_prev, h_prev = state
        
        # Compute input projection: [batch, 1, input_size] -> [batch, 1, 4*hidden_size]
        input_projection = torch.matmul(x, self.input_weights)
        
        # Compute recurrent projection: [batch, 1, hidden_size] -> [batch, 1, 4*hidden_size]
        recurrent_projection = torch.matmul(h_prev, self.recurrent_weights)
        
        # Combine projections
        combined = input_projection + recurrent_projection
        
        # Add bias if present
        if self.use_bias:
            combined = combined + self.bias
        
        # Split into gates: input, forget, cell_candidate, output
        gates = torch.split(combined, self.hidden_size, dim=-1)
        i_gate, f_gate, g_gate, o_gate = gates
        
        # Apply activations
        i_gate = self.recurrent_activation(i_gate)  # Input gate
        f_gate = self.recurrent_activation(f_gate)  # Forget gate
        g_gate = self.activation(g_gate)           # Cell candidate
        o_gate = self.recurrent_activation(o_gate) # Output gate
        
        # Update cell state
        c_new = f_gate * c_prev + i_gate * g_gate
        
        # Update hidden state
        h_new = o_gate * self.activation(c_new)
        
        return h_new, (c_new, h_new)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Step through the LSTM for given input sequence."""
        if x.values.ndim != 3:
            raise ValueError(f'LSTM requires rank 3 input. Got: {x.values.shape}')
        
        # Handle multi-timestep input by processing each timestep
        if x.values.shape[1] == 1:
            # Single timestep - direct cell computation
            h, new_state = self._cell(x.values, state)
            
            # Don't update state on invalid timesteps
            def copy_state_through(new_val: Tensor, old_val: Tensor) -> Tensor:
                mask = x.mask.unsqueeze(-1)  # [batch, 1, 1]
                return torch.where(mask, new_val, old_val)
            
            # Apply mask to state updates
            c_new, h_new = new_state
            c_old, h_old = state
            final_state = (
                copy_state_through(c_new, c_old),
                copy_state_through(h_new, h_old)
            )
            
            return Sequence(h, x.mask), final_state
        else:
            # Multi-timestep input - process step by step
            batch_size, time_steps, _ = x.values.shape
            outputs = []
            current_state = state
            
            for t in range(time_steps):
                # Extract single timestep
                x_t = Sequence(
                    x.values[:, t:t+1, :],
                    x.mask[:, t:t+1]
                )
                
                # Process single timestep
                h_t, current_state = self.step(x_t, current_state, training, constants)
                outputs.append(h_t.values)
            
            # Concatenate outputs
            output_values = torch.cat(outputs, dim=1)
            return Sequence(output_values, x.mask), current_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply LSTM layer-wise (process entire sequence at once)."""
        if x.values.ndim != 3:
            raise ValueError(f'LSTM requires rank 3 input. Got: {x.values.shape}')
        
        # Get initial state if not provided
        if initial_state is None:
            initial_state = self.get_initial_state(
                x.values.shape[0], x.channel_spec, training, constants
            )
        
        # Process sequence step by step
        output, _ = self.step(x, initial_state, training, constants)
        return output


# =============================================================================
# GRU Layer
# =============================================================================

class GRU(BaseRNN):
    """Gated Recurrent Unit (GRU) layer."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 recurrent_activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.activation = activation or torch.tanh
        self.recurrent_activation = recurrent_activation or torch.sigmoid
        
        # Initialize input-to-hidden weights (3 * hidden_size for r, z, n gates)
        self.input_weights = nn.Parameter(
            torch.randn(input_size, 3 * hidden_size) / math.sqrt(input_size)
        )
        
        # Initialize hidden-to-hidden weights
        self.recurrent_weights = nn.Parameter(
            self._orthogonal_init(hidden_size, 3 * hidden_size)
        )
        
        # Initialize bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
    
    def _orthogonal_init(self, rows: int, cols: int) -> Tensor:
        """Initialize weights using orthogonal initialization."""
        tensor = torch.randn(rows, cols)
        
        # Apply orthogonal initialization to each gate separately
        gate_size = cols // 3
        for i in range(3):
            start_idx = i * gate_size
            end_idx = start_idx + gate_size
            gate_tensor = tensor[:, start_idx:end_idx]
            
            if rows >= gate_size:
                q, r = torch.linalg.qr(gate_tensor)
                d = torch.diag(r)
                q *= d.sign().unsqueeze(0).expand_as(q)
                tensor[:, start_idx:end_idx] = q
            else:
                tensor[:, start_idx:end_idx] = F.normalize(gate_tensor, dim=0)
        
        return tensor
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'GRU requires rank 3 input. Got: {input_shape}')
        return (self.hidden_size,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for GRU (hidden state only)."""
        device = next(self.parameters()).device
        dtype = torch.float32  # Always use float32 for state
        
        # Initial hidden state
        h = torch.zeros(batch_size, 1, self.hidden_size, dtype=dtype, device=device)
        
        return h
    
    def _cell(self, x: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """GRU cell computation for a single timestep."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f'Expected [batch, 1, features] input, got: {x.shape}')
        
        # Compute input projection: [batch, 1, input_size] -> [batch, 1, 3*hidden_size]
        input_projection = torch.matmul(x, self.input_weights)
        
        # Compute recurrent projection: [batch, 1, hidden_size] -> [batch, 1, 3*hidden_size]
        recurrent_projection = torch.matmul(h_prev, self.recurrent_weights)
        
        # Split projections into gates
        input_gates = torch.split(input_projection, self.hidden_size, dim=-1)
        recurrent_gates = torch.split(recurrent_projection, self.hidden_size, dim=-1)
        
        # Reset and update gates
        r_input, z_input, n_input = input_gates
        r_recurrent, z_recurrent, n_recurrent = recurrent_gates
        
        # Add bias if present
        if self.use_bias:
            bias_gates = torch.split(self.bias, self.hidden_size, dim=-1)
            r_bias, z_bias, n_bias = bias_gates
            
            r_input = r_input + r_bias
            z_input = z_input + z_bias
            n_input = n_input + n_bias
        
        # Compute reset and update gates
        r_gate = self.recurrent_activation(r_input + r_recurrent)
        z_gate = self.recurrent_activation(z_input + z_recurrent)
        
        # Compute new gate (candidate hidden state)
        n_gate = self.activation(n_input + r_gate * n_recurrent)
        
        # Update hidden state
        h_new = (1 - z_gate) * n_gate + z_gate * h_prev
        
        return h_new, h_new
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Step through the GRU for given input sequence."""
        if x.values.ndim != 3:
            raise ValueError(f'GRU requires rank 3 input. Got: {x.values.shape}')
        
        # Handle multi-timestep input by processing each timestep
        if x.values.shape[1] == 1:
            # Single timestep - direct cell computation
            h, new_state = self._cell(x.values, state)
            
            # Don't update state on invalid timesteps
            mask = x.mask.unsqueeze(-1)  # [batch, 1, 1]
            final_state = torch.where(mask, new_state, state)
            
            return Sequence(h, x.mask), final_state
        else:
            # Multi-timestep input - process step by step
            batch_size, time_steps, _ = x.values.shape
            outputs = []
            current_state = state
            
            for t in range(time_steps):
                # Extract single timestep
                x_t = Sequence(
                    x.values[:, t:t+1, :],
                    x.mask[:, t:t+1]
                )
                
                # Process single timestep
                h_t, current_state = self.step(x_t, current_state, training, constants)
                outputs.append(h_t.values)
            
            # Concatenate outputs
            output_values = torch.cat(outputs, dim=1)
            return Sequence(output_values, x.mask), current_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply GRU layer-wise (process entire sequence at once)."""
        if x.values.ndim != 3:
            raise ValueError(f'GRU requires rank 3 input. Got: {x.values.shape}')
        
        # Get initial state if not provided
        if initial_state is None:
            initial_state = self.get_initial_state(
                x.values.shape[0], x.channel_spec, training, constants
            )
        
        # Process sequence step by step
        output, _ = self.step(x, initial_state, training, constants)
        return output


# =============================================================================
# Vanilla RNN Layer
# =============================================================================

class VanillaRNN(BaseRNN):
    """Vanilla RNN layer with basic recurrent connections."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.activation = activation or torch.tanh
        
        # Initialize input-to-hidden weights
        self.input_weights = nn.Parameter(
            torch.randn(input_size, hidden_size) / math.sqrt(input_size)
        )
        
        # Initialize hidden-to-hidden weights
        self.recurrent_weights = nn.Parameter(
            self._orthogonal_init(hidden_size, hidden_size)
        )
        
        # Initialize bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias', None)
    
    def _orthogonal_init(self, rows: int, cols: int) -> Tensor:
        """Initialize weights using orthogonal initialization."""
        tensor = torch.randn(rows, cols)
        
        if rows >= cols:
            q, r = torch.linalg.qr(tensor)
            d = torch.diag(r)
            q *= d.sign().unsqueeze(0).expand_as(q)
            return q
        else:
            return F.normalize(tensor, dim=0)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'VanillaRNN requires rank 3 input. Got: {input_shape}')
        return (self.hidden_size,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for VanillaRNN (hidden state only)."""
        device = next(self.parameters()).device
        dtype = torch.float32  # Always use float32 for state
        
        # Initial hidden state
        h = torch.zeros(batch_size, 1, self.hidden_size, dtype=dtype, device=device)
        
        return h
    
    def _cell(self, x: Tensor, h_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """Vanilla RNN cell computation for a single timestep."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f'Expected [batch, 1, features] input, got: {x.shape}')
        
        # Compute input projection: [batch, 1, input_size] -> [batch, 1, hidden_size]
        input_projection = torch.matmul(x, self.input_weights)
        
        # Compute recurrent projection: [batch, 1, hidden_size] -> [batch, 1, hidden_size]
        recurrent_projection = torch.matmul(h_prev, self.recurrent_weights)
        
        # Combine projections
        combined = input_projection + recurrent_projection
        
        # Add bias if present
        if self.use_bias:
            combined = combined + self.bias
        
        # Apply activation
        h_new = self.activation(combined)
        
        return h_new, h_new
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Step through the VanillaRNN for given input sequence."""
        if x.values.ndim != 3:
            raise ValueError(f'VanillaRNN requires rank 3 input. Got: {x.values.shape}')
        
        # Handle multi-timestep input by processing each timestep
        if x.values.shape[1] == 1:
            # Single timestep - direct cell computation
            h, new_state = self._cell(x.values, state)
            
            # Don't update state on invalid timesteps
            mask = x.mask.unsqueeze(-1)  # [batch, 1, 1]
            final_state = torch.where(mask, new_state, state)
            
            return Sequence(h, x.mask), final_state
        else:
            # Multi-timestep input - process step by step
            batch_size, time_steps, _ = x.values.shape
            outputs = []
            current_state = state
            
            for t in range(time_steps):
                # Extract single timestep
                x_t = Sequence(
                    x.values[:, t:t+1, :],
                    x.mask[:, t:t+1]
                )
                
                # Process single timestep
                h_t, current_state = self.step(x_t, current_state, training, constants)
                outputs.append(h_t.values)
            
            # Concatenate outputs
            output_values = torch.cat(outputs, dim=1)
            return Sequence(output_values, x.mask), current_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply VanillaRNN layer-wise (process entire sequence at once)."""
        if x.values.ndim != 3:
            raise ValueError(f'VanillaRNN requires rank 3 input. Got: {x.values.shape}')
        
        # Get initial state if not provided
        if initial_state is None:
            initial_state = self.get_initial_state(
                x.values.shape[0], x.channel_spec, training, constants
            )
        
        # Process sequence step by step
        output, _ = self.step(x, initial_state, training, constants)
        return output 