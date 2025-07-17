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
"""Time-varying layers for PyTorch."""

from typing import Any, Callable, Optional, Union, Tuple, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    StatelessPointwise,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    'SequenceEmbedding',
    'SequenceDense',
]


def _check_step_bounds(start_time: int, end_time: int, num_steps: int):
    """Check that step bounds are valid."""
    if start_time < 0:
        raise ValueError(f'start_time must be non-negative, got {start_time}')
    if end_time > num_steps:
        raise ValueError(f'end_time {end_time} exceeds num_steps {num_steps}')


class SequenceEmbedding(SequenceLayer):
    """Computes sequence embeddings of integer input codes.
    
    Provides step-dependent embeddings with num_steps embeddings, where sequence
    step specifies the group index. For steps above num_steps - 1, we use the
    last group (i.e., group_index = num_steps - 1).
    """
    
    def __init__(self,
                 dimension: int,
                 num_embeddings: int,
                 num_steps: int,
                 embedding_init: str = 'uniform',
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.dimension = dimension
        self.num_embeddings = num_embeddings
        self.num_steps = num_steps
        
        # Create embedding table: [num_embeddings * num_steps, dimension]
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings * num_steps,
            embedding_dim=dimension
        )
        
        # Initialize embeddings
        if embedding_init == 'uniform':
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        elif embedding_init == 'normal':
            nn.init.normal_(self.embedding.weight, 0.0, 1.0)
        else:
            raise ValueError(f'Unknown embedding_init: {embedding_init}')
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Returns output shape with embedding dimension appended."""
        return input_shape + (self.dimension,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        """Always returns float32 for embeddings."""
        return torch.float32
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state (current timestep)."""
        # Use CPU as default device since we don't have input tensor context here
        device = torch.device('cpu')
        return torch.zeros((), dtype=torch.int32, device=device)
    
    def _time_wise_embed(self, values: Tensor, start_time: int) -> Tensor:
        """Apply time-wise embedding."""
        # Validate inputs
        if torch.any(values < 0):
            raise ValueError('Out of range lookup index (< 0)')
        if torch.any(values >= self.num_embeddings):
            raise ValueError('Out of range lookup index (>= num_embeddings)')
        
        values = values.long()
        
        batch_size, time_delta = values.shape[:2]
        
        # Create steps tensor
        steps = start_time + torch.arange(time_delta, device=values.device, dtype=torch.long)
        steps = torch.minimum(steps, torch.tensor(self.num_steps - 1, device=values.device, dtype=torch.long))
        
        # Reshape steps to broadcast with values
        steps = steps.view(1, time_delta, *([1] * (values.ndim - 2)))
        
        # The lookup indices are broadcast-added
        lookup_indices = values + steps * self.num_embeddings
        
        # Apply embedding
        embeddings = self.embedding(lookup_indices)
        
        return embeddings
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Apply embedding step-wise."""
        start_time = state.item()
        time_delta = x.shape[1]
        
        # Check bounds
        _check_step_bounds(start_time, start_time + time_delta, self.num_steps)
        
        # Apply time-wise embedding
        embedded_values = self._time_wise_embed(x.values, start_time)
        
        # Create output sequence
        output = Sequence(embedded_values, x.mask)
        
        new_state = torch.tensor(start_time + time_delta, dtype=torch.int32, device=x.values.device)
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply embedding layer-wise."""
        # Apply time-wise embedding starting from time 0
        embedded_values = self._time_wise_embed(x.values, 0)
        
        # Create output sequence
        output = Sequence(embedded_values, x.mask)
        
        return output


class SequenceDense(SequenceLayer):
    """Step-dependent Dense layer.
    
    Provides step-dependent Dense layers up to num_steps, where the sequence
    step determines which Dense layer is used. This layer can be used with
    fixed length sequences, where num_steps is set to the fixed length.
    """
    
    def __init__(self,
                 units: int,
                 num_steps: int,
                 activation: Optional[Union[str, Callable]] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.units = units
        self.num_steps = num_steps
        self.use_bias = use_bias
        
        # Get activation function
        if activation is None:
            self.activation_fn = None
        elif isinstance(activation, str):
            self.activation_fn = getattr(F, activation)
        else:
            self.activation_fn = activation
        
        # Parameters will be created in _build method
        self.kernel = None
        self.bias = None
        self.input_dim = None
        self.built = False
    
    def _build(self, x: Sequence):
        """Build the layer parameters."""
        if self.built:
            return
        
        self.input_dim = x.shape[-1]
        
        # Create kernel: [num_steps, input_dim, units]
        self.kernel = nn.Parameter(torch.empty(self.num_steps, self.input_dim, self.units, 
                                               device=x.values.device, dtype=x.values.dtype))
        
        # Initialize kernel
        if hasattr(nn.init, 'xavier_uniform_'):
            nn.init.xavier_uniform_(self.kernel)
        else:
            nn.init.uniform_(self.kernel, -1.0, 1.0)
        
        # Create bias if needed
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_steps, self.units, 
                                                 device=x.values.device, dtype=x.values.dtype))
        
        self.built = True
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Returns output shape with units as final dimension."""
        return input_shape[:-1] + (self.units,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        """Preserves input dtype."""
        return input_dtype
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state (current timestep)."""
        # Use CPU as default device since we don't have input tensor context here
        device = torch.device('cpu')
        return torch.zeros((), dtype=torch.int32, device=device)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Apply dense layer step-wise."""
        self._build(x)
        
        start_time = state.item()
        time_delta = x.shape[1]
        
        # Check bounds
        _check_step_bounds(start_time, start_time + time_delta, self.num_steps)
        
        # Slice kernel for current timesteps: [time_delta, input_dim, units]
        step_kernel = self.kernel[start_time:start_time + time_delta]
        
        # Apply dense transformation: [batch, time_delta, input_dim] @ [time_delta, input_dim, units]
        # -> [batch, time_delta, units]
        output_values = torch.einsum('bti,tio->bto', x.values, step_kernel)
        
        # Add bias if needed
        if self.use_bias:
            step_bias = self.bias[start_time:start_time + time_delta]
            output_values = output_values + step_bias
        
        # Apply activation
        if self.activation_fn is not None:
            output_values = self.activation_fn(output_values)
        
        # Create output sequence
        output = Sequence(output_values, x.mask)
        
        new_state = torch.tensor(start_time + time_delta, dtype=torch.int32, device=x.values.device)
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply dense layer layer-wise."""
        # For layer-wise execution, we process the entire sequence at once
        # This is equivalent to calling step with initial_state=0
        if initial_state is None:
            initial_state = torch.zeros((), dtype=torch.int32, device=x.values.device)
        
        output, _ = self.step(x, initial_state, training, constants)
        return output 