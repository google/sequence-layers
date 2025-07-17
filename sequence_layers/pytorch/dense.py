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
"""Dense layers for PyTorch."""

import functools
import math
from typing import Any, Callable, Optional, Union, Tuple, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    Stateless,
    StatelessPointwise,
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    # Basic Dense Layers
    'Dense',
    'DenseShaped', 
    'EinsumDense',
    
    # Embedding Layers
    'Embedding',
    'EmbeddingTranspose',
    
    # Gated Units
    'GatedUnit',
    'GatedLinearUnit',
    'GatedTanhUnit',
]


# =============================================================================
# Basic Dense Layers
# =============================================================================

class Dense(Stateless):
    """A basic dense layer."""
    
    def __init__(self, features: int, use_bias: bool = True, 
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.features = features
        self.use_bias = use_bias
        self.activation = activation
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) == 0:
            raise ValueError(f'Dense requires at least rank 3 input. Got: {input_shape=}')
        return tuple(input_shape[:-1]) + (self.features,)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim < 3:
            raise ValueError(f'Dense requires at least rank 3 input. Got: {x.shape=}')
        
        input_features = x.shape[-1]
        
        # Initialize parameters if not already done
        if not hasattr(self, 'weight'):
            self.weight = nn.Parameter(torch.randn(input_features, self.features) / math.sqrt(input_features))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(self.features))
        
        # Apply linear transformation
        def apply_dense(values):
            # Reshape to [batch * time, features] for matmul
            original_shape = values.shape
            values_flat = values.reshape(-1, original_shape[-1])
            
            # Apply dense transformation
            output = torch.matmul(values_flat, self.weight.to(values.dtype))
            
            if self.use_bias:
                output = output + self.bias.to(values.dtype)
            
            if self.activation is not None:
                output = self.activation(output)
            
            # Reshape back to original batch and time dimensions
            output_shape = original_shape[:-1] + (self.features,)
            return output.reshape(output_shape)
        
        # Preserve masked state if no bias or activation are in use
        if self.use_bias or self.activation is not None:
            return x.apply_values(apply_dense)
        else:
            return x.apply_values_masked(apply_dense)


class DenseShaped(Stateless):
    """A dense layer that transforms the channel shape."""
    
    def __init__(self, output_shape: Tuple[int, ...], use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_shape = tuple(output_shape)
        self.use_bias = use_bias
        self.activation = activation
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.output_shape
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        input_channel_shape = x.channel_shape
        output_channel_shape = self.output_shape
        
        # Calculate total input and output dimensions
        input_dim = math.prod(input_channel_shape)
        output_dim = math.prod(output_channel_shape)
        
        # Initialize parameters if not already done
        if not hasattr(self, 'weight'):
            self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / math.sqrt(input_dim))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros(output_dim))
        
        def apply_dense_shaped(values):
            # Flatten input channels
            batch_size, time_size = values.shape[:2]
            values_flat = values.reshape(batch_size, time_size, -1)
            
            # Apply dense transformation
            output = torch.matmul(values_flat, self.weight.to(values.dtype))
            
            if self.use_bias:
                output = output + self.bias.to(values.dtype)
            
            if self.activation is not None:
                output = self.activation(output)
            
            # Reshape to output shape
            output_shape = (batch_size, time_size) + output_channel_shape
            return output.reshape(output_shape)
        
        # Preserve masked state if no bias or activation are in use
        if self.use_bias or self.activation is not None:
            return x.apply_values(apply_dense_shaped)
        else:
            return x.apply_values_masked(apply_dense_shaped)


class EinsumDense(Stateless):
    """A dense layer that transforms the channel shape with an einsum equation."""
    
    def __init__(self, equation: str, output_shape: Tuple[int, ...], 
                 bias_axes: str = "", use_bias: bool = True,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.equation = equation
        self.output_shape = tuple(output_shape)
        self.bias_axes = bias_axes
        self.use_bias = use_bias
        self.activation = activation
        
        # Parse and validate equation
        self._parse_and_validate_equation()
    
    def _parse_and_validate_equation(self):
        """Parse and validate the einsum equation."""
        if '->' not in self.equation:
            raise ValueError(f'equation is not valid for EinsumDense: {self.equation}')
        
        left, output_spec = self.equation.split('->')
        input_spec, kernel_spec = left.split(',')
        
        if not input_spec.startswith('...') or not output_spec.startswith('...'):
            raise ValueError('Equation must be of the form "...X,Y->...Z".')
        
        # Store parsed components
        self.input_spec = input_spec
        self.kernel_spec = kernel_spec
        self.output_spec = output_spec
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.output_shape
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        input_shape = x.shape
        
        # Calculate kernel shape from equation
        input_dims = len(x.channel_shape)
        output_dims = len(self.output_shape)
        
        # Create kernel shape - this is a simplified version
        # In practice, this would need more sophisticated parsing
        kernel_shape = tuple(x.channel_shape) + tuple(self.output_shape)
        
        # Initialize parameters if not already done
        if not hasattr(self, 'weight'):
            fan_in = math.prod(x.channel_shape)
            self.weight = nn.Parameter(torch.randn(kernel_shape) / math.sqrt(fan_in))
            
            if self.use_bias and self.bias_axes:
                bias_shape = tuple(self.output_shape)
                self.bias = nn.Parameter(torch.zeros(bias_shape))
        
        def apply_einsum_dense(values):
            # Apply einsum operation
            output = torch.einsum(self.equation, values, self.weight.to(values.dtype))
            
            if self.use_bias and self.bias_axes:
                output = output + self.bias.to(values.dtype)
            
            if self.activation is not None:
                output = self.activation(output)
            
            return output
        
        # Preserve masked state if no bias or activation are in use
        if (self.use_bias and self.bias_axes) or self.activation is not None:
            return x.apply_values(apply_einsum_dense)
        else:
            return x.apply_values_masked(apply_einsum_dense)


# =============================================================================
# Embedding Layers
# =============================================================================

class Embedding(Stateless):
    """Computes embeddings of integer input codes."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize with unit norm like JAX default
        with torch.no_grad():
            self.embedding.weight.normal_(0, 1.0 / math.sqrt(embedding_dim))
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return tuple(input_shape) + (self.embedding_dim,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        if input_dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            raise ValueError(f'Input to Embedding must be an integer type. Got: {input_dtype}')
        return self.embedding.weight.dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            raise ValueError(f'Input to Embedding must be an integer type. Got: {x.dtype}')
        
        # Apply embedding lookup
        embedded_values = self.embedding(x.values)
        
        return Sequence(embedded_values, x.mask)
    
    def attend(self, x: Sequence) -> Sequence:
        """Attend over the embedding using a query array."""
        if not x.channel_shape:
            raise ValueError('Input query must have a channel dimension.')
        
        # Compute dot product between query and embedding weights
        # x.values: [batch, time, ..., embedding_dim]
        # embedding.weight: [num_embeddings, embedding_dim] 
        attended_values = torch.matmul(x.values, self.embedding.weight.t())
        
        return Sequence(attended_values, x.mask)


class EmbeddingTranspose(Stateless):
    """Wraps a bound Embedding layer to be attended upon (e.g. pre-softmax)."""
    
    def __init__(self, embedding: Embedding, use_bias: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embedding = embedding
        self.use_bias = use_bias
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if not input_shape or input_shape[-1] != self.embedding.embedding_dim:
            raise ValueError(
                "Input query's final channel dimension must be equal to the embedding dimension."
            )
        return tuple(input_shape[:-1]) + (self.embedding.num_embeddings,)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Use the embedding's attend method
        output = self.embedding.attend(x)
        
        if self.use_bias:
            output = output.apply_values(lambda v: v + self.bias.to(v.dtype))
        
        return output


# =============================================================================
# Gated Units
# =============================================================================

class GatedUnit(Stateless, PreservesType):
    """Computes a generalized Gated Unit, reducing the input channels by 2x."""
    
    def __init__(self, feature_activation: Optional[Callable[[Tensor], Tensor]] = None,
                 gate_activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.feature_activation = feature_activation
        self.gate_activation = gate_activation
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) == 0:
            raise ValueError(f'GatedUnit requires at least one input dimension. Got: {input_shape=}')
        
        channels = input_shape[-1]
        if channels % 2 != 0:
            raise ValueError(
                f'Final dimension of input ({input_shape=}) to GatedUnit must have an '
                'even number of channels.'
            )
        return tuple(input_shape[:-1]) + (channels // 2,)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        def apply_gated_unit(values):
            # Split input into feature and gate components
            feature, gate = torch.chunk(values, 2, dim=-1)
            
            if self.feature_activation is not None:
                feature = self.feature_activation(feature)
            if self.gate_activation is not None:
                gate = self.gate_activation(gate)
            
            # Element-wise multiplication
            return feature * gate
        
        return x.apply_values(apply_gated_unit)


class GatedLinearUnit(GatedUnit):
    """Computes a Gated Linear Unit, reducing the input channels by 2x."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            feature_activation=None,
            gate_activation=torch.sigmoid,
            name=name
        )


class GatedTanhUnit(GatedUnit):
    """Computes a Gated Tanh Unit, reducing the input channels by 2x."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            feature_activation=torch.tanh,
            gate_activation=torch.sigmoid,
            name=name
        ) 