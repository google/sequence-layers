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
"""Simple (generally stateless) layers for PyTorch."""

import fractions
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
    StatelessPointwiseFunctor,
    PreservesShape,
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    # Activation layers
    'Abs',
    'Elu', 
    'Exp',
    'Gelu',
    'LeakyRelu',
    'Log',
    'Power',
    'PRelu',
    'Relu',
    'Sigmoid',
    'Softmax',
    'Softplus',
    'Swish',
    'Tanh',
    
    # Transformation layers
    'Add',
    'Affine',
    'Cast',
    'Scale',
    'Translate',
    'Transpose',
    'Reshape',
    'Squeeze',
    'ExpandDims',
    'MoveAxis',
    'SwapAxes',
    'Flatten',
    'Slice',
    'OneHot',
    
    # Utility layers
    'Identity',
    'Lambda',
    'Emit',
    'Dropout',
    'MaskInvalid',
    'Maximum',
    'Minimum',
    'Mod',
    'Logging',
    'OptimizationBarrier',
]


# =============================================================================
# Activation Layers
# =============================================================================

class Abs(StatelessPointwiseFunctor):
    """Absolute value layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.abs(values), mask
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        # The absolute value of complex numbers is a real magnitude
        if input_dtype == torch.complex64:
            return torch.float32
        elif input_dtype == torch.complex128:
            return torch.float64
        else:
            return input_dtype


class Elu(StatelessPointwiseFunctor):
    """An ELU activation layer."""
    
    def __init__(self, alpha: float = 1.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.alpha = alpha
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.elu(values, alpha=self.alpha), mask


class Exp(StatelessPointwiseFunctor):
    """Exponential layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.exp(values), mask


class Gelu(StatelessPointwiseFunctor):
    """A Gaussian Error Linear Unit (GELU) layer."""
    
    def __init__(self, approximate: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.approximate = 'tanh' if approximate else 'none'
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.gelu(values, approximate=self.approximate), mask


class LeakyRelu(StatelessPointwiseFunctor):
    """A Leaky ReLU layer."""
    
    def __init__(self, negative_slope: float = 0.01, name: Optional[str] = None):
        super().__init__(name=name)
        self.negative_slope = negative_slope
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.leaky_relu(values, negative_slope=self.negative_slope), mask


class Log(StatelessPointwiseFunctor):
    """Natural logarithm layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.log(values), mask


class Power(StatelessPointwiseFunctor):
    """Power layer."""
    
    def __init__(self, exponent: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.exponent = exponent
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.pow(values, self.exponent), mask


class PRelu(StatelessPointwiseFunctor):
    """Parametric ReLU, i.e., a Leaky ReLU where the negative slope is learnable."""
    
    def __init__(self, negative_slope_init: float = 0.01, name: Optional[str] = None):
        super().__init__(name=name)
        self.negative_slope = nn.Parameter(torch.tensor(negative_slope_init))
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.prelu(values, self.negative_slope), mask


class Relu(StatelessPointwiseFunctor):
    """A ReLU layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.relu(values), mask


class Sigmoid(StatelessPointwiseFunctor):
    """A sigmoid layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.sigmoid(values), mask


class Softmax(StatelessPointwiseFunctor):
    """A softmax layer."""
    
    def __init__(self, axis: int = -1, name: Optional[str] = None):
        super().__init__(name=name)
        self.axis = axis
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # Validate axis - cannot be applied to batch or time dimension
        axis = self.axis
        if axis < 0:
            axis = values.ndim + axis
        if axis < 2:
            raise ValueError(
                f'The softmax cannot be applied on the batch or time dimension (got '
                f'{self.axis=} for shape={values.shape})'
            )
        return F.softmax(values, dim=self.axis), mask


class Softplus(StatelessPointwiseFunctor):
    """A softplus layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.softplus(values), mask


class Swish(StatelessPointwiseFunctor):
    """A Swish layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return F.silu(values), mask  # SiLU is the same as Swish


class Tanh(StatelessPointwiseFunctor):
    """A tanh layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.tanh(values), mask


# =============================================================================
# Transformation Layers
# =============================================================================

class Add(StatelessPointwiseFunctor):
    """Adds a constant value to the input."""
    
    def __init__(self, value: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.value = value
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return values + self.value, mask


class Affine(Stateless, PreservesType):
    """Learnable additive bias and multiplicative scale."""
    
    def __init__(self, shape: Tuple[int, ...], use_scale: bool = True, 
                 use_bias: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.shape = shape
        self.use_scale = use_scale
        self.use_bias = use_bias
        
        if use_scale:
            self.scale = nn.Parameter(torch.ones(shape))
        else:
            self.register_parameter('scale', None)
            
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(shape))
        else:
            self.register_parameter('bias', None)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Check that the parameters do not have batch or time dimension
        if len(input_shape) < len(self.shape):
            raise ValueError(
                f'The parameter has too many dimensions (input: {len(input_shape)}, '
                f'parameter: {len(self.shape)})'
            )
        
        # Check broadcast compatibility
        try:
            torch.broadcast_shapes(input_shape, self.shape)
        except RuntimeError as e:
            raise ValueError(f'Shapes are not broadcastable: {input_shape} vs {self.shape}') from e
        
        return torch.broadcast_shapes(input_shape, self.shape)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Validate shapes
        _ = self.get_output_shape(x.channel_shape, constants)
        
        values = x.values
        if self.use_scale:
            values = values * self.scale.to(values.dtype)
        if self.use_bias:
            values = values + self.bias.to(values.dtype)
        
        return Sequence(values, x.mask)


class Cast(StatelessPointwiseFunctor):
    """Cast input values to the specified type."""
    
    def __init__(self, dtype: DType, name: Optional[str] = None):
        super().__init__(name=name)
        self.dtype = dtype
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return values.to(self.dtype), mask
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.dtype


class Scale(StatelessPointwiseFunctor):
    """Multiplies input by a constant scale factor."""
    
    def __init__(self, scale: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.scale = scale
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return values * self.scale, mask


class Translate(StatelessPointwiseFunctor):
    """Translates input by a constant offset (alias for Add)."""
    
    def __init__(self, offset: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.offset = offset
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return values + self.offset, mask


class Transpose(Stateless):
    """Transpose layer for swapping dimensions."""
    
    def __init__(self, dim0: int, dim1: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim0 = dim0
        self.dim1 = dim1
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape_list = list(input_shape)
        shape_list[self.dim0], shape_list[self.dim1] = shape_list[self.dim1], shape_list[self.dim0]
        return tuple(shape_list)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimensions to account for batch and time dimensions
        actual_dim0 = self.dim0 + 2 if self.dim0 >= 0 else self.dim0
        actual_dim1 = self.dim1 + 2 if self.dim1 >= 0 else self.dim1
        
        transposed_values = torch.transpose(x.values, actual_dim0, actual_dim1)
        return Sequence(transposed_values, x.mask)


class Reshape(Stateless):
    """Reshape layer for changing channel dimensions."""
    
    def __init__(self, shape: Tuple[int, ...], name: Optional[str] = None):
        super().__init__(name=name)
        self.shape = shape
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.shape
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        batch_size, time_steps = x.values.shape[:2]
        new_shape = (batch_size, time_steps) + self.shape
        reshaped_values = x.values.reshape(new_shape)
        return Sequence(reshaped_values, x.mask)


class Squeeze(Stateless):
    """Squeeze layer for removing singleton dimensions."""
    
    def __init__(self, dim: Optional[int] = None, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = dim
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if self.dim is None:
            return tuple(d for d in input_shape if d != 1)
        else:
            shape_list = list(input_shape)
            if shape_list[self.dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {self.dim} with size {shape_list[self.dim]}")
            return tuple(shape_list[:self.dim] + shape_list[self.dim + 1:])
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if self.dim is None:
            squeezed_values = torch.squeeze(x.values)
        else:
            # Adjust dimension to account for batch and time dimensions
            actual_dim = self.dim + 2 if self.dim >= 0 else self.dim
            squeezed_values = torch.squeeze(x.values, dim=actual_dim)
        
        return Sequence(squeezed_values, x.mask)


class ExpandDims(Stateless):
    """Expand dimensions by adding singleton dimensions."""
    
    def __init__(self, dim: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.dim = dim
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape_list = list(input_shape)
        shape_list.insert(self.dim, 1)
        return tuple(shape_list)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimension to account for batch and time dimensions
        actual_dim = self.dim + 2 if self.dim >= 0 else self.dim
        expanded_values = torch.unsqueeze(x.values, dim=actual_dim)
        return Sequence(expanded_values, x.mask)


class MoveAxis(Stateless):
    """Move axis from source to destination."""
    
    def __init__(self, source: int, destination: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.source = source
        self.destination = destination
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape_list = list(input_shape)
        dim_value = shape_list.pop(self.source)
        shape_list.insert(self.destination, dim_value)
        return tuple(shape_list)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimensions to account for batch and time dimensions
        actual_source = self.source + 2 if self.source >= 0 else self.source
        actual_destination = self.destination + 2 if self.destination >= 0 else self.destination
        
        moved_values = torch.moveaxis(x.values, actual_source, actual_destination)
        return Sequence(moved_values, x.mask)


class SwapAxes(Stateless):
    """Swap two axes (alias for Transpose)."""
    
    def __init__(self, axis1: int, axis2: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.axis1 = axis1
        self.axis2 = axis2
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape_list = list(input_shape)
        shape_list[self.axis1], shape_list[self.axis2] = shape_list[self.axis2], shape_list[self.axis1]
        return tuple(shape_list)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimensions to account for batch and time dimensions
        actual_axis1 = self.axis1 + 2 if self.axis1 >= 0 else self.axis1
        actual_axis2 = self.axis2 + 2 if self.axis2 >= 0 else self.axis2
        
        swapped_values = torch.swapaxes(x.values, actual_axis1, actual_axis2)
        return Sequence(swapped_values, x.mask)


class Flatten(Stateless):
    """Flatten layer for flattening channel dimensions."""
    
    def __init__(self, start_dim: int = 0, end_dim: int = -1, name: Optional[str] = None):
        super().__init__(name=name)
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Calculate the flattened shape
        if self.end_dim == -1:
            end_dim = len(input_shape) - 1
        else:
            end_dim = self.end_dim
        
        flat_size = 1
        for i in range(self.start_dim, end_dim + 1):
            flat_size *= input_shape[i]
        
        new_shape = list(input_shape[:self.start_dim]) + [flat_size] + list(input_shape[end_dim + 1:])
        return tuple(new_shape)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimensions to account for batch and time dimensions
        actual_start_dim = self.start_dim + 2 if self.start_dim >= 0 else self.start_dim
        actual_end_dim = self.end_dim + 2 if self.end_dim >= 0 else self.end_dim
        
        flattened_values = torch.flatten(x.values, start_dim=actual_start_dim, end_dim=actual_end_dim)
        return Sequence(flattened_values, x.mask)


class Slice(Stateless):
    """Slice layer for extracting portions of sequences."""
    
    def __init__(self, start: int, end: Optional[int] = None, step: int = 1, 
                 dim: int = -1, name: Optional[str] = None):
        super().__init__(name=name)
        self.start = start
        self.end = end
        self.step = step
        self.dim = dim
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape_list = list(input_shape)
        dim_size = shape_list[self.dim]
        
        # Calculate slice size
        end = self.end if self.end is not None else dim_size
        slice_size = max(0, (end - self.start + self.step - 1) // self.step)
        
        shape_list[self.dim] = slice_size
        return tuple(shape_list)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # Adjust dimension to account for batch and time dimensions
        actual_dim = self.dim + 2 if self.dim >= 0 else self.dim
        
        # Create slice object
        slice_obj = slice(self.start, self.end, self.step)
        
        # Apply slice to the specified dimension
        sliced_values = torch.index_select(x.values, actual_dim, 
                                         torch.arange(self.start, 
                                                    self.end if self.end is not None else x.values.shape[actual_dim],
                                                    self.step, device=x.values.device))
        
        return Sequence(sliced_values, x.mask)


class OneHot(Stateless):
    """Computes one-hot vector of the input."""
    
    def __init__(self, depth: int, dtype: DType = torch.float32, name: Optional[str] = None):
        super().__init__(name=name)
        self.depth = depth
        self.dtype = dtype
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return tuple(input_shape) + (self.depth,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        integer_types = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
        if input_dtype not in integer_types:
            raise ValueError(f'Input to OneHot must be an integer type. Got: {input_dtype}')
        return self.dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        integer_types = [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]
        if x.dtype not in integer_types:
            raise ValueError(f'Input to OneHot must be an integer type. Got: {x.dtype}')
        
        # Convert to one-hot
        one_hot_values = F.one_hot(x.values.long(), num_classes=self.depth).to(self.dtype)
        
        return Sequence(one_hot_values, x.mask)


# =============================================================================
# Utility Layers
# =============================================================================

class Identity(StatelessPointwise):
    """Identity pass-through of the input."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        return x


class Lambda(Stateless):
    """A SequenceLayer that wraps a Python lambda function."""
    
    def __init__(self, fn: Callable, sequence_input: bool = False, 
                 mask_required: bool = True, name: Optional[str] = None):
        super().__init__(name=name)
        self.fn = fn
        self.sequence_input = sequence_input
        self.mask_required = mask_required
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if self.sequence_input:
            result = self.fn(x)
            if not isinstance(result, Sequence):
                raise ValueError("Function with sequence_input=True must return a Sequence")
            return result
        else:
            result_values = self.fn(x.values)
            if self.mask_required:
                # Function may have changed masking requirements
                return Sequence(result_values, x.mask).mask_invalid()
            else:
                return Sequence(result_values, x.mask)


class Emit(StatelessPointwise):
    """Emit layer that passes through input unchanged but can be used for debugging."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        return x


class Dropout(StatelessPointwise):
    """Dropout layer using PyTorch's dropout."""
    
    def __init__(self, rate: float = 0.5, name: Optional[str] = None):
        super().__init__(name=name)
        self.rate = rate
        self.dropout = nn.Dropout(p=rate)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if training:
            dropped_values = self.dropout(x.values)
        else:
            dropped_values = x.values
        
        return Sequence(dropped_values, x.mask)


class MaskInvalid(StatelessPointwise):
    """Explicitly mask invalid timesteps to zero."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        return x.mask_invalid()


class Maximum(StatelessPointwiseFunctor):
    """Element-wise maximum with a constant."""
    
    def __init__(self, value: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.value = value
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.maximum(values, torch.tensor(self.value, device=values.device, dtype=values.dtype)), mask


class Minimum(StatelessPointwiseFunctor):
    """Element-wise minimum with a constant."""
    
    def __init__(self, value: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.value = value
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.minimum(values, torch.tensor(self.value, device=values.device, dtype=values.dtype)), mask


class Mod(StatelessPointwiseFunctor):
    """Element-wise modulo operation."""
    
    def __init__(self, divisor: float, name: Optional[str] = None):
        super().__init__(name=name)
        self.divisor = divisor
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.remainder(values, self.divisor), mask


class Logging(StatelessPointwise):
    """Logging layer that prints sequence information."""
    
    def __init__(self, message: str = "", name: Optional[str] = None):
        super().__init__(name=name)
        self.message = message
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        x.print(message=self.message)
        return x


class OptimizationBarrier(StatelessPointwise):
    """Optimization barrier layer (no-op in PyTorch)."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        # PyTorch doesn't have optimization barriers like JAX
        return x 