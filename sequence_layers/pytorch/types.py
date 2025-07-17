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
"""PyTorch-specific sequence types."""

import abc
import dataclasses
import enum
import fractions
import functools
import math
import re
from typing import Any, Callable, Iterable, Optional, Union, List, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.internal.types import (
    PaddingMode,
    validate_padding,
    _camel_to_snake,
    Sequence as SequenceBase,
    SequenceArray as SequenceArrayBase,
    SequenceLayer as SequenceLayerBase,
    Emitting as EmittingBase,
    TensorLike,
    DType,
    Shape,
    OpLike,
    SequenceLike,
    State,
    Constants,
    Emits,
)


__all__ = [
    # Core types
    'Sequence',
    'SequenceArray', 
    'SequenceLayer',
    'Emitting',
    'MaskedSequence',
    
    # Utility types
    'ShapeDType',
    'ChannelSpec',
    'PreservesShape',
    'PreservesType',
    'Stateless',
    'StatelessEmitting',
    'StatelessPointwise',
    'StatelessPointwiseFunctor',
    'Steppable',
    
    # Helper functions
    'validate_padding',
    'check_layer',
    'check_step',
    'check_layer_with_emits',
    'check_step_with_emits',
    
    # Re-exports from internal
    'PaddingMode',
    'TensorLike',
    'DType',
    'Shape',
    'OpLike',
    'SequenceLike',
    'State',
    'Constants',
    'Emits',
]

# Type aliases for PyTorch
TensorLike = Union[Tensor, np.ndarray, float, int]
DType = torch.dtype
Shape = Union[torch.Size, Tuple[int, ...], List[int]]


@dataclasses.dataclass(frozen=True)
class ShapeDType:
    """A shape and dtype specification for sequences."""
    shape: Shape
    dtype: DType
    
    def __post_init__(self):
        if isinstance(self.shape, (list, tuple)):
            object.__setattr__(self, 'shape', torch.Size(self.shape))


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
    """A channel specification for sequences."""
    shape: Shape
    dtype: DType
    
    def __post_init__(self):
        if isinstance(self.shape, (list, tuple)):
            object.__setattr__(self, 'shape', torch.Size(self.shape))


class Sequence(SequenceBase):
    """A PyTorch implementation of a sequence with values and mask."""
    
    def __init__(self, values: Tensor, mask: Optional[Tensor] = None):
        """Initialize a sequence with values and optional mask.
        
        Args:
            values: Tensor of shape [batch, time, ...] containing sequence values
            mask: Optional boolean tensor of shape [batch, time] indicating valid timesteps.
                  If None, all timesteps are considered valid.
        """
        self._values = values
        if mask is None:
            # Create a mask with all True values
            mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        self._mask = mask
        
        # Validate shapes
        if values.ndim < 2:
            raise ValueError(f"Values must have at least 2 dimensions (batch, time), got {values.ndim}")
        if mask.shape != values.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} must match values batch and time dimensions {values.shape[:2]}")
    
    @property
    def values(self) -> Tensor:
        """The sequence values tensor."""
        return self._values
    
    @property
    def mask(self) -> Tensor:
        """The sequence mask tensor."""
        return self._mask
    
    @property
    def shape(self) -> torch.Size:
        """The shape of the values tensor."""
        return self._values.shape
    
    @property
    def dtype(self) -> DType:
        """The dtype of the values tensor."""
        return self._values.dtype
    
    @property
    def device(self) -> torch.device:
        """The device of the tensors."""
        return self._values.device
    
    @property
    def channel_shape(self) -> torch.Size:
        """Returns the sequence's channel shape (i.e. excluding batch and time)."""
        return self._values.shape[2:]
    
    @property
    def channel_spec(self) -> ChannelSpec:
        """Returns the channel specification."""
        return ChannelSpec(self.channel_shape, self.dtype)
    
    def __getitem__(self, key) -> 'Sequence':
        """Slice the sequence."""
        if isinstance(key, slice):
            # Single slice - apply to time dimension
            return Sequence(self._values[:, key], self._mask[:, key])
        elif isinstance(key, int):
            # Single integer index - slice batch dimension
            values_sliced = self._values[key]
            mask_sliced = self._mask[key]
            
            # If we sliced a single batch element, we need to add the batch dimension back
            if values_sliced.ndim == self._values.ndim - 1:
                values_sliced = values_sliced.unsqueeze(0)
                mask_sliced = mask_sliced.unsqueeze(0)
            
            return Sequence(values_sliced, mask_sliced)
        elif isinstance(key, tuple):
            if len(key) == 1:
                # Single element tuple - apply to time dimension
                return Sequence(self._values[:, key[0]], self._mask[:, key[0]])
            elif len(key) == 2:
                # Two element tuple - apply to batch and time
                values_sliced = self._values[key[0], key[1]]
                mask_sliced = self._mask[key[0], key[1]]
                
                # Handle dimension preservation for single indexing
                if isinstance(key[0], int) and isinstance(key[1], slice):
                    # Single batch, multiple time steps - add batch dimension back
                    values_sliced = values_sliced.unsqueeze(0)
                    mask_sliced = mask_sliced.unsqueeze(0)
                elif isinstance(key[0], slice) and isinstance(key[1], int):
                    # Multiple batch, single time step - add time dimension back
                    values_sliced = values_sliced.unsqueeze(1)
                    mask_sliced = mask_sliced.unsqueeze(1)
                elif isinstance(key[0], int) and isinstance(key[1], int):
                    # Single batch, single time step - add both dimensions back
                    values_sliced = values_sliced.unsqueeze(0).unsqueeze(0)
                    mask_sliced = mask_sliced.unsqueeze(0).unsqueeze(0)
                
                return Sequence(values_sliced, mask_sliced)
            else:
                # More than 2 elements - apply to values, but only first two to mask
                values_sliced = self._values[key]
                mask_sliced = self._mask[key[:2]]
                
                # Ensure mask has correct dimensions
                if values_sliced.ndim >= 2:
                    return Sequence(values_sliced, mask_sliced)
                else:
                    # If values became 1D, we need to add dimensions
                    if values_sliced.ndim == 1:
                        values_sliced = values_sliced.unsqueeze(0).unsqueeze(0)
                        mask_sliced = mask_sliced.unsqueeze(0).unsqueeze(0)
                    return Sequence(values_sliced, mask_sliced)
        else:
            raise TypeError(f"Invalid key type for sequence indexing: {type(key)}")
    
    def apply(self, map_fn: Callable, *args, **kwargs) -> 'Sequence':
        """Apply a function to both values and mask."""
        new_values, new_mask = map_fn(self._values, self._mask, *args, **kwargs)
        return Sequence(new_values, new_mask)
    
    def apply_values(self, map_fn: Callable, *args, **kwargs) -> 'Sequence':
        """Apply a function to values only, preserving mask."""
        new_values = map_fn(self._values, *args, **kwargs)
        return Sequence(new_values, self._mask)
    
    def apply_values_masked(self, map_fn: Callable, *args, **kwargs) -> 'Sequence':
        """Apply a function to values only where mask is True."""
        # Create a masked version of values
        masked_values = self.mask_invalid()._values
        new_values = map_fn(masked_values, *args, **kwargs)
        return Sequence(new_values, self._mask)
    
    def lengths(self) -> Tensor:
        """Returns the length of each sequence in the batch."""
        return self._mask.sum(dim=1)
    
    def mask_invalid(self) -> 'Sequence':
        """Returns a new Sequence with invalid timesteps replaced with zeros."""
        # Expand mask to match values shape
        expanded_mask = self.expanded_mask()
        # Use torch.where to explicitly set invalid values to zero
        # This handles NaN values correctly: where(mask, value, 0) replaces NaN with 0 when mask is False
        masked_values = torch.where(expanded_mask, self._values, torch.tensor(0.0, dtype=self._values.dtype, device=self._values.device))
        return Sequence(masked_values, self._mask)
    
    def expanded_mask(self) -> Tensor:
        """Returns mask reshaped to the same rank as values."""
        # Expand mask to match values shape for broadcasting
        mask_shape = list(self._mask.shape) + [1] * (self._values.ndim - 2)
        return self._mask.view(mask_shape).expand_as(self._values)
    
    def concatenate(self, other: 'Sequence') -> 'Sequence':
        """Concatenate with another sequence along time dimension."""
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"Batch dimensions must match: {self.shape[0]} vs {other.shape[0]}")
        if self.channel_shape != other.channel_shape:
            raise ValueError(f"Channel shapes must match: {self.channel_shape} vs {other.channel_shape}")
        
        new_values = torch.cat([self._values, other._values], dim=1)
        new_mask = torch.cat([self._mask, other._mask], dim=1)
        return Sequence(new_values, new_mask)
    
    @classmethod
    def concatenate_sequences(cls, sequences: Iterable[SequenceLike]) -> 'Sequence':
        """Concatenate multiple sequences along time dimension."""
        sequences = list(sequences)
        if not sequences:
            raise ValueError("Cannot concatenate empty sequence list")
        
        result = sequences[0]
        for seq in sequences[1:]:
            result = result.concatenate(seq)
        return result
    
    def pad_time(self, pad_left: int, pad_right: int, valid: bool = False, pad_value: float = 0.0) -> 'Sequence':
        """Pad the sequence with timesteps on the left and right.
        
        Args:
            pad_left: Number of timesteps to pad on the left
            pad_right: Number of timesteps to pad on the right  
            valid: If True, padded timesteps are marked as valid in the mask
            pad_value: Value to use for padding
        """
        batch_size = self._values.shape[0]
        
        # Create padding tensors
        values_parts = []
        mask_parts = []
        
        # Left padding
        if pad_left > 0:
            left_values_shape = (batch_size, pad_left) + self._values.shape[2:]
            left_values = torch.full(left_values_shape, pad_value, dtype=self._values.dtype, device=self._values.device)
            left_mask = torch.full((batch_size, pad_left), valid, dtype=torch.bool, device=self._mask.device)
            values_parts.append(left_values)
            mask_parts.append(left_mask)
        
        # Original values
        values_parts.append(self._values)
        mask_parts.append(self._mask)
        
        # Right padding
        if pad_right > 0:
            right_values_shape = (batch_size, pad_right) + self._values.shape[2:]
            right_values = torch.full(right_values_shape, pad_value, dtype=self._values.dtype, device=self._values.device)
            right_mask = torch.full((batch_size, pad_right), valid, dtype=torch.bool, device=self._mask.device)
            values_parts.append(right_values)
            mask_parts.append(right_mask)
        
        # Concatenate along time dimension
        padded_values = torch.cat(values_parts, dim=1)
        padded_mask = torch.cat(mask_parts, dim=1)
        
        return Sequence(padded_values, padded_mask)
    
    def reverse_time(self) -> 'Sequence':
        """Reverse the sequence along the time dimension."""
        return Sequence(
            torch.flip(self._values, dims=[1]),
            torch.flip(self._mask, dims=[1])
        )
    
    def print(self, message: str = "", summarize: bool = True):
        """Print this sequence with an optional message."""
        print(f"{message}Sequence(values={self._values.shape}, mask={self._mask.shape}, dtype={self.dtype})")
        if not summarize:
            print(f"Values:\n{self._values}")
            print(f"Mask:\n{self._mask}")
    
    def to(self, device: torch.device) -> 'Sequence':
        """Move sequence to specified device."""
        return Sequence(self._values.to(device), self._mask.to(device))


class MaskedSequence(Sequence):
    """A sequence that guarantees invalid timesteps are masked to zero."""
    
    def __init__(self, values: Tensor, mask: Optional[Tensor] = None):
        super().__init__(values, mask)
        # Ensure invalid timesteps are masked
        self._values = self.mask_invalid()._values


class SequenceArray(SequenceArrayBase):
    """A PyTorch implementation of SequenceArray for dynamic sequence concatenation."""
    
    def __init__(self, dtype: DType, size: Optional[int] = None, dynamic_size: bool = True):
        """Initialize a SequenceArray.
        
        Args:
            dtype: The dtype of the sequence values
            size: Maximum number of sequences to store
            dynamic_size: Whether to allow dynamic resizing
        """
        self._dtype = dtype
        self._size = size
        self._dynamic_size = dynamic_size
        self._sequences: List[Sequence] = []
        self._current_size = 0
    
    @classmethod
    def new(cls, dtype: DType, size: Optional[int] = None, dynamic_size: Optional[bool] = None) -> 'SequenceArray':
        """Create a new SequenceArray."""
        if dynamic_size is None:
            dynamic_size = size is None
        return cls(dtype, size, dynamic_size)
    
    def write(self, index: int, sequence: Sequence) -> 'SequenceArray':
        """Write a sequence at the specified index."""
        if self._size is not None and index >= self._size and not self._dynamic_size:
            raise IndexError(f"Index {index} exceeds fixed size {self._size}")
        
        # Extend list if needed
        while len(self._sequences) <= index:
            self._sequences.append(None)
        
        self._sequences[index] = sequence
        self._current_size = max(self._current_size, index + 1)
        
        return self
    
    def concat(self) -> Sequence:
        """Concatenate all sequences in the array."""
        valid_sequences = [seq for seq in self._sequences[:self._current_size] if seq is not None]
        if not valid_sequences:
            raise ValueError("No sequences to concatenate")
        
        return Sequence.concatenate_sequences(valid_sequences)


class SequenceLayer(SequenceLayerBase, nn.Module):
    """A PyTorch implementation of SequenceLayer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name or self._default_name()
    
    @property
    def name(self) -> str:
        """The layer's name."""
        return self._name
    
    def forward(self, x: Sequence, training: bool = False, 
                initial_state: Optional[State] = None,
                constants: Optional[Constants] = None) -> Sequence:
        """Forward pass - delegates to layer method."""
        return self.layer(x, training, initial_state, constants)
    
    def step(self, x: Sequence, state: State, training: bool, 
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise."""
        raise NotImplementedError("Subclasses must implement step method")
    
    def layer(self, x: Sequence, training: bool, 
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Process this layer layer-wise."""
        raise NotImplementedError("Subclasses must implement layer method")
    
    def step_with_emits(self, x: Sequence, state: State, training: bool,
                        constants: Optional[Constants] = None) -> Tuple[Sequence, State, Emits]:
        """Process this layer step-wise, producing emitted tensors."""
        outputs, state = self.step(x, state, training, constants)
        return outputs, state, ()
    
    def layer_with_emits(self, x: Sequence, training: bool,
                         initial_state: Optional[State] = None,
                         constants: Optional[Constants] = None) -> Tuple[Sequence, Emits]:
        """Process this layer layer-wise, producing emitted tensors."""
        outputs = self.layer(x, training, initial_state, constants)
        return outputs, ()
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec, 
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns the initial state for this SequenceLayer."""
        # Default implementation returns empty state
        return {}
    
    def get_output_shape_for_sequence(self, x: Sequence, 
                                      constants: Optional[Constants] = None) -> Shape:
        """Returns the output shape this layer produces for the provided Sequence."""
        return self.get_output_shape(x.channel_shape, constants)
    
    def get_output_shape(self, input_shape: Shape, 
                         constants: Optional[Constants] = None) -> Shape:
        """Returns the output shape this layer produces for an input shape."""
        # Default implementation preserves shape
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        """Returns the layer's output dtype."""
        # Default implementation preserves dtype
        return input_dtype
    
    def get_output_spec(self, input_spec: ChannelSpec, 
                        constants: Optional[Constants] = None) -> ChannelSpec:
        """Returns the output specification for this layer."""
        output_shape = self.get_output_shape(input_spec.shape, constants)
        output_dtype = self.get_output_dtype(input_spec.dtype)
        return ChannelSpec(output_shape, output_dtype)
    
    def _yield_emits(self, emits: Emits):
        """Yields (layer, emits) tuples to allow associating emits with layers."""
        yield self, emits


class Emitting(EmittingBase, SequenceLayer):
    """A SequenceLayer that emits auxiliary tensors."""
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Step method delegates to step_with_emits."""
        outputs, state, _ = self.step_with_emits(x, state, training, constants)
        return outputs, state
    
    def layer(self, x: Sequence, training: bool,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Layer method delegates to layer_with_emits."""
        outputs, _ = self.layer_with_emits(x, training, initial_state, constants)
        return outputs


# Convenience base classes for common layer types
class PreservesShape(SequenceLayer):
    """Base class for layers that preserve input shape."""
    
    def get_output_shape(self, input_shape: Shape, 
                         constants: Optional[Constants] = None) -> Shape:
        return input_shape


class PreservesType(SequenceLayer):
    """Base class for layers that preserve input type."""
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype


class Stateless(PreservesShape, PreservesType):
    """Base class for stateless layers."""
    
    def supports_step(self) -> bool:
        return True
    
    def get_initial_state(self, batch_size: int, input_spec: ChannelSpec, 
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        return {}
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        output = self.layer(x, training, None, constants)
        return output, state


class StatelessEmitting(Emitting, Stateless):
    """Base class for stateless layers that emit auxiliary tensors."""
    pass


class StatelessPointwise(Stateless):
    """Base class for stateless pointwise layers."""
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1)


class StatelessPointwiseFunctor(StatelessPointwise):
    """Base class for stateless pointwise layers that apply a function."""
    
    def fn(self, values: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Function to apply to values and mask."""
        raise NotImplementedError("Subclasses must implement fn method")
    
    def layer(self, x: Sequence, training: bool,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply the function to the sequence."""
        return x.apply(self.fn)


class Steppable(SequenceLayer):
    """Base class for layers that support step-wise execution."""
    
    def supports_step(self) -> bool:
        return True


# Utility functions for testing and validation
def check_layer(layer: SequenceLayer, x: Sequence, training: bool = False,
                constants: Optional[Constants] = None) -> Sequence:
    """Check that a layer produces valid output."""
    output = layer.layer(x, training, None, constants)
    if not isinstance(output, Sequence):
        raise TypeError(f"Layer must return Sequence, got {type(output)}")
    return output


def check_step(layer: SequenceLayer, x: Sequence, state: State, training: bool = False,
               constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
    """Check that a layer step produces valid output."""
    if not layer.supports_step():
        raise ValueError(f"Layer {layer} does not support step-wise execution")
    
    output, new_state = layer.step(x, state, training, constants)
    if not isinstance(output, Sequence):
        raise TypeError(f"Layer step must return Sequence, got {type(output)}")
    return output, new_state


def check_layer_with_emits(layer: SequenceLayer, x: Sequence, training: bool = False,
                           constants: Optional[Constants] = None) -> Tuple[Sequence, Emits]:
    """Check that a layer with emits produces valid output."""
    output, emits = layer.layer_with_emits(x, training, None, constants)
    if not isinstance(output, Sequence):
        raise TypeError(f"Layer must return Sequence, got {type(output)}")
    return output, emits


def check_step_with_emits(layer: SequenceLayer, x: Sequence, state: State, training: bool = False,
                          constants: Optional[Constants] = None) -> Tuple[Sequence, State, Emits]:
    """Check that a layer step with emits produces valid output."""
    if not layer.supports_step():
        raise ValueError(f"Layer {layer} does not support step-wise execution")
    
    output, new_state, emits = layer.step_with_emits(x, state, training, constants)
    if not isinstance(output, Sequence):
        raise TypeError(f"Layer step must return Sequence, got {type(output)}")
    return output, new_state, emits 