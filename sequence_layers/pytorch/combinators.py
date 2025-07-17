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
"""Combinator layers for PyTorch."""

import enum
import fractions
import math
from typing import Any, Callable, Optional, Union, Tuple, List, Sequence as TypingSequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    Emitting,
    StatelessPointwise,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    # Core combinators
    'Sequential',
    'Parallel',
    'Residual',
    'Repeat',
    
    # Utility enums
    'CombinationMode',
    
    # Wrapper mixins
    'WrapperMixin',
]


@enum.unique
class CombinationMode(enum.Enum):
    """The type of combination to perform for parallel layers."""
    
    # Stack output of each parallel layer.
    STACK = 'stack'
    # Broadcast-add the output of each parallel layer.
    ADD = 'add'
    # Broadcast-mean the output of each parallel layer.
    MEAN = 'mean'


class WrapperMixin:
    """A wrapper mixin where the wrapped layer properties are unchanged.
    
    Used for layers that modify the wrapped layer's behavior in a way that does
    not change any of the layer properties or utility functions.
    """
    
    child_layer: SequenceLayer
    
    @property
    def supports_step(self) -> bool:
        return self.child_layer.supports_step
    
    @property
    def block_size(self) -> int:
        return self.child_layer.block_size
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return self.child_layer.output_ratio
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.child_layer.get_output_shape(input_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.child_layer.get_output_dtype(input_dtype)
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        return self.child_layer.get_initial_state(batch_size, channel_spec, training, constants)


class Sequential(SequenceLayer):
    """A combinator that processes SequenceLayers serially."""
    
    def __init__(self, layers: TypingSequence[SequenceLayer], name: Optional[str] = None):
        super().__init__(name=name)
        self.layers = nn.ModuleList(layers)
        
        # Validate that all layers are SequenceLayer instances
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SequenceLayer):
                raise ValueError(f'Layer {i} is not a SequenceLayer: {type(layer)}')
    
    @property
    def supports_step(self) -> bool:
        return all(layer.supports_step for layer in self.layers)
    
    @property
    def block_size(self) -> int:
        """The block size is the LCM of all layer block sizes."""
        block_size = 1
        for layer in self.layers:
            block_size = math.lcm(block_size, layer.block_size)
        return block_size
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        """The output ratio is the product of all layer output ratios."""
        ratio = fractions.Fraction(1)
        for layer in self.layers:
            ratio *= layer.output_ratio
        return ratio
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        shape = input_shape
        for layer in self.layers:
            shape = layer.get_output_shape(shape, constants)
        return shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        dtype = input_dtype
        for layer in self.layers:
            dtype = layer.get_output_dtype(dtype)
        return dtype
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state for all layers."""
        spec = channel_spec
        states = []
        for layer in self.layers:
            state = layer.get_initial_state(batch_size, spec, training, constants)
            states.append(state)
            
            # Update spec for next layer
            if hasattr(layer, 'get_output_spec'):
                spec = layer.get_output_spec(spec, constants)
            else:
                # Fallback: create new spec from current layer's output
                output_shape = layer.get_output_shape(spec.shape, constants)
                output_dtype = layer.get_output_dtype(spec.dtype)
                spec = ChannelSpec(output_shape, output_dtype)
        
        return tuple(states)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Applies the layers to a block of inputs step-wise."""
        if not isinstance(state, (tuple, list)) or len(state) != len(self.layers):
            raise ValueError(f'Expected state to be a tuple of length {len(self.layers)}, got: {type(state)}')
        
        new_states = []
        for i, (layer, layer_state) in enumerate(zip(self.layers, state)):
            x, new_state = layer.step(x, layer_state, training, constants)
            new_states.append(new_state)
        
        return x, tuple(new_states)
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Applies the layers to the input sequence layer-wise."""
        for layer in self.layers:
            x = layer.layer(x, training, initial_state, constants)
        return x


class Parallel(SequenceLayer):
    """Applies a sequence of layers in parallel and combines their outputs."""
    
    def __init__(self, 
                 layers: TypingSequence[SequenceLayer], 
                 combination: CombinationMode = CombinationMode.STACK,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.layers = nn.ModuleList(layers)
        self.combination = combination
        
        # Validate that all layers are SequenceLayer instances
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SequenceLayer):
                raise ValueError(f'Layer {i} is not a SequenceLayer: {type(layer)}')
        
        # Validate that all layers have the same output ratio
        if self.layers:
            first_ratio = self.layers[0].output_ratio
            for i, layer in enumerate(self.layers[1:], 1):
                if layer.output_ratio != first_ratio:
                    raise ValueError(f'All layers must have the same output ratio. '
                                   f'Layer 0 has ratio {first_ratio}, layer {i} has ratio {layer.output_ratio}')
    
    @property
    def supports_step(self) -> bool:
        return all(layer.supports_step for layer in self.layers)
    
    @property
    def block_size(self) -> int:
        """The block size is the LCM of all layer block sizes."""
        if not self.layers:
            return 1
        
        block_size = 1
        for layer in self.layers:
            block_size = math.lcm(block_size, layer.block_size)
        return block_size
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        """All layers must have the same output ratio."""
        if not self.layers:
            return fractions.Fraction(1)
        return self.layers[0].output_ratio
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if not self.layers:
            return input_shape
        
        # Get output shapes from all layers
        output_shapes = []
        for layer in self.layers:
            output_shapes.append(layer.get_output_shape(input_shape, constants))
        
        # Validate that all output shapes are the same for non-stack combinations
        if self.combination != CombinationMode.STACK:
            first_shape = output_shapes[0]
            for i, shape in enumerate(output_shapes[1:], 1):
                if shape != first_shape:
                    raise ValueError(f'All layers must have the same output shape for {self.combination} combination. '
                                   f'Layer 0 has shape {first_shape}, layer {i} has shape {shape}')
            return first_shape
        else:
            # For STACK combination, add a new dimension at the beginning
            first_shape = output_shapes[0]
            return (len(self.layers),) + first_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        if not self.layers:
            return input_dtype
        
        # Get output dtypes from all layers
        output_dtypes = []
        for layer in self.layers:
            output_dtypes.append(layer.get_output_dtype(input_dtype))
        
        # For combination modes other than STACK, find common dtype
        if self.combination != CombinationMode.STACK:
            # Find the result dtype that can represent all output dtypes
            result_dtype = output_dtypes[0]
            for dtype in output_dtypes[1:]:
                # Use torch.result_type to find compatible dtype
                dummy_tensor1 = torch.tensor(0, dtype=result_dtype)
                dummy_tensor2 = torch.tensor(0, dtype=dtype)
                result_dtype = torch.result_type(dummy_tensor1, dummy_tensor2)
            return result_dtype
        else:
            # For STACK, return the first layer's output dtype
            return output_dtypes[0]
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state for all layers."""
        states = []
        for layer in self.layers:
            state = layer.get_initial_state(batch_size, channel_spec, training, constants)
            states.append(state)
        return tuple(states)
    
    def _combine_outputs(self, outputs: List[Sequence]) -> Sequence:
        """Combine outputs from parallel layers according to combination mode."""
        if not outputs:
            # If no outputs, this should not happen in normal operation
            # but we'll handle it gracefully by returning None
            return None
        
        # All outputs should have the same mask structure
        first_mask = outputs[0].mask
        
        if self.combination == CombinationMode.STACK:
            # For STACK, we stack along the channel dimension
            # Each output has shape (batch, time, ...)
            # We want to stack them to get (batch, time, num_layers, ...)
            combined_values = torch.stack([output.values for output in outputs], dim=2)
            combined_mask = first_mask  # Mask shape stays (batch, time)
        elif self.combination == CombinationMode.ADD:
            # Sum across the outputs
            stacked_values = torch.stack([output.values for output in outputs], dim=0)
            combined_values = torch.sum(stacked_values, dim=0)
            combined_mask = first_mask
        elif self.combination == CombinationMode.MEAN:
            # Mean across the outputs
            stacked_values = torch.stack([output.values for output in outputs], dim=0)
            combined_values = torch.mean(stacked_values, dim=0)
            combined_mask = first_mask
        else:
            raise ValueError(f"Unknown combination mode: {self.combination}")
        
        return Sequence(combined_values, combined_mask)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Applies the layers to a block of inputs in parallel."""
        if not self.layers:
            # Empty parallel returns the input unchanged
            return x, ()
        
        if not isinstance(state, (tuple, list)) or len(state) != len(self.layers):
            raise ValueError(f'Expected state to be a tuple of length {len(self.layers)}, got: {type(state)}')
        
        outputs = []
        new_states = []
        
        for layer, layer_state in zip(self.layers, state):
            output, new_state = layer.step(x, layer_state, training, constants)
            outputs.append(output)
            new_states.append(new_state)
        
        combined_output = self._combine_outputs(outputs)
        return combined_output, tuple(new_states)
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Applies the layers to the input sequence in parallel."""
        if not self.layers:
            # Empty parallel returns the input unchanged
            return x
        
        outputs = []
        
        for layer in self.layers:
            output = layer.layer(x, training, initial_state, constants)
            outputs.append(output)
        
        return self._combine_outputs(outputs)


class Residual(SequenceLayer):
    """A residual wrapper that computes `y = layers(x) + shortcut(x)`."""
    
    def __init__(self, 
                 layers: Union[SequenceLayer, TypingSequence[SequenceLayer]],
                 shortcut: Optional[SequenceLayer] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Handle both single layer and sequence of layers
        if isinstance(layers, SequenceLayer):
            self.main_layers = layers
        else:
            self.main_layers = Sequential(layers)
        
        # Default shortcut is identity
        if shortcut is None:
            from sequence_layers.pytorch.simple import Identity
            self.shortcut = Identity()
        else:
            self.shortcut = shortcut
        
        # Validate that output ratios match
        if self.main_layers.output_ratio != self.shortcut.output_ratio:
            raise ValueError(f'Main layers and shortcut must have the same output ratio. '
                           f'Main: {self.main_layers.output_ratio}, Shortcut: {self.shortcut.output_ratio}')
    
    @property
    def supports_step(self) -> bool:
        return self.main_layers.supports_step and self.shortcut.supports_step
    
    @property
    def block_size(self) -> int:
        return math.lcm(self.main_layers.block_size, self.shortcut.block_size)
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return self.main_layers.output_ratio
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        main_shape = self.main_layers.get_output_shape(input_shape, constants)
        shortcut_shape = self.shortcut.get_output_shape(input_shape, constants)
        
        # Shapes must be compatible for addition
        if main_shape != shortcut_shape:
            # Try to see if they're broadcastable
            try:
                # Create dummy tensors to test broadcastability
                dummy_main = torch.zeros((1, 1) + main_shape)
                dummy_shortcut = torch.zeros((1, 1) + shortcut_shape)
                result = dummy_main + dummy_shortcut
                return result.shape[2:]  # Remove batch and time dimensions
            except RuntimeError:
                raise ValueError(f'Main output shape {main_shape} and shortcut output shape {shortcut_shape} '
                               f'are not compatible for addition')
        
        return main_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        main_dtype = self.main_layers.get_output_dtype(input_dtype)
        shortcut_dtype = self.shortcut.get_output_dtype(input_dtype)
        
        # Find compatible dtype for addition
        dummy_main = torch.tensor(0, dtype=main_dtype)
        dummy_shortcut = torch.tensor(0, dtype=shortcut_dtype)
        result_dtype = torch.result_type(dummy_main, dummy_shortcut)
        
        return result_dtype
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        main_state = self.main_layers.get_initial_state(batch_size, channel_spec, training, constants)
        shortcut_state = self.shortcut.get_initial_state(batch_size, channel_spec, training, constants)
        return (main_state, shortcut_state)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Applies residual connection step-wise."""
        main_state, shortcut_state = state
        
        # Process main path
        main_output, new_main_state = self.main_layers.step(x, main_state, training, constants)
        
        # Process shortcut path
        shortcut_output, new_shortcut_state = self.shortcut.step(x, shortcut_state, training, constants)
        
        # Add them together
        output_values = main_output.values + shortcut_output.values
        
        # Use the main path's mask (they should be the same)
        output = Sequence(output_values, main_output.mask)
        
        return output, (new_main_state, new_shortcut_state)
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Applies residual connection layer-wise."""
        # Process main path
        main_output = self.main_layers.layer(x, training, initial_state, constants)
        
        # Process shortcut path
        shortcut_output = self.shortcut.layer(x, training, initial_state, constants)
        
        # Add them together
        output_values = main_output.values + shortcut_output.values
        
        # Use the main path's mask
        output = Sequence(output_values, main_output.mask)
        
        return output


class Repeat(SequenceLayer):
    """A combinator that repeats the specified SequenceLayer N times."""
    
    def __init__(self, 
                 layer: SequenceLayer, 
                 num_repeats: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        if num_repeats <= 0:
            raise ValueError(f'num_repeats must be positive, got: {num_repeats}')
        
        self.child_layer = layer
        self.num_repeats = num_repeats
        
        # Validate that the layer preserves shape and dtype (required for repeat)
        if layer.output_ratio != fractions.Fraction(1):
            raise ValueError(f'Repeated layer must have output_ratio of 1, got: {layer.output_ratio}')
    
    @property
    def supports_step(self) -> bool:
        return self.child_layer.supports_step
    
    @property
    def block_size(self) -> int:
        return self.child_layer.block_size
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return self.child_layer.output_ratio
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Since we require output_ratio = 1, the shape should be preserved
        return self.child_layer.get_output_shape(input_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.child_layer.get_output_dtype(input_dtype)
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        # Get the initial state from the child layer
        return self.child_layer.get_initial_state(batch_size, channel_spec, training, constants)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Applies the layer num_repeats times step-wise."""
        current_x = x
        current_state = state
        
        for _ in range(self.num_repeats):
            current_x, current_state = self.child_layer.step(current_x, current_state, training, constants)
        
        return current_x, current_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Applies the layer num_repeats times layer-wise."""
        current_x = x
        
        for _ in range(self.num_repeats):
            current_x = self.child_layer.layer(current_x, training, initial_state, constants)
        
        return current_x 