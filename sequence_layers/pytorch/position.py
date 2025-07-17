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
"""Position encoding layers for PyTorch."""

import math
from typing import Any, Callable, Optional, Union, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    StatelessPointwise,
    PreservesShape,
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    'AddTimingSignal',
    'ApplyRotaryPositionalEncoding',
]


def _get_timing_signal_1d(
    length: int,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    start_index: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> Tensor:
    """Gets a bunch of sinusoids of different frequencies.
    
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    
    Args:
        length: Length of timing signal sequence.
        channels: Size of timing embeddings to create.
        min_timescale: Minimum timescale.
        max_timescale: Maximum timescale.
        start_index: Index of first position.
        dtype: Data type of the returned timing signal.
        device: Device for the returned timing signal.
    
    Returns:
        A tensor of timing signals [1, length, channels].
    """
    position = torch.arange(length, dtype=dtype, device=device) + start_index
    num_timescales = channels // 2
    
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
    
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=dtype, device=device) * -log_timescale_increment
    )
    
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    
    # Pad with zeros if channels is odd
    if channels % 2 == 1:
        signal = torch.cat([signal, torch.zeros(length, 1, dtype=dtype, device=device)], dim=1)
    
    return signal.unsqueeze(0)  # Add batch dimension


class AddTimingSignal(PreservesShape, PreservesType, SequenceLayer):
    """Adds sinusoids at varying frequencies to the input channels dimension."""
    
    def __init__(self,
                 min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4,
                 trainable_scale: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.trainable_scale = trainable_scale
        
        # Flag to track if scale has been initialized
        self._scale_initialized = False
        
    def _initialize_scale(self, device: torch.device, dtype: torch.dtype):
        """Initialize scale parameter with proper device and dtype."""
        if not self._scale_initialized:
            if self.trainable_scale:
                self.scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
            else:
                # Use self.register_buffer to create a non-trainable scale
                scale_buffer = torch.ones(1, device=device, dtype=dtype)
                self.register_buffer('scale', scale_buffer)
            self._scale_initialized = True
    
    def _check_inputs(self, x: Sequence):
        """Check that input is floating point."""
        if not x.values.dtype.is_floating_point:
            raise ValueError(f'{type(self).__name__} requires floating point argument.')
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state (current timestep for batch)."""
        # Use CPU as default device since we don't have input tensor context here
        device = torch.device('cpu')
        return torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Apply timing signal step-wise."""
        self._check_inputs(x)
        
        # Initialize scale parameter with proper device/dtype
        self._initialize_scale(x.values.device, x.values.dtype)
        
        current_time = torch.max(state)
        time_steps = x.shape[1]
        
        # Get total number of elements in channel shape
        channel_shape = x.channel_shape
        num_elements = 1
        for dim in channel_shape:
            num_elements *= dim
        
        # Generate timing signal
        timing_signal = _get_timing_signal_1d(
            time_steps,
            num_elements,
            min_timescale=self.min_timescale,
            max_timescale=self.max_timescale,
            start_index=current_time.item(),
            dtype=x.values.dtype,
            device=x.values.device
        )
        
        # Reshape to match channel shape
        timing_signal = timing_signal.view(1, time_steps, *channel_shape)
        
        # Apply scale if available
        if self._scale_initialized:
            timing_signal = timing_signal * self.scale
        
        # Add timing signal to input
        output_values = x.values + timing_signal
        output = Sequence(output_values, x.mask)
        
        new_state = state + time_steps
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply timing signal layer-wise."""
        self._check_inputs(x)
        
        # Initialize scale parameter with proper device/dtype
        self._initialize_scale(x.values.device, x.values.dtype)
        
        time_steps = x.shape[1]
        
        # Get total number of elements in channel shape
        channel_shape = x.channel_shape
        num_elements = 1
        for dim in channel_shape:
            num_elements *= dim
        
        # Generate timing signal
        timing_signal = _get_timing_signal_1d(
            time_steps,
            num_elements,
            min_timescale=self.min_timescale,
            max_timescale=self.max_timescale,
            start_index=0,
            dtype=x.values.dtype,
            device=x.values.device
        )
        
        # Reshape to match channel shape
        timing_signal = timing_signal.view(1, time_steps, *channel_shape)
        
        # Apply scale if available
        if self._scale_initialized:
            timing_signal = timing_signal * self.scale
        
        # Add timing signal to input
        output_values = x.values + timing_signal
        output = Sequence(output_values, x.mask)
        
        return output


class ApplyRotaryPositionalEncoding(PreservesShape, PreservesType, SequenceLayer):
    """Applies Rotary Positional Encodings (RoPE) to the sequence.
    
    See https://arxiv.org/abs/2104.09864 for details.
    """
    
    def __init__(self,
                 max_wavelength: float = 1.0e4,
                 axis: int = -1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.max_wavelength = max_wavelength
        self.axis = axis
    
    def _check_inputs(self, x: Sequence):
        """Check that input is floating point and axis dimension is even."""
        if not x.values.dtype.is_floating_point:
            raise ValueError(f'{type(self).__name__} requires floating point argument.')
        
        # Check that the axis dimension is even
        axis_dim = x.channel_shape[self.axis] if self.axis >= 0 else x.channel_shape[len(x.channel_shape) + self.axis]
        if axis_dim % 2 != 0:
            raise ValueError(f'{type(self).__name__} requires axis dimension to be even, got {axis_dim}')
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns initial state (current timestep for batch)."""
        # Use CPU as default device since we don't have input tensor context here
        device = torch.device('cpu')
        return torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
    
    def _apply_rope(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply rotary positional encoding to tensor."""
        # Resolve axis
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        
        # Get axis dimension
        axis_dim = x.shape[axis]
        
        # Create frequency exponents
        freq_exponents = (2.0 / axis_dim) * torch.arange(axis_dim // 2, dtype=x.dtype, device=x.device)
        timescale = self.max_wavelength ** freq_exponents
        
        # Expand positions to match input shape
        positions = positions.float().to(x.device)
        while positions.ndim < x.ndim:
            positions = positions.unsqueeze(-1)
        
        # Compute angles
        angles = positions / timescale
        
        # Compute sin and cos
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        
        # Split input along axis
        x1, x2 = torch.chunk(x, 2, dim=axis)
        
        # Apply rotation
        rotated_x1 = x1 * cos_angles - x2 * sin_angles
        rotated_x2 = x2 * cos_angles + x1 * sin_angles
        
        # Concatenate back
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=axis)
        
        return rotated_x
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Apply RoPE step-wise."""
        self._check_inputs(x)
        
        time_steps = x.shape[1]
        
        # Get positions for the batch
        positions = state + torch.arange(time_steps, dtype=torch.int32, device=x.values.device)
        
        # Apply RoPE
        rotated_values = self._apply_rope(x.values, positions)
        output = Sequence(rotated_values, x.mask)
        
        new_state = state + time_steps
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply RoPE layer-wise."""
        self._check_inputs(x)
        
        time_steps = x.shape[1]
        
        # Create positions
        positions = torch.arange(time_steps, dtype=torch.int32, device=x.values.device).unsqueeze(0)
        
        # Apply RoPE
        rotated_values = self._apply_rope(x.values, positions)
        output = Sequence(rotated_values, x.mask)
        
        return output 