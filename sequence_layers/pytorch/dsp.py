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
"""DSP layers for PyTorch."""

import fractions
import math
from typing import Any, Callable, Optional, Union, Tuple, List, Literal
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    Stateless,
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
    # Basic DSP layers
    'Delay',
    'Frame',
    'OverlapAdd',
    'Window',
    
    # FFT family
    'FFT',
    'IFFT',
    'RFFT',
    'IRFFT',
    
    # STFT family
    'STFT',
    'InverseSTFT',
    
    # Spectral processing
    'LinearToMelSpectrogram',
]


# =============================================================================
# Utility Functions
# =============================================================================

def _validate_fft_params(fft_length: Optional[int], padding: str, axis: int):
    """Validate FFT parameters."""
    if fft_length is not None and fft_length <= 0:
        raise ValueError(f'fft_length must be positive, got: {fft_length}')
    if padding not in ('center', 'right'):
        raise ValueError(f'padding must be "center" or "right", got: {padding}')
    if axis in (0, 1):
        raise ValueError(f'Computing FFTs over batch or time dimension is not allowed, got axis: {axis}')


def _pad_or_truncate_for_fft(x: Sequence, fft_length: int, padding: str, axis: int) -> Sequence:
    """Pad or truncate input for FFT."""
    current_length = x.shape[axis]
    
    if current_length == fft_length:
        return x
    elif current_length < fft_length:
        # Pad
        pad_amount = fft_length - current_length
        if padding == 'right':
            pad_spec = [0] * (2 * x.values.ndim)
            pad_spec[2 * (x.values.ndim - 1 - axis)] = pad_amount
        else:  # left padding
            pad_spec = [0] * (2 * x.values.ndim)
            pad_spec[2 * (x.values.ndim - 1 - axis) + 1] = pad_amount
        
        padded_values = F.pad(x.values, pad_spec)
        return Sequence(padded_values, x.mask)
    else:
        # Truncate
        if padding == 'right':
            # Take the first fft_length elements
            slices = [slice(None)] * x.values.ndim
            slices[axis] = slice(0, fft_length)
        else:  # left padding
            # Take the last fft_length elements
            slices = [slice(None)] * x.values.ndim
            slices[axis] = slice(current_length - fft_length, current_length)
        
        truncated_values = x.values[tuple(slices)]
        return Sequence(truncated_values, x.mask)


# Fix ndim attribute references and other bugs
def _normalize_axis(axis: int, ndim: int) -> int:
    """Normalize axis to positive index."""
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f'Axis {axis} is out of range for tensor with {ndim} dimensions')
    return axis


# =============================================================================
# Window Functions
# =============================================================================

def hann_window(window_length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a Hann window."""
    return torch.hann_window(window_length, periodic=False, dtype=dtype)


def hamming_window(window_length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a Hamming window."""
    return torch.hamming_window(window_length, periodic=False, dtype=dtype)


def blackman_window(window_length: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a Blackman window."""
    return torch.blackman_window(window_length, periodic=False, dtype=dtype)


# =============================================================================
# Basic DSP Layers
# =============================================================================

class Delay(SequenceLayer):
    """A layer that delays its input by length timesteps."""
    
    def __init__(self, length: int, name: Optional[str] = None):
        super().__init__(name=name)
        if length < 0:
            raise ValueError(f'Negative delay ({length}) is not supported by Delay layer.')
        self.length = length
    
    @property
    def supports_step(self) -> bool:
        return True
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1, 1)
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        if self.length == 0:
            return None
        
        # Create buffer for delayed values
        state_shape = (batch_size, self.length, *channel_spec.shape)
        state_values = torch.zeros(state_shape, dtype=channel_spec.dtype)
        state_mask = torch.ones(batch_size, self.length, dtype=torch.bool)
        
        return Sequence(state_values, state_mask)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        if self.length == 0:
            return x, state
        
        # Concatenate state and input
        if state is None:
            state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
        
        combined = state.concatenate(x)
        
        # Split into output and new state
        output = combined[:, :x.shape[1]]
        new_state = combined[:, x.shape[1]:]
        
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        if self.length == 0:
            return x
        
        # Pad at the beginning and truncate at the end
        padded = x.pad_time(self.length, 0, valid=True)
        return padded[:, :x.shape[1]]


class Frame(SequenceLayer):
    """Produce a sequence of overlapping frames of the input sequence."""
    
    def __init__(self, 
                 frame_length: int,
                 frame_step: int,
                 padding: str = 'reverse_causal_valid',
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        if frame_length <= 0:
            raise ValueError(f'frame_length must be positive, got: {frame_length}')
        if frame_step <= 0:
            raise ValueError(f'frame_step must be positive, got: {frame_step}')
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.padding = padding
        
        # Validate padding
        valid_padding_modes = ['causal', 'reverse_causal', 'causal_valid', 'reverse_causal_valid', 'semicausal', 'valid']
        if padding not in valid_padding_modes:
            raise ValueError(f'Invalid padding mode: {padding}. Must be one of {valid_padding_modes}')
    
    @property
    def supports_step(self) -> bool:
        return self.padding in ['causal', 'reverse_causal', 'causal_valid', 'reverse_causal_valid', 'semicausal']
    
    @property
    def block_size(self) -> int:
        return self.frame_step
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1, self.frame_step)
    
    @property
    def input_latency(self) -> int:
        """Number of timesteps of input latency for this layer."""
        if self.padding == 'reverse_causal':
            return self.frame_length - 1
        return 0
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        if not self.supports_step:
            return None
        
        # Buffer for causal padding
        buffer_width = self.frame_length - 1
        if buffer_width <= 0:
            return None
        
        # For reverse_causal, we don't pre-allocate padding - start with empty state
        # The padding will be handled differently in step-wise execution
        if self.padding == 'reverse_causal':
            return None
        
        # For other causal modes, pre-allocate buffer
        buffer_shape = (batch_size, buffer_width, *channel_spec.shape)
        buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype)
        buffer_mask = torch.zeros(batch_size, buffer_width, dtype=torch.bool)
        
        return Sequence(buffer_values, buffer_mask)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return (self.frame_length,) + input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        if not self.supports_step:
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
        
        # Handle different padding modes
        if self.padding in ['causal', 'causal_valid', 'semicausal']:
            # For causal modes, concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Extract frames
            frames = []
            frame_masks = []
            for i in range(0, combined.shape[1] - self.frame_length + 1, self.frame_step):
                frame = combined[:, i:i + self.frame_length]
                frames.append(frame.values)
                # Create mask for this frame (all True if we have a valid frame)
                frame_mask = torch.ones(frame.shape[0], dtype=torch.bool, device=frame.device)
                frame_masks.append(frame_mask)
            
            if frames:
                # Stack frames: (batch, num_frames, frame_length, channels)
                output_values = torch.stack(frames, dim=1)
                output_mask = torch.stack(frame_masks, dim=1)
                output = Sequence(output_values, output_mask)
            else:
                # No frames produced
                batch_size = x.shape[0]
                output_shape = (batch_size, 0, self.frame_length, *x.channel_shape)
                output_values = torch.empty(output_shape, dtype=x.dtype, device=x.device)
                output_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=x.device)
                output = Sequence(output_values, output_mask)
            
            # Update state
            if self.frame_length > 1:
                buffer_width = self.frame_length - 1
                new_state = combined[:, -buffer_width:]
            else:
                new_state = None
                
        elif self.padding == 'reverse_causal':
            # For reverse causal, we need future context - delay output
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # For reverse causal padding, we need to delay output until we have
            # enough future context. We should output all frames that become ready
            # with the current input batch.
            
            frames = []
            frame_masks = []
            
            # We can only produce frames at positions where we have enough future context
            total_timesteps_seen = combined.shape[1]
            current_input_timesteps = x.shape[1]
            
            # For step-wise execution, we need to output all frames that become ready
            # with this input batch. A frame at position i becomes ready when we have
            # seen at least timestep i + frame_length - 1
            
            # The frames that become ready with this input are those at positions:
            # max(0, total_timesteps_seen - current_input_timesteps - frame_length + 1) to (total_timesteps_seen - frame_length)
            start_pos = max(0, total_timesteps_seen - current_input_timesteps - self.frame_length + 1)
            end_pos = total_timesteps_seen - self.frame_length + 1
            
            # Generate all frames that are ready and at valid frame_step positions
            for frame_pos in range(start_pos, end_pos):
                if frame_pos >= 0 and frame_pos % self.frame_step == 0:
                    frame = combined[:, frame_pos:frame_pos + self.frame_length]
                    frames.append(frame.values)
                    frame_mask = torch.ones(frame.shape[0], dtype=torch.bool, device=frame.device)
                    frame_masks.append(frame_mask)
            
            if frames:
                output_values = torch.stack(frames, dim=1)
                output_mask = torch.stack(frame_masks, dim=1)
                output = Sequence(output_values, output_mask)
            else:
                # No frames ready this step
                batch_size = x.shape[0]
                output_shape = (batch_size, 0, self.frame_length, *x.channel_shape)
                output_values = torch.empty(output_shape, dtype=x.dtype, device=x.device)
                output_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=x.device)
                output = Sequence(output_values, output_mask)
            
            # Update state: keep enough context for future frames
            # We need to keep at least frame_length timesteps
            if combined.shape[1] > self.frame_length:
                # Keep only what we need to avoid growing state indefinitely
                keep_size = min(combined.shape[1], self.frame_length + self.frame_step)
                new_state = combined[:, -keep_size:]
            else:
                new_state = combined
                
        else:
            raise ValueError(f"Unsupported padding mode for step-wise execution: {self.padding}")
        
        return output, new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # Apply padding based on mode
        if self.padding == 'causal':
            x_padded = x.pad_time(self.frame_length - 1, 0, valid=False)
        elif self.padding == 'reverse_causal':
            x_padded = x.pad_time(0, self.frame_length - 1, valid=False)
        elif self.padding == 'semicausal':
            left_pad = (self.frame_length - 1) // 2
            right_pad = self.frame_length - 1 - left_pad
            x_padded = x.pad_time(left_pad, right_pad, valid=False)
        else:
            x_padded = x
        
        # Extract frames using unfold
        # unfold(dimension, size, step) -> (batch, num_frames, channels, frame_length)
        unfolded = x_padded.values.unfold(1, self.frame_length, self.frame_step)
        
        # Transpose to get (batch, num_frames, frame_length, channels)
        # This gives us channel_shape (frame_length, channels)
        transposed = unfolded.transpose(-1, -2)
        
        # Create output mask
        batch_size, num_frames = transposed.shape[:2]
        output_mask = torch.ones(batch_size, num_frames, dtype=torch.bool, device=x.device)
        
        return Sequence(transposed, output_mask)


class OverlapAdd(SequenceLayer):
    """Overlap-add reconstruction from frames."""
    
    def __init__(self, 
                 frame_length: int,
                 frame_step: int,
                 padding: str = 'causal',
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        if frame_length <= 0:
            raise ValueError(f'frame_length must be positive, got: {frame_length}')
        if frame_step <= 0:
            raise ValueError(f'frame_step must be positive, got: {frame_step}')
        if frame_length < frame_step:
            raise ValueError('frame_length must be at least frame_step.')
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.padding = padding
    
    @property
    def supports_step(self) -> bool:
        return self.padding == 'causal'
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(self.frame_step, 1)
    
    @property
    def output_latency(self) -> int:
        """Number of additional timesteps produced by layer-wise execution compared to step-wise."""
        return self.frame_length - self.frame_step
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        if not self.supports_step:
            return None
        
        # Buffer for overlap-add
        buffer_width = max(0, self.frame_length - self.frame_step)
        if buffer_width <= 0:
            return None
        
        # Expected input shape is (frame_length, ...)
        output_shape = channel_spec.shape[1:]  # Remove frame_length dimension
        buffer_shape = (batch_size, buffer_width, *output_shape)
        buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype)
        
        return buffer_values
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Input shape is (frame_length, ...), output shape is (...)
        if not input_shape or input_shape[0] != self.frame_length:
            raise ValueError(f'Expected input shape (frame_length={self.frame_length}, ...), got: {input_shape}')
        return input_shape[1:]
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        if not self.supports_step:
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
        
        # x has shape (batch, time, frame_length, ...)
        batch_size, time_steps, frame_length, *channel_shape = x.shape
        
        # For step-wise execution, we need to produce the same total timesteps as layer-wise
        # Layer-wise produces: (total_frames - 1) * frame_step + frame_length
        # But step-wise doesn't know total_frames, so we use a different approach
        
        # Each frame produces frame_step timesteps, but we need to handle the final overlap
        # We'll produce extra timesteps to match layer-wise behavior
        base_output_length = time_steps * self.frame_step
        
        # Add extra timesteps to account for the final frame overlap
        # This ensures step-wise produces the same total timesteps as layer-wise
        extra_timesteps = self.frame_length - self.frame_step
        output_length = base_output_length
        
        output_shape = (batch_size, output_length, *channel_shape)
        output_values = x.values.new_zeros(output_shape)
        
        # Process each frame using overlap-add
        for t in range(time_steps):
            frame = x.values[:, t, :, ...]  # (batch, frame_length, ...)
            frame_start = t * self.frame_step
            
            # Add state contribution (only for the first frame)
            if t == 0 and state is not None:
                state_length = state.shape[1]
                # Ensure state has the same dtype as output
                if state.dtype != output_values.dtype:
                    state = state.to(output_values.dtype)
                
                # Add overlap from previous step's state
                overlap_length = min(state_length, self.frame_step, output_length - frame_start)
                if overlap_length > 0:
                    output_values[:, frame_start:frame_start + overlap_length, ...] += state[:, :overlap_length, ...]
            
            # Add frame contribution - only contribute frame_step timesteps, not frame_length
            frame_contribution_length = min(self.frame_step, output_length - frame_start)
            if frame_contribution_length > 0:
                output_values[:, frame_start:frame_start + frame_contribution_length, ...] += frame[:, :frame_contribution_length, ...]
        
        # Update state for next step
        if self.frame_length > self.frame_step and time_steps > 0:
            # Keep the tail of the last frame for overlap with next step
            last_frame = x.values[:, -1, :, ...]
            tail_length = self.frame_length - self.frame_step
            
            if tail_length > 0:
                new_state = last_frame[:, self.frame_step:self.frame_step + tail_length, ...]
                
                # Pad with zeros if necessary
                if new_state.shape[1] < tail_length:
                    pad_length = tail_length - new_state.shape[1]
                    padding = torch.zeros(batch_size, pad_length, *channel_shape,
                                        dtype=new_state.dtype, device=new_state.device)
                    new_state = torch.cat([new_state, padding], dim=1)
            else:
                new_state = None
        else:
            new_state = None
        
        # Create output mask
        output_mask = torch.ones(batch_size, output_length, dtype=torch.bool, device=x.device)
        
        return Sequence(output_values, output_mask), new_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # x has shape (batch, time, frame_length, ...)
        batch_size, time_steps, frame_length, *channel_shape = x.shape
        
        # Calculate output length
        output_length = (time_steps - 1) * self.frame_step + frame_length
        output_shape = (batch_size, output_length, *channel_shape)
        # Use new_zeros to preserve gradient tracking from input
        output_values = x.values.new_zeros(output_shape)
        
        # Overlap-add each frame
        for t in range(time_steps):
            frame = x.values[:, t, :, ...]  # (batch, frame_length, ...)
            start_pos = t * self.frame_step
            end_pos = start_pos + frame_length
            
            output_values[:, start_pos:end_pos, ...] += frame
        
        # Create output mask
        output_mask = torch.ones(batch_size, output_length, dtype=torch.bool, device=x.device)
        
        return Sequence(output_values, output_mask)


class Window(StatelessPointwise):
    """Apply a window function to the input."""
    
    def __init__(self, 
                 window_fn: Callable[[int], torch.Tensor],
                 axis: int = -1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.window_fn = window_fn
        self.axis = axis
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # Normalize axis
        axis = _normalize_axis(self.axis, x.values.ndim)
        window_length = x.shape[axis]
        
        # Create window
        window = self.window_fn(window_length).to(x.device)
        
        # Reshape window to broadcast correctly
        window_shape = [1] * x.values.ndim
        window_shape[axis] = window_length
        window = window.reshape(window_shape)
        
        # Apply window
        windowed_values = x.values * window
        
        return Sequence(windowed_values, x.mask)


# =============================================================================
# FFT Family
# =============================================================================

class FFT(Stateless):
    """Fast Fourier Transform layer."""
    
    def __init__(self, 
                 fft_length: Optional[int] = None,
                 axis: int = -1,
                 padding: str = 'right',
                 name: Optional[str] = None):
        super().__init__(name=name)
        _validate_fft_params(fft_length, padding, axis)
        
        self.fft_length = fft_length
        self.axis = axis
        self.padding = padding
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        output_shape = list(input_shape)
        axis = _normalize_axis(self.axis, len(input_shape) + 2) - 2  # Account for batch and time dims
        output_shape[axis] = self.fft_length or input_shape[axis]
        return tuple(output_shape)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        # FFT always produces complex output
        if input_dtype == torch.float32:
            return torch.complex64
        elif input_dtype == torch.float64:
            return torch.complex128
        else:
            return torch.complex64
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim <= 2:
            raise ValueError('FFT requires an input of rank at least 3.')
        
        # Normalize axis
        axis = _normalize_axis(self.axis, x.values.ndim)
        
        # Determine FFT length
        fft_length = self.fft_length or x.shape[axis]
        
        # Pad or truncate input
        x_processed = _pad_or_truncate_for_fft(x, fft_length, self.padding, axis)
        
        # Apply FFT
        fft_values = torch.fft.fft(x_processed.values, n=fft_length, dim=axis)
        
        return Sequence(fft_values, x_processed.mask)


class IFFT(Stateless):
    """Inverse Fast Fourier Transform layer."""
    
    def __init__(self, 
                 fft_length: Optional[int] = None,
                 frame_length: Optional[int] = None,
                 axis: int = -1,
                 padding: str = 'right',
                 name: Optional[str] = None):
        super().__init__(name=name)
        _validate_fft_params(fft_length, padding, axis)
        
        self.fft_length = fft_length
        self.frame_length = frame_length
        self.axis = axis
        self.padding = padding
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        output_shape = list(input_shape)
        axis = _normalize_axis(self.axis, len(input_shape) + 2) - 2  # Account for batch and time dims
        output_shape[axis] = self.frame_length or input_shape[axis]
        return tuple(output_shape)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        # IFFT produces real output for complex input
        if input_dtype == torch.complex64:
            return torch.float32
        elif input_dtype == torch.complex128:
            return torch.float64
        else:
            return torch.float32
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim <= 2:
            raise ValueError('IFFT requires an input of rank at least 3.')
        
        # Normalize axis
        axis = _normalize_axis(self.axis, x.values.ndim)
        
        # Apply IFFT
        ifft_values = torch.fft.ifft(x.values, n=self.fft_length, dim=axis)
        
        # Convert to real if input was complex
        if x.dtype.is_complex:
            ifft_values = ifft_values.real
        
        # Pad or truncate to frame_length
        if self.frame_length is not None:
            x_processed = _pad_or_truncate_for_fft(
                Sequence(ifft_values, x.mask), 
                self.frame_length, 
                self.padding, 
                axis
            )
            return x_processed
        
        return Sequence(ifft_values, x.mask)


class RFFT(Stateless):
    """Real Fast Fourier Transform layer."""
    
    def __init__(self, 
                 fft_length: Optional[int] = None,
                 axis: int = -1,
                 padding: str = 'right',
                 name: Optional[str] = None):
        super().__init__(name=name)
        _validate_fft_params(fft_length, padding, axis)
        
        self.fft_length = fft_length
        self.axis = axis
        self.padding = padding
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        output_shape = list(input_shape)
        axis = _normalize_axis(self.axis, len(input_shape) + 2) - 2  # Account for batch and time dims
        fft_length = self.fft_length or input_shape[axis]
        output_shape[axis] = fft_length // 2 + 1
        return tuple(output_shape)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        # RFFT produces complex output
        if input_dtype == torch.float32:
            return torch.complex64
        elif input_dtype == torch.float64:
            return torch.complex128
        else:
            return torch.complex64
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim <= 2:
            raise ValueError('RFFT requires an input of rank at least 3.')
        
        # Normalize axis
        axis = _normalize_axis(self.axis, x.values.ndim)
        
        # Determine FFT length
        fft_length = self.fft_length or x.shape[axis]
        
        # Pad or truncate input
        x_processed = _pad_or_truncate_for_fft(x, fft_length, self.padding, axis)
        
        # Apply RFFT
        rfft_values = torch.fft.rfft(x_processed.values, n=fft_length, dim=axis)
        
        return Sequence(rfft_values, x_processed.mask)


class IRFFT(Stateless):
    """Inverse Real Fast Fourier Transform layer."""
    
    def __init__(self, 
                 fft_length: Optional[int] = None,
                 frame_length: Optional[int] = None,
                 axis: int = -1,
                 padding: str = 'right',
                 name: Optional[str] = None):
        super().__init__(name=name)
        _validate_fft_params(fft_length, padding, axis)
        
        self.fft_length = fft_length
        self.frame_length = frame_length
        self.axis = axis
        self.padding = padding
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        output_shape = list(input_shape)
        axis = _normalize_axis(self.axis, len(input_shape) + 2) - 2  # Account for batch and time dims
        # If fft_length is specified, use it; otherwise infer from input
        if self.fft_length is not None:
            output_length = self.fft_length
        else:
            output_length = 2 * (input_shape[axis] - 1)
        output_shape[axis] = self.frame_length or output_length
        return tuple(output_shape)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        # IRFFT produces real output
        if input_dtype == torch.complex64:
            return torch.float32
        elif input_dtype == torch.complex128:
            return torch.float64
        else:
            return torch.float32
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim <= 2:
            raise ValueError('IRFFT requires an input of rank at least 3.')
        
        # Normalize axis
        axis = _normalize_axis(self.axis, x.values.ndim)
        
        # Apply IRFFT
        irfft_values = torch.fft.irfft(x.values, n=self.fft_length, dim=axis)
        
        # Pad or truncate to frame_length
        if self.frame_length is not None:
            x_processed = _pad_or_truncate_for_fft(
                Sequence(irfft_values, x.mask), 
                self.frame_length, 
                self.padding, 
                axis
            )
            return x_processed
        
        return Sequence(irfft_values, x.mask)


# =============================================================================
# STFT Family
# =============================================================================

class STFT(SequenceLayer):
    """Short-time Fourier Transform layer."""
    
    def __init__(self,
                 frame_length: int,
                 frame_step: int,
                 fft_length: int,
                 window_fn: Optional[Callable[[int], torch.Tensor]] = None,
                 time_padding: str = 'reverse_causal_valid',
                 fft_padding: str = 'right',
                 output_magnitude: bool = False,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.window_fn = window_fn or hann_window
        self.time_padding = time_padding
        self.fft_padding = fft_padding
        self.output_magnitude = output_magnitude
        
        # Create sub-layers
        self.framer = Frame(frame_length, frame_step, time_padding)
        self.rfft = RFFT(fft_length, axis=2, padding=fft_padding)
    
    @property
    def supports_step(self) -> bool:
        return self.framer.supports_step
    
    @property
    def block_size(self) -> int:
        return self.framer.block_size
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return self.framer.output_ratio
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        return self.framer.get_initial_state(batch_size, channel_spec, training, constants)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Frame output shape: (frame_length, ...)
        frame_shape = self.framer.get_output_shape(input_shape, constants)
        # RFFT output shape: (fft_length // 2 + 1, ...)
        return self.rfft.get_output_shape(frame_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        rfft_dtype = self.rfft.get_output_dtype(input_dtype)
        if self.output_magnitude:
            # Convert complex to real for magnitude
            if rfft_dtype == torch.complex64:
                return torch.float32
            elif rfft_dtype == torch.complex128:
                return torch.float64
            else:
                return torch.float32
        return rfft_dtype
    
    def _apply_window(self, x: Sequence) -> Sequence:
        """Apply window function to frames."""
        # x has shape (batch, time, frame_length, ...)
        if self.window_fn is None:
            return x
        
        window = self.window_fn(self.frame_length).to(x.device)
        # Reshape window to broadcast: (1, 1, frame_length, 1, ...)
        window_shape = [1, 1, self.frame_length] + [1] * (x.values.ndim - 3)
        window = window.reshape(window_shape)
        
        windowed_values = x.values * window
        return Sequence(windowed_values, x.mask)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        # Frame the input
        framed, state = self.framer.step(x, state, training, constants)
        
        # Apply window
        windowed = self._apply_window(framed)
        
        # Apply RFFT
        stft_output, _ = self.rfft.step(windowed, None, training, constants)
        
        # Apply magnitude if requested
        if self.output_magnitude:
            stft_values = torch.abs(stft_output.values)
            stft_output = Sequence(stft_values, stft_output.mask)
        
        return stft_output, state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # Frame the input
        framed = self.framer.layer(x, training, initial_state, constants)
        
        # Apply window
        windowed = self._apply_window(framed)
        
        # Apply RFFT
        stft_output = self.rfft.layer(windowed, training, None, constants)
        
        # Apply magnitude if requested
        if self.output_magnitude:
            stft_values = torch.abs(stft_output.values)
            stft_output = Sequence(stft_values, stft_output.mask)
        
        return stft_output


class InverseSTFT(SequenceLayer):
    """Inverse Short-time Fourier Transform layer."""
    
    def __init__(self,
                 frame_length: int,
                 frame_step: int,
                 fft_length: int,
                 window_fn: Optional[Callable[[int], torch.Tensor]] = None,
                 time_padding: str = 'causal',
                 fft_padding: str = 'right',
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.window_fn = window_fn or hann_window
        self.time_padding = time_padding
        self.fft_padding = fft_padding
        
        # Create sub-layers
        self.irfft = IRFFT(fft_length, frame_length, axis=2, padding=fft_padding)
        self.overlap_add = OverlapAdd(frame_length, frame_step, time_padding)
    
    @property
    def supports_step(self) -> bool:
        return self.overlap_add.supports_step
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return self.overlap_add.output_ratio
    
    @property
    def output_latency(self) -> int:
        """Number of additional timesteps produced by layer-wise execution compared to step-wise."""
        return self.overlap_add.output_latency
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        # Channel spec should be (fft_length // 2 + 1, ...)
        # We need to get the output spec of IRFFT for overlap_add
        irfft_shape = (self.frame_length,) + channel_spec.shape[1:]
        irfft_spec = ChannelSpec(irfft_shape, channel_spec.dtype)
        return self.overlap_add.get_initial_state(batch_size, irfft_spec, training, constants)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Input shape: (fft_length // 2 + 1, ...)
        # IRFFT output shape: (frame_length, ...)
        irfft_shape = self.irfft.get_output_shape(input_shape, constants)
        # OverlapAdd output shape: (...)
        return self.overlap_add.get_output_shape(irfft_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.irfft.get_output_dtype(input_dtype)
    
    def _apply_window(self, x: Sequence) -> Sequence:
        """Apply window function to frames."""
        # x has shape (batch, time, frame_length, ...)
        if self.window_fn is None:
            return x
        
        window = self.window_fn(self.frame_length).to(x.device)
        # Reshape window to broadcast: (1, 1, frame_length, 1, ...)
        window_shape = [1, 1, self.frame_length] + [1] * (x.values.ndim - 3)
        window = window.reshape(window_shape)
        
        windowed_values = x.values * window
        return Sequence(windowed_values, x.mask)
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        # Apply IRFFT
        irfft_output, _ = self.irfft.step(x, None, training, constants)
        
        # Apply window
        windowed = self._apply_window(irfft_output)
        
        # Apply overlap-add
        output, state = self.overlap_add.step(windowed, state, training, constants)
        
        return output, state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # Apply IRFFT
        irfft_output = self.irfft.layer(x, training, None, constants)
        
        # Apply window
        windowed = self._apply_window(irfft_output)
        
        # Apply overlap-add
        output = self.overlap_add.layer(windowed, training, initial_state, constants)
        
        return output


# =============================================================================
# Spectral Processing
# =============================================================================

class LinearToMelSpectrogram(Stateless):
    """Converts linear-scale spectrogram to mel-scale spectrogram."""
    
    def __init__(self,
                 num_mel_bins: int,
                 sample_rate: float,
                 lower_edge_hertz: float,
                 upper_edge_hertz: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.num_mel_bins = num_mel_bins
        self.sample_rate = sample_rate
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        
        # This would typically be computed in a setup method
        # For now, we'll compute it in the first forward pass
        self.mel_weight_matrix = None
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        # Replace the frequency dimension with mel bins
        output_shape = list(input_shape)
        output_shape[0] = self.num_mel_bins  # Assuming frequency is first dimension
        return tuple(output_shape)
    
    def _build_mel_weight_matrix(self, num_spectrogram_bins: int) -> torch.Tensor:
        """Build the mel weight matrix for spectral conversion."""
        # This is a simplified implementation
        # In practice, you'd use a proper mel-scale conversion
        
        # Linear frequency bins
        linear_freqs = torch.linspace(0, self.sample_rate / 2, num_spectrogram_bins)
        
        # Mel frequency bins
        mel_freqs = torch.linspace(
            self._hertz_to_mel(self.lower_edge_hertz),
            self._hertz_to_mel(self.upper_edge_hertz),
            self.num_mel_bins + 2
        )
        mel_freqs_hz = self._mel_to_hertz(mel_freqs)
        
        # Create triangular filters
        mel_weight_matrix = torch.zeros(self.num_mel_bins, num_spectrogram_bins)
        
        for i in range(self.num_mel_bins):
            left = mel_freqs_hz[i]
            center = mel_freqs_hz[i + 1]
            right = mel_freqs_hz[i + 2]
            
            # Triangular filter
            for j, freq in enumerate(linear_freqs):
                if left <= freq <= center:
                    mel_weight_matrix[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    mel_weight_matrix[i, j] = (right - freq) / (right - center)
        
        return mel_weight_matrix
    
    def _hertz_to_mel(self, hertz: float) -> float:
        """Convert frequency in Hertz to mel scale."""
        return 2595.0 * math.log10(1.0 + hertz / 700.0)
    
    def _mel_to_hertz(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel scale to frequency in Hertz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        # x has shape (batch, time, num_frequency_bins, ...)
        # We assume the frequency dimension is the third dimension (index 2)
        
        if self.mel_weight_matrix is None:
            num_spectrogram_bins = x.shape[2]
            self.mel_weight_matrix = self._build_mel_weight_matrix(num_spectrogram_bins)
            self.mel_weight_matrix = self.mel_weight_matrix.to(x.device)
        
        # Apply mel transformation
        # x.values has shape (batch, time, num_frequency_bins, ...)
        # mel_weight_matrix has shape (num_mel_bins, num_frequency_bins)
        
        # Reshape for matrix multiplication
        original_shape = x.values.shape
        batch_size, time_steps, num_freq_bins = original_shape[:3]
        channel_shape = original_shape[3:]
        
        # Reshape to (batch * time * channels, num_frequency_bins)
        x_reshaped = x.values.reshape(-1, num_freq_bins)
        
        # Apply mel transformation
        mel_output = torch.matmul(x_reshaped, self.mel_weight_matrix.t())
        
        # Reshape back to (batch, time, num_mel_bins, ...)
        mel_output = mel_output.reshape(batch_size, time_steps, self.num_mel_bins, *channel_shape)
        
        return Sequence(mel_output, x.mask) 