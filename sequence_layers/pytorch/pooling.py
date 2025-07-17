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
"""Pooling layers for PyTorch."""

import fractions
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
    PreservesShape,
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    # 1D pooling layers
    'MaxPooling1D',
    'AveragePooling1D',
    'MinPooling1D',
    
    # 2D pooling layers
    'MaxPooling2D',
    'AveragePooling2D',
    'MinPooling2D',
    
    # Global pooling layers
    'GlobalMaxPooling',
    'GlobalAveragePooling',
]


# =============================================================================
# Utility Functions
# =============================================================================

def _get_pad_value_for_pooling(pool_type: str, dtype: torch.dtype) -> float:
    """Get the appropriate padding value for different pooling types."""
    if pool_type == 'max':
        if dtype.is_floating_point:
            return -float('inf')
        else:
            return torch.iinfo(dtype).min
    elif pool_type == 'min':
        if dtype.is_floating_point:
            return float('inf')
        else:
            return torch.iinfo(dtype).max
    elif pool_type == 'avg':
        return 0.0
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")


def _mask_invalid_with_value(x: Sequence, pad_value: float) -> Sequence:
    """Mask invalid timesteps with a specific pad value."""
    if x.mask is None:
        return x
    
    # Create expanded mask to match values shape
    expanded_mask = x.expanded_mask()
    
    # Replace invalid timesteps with pad_value
    masked_values = torch.where(
        expanded_mask,
        x.values,
        torch.full_like(x.values, pad_value)
    )
    
    return Sequence(masked_values, x.mask)


def _apply_explicit_padding(x: Tensor, padding: Tuple[int, int], pad_value: float) -> Tensor:
    """Apply explicit padding to a tensor."""
    if padding[0] == 0 and padding[1] == 0:
        return x
    
    # Pad format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # For 3D tensors (B, T, C), we pad the time dimension
    pad_spec = (0, 0, padding[0], padding[1], 0, 0)
    return F.pad(x, pad_spec, mode='constant', value=pad_value)


def _compute_conv_mask(
    mask: Tensor,
    pool_size: int,
    stride: int,
    dilation_rate: int,
    padding_mode: str,
    is_step: bool = False
) -> Tensor:
    """Compute mask for pooled output."""
    import math
    
    batch_size, input_length = mask.shape
    
    if padding_mode == 'causal':
        if is_step:
            # In step mode, output has same time dimension as input
            return mask
        else:
            # In layer mode, mask length is preserved for causal padding
            # But actual output length depends on stride
            if stride == 1:
                return mask
            else:
                # For strided causal, we need to downsample the mask
                output_length = input_length // stride
                return mask[:, :output_length * stride:stride]
    elif padding_mode == 'valid':
        # For valid padding, PyTorch uses ceil((input_length - pool_size + 1) / stride)
        if is_step:
            # In step mode, we assume we're processing one timestep at a time
            return mask
        else:
            output_length = max(0, math.ceil((input_length - pool_size + 1) / stride))
            if output_length == 0:
                return torch.zeros(batch_size, 0, dtype=mask.dtype, device=mask.device)
            # Create output mask based on input mask
            # For simplicity, we'll use a conservative approach
            output_mask = torch.zeros(batch_size, output_length, dtype=mask.dtype, device=mask.device)
            for i in range(output_length):
                # Check if the pooling window has any valid timesteps
                start_idx = i * stride
                end_idx = min(start_idx + pool_size, input_length)
                if start_idx < input_length:
                    window_valid = torch.any(mask[:, start_idx:end_idx], dim=1)
                    output_mask[:, i] = window_valid
            return output_mask
    elif padding_mode == 'same':
        # For same padding, output length is input_length // stride
        if is_step:
            return mask
        else:
            output_length = input_length // stride
            if output_length == 0:
                return torch.zeros(batch_size, 0, dtype=mask.dtype, device=mask.device)
            # Create output mask based on input mask
            output_mask = torch.zeros(batch_size, output_length, dtype=mask.dtype, device=mask.device)
            for i in range(output_length):
                # For same padding, we need to consider the center of the pooling window
                center_idx = i * stride
                start_idx = max(0, center_idx - pool_size // 2)
                end_idx = min(input_length, center_idx + (pool_size + 1) // 2)
                if start_idx < end_idx:
                    window_valid = torch.any(mask[:, start_idx:end_idx], dim=1)
                    output_mask[:, i] = window_valid
            return output_mask
    else:
        # For other padding modes, use input mask (simplified)
        return mask


def _explicit_padding_for_mode(
    padding_mode: str,
    pool_size: int,
    stride: int,
    dilation_rate: int
) -> Tuple[int, int]:
    """Compute explicit padding for a given padding mode."""
    if padding_mode == 'valid':
        return (0, 0)
    elif padding_mode == 'same':
        # For same padding, we need to pad to keep the output size the same
        effective_pool_size = pool_size + (pool_size - 1) * (dilation_rate - 1)
        total_padding = effective_pool_size - 1
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        return (left_padding, right_padding)
    elif padding_mode == 'causal':
        # For causal padding, we pad on the left only
        effective_pool_size = pool_size + (pool_size - 1) * (dilation_rate - 1)
        return (effective_pool_size - 1, 0)
    else:
        # For other modes, return no padding for now
        return (0, 0)


# =============================================================================
# Base Pooling Classes
# =============================================================================

class BasePooling1D(PreservesShape, PreservesType, SequenceLayer):
    """Base class for 1D pooling layers."""
    
    def __init__(self,
                 pool_size: int,
                 stride: int = 1,
                 padding: str = 'valid',
                 dilation_rate: int = 1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        if pool_size <= 0:
            raise ValueError(f'pool_size must be positive, got: {pool_size}')
        if stride <= 0:
            raise ValueError(f'stride must be positive, got: {stride}')
        if dilation_rate <= 0:
            raise ValueError(f'dilation_rate must be positive, got: {dilation_rate}')
        
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate
        
        # Validate padding mode
        valid_padding_modes = ['valid', 'same', 'causal', 'causal_valid', 'reverse_causal', 'semicausal']
        if padding not in valid_padding_modes:
            raise ValueError(f'Invalid padding mode: {padding}. Must be one of {valid_padding_modes}')
        
        # Compute buffer width for step-wise execution
        self._buffer_width = max(0, pool_size - 1) if padding == 'causal' else 0
    
    @property
    def supports_step(self) -> bool:
        """Whether this layer supports step-wise execution."""
        return self.padding in ['causal', 'reverse_causal', 'semicausal']
    
    @property
    def block_size(self) -> int:
        """Block size for step-wise execution."""
        return self.stride
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        """Output ratio for step-wise execution."""
        return fractions.Fraction(1, self.stride)
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step or self._buffer_width == 0:
            return None
        
        # Create buffer for causal padding
        buffer_shape = (batch_size, self._buffer_width, *channel_spec.shape)
        # Initialize buffer with appropriate padding values instead of zeros
        pad_value = self._get_pad_value(channel_spec.dtype)
        buffer_values = torch.full(buffer_shape, pad_value, dtype=channel_spec.dtype)
        # Buffer mask should be False since these are padding values
        buffer_mask = torch.zeros(batch_size, self._buffer_width, dtype=torch.bool)
        
        return Sequence(buffer_values, buffer_mask)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Get output shape."""
        if len(input_shape) != 1:
            raise ValueError(f'1D pooling requires 1D input shape, got: {input_shape}')
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        """Get output dtype."""
        return input_dtype
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for this pooling type."""
        raise NotImplementedError("Subclasses must implement _get_pad_value")
    
    def _apply_pooling(self, x: Tensor, padding: Tuple[int, int]) -> Tensor:
        """Apply pooling operation."""
        raise NotImplementedError("Subclasses must implement _apply_pooling")
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise."""
        if not self.supports_step:
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
        
        # Replace masked values with pad value
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _mask_invalid_with_value(x, pad_value)
        
        if self._buffer_width > 0:
            # Concatenate with previous buffer
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate state and current input
            combined = state.concatenate(x_padded)
        else:
            combined = x_padded
        
        # Apply pooling
        output_values = self._apply_pooling(combined.values, (0, 0))
        
        # For step-wise execution, we need to handle the output shape carefully
        # The pooling operation might produce multiple timesteps, but we need to
        # return only the relevant ones based on the stride
        if self.stride > 1:
            # For strided pooling, we may not produce output every step
            # We need to handle this based on the current step position
            # For now, take the last timestep(s) that correspond to the current step
            if output_values.shape[1] > 0:
                # Take only the timesteps that correspond to the current step
                output_values = output_values[:, -1:, :]  # Take last timestep
            else:
                # No output for this step
                output_values = torch.empty(x.shape[0], 0, *x.channel_shape, dtype=x.dtype, device=x.device)
        
        # Create output mask based on whether we have output
        if output_values.shape[1] > 0:
            output_mask = torch.ones(x.shape[0], output_values.shape[1], dtype=torch.bool, device=x.device)
        else:
            output_mask = torch.zeros(x.shape[0], 0, dtype=torch.bool, device=x.device)
        
        # Update state for next step
        if self._buffer_width > 0:
            next_state = combined[:, -self._buffer_width:] if self._buffer_width > 0 else None
        else:
            next_state = None
        
        return Sequence(output_values, output_mask), next_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Process this layer."""
        # Get padding for this mode
        padding = _explicit_padding_for_mode(self.padding, self.pool_size, self.stride, self.dilation_rate)
        
        # Replace masked values with pad value
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _mask_invalid_with_value(x, pad_value)
        
        # Apply pooling
        output_values = self._apply_pooling(x_padded.values, padding)
        
        # Compute output mask
        output_mask = _compute_conv_mask(
            x.mask, self.pool_size, self.stride, self.dilation_rate, self.padding, is_step=False
        )
        
        return Sequence(output_values, output_mask)


class BasePooling2D(PreservesType, SequenceLayer):
    """Base class for 2D pooling layers."""
    
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 time_padding: str = 'valid',
                 spatial_padding: str = 'same',
                 dilation_rate: Union[int, Tuple[int, int]] = 1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Normalize 2-tuples
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)
        
        if len(pool_size) != 2:
            raise ValueError(f'pool_size must be int or 2-tuple, got: {pool_size}')
        if len(stride) != 2:
            raise ValueError(f'stride must be int or 2-tuple, got: {stride}')
        if len(dilation_rate) != 2:
            raise ValueError(f'dilation_rate must be int or 2-tuple, got: {dilation_rate}')
        
        if any(p <= 0 for p in pool_size):
            raise ValueError(f'pool_size must be positive, got: {pool_size}')
        if any(s <= 0 for s in stride):
            raise ValueError(f'stride must be positive, got: {stride}')
        if any(d <= 0 for d in dilation_rate):
            raise ValueError(f'dilation_rate must be positive, got: {dilation_rate}')
        
        self.pool_size = pool_size
        self.stride = stride
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.dilation_rate = dilation_rate
        
        # Validate padding modes
        valid_padding_modes = ['valid', 'same', 'causal', 'causal_valid', 'reverse_causal', 'semicausal']
        if time_padding not in valid_padding_modes:
            raise ValueError(f'Invalid time_padding mode: {time_padding}. Must be one of {valid_padding_modes}')
        if spatial_padding not in valid_padding_modes:
            raise ValueError(f'Invalid spatial_padding mode: {spatial_padding}. Must be one of {valid_padding_modes}')
        
        # Compute buffer width for step-wise execution
        self._buffer_width = max(0, pool_size[0] - 1) if time_padding == 'causal' else 0
    
    @property
    def supports_step(self) -> bool:
        """Whether this layer supports step-wise execution."""
        return self.time_padding in ['causal', 'reverse_causal', 'semicausal']
    
    @property
    def block_size(self) -> int:
        """Block size for step-wise execution."""
        return self.stride[0]
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        """Output ratio for step-wise execution."""
        return fractions.Fraction(1, self.stride[0])
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step or self._buffer_width == 0:
            return None
        
        # Create buffer for causal padding
        # For 2D, we need height and channels
        if len(channel_spec.shape) != 2:
            raise ValueError(f'2D pooling requires 2D channel shape, got: {channel_spec.shape}')
        
        height, channels = channel_spec.shape
        buffer_shape = (batch_size, self._buffer_width, height, channels)
        # Initialize buffer with appropriate padding values instead of zeros
        pad_value = self._get_pad_value(channel_spec.dtype)
        buffer_values = torch.full(buffer_shape, pad_value, dtype=channel_spec.dtype)
        # Buffer mask should be False since these are padding values
        buffer_mask = torch.zeros(batch_size, self._buffer_width, dtype=torch.bool)
        
        return Sequence(buffer_values, buffer_mask)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Get output shape."""
        if len(input_shape) != 2:
            raise ValueError(f'2D pooling requires 2D input shape, got: {input_shape}')
        
        height, channels = input_shape
        
        # Compute spatial output size
        spatial_padding = _explicit_padding_for_mode(
            self.spatial_padding, self.pool_size[1], self.stride[1], self.dilation_rate[1]
        )
        
        # Simplified output size calculation
        output_height = (height + spatial_padding[0] + spatial_padding[1] - self.pool_size[1]) // self.stride[1] + 1
        
        return (output_height, channels)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        """Get output dtype."""
        return input_dtype
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for this pooling type."""
        raise NotImplementedError("Subclasses must implement _get_pad_value")
    
    def _apply_pooling(self, x: Tensor, time_padding: Tuple[int, int], spatial_padding: Tuple[int, int]) -> Tensor:
        """Apply pooling operation."""
        raise NotImplementedError("Subclasses must implement _apply_pooling")
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise."""
        if not self.supports_step:
            raise ValueError(f"Step-wise execution not supported for time padding mode: {self.time_padding}")
        
        # Replace masked values with pad value
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _mask_invalid_with_value(x, pad_value)
        
        # In step mode, time padding is handled by buffer
        time_padding = (0, 0)
        spatial_padding = _explicit_padding_for_mode(
            self.spatial_padding, self.pool_size[1], self.stride[1], self.dilation_rate[1]
        )
        
        if self._buffer_width > 0:
            # Concatenate with previous buffer
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate state and current input
            combined = state.concatenate(x_padded)
        else:
            combined = x_padded
        
        # Apply pooling
        output_values = self._apply_pooling(combined.values, time_padding, spatial_padding)
        
        # For step-wise execution, we need to handle the output shape carefully
        # The pooling operation might produce multiple timesteps, but we need to
        # return only the relevant ones based on the stride
        if self.stride[0] > 1:
            # For strided pooling, we may not produce output every step
            # We need to handle this based on the current step position
            # For now, take the last timestep(s) that correspond to the current step
            if output_values.shape[1] > 0:
                # Take only the timesteps that correspond to the current step
                output_values = output_values[:, -1:, :]  # Take last timestep
            else:
                # No output for this step
                output_values = torch.empty(x.shape[0], 0, *x.channel_shape, dtype=x.dtype, device=x.device)
        
        # Create output mask based on whether we have output
        if output_values.shape[1] > 0:
            output_mask = torch.ones(x.shape[0], output_values.shape[1], dtype=torch.bool, device=x.device)
        else:
            output_mask = torch.zeros(x.shape[0], 0, dtype=torch.bool, device=x.device)
        
        # Update state for next step
        if self._buffer_width > 0:
            next_state = combined[:, -self._buffer_width:] if self._buffer_width > 0 else None
        else:
            next_state = None
        
        return Sequence(output_values, output_mask), next_state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Process this layer."""
        # Get padding for both dimensions
        time_padding = _explicit_padding_for_mode(
            self.time_padding, self.pool_size[0], self.stride[0], self.dilation_rate[0]
        )
        spatial_padding = _explicit_padding_for_mode(
            self.spatial_padding, self.pool_size[1], self.stride[1], self.dilation_rate[1]
        )
        
        # Replace masked values with pad value
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _mask_invalid_with_value(x, pad_value)
        
        # Apply pooling
        output_values = self._apply_pooling(x_padded.values, time_padding, spatial_padding)
        
        # Compute output mask
        output_mask = _compute_conv_mask(
            x.mask, self.pool_size[0], self.stride[0], self.dilation_rate[0], self.time_padding, is_step=False
        )
        
        return Sequence(output_values, output_mask)


# =============================================================================
# 1D Pooling Layers
# =============================================================================

class MaxPooling1D(BasePooling1D):
    """1D max pooling layer."""
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for max pooling."""
        return _get_pad_value_for_pooling('max', dtype)
    
    def _apply_pooling(self, x: Tensor, padding: Tuple[int, int]) -> Tensor:
        """Apply max pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _apply_explicit_padding(x, padding, pad_value)
        
        # Apply max pooling
        # PyTorch max_pool1d expects (N, C, L) but we have (N, L, C)
        # So we need to transpose, pool, and transpose back
        x_transposed = x_padded.transpose(1, 2)  # (N, L, C) -> (N, C, L)
        
        pooled_transposed = F.max_pool1d(
            x_transposed,
            kernel_size=self.pool_size,
            stride=self.stride,
            padding=0,  # We handle padding explicitly
            dilation=self.dilation_rate
        )
        
        # Transpose back to (N, L, C)
        return pooled_transposed.transpose(1, 2)


class AveragePooling1D(BasePooling1D):
    """1D average pooling layer."""
    
    def __init__(self,
                 pool_size: int,
                 stride: int = 1,
                 padding: str = 'valid',
                 dilation_rate: int = 1,
                 masked_average: bool = False,
                 name: Optional[str] = None):
        super().__init__(pool_size, stride, padding, dilation_rate, name)
        self.masked_average = masked_average
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for average pooling."""
        return _get_pad_value_for_pooling('avg', dtype)
    
    def _apply_pooling(self, x: Tensor, padding: Tuple[int, int]) -> Tensor:
        """Apply average pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _apply_explicit_padding(x, padding, pad_value)
        
        # Apply average pooling
        # PyTorch avg_pool1d expects (N, C, L) but we have (N, L, C)
        # So we need to transpose, pool, and transpose back
        x_transposed = x_padded.transpose(1, 2)  # (N, L, C) -> (N, C, L)
        
        pooled_transposed = F.avg_pool1d(
            x_transposed,
            kernel_size=self.pool_size,
            stride=self.stride,
            padding=0,  # We handle padding explicitly
            count_include_pad=False  # Don't include padding in average
        )
        
        # Transpose back to (N, L, C)
        return pooled_transposed.transpose(1, 2)


class MinPooling1D(BasePooling1D):
    """1D min pooling layer."""
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for min pooling."""
        return _get_pad_value_for_pooling('min', dtype)
    
    def _apply_pooling(self, x: Tensor, padding: Tuple[int, int]) -> Tensor:
        """Apply min pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        x_padded = _apply_explicit_padding(x, padding, pad_value)
        
        # Apply min pooling (negative max pooling)
        # PyTorch max_pool1d expects (N, C, L) but we have (N, L, C)
        # So we need to transpose, pool, and transpose back
        x_transposed = (-x_padded).transpose(1, 2)  # (N, L, C) -> (N, C, L) and negate
        
        pooled_transposed = F.max_pool1d(
            x_transposed,
            kernel_size=self.pool_size,
            stride=self.stride,
            padding=0,  # We handle padding explicitly
            dilation=self.dilation_rate
        )
        
        # Transpose back to (N, L, C) and negate back
        return (-pooled_transposed).transpose(1, 2)


# =============================================================================
# 2D Pooling Layers
# =============================================================================

class MaxPooling2D(BasePooling2D):
    """2D max pooling layer."""
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for max pooling."""
        return _get_pad_value_for_pooling('max', dtype)
    
    def _apply_pooling(self, x: Tensor, time_padding: Tuple[int, int], spatial_padding: Tuple[int, int]) -> Tensor:
        """Apply max pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        
        # Pad time dimension
        if time_padding[0] > 0 or time_padding[1] > 0:
            x = F.pad(x, (0, 0, 0, 0, time_padding[0], time_padding[1]), mode='constant', value=pad_value)
        
        # Pad spatial dimension
        if spatial_padding[0] > 0 or spatial_padding[1] > 0:
            x = F.pad(x, (0, 0, spatial_padding[0], spatial_padding[1], 0, 0), mode='constant', value=pad_value)
        
        # Apply max pooling
        # PyTorch expects (B, C, H, W) but we have (B, T, H, C)
        # We need to reshape to (B*T, C, H, 1) or use unfold
        batch_size, time_steps, height, channels = x.shape
        
        # Reshape to (B*T, C, H, 1)
        x_reshaped = x.permute(0, 1, 3, 2).reshape(batch_size * time_steps, channels, height, 1)
        
        # Apply 2D max pooling
        pooled = F.max_pool2d(
            x_reshaped,
            kernel_size=(self.pool_size[1], 1),  # Only pool over spatial dimension
            stride=(self.stride[1], 1),
            padding=0,
            dilation=(self.dilation_rate[1], 1)
        )
        
        # Reshape back and apply time pooling
        _, _, new_height, _ = pooled.shape
        pooled = pooled.reshape(batch_size, time_steps, channels, new_height).permute(0, 1, 3, 2)
        
        # Apply time pooling
        if self.pool_size[0] > 1:
            pooled = F.max_pool1d(
                pooled.permute(0, 2, 3, 1).reshape(batch_size * new_height * channels, time_steps, 1).squeeze(-1),
                kernel_size=self.pool_size[0],
                stride=self.stride[0],
                padding=0,
                dilation=self.dilation_rate[0]
            ).unsqueeze(-1).reshape(batch_size, new_height, channels, -1).permute(0, 3, 1, 2)
        
        return pooled


class AveragePooling2D(BasePooling2D):
    """2D average pooling layer."""
    
    def __init__(self,
                 pool_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 time_padding: str = 'valid',
                 spatial_padding: str = 'same',
                 dilation_rate: Union[int, Tuple[int, int]] = 1,
                 masked_average: bool = False,
                 name: Optional[str] = None):
        super().__init__(pool_size, stride, time_padding, spatial_padding, dilation_rate, name)
        self.masked_average = masked_average
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for average pooling."""
        return _get_pad_value_for_pooling('avg', dtype)
    
    def _apply_pooling(self, x: Tensor, time_padding: Tuple[int, int], spatial_padding: Tuple[int, int]) -> Tensor:
        """Apply average pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        
        # Pad time dimension
        if time_padding[0] > 0 or time_padding[1] > 0:
            x = F.pad(x, (0, 0, 0, 0, time_padding[0], time_padding[1]), mode='constant', value=pad_value)
        
        # Pad spatial dimension
        if spatial_padding[0] > 0 or spatial_padding[1] > 0:
            x = F.pad(x, (0, 0, spatial_padding[0], spatial_padding[1], 0, 0), mode='constant', value=pad_value)
        
        # Apply average pooling (similar to max pooling but with avg_pool2d)
        batch_size, time_steps, height, channels = x.shape
        
        # Reshape to (B*T, C, H, 1)
        x_reshaped = x.permute(0, 1, 3, 2).reshape(batch_size * time_steps, channels, height, 1)
        
        # Apply 2D average pooling
        pooled = F.avg_pool2d(
            x_reshaped,
            kernel_size=(self.pool_size[1], 1),  # Only pool over spatial dimension
            stride=(self.stride[1], 1),
            padding=0,
            count_include_pad=False
        )
        
        # Reshape back and apply time pooling
        _, _, new_height, _ = pooled.shape
        pooled = pooled.reshape(batch_size, time_steps, channels, new_height).permute(0, 1, 3, 2)
        
        # Apply time pooling
        if self.pool_size[0] > 1:
            pooled = F.avg_pool1d(
                pooled.permute(0, 2, 3, 1).reshape(batch_size * new_height * channels, time_steps, 1).squeeze(-1),
                kernel_size=self.pool_size[0],
                stride=self.stride[0],
                padding=0,
                count_include_pad=False
            ).unsqueeze(-1).reshape(batch_size, new_height, channels, -1).permute(0, 3, 1, 2)
        
        return pooled


class MinPooling2D(BasePooling2D):
    """2D min pooling layer."""
    
    def _get_pad_value(self, dtype: torch.dtype) -> float:
        """Get padding value for min pooling."""
        return _get_pad_value_for_pooling('min', dtype)
    
    def _apply_pooling(self, x: Tensor, time_padding: Tuple[int, int], spatial_padding: Tuple[int, int]) -> Tensor:
        """Apply min pooling."""
        # Apply padding
        pad_value = self._get_pad_value(x.dtype)
        
        # Pad time dimension
        if time_padding[0] > 0 or time_padding[1] > 0:
            x = F.pad(x, (0, 0, 0, 0, time_padding[0], time_padding[1]), mode='constant', value=pad_value)
        
        # Pad spatial dimension
        if spatial_padding[0] > 0 or spatial_padding[1] > 0:
            x = F.pad(x, (0, 0, spatial_padding[0], spatial_padding[1], 0, 0), mode='constant', value=pad_value)
        
        # Apply min pooling (negative max pooling)
        batch_size, time_steps, height, channels = x.shape
        
        # Reshape to (B*T, C, H, 1)
        x_reshaped = (-x).permute(0, 1, 3, 2).reshape(batch_size * time_steps, channels, height, 1)
        
        # Apply 2D max pooling on negated values
        pooled = F.max_pool2d(
            x_reshaped,
            kernel_size=(self.pool_size[1], 1),
            stride=(self.stride[1], 1),
            padding=0,
            dilation=(self.dilation_rate[1], 1)
        )
        
        # Reshape back and apply time pooling
        _, _, new_height, _ = pooled.shape
        pooled = (-pooled).reshape(batch_size, time_steps, channels, new_height).permute(0, 1, 3, 2)
        
        # Apply time pooling
        if self.pool_size[0] > 1:
            pooled = -F.max_pool1d(
                (-pooled).permute(0, 2, 3, 1).reshape(batch_size * new_height * channels, time_steps, 1).squeeze(-1),
                kernel_size=self.pool_size[0],
                stride=self.stride[0],
                padding=0,
                dilation=self.dilation_rate[0]
            ).unsqueeze(-1).reshape(batch_size, new_height, channels, -1).permute(0, 3, 1, 2)
        
        return pooled


# =============================================================================
# Global Pooling Layers
# =============================================================================

class GlobalMaxPooling(StatelessPointwise):
    """Global max pooling layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    @property
    def supports_step(self) -> bool:
        """Global pooling doesn't support step-wise execution."""
        return False
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Get output shape."""
        # For global pooling over time, we go from (channels,) to (channels,)
        # but the time dimension is reduced to 1
        return input_shape
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply global max pooling."""
        # Apply masking
        pad_value = _get_pad_value_for_pooling('max', x.dtype)
        x_masked = _mask_invalid_with_value(x, pad_value)
        
        # Pool over time dimension
        pooled_values, _ = torch.max(x_masked.values, dim=1, keepdim=True)
        
        # For global pooling, create a mask with single timestep
        output_mask = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
        
        return Sequence(pooled_values, output_mask)


class GlobalAveragePooling(StatelessPointwise):
    """Global average pooling layer."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    @property
    def supports_step(self) -> bool:
        """Global pooling doesn't support step-wise execution."""
        return False
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Get output shape."""
        # For global pooling over time, we go from (channels,) to (channels,)
        # but the time dimension is reduced to 1
        return input_shape
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply global average pooling."""
        # First mask invalid values to prevent NaN propagation
        x = x.mask_invalid()
        
        # Compute average only over valid timesteps
        if x.mask is not None:
            # Expand mask to match values shape
            mask_expanded = x.mask.unsqueeze(-1).expand_as(x.values).float()
            
            # Sum over valid timesteps
            sum_values = torch.sum(x.values * mask_expanded, dim=1, keepdim=True)
            
            # Count valid timesteps
            valid_count = torch.sum(mask_expanded, dim=1, keepdim=True)
            valid_count = torch.clamp(valid_count, min=1.0)  # Avoid division by zero
            
            # Compute average
            pooled_values = sum_values / valid_count
            
            # Handle edge case where all timesteps are invalid
            # In this case, set output to zero
            all_invalid = torch.sum(x.mask, dim=1, keepdim=True) == 0
            if torch.any(all_invalid):
                all_invalid_expanded = all_invalid.unsqueeze(-1).expand_as(pooled_values)
                pooled_values = torch.where(all_invalid_expanded, 
                                          torch.zeros_like(pooled_values), 
                                          pooled_values)
        else:
            # No mask, simple average
            pooled_values = torch.mean(x.values, dim=1, keepdim=True)
        
        # For global pooling, create a mask with single timestep
        # If all input timesteps were invalid, mark output as invalid too
        if x.mask is not None:
            output_mask = torch.sum(x.mask, dim=1, keepdim=True) > 0
        else:
            output_mask = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
        
        return Sequence(pooled_values, output_mask) 