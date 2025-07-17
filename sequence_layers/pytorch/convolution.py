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
"""Convolutional layers for PyTorch."""

import abc
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
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
    PaddingMode,
)


__all__ = [
    # 1D Convolutions
    'Conv1D',
    'DepthwiseConv1D',
    'Conv1DTranspose',
    
    # 2D Convolutions
    'Conv2D',
    'Conv2DTranspose',
    
    # 3D Convolutions
    'Conv3D',
]


# =============================================================================
# Utility Functions
# =============================================================================

def _compute_conv_output_length(input_length: int, kernel_size: int, 
                               stride: int, dilation: int, padding: Tuple[int, int]) -> int:
    """Compute the output length after convolution."""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padded_length = input_length + padding[0] + padding[1]
    return (padded_length - effective_kernel_size) // stride + 1


def _compute_conv_mask(mask: Tensor, kernel_size: int, stride: int, 
                      dilation: int, padding: str, use_logical_or: bool = False) -> Tensor:
    """Compute the output mask after convolution."""
    if kernel_size == 1:
        return mask
    
    # Get padding values to compute correct output length
    pad_left, pad_right = _get_conv_padding_for_mode(padding, kernel_size, stride, dilation)
    
    # Compute expected output length
    input_length = mask.shape[1]
    padded_length = input_length + pad_left + pad_right
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    output_length = (padded_length - effective_kernel_size) // stride + 1
    
    # For simplicity, we'll use a conservative approach
    # In a full implementation, we'd need to track which output positions
    # correspond to valid input positions based on the receptive field
    
    if padding in ['same', 'causal']:
        # For 'same' and 'causal', output length should match input length
        # For stride > 1, output length is input_length // stride
        if stride == 1:
            output_mask = mask.clone()
        else:
            # Downsample mask by stride
            output_mask = mask[:, ::stride]
    elif padding in ['valid', 'causal_valid']:
        # For 'valid' modes, output is shorter
        if stride == 1:
            output_mask = mask[:, :output_length]
        else:
            # Downsample mask by stride and truncate
            downsampled_mask = mask[:, ::stride]
            output_mask = downsampled_mask[:, :output_length]
    else:
        # For other padding modes, create a conservative mask
        output_mask = torch.ones(mask.shape[0], output_length, dtype=torch.bool, device=mask.device)
        
        # Apply input mask constraints
        if stride == 1:
            # For stride 1, we can be more precise
            if output_length <= mask.shape[1]:
                output_mask = mask[:, :output_length]
            else:
                # Pad mask if output is longer
                output_mask = torch.ones(mask.shape[0], output_length, dtype=torch.bool, device=mask.device)
                output_mask[:, :mask.shape[1]] = mask
        else:
            # For stride > 1, downsample
            downsampled_mask = mask[:, ::stride]
            if output_length <= downsampled_mask.shape[1]:
                output_mask = downsampled_mask[:, :output_length]
            else:
                output_mask = torch.ones(mask.shape[0], output_length, dtype=torch.bool, device=mask.device)
                output_mask[:, :downsampled_mask.shape[1]] = downsampled_mask
    
    return output_mask


def _get_conv_padding_for_mode(padding_mode: str, kernel_size: int, 
                              stride: int, dilation: int) -> Tuple[int, int]:
    """Get explicit padding values for different padding modes."""
    if padding_mode == 'valid':
        return (0, 0)
    elif padding_mode == 'same':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        pad_total = max(0, effective_kernel_size - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return (pad_left, pad_right)
    elif padding_mode == 'causal':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        return (effective_kernel_size - 1, 0)
    elif padding_mode == 'causal_valid':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        return (effective_kernel_size - 1, 0)
    elif padding_mode == 'reverse_causal':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        return (0, effective_kernel_size - 1)
    elif padding_mode == 'reverse_causal_valid':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        return (0, effective_kernel_size - 1)
    elif padding_mode == 'semicausal':
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        pad_total = max(0, effective_kernel_size - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return (pad_left, pad_right)
    else:
        raise ValueError(f'Unknown padding mode: {padding_mode}')


# =============================================================================
# Base Convolution Class
# =============================================================================

class BaseConv(SequenceLayer):
    """Shared base logic for convolution layers."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    
    @property
    @abc.abstractmethod
    def _kernel_size(self) -> Tuple[int, ...]:
        pass
    
    @property
    @abc.abstractmethod
    def _strides(self) -> Tuple[int, ...]:
        pass
    
    @property
    @abc.abstractmethod
    def _dilation_rate(self) -> Tuple[int, ...]:
        pass
    
    @property
    @abc.abstractmethod
    def _paddings(self) -> Tuple[str, ...]:
        pass
    
    def supports_step(self) -> bool:
        """Check if the layer supports step-wise execution."""
        return self._paddings[0] in [
            'causal_valid', 'reverse_causal_valid', 
            'causal', 'reverse_causal', 'semicausal'
        ]
    
    @property
    def block_size(self) -> int:
        return self._strides[0]
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(1, self._strides[0])


# =============================================================================
# 1D Convolution Layers
# =============================================================================

class Conv1D(BaseConv):
    """A 1D strided or dilated convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, 
                 padding: str = 'valid', groups: int = 1,
                 use_bias: bool = True, use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels // groups, kernel_size
        ) / math.sqrt(in_channels // groups * kernel_size))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return (self.kernel_size,)
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return (self.stride,)
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return (self.dilation,)
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.padding,)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'Conv1D requires rank 3 input. Got: {input_shape}')
        return (self.out_channels,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # The buffer maintains (kernel_size - 1) past timesteps, so we take the last few outputs
            if self.stride == 1:
                # For stride 1, output has same length as input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided convolution, we need to handle output timing carefully
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] // self.stride):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, channel_spec.shape[0])
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 3:
            raise ValueError(f'Conv1D requires rank 3 input. Got: {x.shape}')
        
        # Mask inputs if receptive field is greater than 1
        if self.kernel_size > 1:
            x = x.mask_invalid()
        
        # Get padding values
        pad_left, pad_right = _get_conv_padding_for_mode(
            self.padding, self.kernel_size, self.stride, self.dilation
        )
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1) / torch.norm(
                weight, dim=[1, 2], keepdim=True
            )
        
        # Apply convolution
        def apply_conv(values):
            # Convert to PyTorch conv format: [B, T, D] -> [B, D, T]
            values = values.transpose(1, 2)  # [B, D, T]
            
            # Apply padding to time dimension (last dimension)
            if pad_left > 0 or pad_right > 0:
                values = F.pad(values, (pad_left, pad_right), mode='constant', value=0)
            
            # Apply convolution
            output = F.conv1d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            return output.transpose(1, 2)  # [B, T, D]
        
        # Apply the transformation
        values = apply_conv(x.values)
        
        # Compute output mask
        mask = _compute_conv_mask(
            x.mask, self.kernel_size, self.stride, self.dilation, self.padding
        )
        
        # Adjust mask length to match output
        if mask.shape[1] != values.shape[1]:
            mask = mask[:, :values.shape[1]]
        
        # Return appropriate sequence type
        if self.use_bias or self.kernel_size > 1:
            return Sequence(values, mask)
        else:
            return type(x)(values, mask)


class DepthwiseConv1D(BaseConv):
    """A 1D depthwise strided or dilated convolution layer."""
    
    def __init__(self, in_channels: int, kernel_size: int, 
                 stride: int = 1, depth_multiplier: int = 1,
                 dilation: int = 1, padding: str = 'valid',
                 use_bias: bool = True, use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.depth_multiplier = depth_multiplier
        self.dilation = dilation
        self.padding = padding
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        self.out_channels = in_channels * depth_multiplier
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            in_channels * depth_multiplier, 1, kernel_size
        ) / math.sqrt(kernel_size))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(self.out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return (self.kernel_size,)
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return (self.stride,)
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return (self.dilation,)
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.padding,)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'DepthwiseConv1D requires rank 3 input. Got: {input_shape}')
        return (self.out_channels,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 3:
            raise ValueError(f'DepthwiseConv1D requires rank 3 input. Got: {x.shape}')
        
        # Mask inputs if receptive field is greater than 1
        if self.kernel_size > 1:
            x = x.mask_invalid()
        
        # Get padding values
        pad_left, pad_right = _get_conv_padding_for_mode(
            self.padding, self.kernel_size, self.stride, self.dilation
        )
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1) / torch.norm(
                weight, dim=[1, 2], keepdim=True
            )
        
        # Apply convolution
        def apply_conv(values):
            # Convert to PyTorch conv format: [B, T, D] -> [B, D, T]
            values = values.transpose(1, 2)  # [B, D, T]
            
            # Apply padding to time dimension (last dimension)
            if pad_left > 0 or pad_right > 0:
                values = F.pad(values, (pad_left, pad_right), mode='constant', value=0)
            
            # Apply depthwise convolution
            output = F.conv1d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.in_channels
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            return output.transpose(1, 2)  # [B, T, D]
        
        # Apply the transformation
        values = apply_conv(x.values)
        
        # Compute output mask
        mask = _compute_conv_mask(
            x.mask, self.kernel_size, self.stride, self.dilation, self.padding
        )
        
        # Adjust mask length to match output
        if mask.shape[1] != values.shape[1]:
            mask = mask[:, :values.shape[1]]
        
        # Return appropriate sequence type
        if self.use_bias or self.kernel_size > 1:
            return Sequence(values, mask)
        else:
            return type(x)(values, mask)
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # The buffer maintains (kernel_size - 1) past timesteps, so we take the last few outputs
            if self.stride == 1:
                # For stride 1, output has same length as input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided convolution, we need to handle output timing carefully
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] // self.stride):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, channel_spec.shape[0])
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None


class Conv1DTranspose(BaseConv):
    """A 1D transposed convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, dilation: int = 1, 
                 padding: str = 'valid', groups: int = 1,
                 use_bias: bool = True, use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            in_channels, out_channels // groups, kernel_size
        ) / math.sqrt(in_channels * kernel_size))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return (self.kernel_size,)
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return (self.stride,)
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return (self.dilation,)
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.padding,)
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(self.stride, 1)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 1:
            raise ValueError(f'Conv1DTranspose requires rank 3 input. Got: {input_shape}')
        return (self.out_channels,)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 3:
            raise ValueError(f'Conv1DTranspose requires rank 3 input. Got: {x.shape}')
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1) / torch.norm(
                weight, dim=[1, 2], keepdim=True
            )
        
        # Apply transposed convolution
        def apply_conv_transpose(values):
            # Convert to PyTorch conv format: [B, T, D] -> [B, D, T]
            values = values.transpose(1, 2)  # [B, D, T]
            
            # Apply transposed convolution
            output = F.conv_transpose1d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            return output.transpose(1, 2)  # [B, T, D]
        
        # Apply the transformation
        values = apply_conv_transpose(x.values)
        
        # For transposed convolution, we need to upsample the mask
        mask = x.mask
        if self.stride > 1:
            mask = mask.repeat_interleave(self.stride, dim=1)
        
        # Adjust mask length to match output
        if mask.shape[1] > values.shape[1]:
            mask = mask[:, :values.shape[1]]
        elif mask.shape[1] < values.shape[1]:
            # Pad mask if needed
            pad_size = values.shape[1] - mask.shape[1]
            mask = F.pad(mask, (0, pad_size), mode='constant', value=True)
        
        return Sequence(values, mask)
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # For transposed convolution, the output is typically longer than input
            if self.stride == 1:
                # For stride 1, output has same or longer length than input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided transposed convolution, output can be much longer
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] * self.stride):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, channel_spec.shape[0])
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None


# =============================================================================
# 2D Convolution Layers
# =============================================================================

class Conv2D(BaseConv):
    """A 2D strided or dilated convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 time_padding: str = 'valid', 
                 spatial_padding: Union[str, Tuple[int, int]] = 'same',
                 groups: int = 1, use_bias: bool = True,
                 use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels // groups, *self.kernel_size
        ) / math.sqrt(in_channels // groups * math.prod(self.kernel_size)))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return self.kernel_size
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return self.stride
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return self.dilation
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.time_padding, str(self.spatial_padding))
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 2:
            raise ValueError(f'Conv2D requires rank 4 input. Got: {input_shape}')
        
        # Calculate output spatial dimensions based on stride and padding
        # For spatial dimensions, we need to account for stride effects
        spatial_in = input_shape[0]  # Input spatial dimension
        
        # For causal time padding, preserve spatial dimensions in channel spec
        # For other time padding modes, apply stride effects
        if self.time_padding == 'causal':
            # For causal padding, preserve spatial dimension (used in streaming)
            spatial_out = spatial_in
        else:
            # For non-causal padding, apply stride effects
            # Calculate output spatial dimension based on stride
            # For 'same' padding with stride > 1, output = input // stride
            # For 'valid' padding, need to calculate based on kernel size
            if isinstance(self.spatial_padding, str):
                if self.spatial_padding == 'same':
                    # For 'same' padding, output size = input size // stride
                    spatial_out = (spatial_in + self.stride[1] - 1) // self.stride[1]
                elif self.spatial_padding == 'valid':
                    # For 'valid' padding, output size = (input - kernel + 1) // stride
                    spatial_out = (spatial_in - self.kernel_size[1] + 1 + self.stride[1] - 1) // self.stride[1]
                else:
                    # For other padding modes, use stride-based calculation
                    spatial_out = (spatial_in + self.stride[1] - 1) // self.stride[1]
            else:
                # For explicit padding, use stride-based calculation
                spatial_out = (spatial_in + self.stride[1] - 1) // self.stride[1]
        
        return (spatial_out, self.out_channels)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.time_padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.time_padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # The buffer maintains (kernel_size - 1) past timesteps, so we take the last few outputs
            if self.stride[0] == 1:
                # For stride 1, output has same length as input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided convolution, we need to handle output timing carefully
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] // self.stride[0]):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.time_padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.time_padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.time_padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, *channel_spec.shape)
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 4:
            raise ValueError(f'Conv2D requires rank 4 input. Got: {x.shape}')
        
        # Mask inputs if receptive field is greater than 1
        if self.kernel_size[0] > 1:
            x = x.mask_invalid()
        
        # Get padding values
        time_pad = _get_conv_padding_for_mode(
            self.time_padding, self.kernel_size[0], self.stride[0], self.dilation[0]
        )
        
        if isinstance(self.spatial_padding, str):
            spatial_pad = _get_conv_padding_for_mode(
                self.spatial_padding, self.kernel_size[1], self.stride[1], self.dilation[1]
            )
        else:
            spatial_pad = self.spatial_padding
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1, 1) / torch.norm(
                weight, dim=[1, 2, 3], keepdim=True
            )
        
        # Apply convolution
        def apply_conv(values):
            # Input is [B, T, H, D] -> Convert to [B, D, T, H] for conv2d
            values = values.permute(0, 3, 1, 2)
            
            # Apply padding: [height_left, height_right, time_left, time_right]
            padding = (spatial_pad[0], spatial_pad[1], time_pad[0], time_pad[1])
            if sum(padding) > 0:
                values = F.pad(values, padding, mode='constant', value=0)
            
            # Apply convolution
            output = F.conv2d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            # Convert back to [B, T, H, D]
            return output.permute(0, 2, 3, 1)
        
        # Apply the transformation
        values = apply_conv(x.values)
        
        # Compute output mask (only time dimension affects mask)
        mask = _compute_conv_mask(
            x.mask, self.kernel_size[0], self.stride[0], self.dilation[0], self.time_padding
        )
        
        # Adjust mask length to match output
        if mask.shape[1] != values.shape[1]:
            mask = mask[:, :values.shape[1]]
        
        # Return appropriate sequence type
        if self.use_bias or self.kernel_size[0] > 1:
            return Sequence(values, mask)
        else:
            return type(x)(values, mask)


class Conv2DTranspose(BaseConv):
    """A 2D transposed convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 time_padding: str = 'valid', 
                 spatial_padding: Union[str, Tuple[int, int]] = 'same',
                 groups: int = 1, use_bias: bool = True,
                 use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            in_channels, out_channels // groups, *self.kernel_size
        ) / math.sqrt(in_channels * math.prod(self.kernel_size)))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return self.kernel_size
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return self.stride
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return self.dilation
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.time_padding, str(self.spatial_padding))
    
    @property
    def block_size(self) -> int:
        return 1
    
    @property
    def output_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(self.stride[0], 1)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 2:
            raise ValueError(f'Conv2DTranspose requires rank 4 input. Got: {input_shape}')
        # For transposed convolution, calculate the actual output shape based on PyTorch's behavior
        # PyTorch's conv_transpose2d formula: output_size = (input_size - 1) * stride + kernel_size
        spatial_out = (input_shape[0] - 1) * self.stride[1] + self.kernel_size[1]
        return (spatial_out, self.out_channels)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 4:
            raise ValueError(f'Conv2DTranspose requires rank 4 input. Got: {x.shape}')
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1, 1) / torch.norm(
                weight, dim=[1, 2, 3], keepdim=True
            )
        
        # Apply transposed convolution
        def apply_conv_transpose(values):
            # Input is [B, T, H, D] -> Convert to [B, D, T, H] for conv_transpose2d
            values = values.permute(0, 3, 1, 2)
            
            # Apply transposed convolution
            output = F.conv_transpose2d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            # Convert back to [B, T, H, D]
            return output.permute(0, 2, 3, 1)
        
        # Apply the transformation
        values = apply_conv_transpose(x.values)
        
        # For transposed convolution, we need to upsample the mask
        mask = x.mask
        if self.stride[0] > 1:
            mask = mask.repeat_interleave(self.stride[0], dim=1)
        
        # Adjust mask length to match output
        if mask.shape[1] > values.shape[1]:
            mask = mask[:, :values.shape[1]]
        elif mask.shape[1] < values.shape[1]:
            # Pad mask if needed
            pad_size = values.shape[1] - mask.shape[1]
            mask = F.pad(mask, (0, pad_size), mode='constant', value=True)
        
        return Sequence(values, mask)
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.time_padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.time_padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # For transposed convolution, the output is typically longer than input
            if self.stride[0] == 1:
                # For stride 1, output has same or longer length than input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided transposed convolution, output can be much longer
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] * self.stride[0]):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.time_padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.time_padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.time_padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, *channel_spec.shape)
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None


# =============================================================================
# 3D Convolution Layers
# =============================================================================

class Conv3D(BaseConv):
    """A 3D strided or dilated convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 time_padding: str = 'valid', 
                 spatial_padding: Tuple[Union[str, Tuple[int, int]], Union[str, Tuple[int, int]]] = ('same', 'same'),
                 groups: int = 1, use_bias: bool = True,
                 use_weight_norm: bool = False,
                 activation: Optional[Callable[[Tensor], Tensor]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.time_padding = time_padding
        self.spatial_padding = spatial_padding
        self.groups = groups
        self.use_bias = use_bias
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        
        # Validate parameters
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels // groups, *self.kernel_size
        ) / math.sqrt(in_channels // groups * math.prod(self.kernel_size)))
        
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if use_weight_norm:
            self.weight_scale = nn.Parameter(torch.ones(out_channels))
    
    @property
    def _kernel_size(self) -> Tuple[int, ...]:
        return self.kernel_size
    
    @property
    def _strides(self) -> Tuple[int, ...]:
        return self.stride
    
    @property
    def _dilation_rate(self) -> Tuple[int, ...]:
        return self.dilation
    
    @property
    def _paddings(self) -> Tuple[str, ...]:
        return (self.time_padding, str(self.spatial_padding[0]), str(self.spatial_padding[1]))
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        if len(input_shape) != 3:
            raise ValueError(f'Conv3D requires rank 5 input. Got: {input_shape}')
        
        # Calculate output spatial dimensions based on stride and padding
        # For spatial dimensions, we need to account for stride effects
        spatial_in_0 = input_shape[0]  # Input first spatial dimension
        spatial_in_1 = input_shape[1]  # Input second spatial dimension
        
        # For causal time padding, preserve spatial dimensions in channel spec
        # For other time padding modes, apply stride effects
        if self.time_padding == 'causal':
            # For causal padding, preserve spatial dimensions (used in streaming)
            spatial_out_0 = spatial_in_0
            spatial_out_1 = spatial_in_1
        else:
            # For non-causal padding, apply stride effects
            # Calculate output spatial dimensions based on stride
            # For 'same' padding with stride > 1, output = input // stride
            # For 'valid' padding, need to calculate based on kernel size
            
            # Calculate first spatial dimension output
            if isinstance(self.spatial_padding[0], str):
                if self.spatial_padding[0] == 'same':
                    spatial_out_0 = (spatial_in_0 + self.stride[1] - 1) // self.stride[1]
                elif self.spatial_padding[0] == 'valid':
                    spatial_out_0 = (spatial_in_0 - self.kernel_size[1] + 1 + self.stride[1] - 1) // self.stride[1]
                else:
                    spatial_out_0 = (spatial_in_0 + self.stride[1] - 1) // self.stride[1]
            else:
                spatial_out_0 = (spatial_in_0 + self.stride[1] - 1) // self.stride[1]
            
            # Calculate second spatial dimension output
            if isinstance(self.spatial_padding[1], str):
                if self.spatial_padding[1] == 'same':
                    spatial_out_1 = (spatial_in_1 + self.stride[2] - 1) // self.stride[2]
                elif self.spatial_padding[1] == 'valid':
                    spatial_out_1 = (spatial_in_1 - self.kernel_size[2] + 1 + self.stride[2] - 1) // self.stride[2]
                else:
                    spatial_out_1 = (spatial_in_1 + self.stride[2] - 1) // self.stride[2]
            else:
                spatial_out_1 = (spatial_in_1 + self.stride[2] - 1) // self.stride[2]
        
        return (spatial_out_0, spatial_out_1, self.out_channels)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise for causal padding."""
        if not self.supports_step():
            raise ValueError(f"Layer with padding '{self.time_padding}' does not support step-wise execution")
        
        # For causal padding modes, we need to maintain history
        if self.time_padding in ['causal', 'causal_valid']:
            # Initialize state if needed
            if state is None:
                state = self.get_initial_state(x.shape[0], x.channel_spec, training, constants)
            
            # Concatenate with buffer state
            if state is not None:
                combined = state.concatenate(x)
            else:
                combined = x
            
            # Apply convolution to combined sequence
            output = self.layer(combined, training=training, constants=constants)
            
            # For step-wise execution, we need to return only the output corresponding to new input
            # The buffer maintains (kernel_size - 1) past timesteps, so we take the last few outputs
            if self.stride[0] == 1:
                # For stride 1, output has same length as input
                new_output = output[:, -x.shape[1]:]
            else:
                # For strided convolution, we need to handle output timing carefully
                # This is a simplified approach - in practice, we'd need more sophisticated buffering
                new_output = output[:, -max(1, x.shape[1] // self.stride[0]):]
            
            # Update state: keep last (kernel_size - 1) timesteps
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                new_state = combined[:, -buffer_size:]
            else:
                new_state = None
            
            return new_output, new_state
            
        elif self.time_padding in ['reverse_causal', 'semicausal']:
            # For other causal modes, implement appropriate buffering
            # For now, use simplified approach
            output = self.layer(x, training=training, constants=constants)
            return output, state
        
        else:
            # For non-causal modes, step-wise execution is not supported
            raise ValueError(f"Step-wise execution not supported for padding mode: {self.time_padding}")
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Get initial state for step-wise execution."""
        if not self.supports_step():
            return None
        
        # For causal padding, we need to buffer past inputs
        if self.time_padding in ['causal', 'causal_valid']:
            buffer_size = self.kernel_size[0] - 1
            if buffer_size > 0:
                # Create buffer for past inputs
                buffer_shape = (batch_size, buffer_size, *channel_spec.shape)
                buffer_values = torch.zeros(buffer_shape, dtype=channel_spec.dtype, device=torch.device('cpu'))
                buffer_mask = torch.zeros(batch_size, buffer_size, dtype=torch.bool, device=torch.device('cpu'))
                return Sequence(buffer_values, buffer_mask)
        
        return None
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        if x.values.ndim != 5:
            raise ValueError(f'Conv3D requires rank 5 input. Got: {x.shape}')
        
        # Mask inputs if receptive field is greater than 1
        if self.kernel_size[0] > 1:
            x = x.mask_invalid()
        
        # Get padding values
        time_pad = _get_conv_padding_for_mode(
            self.time_padding, self.kernel_size[0], self.stride[0], self.dilation[0]
        )
        
        # Get spatial padding values
        spatial_pad_0 = _get_conv_padding_for_mode(
            str(self.spatial_padding[0]), self.kernel_size[1], self.stride[1], self.dilation[1]
        ) if isinstance(self.spatial_padding[0], str) else self.spatial_padding[0]
        
        spatial_pad_1 = _get_conv_padding_for_mode(
            str(self.spatial_padding[1]), self.kernel_size[2], self.stride[2], self.dilation[2]
        ) if isinstance(self.spatial_padding[1], str) else self.spatial_padding[1]
        
        # Apply weight normalization if enabled
        weight = self.weight
        if self.use_weight_norm:
            weight = weight * self.weight_scale.view(-1, 1, 1, 1, 1) / torch.norm(
                weight, dim=[1, 2, 3, 4], keepdim=True
            )
        
        # Apply convolution
        def apply_conv(values):
            # Input is [B, T, H, W, D] -> Convert to [B, D, T, H, W] for conv3d
            values = values.permute(0, 4, 1, 2, 3)
            
            # Apply padding: [width_left, width_right, height_left, height_right, time_left, time_right]
            padding = (spatial_pad_1[0], spatial_pad_1[1], spatial_pad_0[0], spatial_pad_0[1], time_pad[0], time_pad[1])
            if sum(padding) > 0:
                values = F.pad(values, padding, mode='constant', value=0)
            
            # Apply convolution
            output = F.conv3d(
                values,
                weight,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups
            )
            
            # Apply activation if specified
            if self.activation is not None:
                output = self.activation(output)
            
            # Convert back to [B, T, H, W, D]
            return output.permute(0, 2, 3, 4, 1)
        
        # Apply the transformation
        values = apply_conv(x.values)
        
        # Compute output mask (only time dimension affects mask)
        mask = _compute_conv_mask(
            x.mask, self.kernel_size[0], self.stride[0], self.dilation[0], self.time_padding
        )
        
        # Adjust mask length to match output
        if mask.shape[1] != values.shape[1]:
            mask = mask[:, :values.shape[1]]
        
        # Return appropriate sequence type
        if self.use_bias or self.kernel_size[0] > 1:
            return Sequence(values, mask)
        else:
            return type(x)(values, mask) 