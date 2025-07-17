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
"""Normalization layers for PyTorch."""

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
    # Standard normalization layers
    'LayerNorm',
    'RMSNorm',
    'BatchNorm',
    'GroupNorm',
    'InstanceNorm',
    
    # Sequence-specific normalization layers
    'SequenceLayerNorm',
    'MaskedBatchNorm',
]


# =============================================================================
# Utility Functions
# =============================================================================

def _validate_and_normalize_axes(axis: Union[int, List[int]], shape: torch.Size) -> List[int]:
    """Validate and normalize axis indices for the given shape."""
    if isinstance(axis, int):
        axis = [axis]
    
    normalized_axes = []
    for ax in axis:
        if ax < 0:
            ax = len(shape) + ax
        if ax < 0 or ax >= len(shape):
            raise ValueError(f"Axis {ax} is out of bounds for shape {shape}")
        normalized_axes.append(ax)
    
    return normalized_axes


def _compute_masked_moments(values: Tensor, mask: Tensor, axis: List[int], 
                           keepdims: bool = True) -> Tuple[Tensor, Tensor]:
    """Compute mean and variance for masked sequences."""
    # Expand mask to match values shape
    expanded_mask = mask
    while expanded_mask.ndim < values.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    
    # Create a mask that matches the values shape
    mask_values = expanded_mask.expand_as(values)
    
    # Set masked values to 0
    masked_values = values * mask_values.float()
    
    # Count valid elements
    valid_count = mask_values.float().sum(dim=axis, keepdim=keepdims)
    valid_count = torch.clamp(valid_count, min=1.0)  # Avoid division by zero
    
    # Compute mean
    mean = masked_values.sum(dim=axis, keepdim=keepdims) / valid_count
    
    # Compute variance
    if keepdims:
        centered = masked_values - mean
    else:
        # Expand mean to match values shape
        mean_expanded = mean
        for ax in sorted(axis):
            mean_expanded = mean_expanded.unsqueeze(ax)
        centered = masked_values - mean_expanded
    
    variance = (centered * centered * mask_values.float()).sum(dim=axis, keepdim=keepdims) / valid_count
    
    return mean, variance


# =============================================================================
# Standard Normalization Layers
# =============================================================================

class LayerNorm(StatelessPointwise):
    """Layer normalization for sequences.
    
    Applies layer normalization to the specified dimensions of the input.
    This implementation is sequence-aware and only considers valid timesteps.
    """
    
    def __init__(self, 
                 normalized_shape: Union[int, List[int]],
                 axis: Union[int, List[int]] = -1,
                 epsilon: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Handle different input formats
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        
        self.normalized_shape = normalized_shape
        self.axis = axis if isinstance(axis, list) else [axis]
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine
        self.bias = bias
        
        # Create parameters
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias_param = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias_param', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias_param', None)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        
        # Validate axis
        normalized_axes = _validate_and_normalize_axes(self.axis, x.values.shape)
        
        # Compute moments
        mean = torch.mean(x.values, dim=normalized_axes, keepdim=True)
        variance = torch.var(x.values, dim=normalized_axes, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (x.values - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply affine transformation
        if self.elementwise_affine:
            # Reshape weight and bias to broadcast correctly
            weight_shape = [1] * x.values.ndim
            for ax in normalized_axes:
                weight_shape[ax] = x.values.shape[ax]
            
            weight = self.weight.view(weight_shape)
            normalized = normalized * weight
            
            if self.bias_param is not None:
                bias = self.bias_param.view(weight_shape)
                normalized = normalized + bias
        
        return type(x)(normalized, x.mask)


class RMSNorm(StatelessPointwise):
    """Root Mean Square Layer Normalization.
    
    A simplified version of LayerNormalization used in T5.
    No mean statistics or offset terms are included.
    """
    
    def __init__(self, 
                 normalized_shape: Union[int, List[int]],
                 axis: Union[int, List[int]] = -1,
                 epsilon: float = 1e-6,
                 elementwise_affine: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Handle different input formats
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        
        self.normalized_shape = normalized_shape
        self.axis = axis if isinstance(axis, list) else [axis]
        self.epsilon = epsilon
        self.elementwise_affine = elementwise_affine
        
        # Create parameters
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        
        # Validate axis
        normalized_axes = _validate_and_normalize_axes(self.axis, x.values.shape)
        
        # Compute RMS
        mean_square = torch.mean(x.values ** 2, dim=normalized_axes, keepdim=True)
        rms = torch.sqrt(mean_square + self.epsilon)
        
        # Normalize
        normalized = x.values / rms
        
        # Apply scale
        if self.elementwise_affine:
            # Reshape weight to broadcast correctly
            weight_shape = [1] * x.values.ndim
            for ax in normalized_axes:
                weight_shape[ax] = x.values.shape[ax]
            
            weight = self.weight.view(weight_shape)
            normalized = normalized * weight
        
        return type(x)(normalized, x.mask)


class BatchNorm(StatelessPointwise):
    """Batch normalization for sequences.
    
    Applies batch normalization to the channels dimension of input sequences.
    This implementation is sequence-aware and only considers valid timesteps.
    
    Note: Step-wise training is not supported since it cannot be made identical
    to layer-wise training (it's not causal).
    """
    
    def __init__(self, 
                 num_features: int,
                 axis: int = -1,
                 epsilon: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.num_features = num_features
        self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Create parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    @property
    def supports_step(self) -> bool:
        """BatchNorm supports step-wise execution only when using running statistics."""
        return self.track_running_stats
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise.
        
        Step-wise training is not supported for BatchNorm since it cannot be made
        identical to layer-wise training (it's not causal).
        
        Step-wise inference is only supported when track_running_stats=True.
        """
        if not self.supports_step:
            raise ValueError('BatchNorm does not support step-wise execution when track_running_stats=False. Use layer-wise execution instead.')
        
        if training:
            raise ValueError('Step-wise training is not supported for BatchNormalization.')
        return self.layer(x, training=training, constants=constants), state
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        
        # Mask invalid values to prevent NaN propagation from padded timesteps
        x = x.mask_invalid()
        
        # Normalize axis
        axis = self.axis if self.axis >= 0 else len(x.values.shape) + self.axis
        
        if training:
            # Compute batch statistics from valid timesteps only
            if x.mask is not None:
                # Use masked moments computation
                reduction_axes = [i for i in range(x.values.ndim) if i != axis]
                mean, variance = _compute_masked_moments(x.values, x.mask, reduction_axes, keepdims=False)
            else:
                # Standard batch normalization
                reduction_axes = [i for i in range(x.values.ndim) if i != axis]
                mean = torch.mean(x.values, dim=reduction_axes)
                variance = torch.var(x.values, dim=reduction_axes, unbiased=False)
            
            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                    else:
                        exponential_average_factor = self.momentum
                    
                    self.running_mean = ((1 - exponential_average_factor) * self.running_mean + 
                                       exponential_average_factor * mean)
                    self.running_var = ((1 - exponential_average_factor) * self.running_var + 
                                      exponential_average_factor * variance)
            
            # Use computed statistics
            norm_mean = mean
            norm_var = variance
        else:
            # Use running statistics
            if self.track_running_stats:
                norm_mean = self.running_mean
                norm_var = self.running_var
            else:
                # Fallback to computing from current batch using masked moments
                reduction_axes = [i for i in range(x.values.ndim) if i != axis]
                if x.mask is not None:
                    # Use masked moments computation for proper padding invariance
                    norm_mean, norm_var = _compute_masked_moments(x.values, x.mask, reduction_axes, keepdims=False)
                else:
                    # Standard batch normalization
                    norm_mean = torch.mean(x.values, dim=reduction_axes)
                    norm_var = torch.var(x.values, dim=reduction_axes, unbiased=False)
        
        # Reshape for broadcasting
        broadcast_shape = [1] * x.values.ndim
        broadcast_shape[axis] = self.num_features
        
        norm_mean = norm_mean.view(broadcast_shape)
        norm_var = norm_var.view(broadcast_shape)
        
        # Normalize
        normalized = (x.values - norm_mean) / torch.sqrt(norm_var + self.epsilon)
        
        # Apply affine transformation
        if self.affine:
            weight = self.weight.view(broadcast_shape)
            bias = self.bias.view(broadcast_shape)
            normalized = normalized * weight + bias
        
        return type(x)(normalized, x.mask)


class GroupNorm(StatelessPointwise):
    """Group normalization for sequences.
    
    Applies group normalization to the input by dividing channels into groups
    and computing statistics within each group.
    """
    
    def __init__(self, 
                 num_groups: int,
                 num_channels: int,
                 epsilon: float = 1e-5,
                 affine: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.affine = affine
        
        # Create parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise.
        
        GroupNorm does not support step-wise execution because it requires
        statistics computed over the entire sequence, not individual steps.
        """
        raise ValueError('GroupNorm does not support step-wise execution. Use layer-wise execution instead.')
    
    @property
    def supports_step(self) -> bool:
        """GroupNorm does not support step-wise execution."""
        return False
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        
        # Mask invalid values to prevent NaN propagation from padded timesteps
        x = x.mask_invalid()
        
        # Assume channel dimension is the last dimension
        values = x.values
        original_shape = values.shape
        
        # Reshape to separate groups
        # From [batch, time, ..., channels] to [batch, time, ..., groups, channels_per_group]
        group_shape = list(original_shape[:-1]) + [self.num_groups, self.num_channels // self.num_groups]
        values = values.view(group_shape)
        
        # Compute group statistics using masked moments if mask is available
        # Reduce over all dimensions except batch and groups
        reduction_axes = list(range(1, len(group_shape) - 1))  # All except batch and groups
        
        if x.mask is not None:
            # Use masked moments computation for proper padding invariance
            mean, variance = _compute_masked_moments(values, x.mask, reduction_axes, keepdims=True)
        else:
            mean = torch.mean(values, dim=reduction_axes, keepdim=True)
            variance = torch.var(values, dim=reduction_axes, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (values - mean) / torch.sqrt(variance + self.epsilon)
        
        # Reshape back to original shape
        normalized = normalized.view(original_shape)
        
        # Apply affine transformation
        if self.affine:
            normalized = normalized * self.weight + self.bias
        
        return Sequence(normalized, x.mask)


class InstanceNorm(StatelessPointwise):
    """Instance normalization for sequences.
    
    Applies instance normalization by computing statistics for each sample
    and channel independently.
    """
    
    def __init__(self, 
                 num_features: int,
                 epsilon: float = 1e-5,
                 affine: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.num_features = num_features
        self.epsilon = epsilon
        self.affine = affine
        
        # Create parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise.
        
        InstanceNorm does not support step-wise execution because it requires
        statistics computed over the entire sequence, not individual steps.
        """
        raise ValueError('InstanceNorm does not support step-wise execution. Use layer-wise execution instead.')
    
    @property
    def supports_step(self) -> bool:
        """InstanceNorm does not support step-wise execution."""
        return False
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return input_shape
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return input_dtype
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        
        # Mask invalid values to prevent NaN propagation from padded timesteps
        x = x.mask_invalid()
        
        # Compute instance statistics
        # Reduce over all dimensions except batch and channel
        reduction_axes = list(range(1, x.values.ndim - 1))  # All except batch and channel
        
        if x.mask is not None:
            # Use masked moments computation
            mean, variance = _compute_masked_moments(x.values, x.mask, reduction_axes, keepdims=True)
        else:
            mean = torch.mean(x.values, dim=reduction_axes, keepdim=True)
            variance = torch.var(x.values, dim=reduction_axes, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (x.values - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply affine transformation
        if self.affine:
            normalized = normalized * self.weight + self.bias
        
        return Sequence(normalized, x.mask)


# =============================================================================
# Sequence-Specific Normalization Layers
# =============================================================================

class SequenceLayerNorm(StatelessPointwise):
    """Sequence-aware layer normalization.
    
    This is an alias for LayerNorm that emphasizes sequence awareness.
    """
    
    def __init__(self, 
                 normalized_shape: Union[int, List[int]],
                 axis: Union[int, List[int]] = -1,
                 epsilon: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.layer_norm = LayerNorm(
            normalized_shape=normalized_shape,
            axis=axis,
            epsilon=epsilon,
            elementwise_affine=elementwise_affine,
            bias=bias,
            name=name
        )
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.layer_norm.get_output_shape(input_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.layer_norm.get_output_dtype(input_dtype)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        return self.layer_norm.layer(x, training, initial_state, constants)


class MaskedBatchNorm(StatelessPointwise):
    """Masked batch normalization for sequences.
    
    This is an alias for BatchNorm that emphasizes mask awareness.
    """
    
    def __init__(self, 
                 num_features: int,
                 axis: int = -1,
                 epsilon: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.batch_norm = BatchNorm(
            num_features=num_features,
            axis=axis,
            epsilon=epsilon,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            name=name
        )
    
    def step(self, x: Sequence, state: State, training: bool,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Process this layer step-wise by delegating to BatchNorm."""
        return self.batch_norm.step(x, state, training, constants)
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        return self.batch_norm.get_output_shape(input_shape, constants)
    
    def get_output_dtype(self, input_dtype: DType) -> DType:
        return self.batch_norm.get_output_dtype(input_dtype)
    
    def layer(self, x: Sequence, training: bool = False, 
              initial_state: Optional[State] = None, 
              constants: Optional[Constants] = None) -> Sequence:
        return self.batch_norm.layer(x, training, initial_state, constants) 