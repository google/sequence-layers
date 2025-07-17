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
"""Conditioning layers for PyTorch."""

import abc
import enum
from typing import Any, Callable, Optional, Union, Tuple, Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sequence_layers.pytorch.types import (
    Sequence,
    SequenceLayer,
    PreservesType,
    ChannelSpec,
    Shape,
    DType,
    Constants,
    State,
)


__all__ = [
    'Conditioning',
    'ProjectionMode',
    'CombinationMode',
]


def _get_conditioning(
    layer: SequenceLayer,
    conditioning_name: str,
    constants: Optional[Constants],
) -> Union[Tensor, Sequence]:
    """Gets the conditioning from constants and does basic validation."""
    if constants is None:
        raise ValueError(
            f'{layer} requires the conditioning to be provided via '
            f'constants, got: {constants}'
        )
    conditioning = constants.get(conditioning_name)
    if conditioning is None:
        raise ValueError(
            f'{layer} expected {conditioning_name} to be present in '
            f'constants, got: {constants}'
        )
    elif not isinstance(conditioning, (torch.Tensor, Sequence)):
        raise ValueError(
            f'Unexpected conditioning "{conditioning_name}" having '
            f'type: {type(conditioning)}: {conditioning=}'
        )
    
    return conditioning


@enum.unique
class ProjectionMode(enum.Enum):
    """The type of projection to perform."""
    
    # No projection.
    IDENTITY = 'identity'
    # Dense projection from conditioning to same shape as input.
    LINEAR = 'linear'
    # Dense projection from conditioning to [2, input_shape].
    LINEAR_AFFINE = 'linear_affine'


@enum.unique
class CombinationMode(enum.Enum):
    """The type of combination to perform."""
    
    # Broadcast-add conditioning.
    ADD = 'add'
    # Broadcast-concat conditioning.
    CONCAT = 'concat'
    # Affine conditioning. Requires LINEAR_AFFINE projection.
    AFFINE = 'affine'
    # Affine shift conditioning. Requires LINEAR projection.
    AFFINE_SHIFT = 'affine_shift'
    # Affine scale conditioning. Requires LINEAR projection.
    AFFINE_SCALE = 'affine_scale'
    # Broadcast-multiply conditioning.
    MUL = 'mul'
    # Broadcast-concat conditioning via prepending.
    CONCAT_BEFORE = 'concat_before'


class BaseConditioning(PreservesType, SequenceLayer, metaclass=abc.ABCMeta):
    """Base class for conditioning types."""
    
    def __init__(self,
                 conditioning_name: str,
                 projection: ProjectionMode,
                 combination: CombinationMode,
                 projection_channel_shape: Optional[Shape] = None,
                 affine_scale_offset: float = 1.0,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.conditioning_name = conditioning_name
        self.projection = projection
        self.combination = combination
        self.projection_channel_shape = projection_channel_shape
        self.affine_scale_offset = affine_scale_offset
        
        # Validate combination/projection compatibility
        self._validate()
        
        # Projection layer (if needed)
        self.projection_layer = None
        self.built = False
    
    def _validate(self):
        """Validate combination/projection compatibility."""
        if (self.combination == CombinationMode.AFFINE and 
            self.projection != ProjectionMode.LINEAR_AFFINE):
            raise ValueError('AFFINE combination requires LINEAR_AFFINE projection.')
        if (self.combination == CombinationMode.AFFINE_SHIFT and 
            self.projection != ProjectionMode.LINEAR):
            raise ValueError('AFFINE_SHIFT combination requires LINEAR projection.')
        if (self.combination == CombinationMode.AFFINE_SCALE and 
            self.projection != ProjectionMode.LINEAR):
            raise ValueError('AFFINE_SCALE combination requires LINEAR projection.')
        if (self.combination == CombinationMode.MUL and 
            self.projection not in [ProjectionMode.LINEAR, ProjectionMode.IDENTITY]):
            raise ValueError('MUL combination requires LINEAR or IDENTITY projection.')
    
    def _get_conditioning_channel_shape(self, constants: Optional[Constants]) -> Shape:
        """Get the channel shape of the conditioning."""
        conditioning = _get_conditioning(self, self.conditioning_name, constants)
        if isinstance(conditioning, Sequence):
            return conditioning.channel_shape
        else:
            return conditioning.shape[2:]  # Skip batch and time dimensions
    
    def _projected_condition_shape(self, input_shape: Shape, condition_shape: Shape) -> Shape:
        """Get the projected condition shape."""
        projection_channel_shape = self.projection_channel_shape
        if projection_channel_shape is None:
            projection_channel_shape = input_shape
        
        if self.projection == ProjectionMode.IDENTITY:
            return condition_shape
        elif self.projection == ProjectionMode.LINEAR:
            return projection_channel_shape
        elif self.projection == ProjectionMode.LINEAR_AFFINE:
            return (2,) + projection_channel_shape
        else:
            raise ValueError(f'Unsupported projection: {self.projection}')
    
    def _build_projection(self, input_shape: Shape, constants: Optional[Constants]):
        """Build the projection layer if needed."""
        if self.built:
            return
        
        if self.projection == ProjectionMode.IDENTITY:
            self.projection_layer = None
        else:
            condition_shape = self._get_conditioning_channel_shape(constants)
            projected_shape = self._projected_condition_shape(input_shape, condition_shape)
            
            # Calculate input and output sizes
            condition_size = 1
            for dim in condition_shape:
                condition_size *= dim
            
            projected_size = 1
            for dim in projected_shape:
                projected_size *= dim
            
            # Create linear layer
            self.projection_layer = nn.Linear(condition_size, projected_size)
        
        self.built = True
    
    def _apply_projection(self, conditioning: Union[Tensor, Sequence], input_shape: Shape) -> Tensor:
        """Apply projection to conditioning."""
        if isinstance(conditioning, Sequence):
            conditioning_values = conditioning.values
        else:
            conditioning_values = conditioning
        
        if self.projection == ProjectionMode.IDENTITY:
            return conditioning_values
        
        # Flatten conditioning for projection
        batch_size, time_steps = conditioning_values.shape[:2]
        conditioning_flat = conditioning_values.view(batch_size * time_steps, -1)
        
        # Apply projection
        projected_flat = self.projection_layer(conditioning_flat)
        
        # Reshape back
        projected_shape = self._projected_condition_shape(input_shape, conditioning_values.shape[2:])
        projected = projected_flat.view(batch_size, time_steps, *projected_shape)
        
        return projected
    
    def _apply_combination(self, x: Sequence, projected_conditioning: Tensor) -> Sequence:
        """Apply combination of input and projected conditioning."""
        if self.combination == CombinationMode.ADD:
            combined_values = x.values + projected_conditioning
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.CONCAT:
            combined_values = torch.cat([x.values, projected_conditioning], dim=-1)
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.CONCAT_BEFORE:
            combined_values = torch.cat([projected_conditioning, x.values], dim=-1)
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.MUL:
            combined_values = x.values * projected_conditioning
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.AFFINE:
            # Split projected conditioning into scale and shift
            scale, shift = torch.chunk(projected_conditioning, 2, dim=2)
            # Remove the split dimension: (batch, time, 1, channels) -> (batch, time, channels)
            scale = scale.squeeze(2)
            shift = shift.squeeze(2)
            scale = scale + self.affine_scale_offset
            combined_values = x.values * scale + shift
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.AFFINE_SHIFT:
            combined_values = x.values + projected_conditioning
            return Sequence(combined_values, x.mask)
        
        elif self.combination == CombinationMode.AFFINE_SCALE:
            scale = projected_conditioning + self.affine_scale_offset
            combined_values = x.values * scale
            return Sequence(combined_values, x.mask)
        
        else:
            raise ValueError(f'Unknown combination mode: {self.combination}')
    
    def get_output_shape(self, input_shape: Shape, constants: Optional[Constants] = None) -> Shape:
        """Get output shape after conditioning."""
        if self.combination in [CombinationMode.CONCAT, CombinationMode.CONCAT_BEFORE]:
            condition_shape = self._get_conditioning_channel_shape(constants)
            projected_shape = self._projected_condition_shape(input_shape, condition_shape)
            
            # Calculate new channel dimension
            input_channels = input_shape[-1] if input_shape else 1
            projected_channels = projected_shape[-1] if projected_shape else 1
            
            return input_shape[:-1] + (input_channels + projected_channels,)
        else:
            return input_shape
    
    def get_initial_state(self, batch_size: int, channel_spec: ChannelSpec,
                         training: bool = False, constants: Optional[Constants] = None) -> State:
        """Returns empty state (stateless layer)."""
        return ()
    
    def step(self, x: Sequence, state: State, training: bool = False,
             constants: Optional[Constants] = None) -> Tuple[Sequence, State]:
        """Apply conditioning step-wise."""
        self._build_projection(x.channel_shape, constants)
        
        # Get conditioning
        conditioning = _get_conditioning(self, self.conditioning_name, constants)
        
        # Apply projection
        projected_conditioning = self._apply_projection(conditioning, x.channel_shape)
        
        # Apply combination
        output = self._apply_combination(x, projected_conditioning)
        
        return output, state
    
    def layer(self, x: Sequence, training: bool = False,
              initial_state: Optional[State] = None,
              constants: Optional[Constants] = None) -> Sequence:
        """Apply conditioning layer-wise."""
        output, _ = self.step(x, (), training, constants)
        return output


class Conditioning(BaseConditioning):
    """Conditioning layer with configurable projection and combination modes."""
    
    def __init__(self,
                 conditioning_name: str,
                 projection: ProjectionMode = ProjectionMode.IDENTITY,
                 combination: CombinationMode = CombinationMode.ADD,
                 projection_channel_shape: Optional[Shape] = None,
                 affine_scale_offset: float = 1.0,
                 name: Optional[str] = None):
        super().__init__(
            conditioning_name=conditioning_name,
            projection=projection,
            combination=combination,
            projection_channel_shape=projection_channel_shape,
            affine_scale_offset=affine_scale_offset,
            name=name
        ) 