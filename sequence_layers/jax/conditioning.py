# Copyright 2024 Google LLC
#
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
"""Conditioning layers."""

import abc
import dataclasses
import enum

import flax.linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import dense
from sequence_layers.jax import types
from sequence_layers.jax import utils


__all__ = (
    # go/keep-sorted start
    'Conditioning',
    # go/keep-sorted end
)


def _get_conditioning(
    layer: types.SequenceLayer,
    conditioning_name: str,
    constants: types.Constants | None,
) -> jax.Array | types.Sequence:
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
  elif not isinstance(conditioning, types.ARRAY_LIKE_TYPES) and not isinstance(
      conditioning, types.Sequence
  ):
    raise ValueError(
        f'Unexpected conditioning "{conditioning_name}" having '
        f'type: {type(conditioning)}: {conditioning=}'
    )

  return conditioning


class BaseConditioning(
    types.PreservesType, types.SequenceLayer, metaclass=abc.ABCMeta
):
  """Base class for conditioning types."""

  @enum.unique
  class Projection(enum.Enum):
    """The type of projection to perform."""

    # No projection.
    IDENTITY = 1
    # Dense projection from every element of c at a given time step, to a tensor
    # of the same shape as x at given time step (c.channel_shape.num_elements()
    # to x.channel_shape.num_elements()).
    LINEAR = 2
    # Dense projection from every element of c at a given time step, to a tensor
    # of shape [2, x.shape...] at given time step (
    # c.channel_shape.num_elements() to 2 * x.channel_shape.num_elements()).
    LINEAR_AFFINE = 3

  @enum.unique
  class Combination(enum.Enum):
    """The type of combination to perform."""

    # Broadcast-add conditioning.
    ADD = 1
    # Broadcast-concat conditioning.
    CONCAT = 2
    # Affine conditioning. Requires LINEAR_AFFINE projection.
    AFFINE = 3

  def _projected_condition_shape(
      self, input_shape: types.Shape, condition_shape: types.Shape
  ) -> types.Shape:
    projection_channel_shape = self._projection_channel_shape
    if projection_channel_shape is None:
      projection_channel_shape = input_shape

    match self._projection:
      case self.Projection.IDENTITY:
        return condition_shape
      case self.Projection.LINEAR:
        return projection_channel_shape
      case self.Projection.LINEAR_AFFINE:
        return (2,) + projection_channel_shape
      case _:
        raise ValueError(f'Unsupported projection: {self._projection}')

  def _get_conditioning_channel_shape(
      self,
      constants: types.Constants | None,
  ) -> types.Shape:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if isinstance(conditioning, types.Sequence):
      return conditioning.channel_shape
    else:
      return conditioning.shape[1:]

  @property
  @abc.abstractmethod
  def _conditioning_name(self) -> str:
    pass

  @property
  @abc.abstractmethod
  def _projection(self) -> Projection:
    pass

  @property
  @abc.abstractmethod
  def _projection_channel_shape(self) -> types.Shape | None:
    pass

  @property
  @abc.abstractmethod
  def _combination(self) -> Combination:
    pass

  @property
  @abc.abstractmethod
  def _dtype(self) -> types.DType | None:
    pass

  @property
  @abc.abstractmethod
  def _param_dtype(self) -> types.DType:
    pass

  def _validate(self):
    if (
        self._combination == self.Combination.AFFINE
        and self._projection != self.Projection.LINEAR_AFFINE
    ):
      raise ValueError('AFFINE combination requires LINEAR_AFFINE projection.')

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    self._validate()
    conditioning_channel_shape = self._get_conditioning_channel_shape(constants)
    projected_conditioning_shape = self._projected_condition_shape(
        input_shape, conditioning_channel_shape
    )
    match self._combination:
      case self.Combination.ADD:
        return jnp.broadcast_shapes(input_shape, projected_conditioning_shape)
      case self.Combination.CONCAT:
        input_inner_dim = input_shape[-1] if input_shape else 1
        projected_conditioning_inner_dim = (
            projected_conditioning_shape[-1]
            if projected_conditioning_shape
            else 1
        )
        output_inner_dim = input_inner_dim + projected_conditioning_inner_dim
        output_outer_shape = jnp.broadcast_shapes(
            input_shape[:-1], projected_conditioning_shape[:-1]
        )
        return output_outer_shape + (output_inner_dim,)
      case self.Combination.AFFINE:
        projected_conditioning_shape = projected_conditioning_shape[1:]
        return jnp.broadcast_shapes(input_shape, projected_conditioning_shape)
      case _:
        raise ValueError(f'Unsupported combination: {self._combination}')

  def _project_and_combine(
      self, x: types.Sequence, conditioning: types.Sequence, training: bool
  ):
    return self._combine(x, self._project(x, conditioning, training))

  @nn.compact
  def _project(
      self, x: types.Sequence, conditioning: types.Sequence, training: bool
  ) -> types.Sequence:
    self._validate()
    projection_channel_shape = self._projection_channel_shape
    if projection_channel_shape is None:
      projection_channel_shape = x.channel_shape
    match self._projection:
      case self.Projection.IDENTITY:
        return conditioning
      case self.Projection.LINEAR:
        return (
            dense.DenseShaped.Config(
                projection_channel_shape,
                param_dtype=self._param_dtype,
                dtype=self._dtype,
                name='dense',
            )
            .make()
            .layer(conditioning, training=training)
        )
      case self.Projection.LINEAR_AFFINE:
        return (
            dense.DenseShaped.Config(
                (2,) + projection_channel_shape,
                param_dtype=self._param_dtype,
                dtype=self._dtype,
                name='dense',
            )
            .make()
            .layer(conditioning, training=training)
        )
      case _:
        raise ValueError(f'Unsupported projection: {self._projection}')

  @nn.nowrap
  def _combine(
      self, x: types.Sequence, conditioning: types.Sequence
  ) -> types.Sequence:
    self._validate()

    def _affine_fn(x, conditioning):
      scale, shift = utils.sequence_unstack(conditioning, axis=2)
      # Offset scale by 1.0
      scale = scale.apply_values(lambda v: v + 1.0)
      return utils.sequence_broadcast_affine(x, scale, shift)

    match self._combination:
      case self.Combination.ADD:
        combine_fn = utils.sequence_broadcast_add
      case self.Combination.CONCAT:
        combine_fn = utils.sequence_broadcast_concat
      case self.Combination.AFFINE:
        combine_fn = _affine_fn
      case _:
        raise ValueError(f'Unsupported combination: {self._combination}')
    return combine_fn(x, conditioning)


def _tensor_to_fake_sequence(t: jax.Array) -> types.MaskedSequence:
  batch_size = t.shape[0]
  return types.MaskedSequence(
      t[:, jnp.newaxis],
      jnp.full([batch_size, 1], True),
  )


class Conditioning(BaseConditioning):
  """Conditions the sequence x on a conditioning sequence c.

  Conditioning is done in a time-synchronized way, where each time step of x is
  conditioned on the corresponding time step of c.

  Conditioning is defined as Combine(x, Project(c)), where supported projection
  and combination functions are available in Projection and Combination enums
  above.

  All valid broadcasts between x and c channel_shapes are supported, e.g.:
    - Projection=IDENTITY, Combination=ADD:
      - (x: [b, t_x, 3], c: [b, t_c, 3]) -> c_x: [b, t_x, 3]
      - (x: [b, t_x, 3, 1, 5], c: [b, t_c, 2, 5]) -> c_x: [b, t_x, 3, 2, 5]
    - Projection=IDENTITY, Combination=CONCAT:
      - (x: [b, t_x, 3], c: [b, t_c, 7]) -> c_x: [b, t_x, 10]
      - (x: [b, t_x, 3, 1, 5], c: [b, t_c, 2, 7]) -> c_x: [b, t_x, 3, 2, 12]
    - Projection=LINEAR, Combination=ADD:
      - (x: [b, t_x, 3], c: [b, t_c, 7]) -> c_x: [b, t_x, 3]
      - (x: [b, t_x, 3, 1, 5], c: [b, t_c, 2, 7]) -> c_x: [b, t_x, 3, 1, 5]
    - Projection=LINEAR, Combination=CONCAT:
      - (x: [b, t_x, 3], c: [b, t_c, 7]) -> c_x: [b, t_x, 6]
      - (x: [b, t_x, 3, 1, 5], c: [b, t_c, 2, 7]) -> c_x: [b, t_x, 3, 1, 10]

  Where t_x = t_c. For t_x != t_c, see UpsampleConditioning.
  TODO(rryan): Implement UpsampleConditioning.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Conditioning."""

    # The name of the conditioning sequence or array in the constants
    # dictionary.
    conditioning_name: str
    # The type of projection to perform to project the conditioning before
    # combination.
    projection: BaseConditioning.Projection
    # The type of combination to perform between the projected conditioning and
    # the input sequence.
    combination: BaseConditioning.Combination
    # If projection is LINEAR or LINEAR_AFFINE, the channel shape to project the
    # conditioning to. If unspecified, projects to the input sequence's channel
    # shape.
    projection_channel_shape: types.Shape | None = None
    # If true, the conditioning sequence is expected to be streamed at the same
    # block_size as the input sequence.
    streaming: bool = False
    # The dtype to use for layer compute.
    dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'Conditioning':
      return Conditioning(self, name=self.name)

  config: Config

  @property
  def _conditioning_name(self) -> str:
    return self.config.conditioning_name

  @property
  def _projection(self) -> BaseConditioning.Projection:
    return self.config.projection

  @property
  def _projection_channel_shape(self) -> types.Shape | None:
    return self.config.projection_channel_shape

  @property
  def _combination(self) -> BaseConditioning.Combination:
    return self.config.combination

  @property
  def _dtype(self) -> types.DType | None:
    return self.config.dtype

  @property
  def _param_dtype(self) -> types.DType:
    return self.config.param_dtype

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if isinstance(conditioning, types.Sequence) and not self.config.streaming:
      return jnp.zeros([batch_size], jnp.int32)
    else:
      return ()

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if not isinstance(conditioning, types.Sequence):
      conditioning = _tensor_to_fake_sequence(conditioning)
    broadcasted_shape = jnp.broadcast_shapes(
        x.shape[:2], conditioning.shape[:2]
    )
    utils.assert_is_compatible_with(x.shape[:2], broadcasted_shape)
    return self._project_and_combine(x, conditioning, training)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    conditioning_batch_shape = conditioning.shape[:1]
    if not isinstance(conditioning, types.Sequence):
      conditioning = _tensor_to_fake_sequence(conditioning)
    elif not self.config.streaming:
      # If not streaming, time slice the conditioning.
      time_index = state
      step_size = x.shape[1]
      conditioning = utils.batched_time_slice(
          conditioning, time_index, step_size
      )
      state = time_index + step_size
    utils.assert_is_compatible_with(x.shape[:1], conditioning_batch_shape)
    conditioned_x = self._project_and_combine(x, conditioning, training)
    return conditioned_x, state
