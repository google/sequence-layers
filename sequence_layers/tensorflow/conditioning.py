# Copyright 2023 Google LLC
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

import enum
from typing import Optional, Tuple, Union

from sequence_layers.tensorflow import dense
from sequence_layers.tensorflow import simple
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


def _get_conditioning(
    layer: types.SequenceLayer,
    conditioning_name: str,
    constants: types.Constants,
) -> Union[tf.Tensor, types.Sequence]:
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
  elif isinstance(conditioning, tf.Tensor):
    # Conditioning is expected to be [batch_size, ...] where ... is conditioning
    # information for the channels of the input sequence. Bare tensors cannot be
    # treated as sequences.
    conditioning.shape[1:].assert_is_fully_defined()
  elif isinstance(conditioning, types.Sequence):
    conditioning.channel_shape.assert_is_fully_defined()
  else:
    raise ValueError(f'Unexpected conditioning of type: {type(conditioning)}')
  return conditioning


def _time_slice(
    x: types.Sequence,
    begin: tf.Tensor,
    size: tf.Tensor,
    name: Optional[str] = None,
) -> types.Sequence:
  """Gathers a slice of time dimension of the sequence."""
  with tf.name_scope(name or 'time_slice'):
    begin = tf.convert_to_tensor(begin, name='begin')
    size = tf.convert_to_tensor(size, name='size')
    begin.shape.assert_has_rank(0)
    size.shape.assert_has_rank(0)
    time = utils.smart_dimension_size(x.values, 1)

    available = tf.maximum(time - begin, 0)
    valid_size = tf.minimum(size, available)

    # Use tf.range(size) and clip indices that are out of range instead of
    # tf.range(valid_size) so that shapes are fixed (therefore TPU compatible).
    # TODO(rryan): Handle zero length?
    time_indices = tf.minimum(begin + tf.range(size), tf.maximum(0, time - 1))
    time_indices.shape.assert_has_rank(1)

    # Make a mask that is zero if the index went beyond the end.
    valid_mask = tf.sequence_mask([valid_size], size, dtype=types.MASK_DTYPE)

    return types.Sequence(
        tf.gather(x.values, time_indices, axis=1),
        valid_mask * tf.gather(x.mask, time_indices, axis=1),
    )


def _upsample_tensor(x: tf.Tensor, upsample_ratio: int) -> tf.Tensor:
  """Upsample x by a factor of upsample_ratio (simply repeating)."""
  with tf.name_scope('upsample_tensor'):
    x = tf.convert_to_tensor(x)
    x.shape.with_rank_at_least(2)
    if upsample_ratio == 1:
      return x
    return tf.repeat(x, upsample_ratio, axis=1)


def _tensor_to_fake_sequence(t: tf.Tensor) -> types.Sequence:
  batch_size = utils.smart_dimension_size(t, 0)
  return types.Sequence(
      t[:, tf.newaxis],
      tf.fill([batch_size, 1], tf.constant(1.0, dtype=types.MASK_DTYPE)),
  )


class BaseConditioning(types.SequenceLayer):
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

  def __init__(
      self,
      conditioning_name: str | None,
      projection: Projection,
      combination: Combination,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._conditioning_name = conditioning_name
    self._projection = projection
    self._combination = combination
    self._dense_shaped = None
    self._projection_fn = None
    self._built = False

    if (
        combination == self.Combination.AFFINE
        and projection != self.Projection.LINEAR_AFFINE
    ):
      raise ValueError('AFFINE combination requires LINEAR_AFFINE projection.')

  def _projected_condition_shape(
      self, input_shape: tf.TensorShape, condition_shape: tf.TensorShape
  ) -> tf.TensorShape:
    return {
        self.Projection.IDENTITY: condition_shape,
        self.Projection.LINEAR: input_shape,
        self.Projection.LINEAR_AFFINE: tf.TensorShape([2]).concatenate(
            input_shape
        ),
    }[self._projection]

  def _get_conditioning_channel_shape(
      self, constants: types.Constants
  ) -> tf.TensorShape:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    # PyType is confused by isinstance(conditioning, tf.Tensor).
    if isinstance(conditioning, types.Sequence):
      return conditioning.channel_shape
    else:
      return conditioning.shape[1:]

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    input_shape.assert_is_fully_defined()
    conditioning_channel_shape = self._get_conditioning_channel_shape(constants)
    projected_conditioning_shape = self._projected_condition_shape(
        input_shape, conditioning_channel_shape
    )
    if self._combination == self.Combination.ADD:
      return tf.broadcast_static_shape(
          input_shape, projected_conditioning_shape
      )
    elif self._combination == self.Combination.CONCAT:
      input_inner_dim = input_shape[-1] if input_shape.rank else 1
      projected_conditioning_inner_dim = (
          projected_conditioning_shape[-1]
          if projected_conditioning_shape.rank
          else 1
      )
      output_inner_dim = input_inner_dim + projected_conditioning_inner_dim
      output_outer_shape = tf.broadcast_static_shape(
          input_shape[:-1], projected_conditioning_shape[:-1]
      )
      return output_outer_shape + output_inner_dim
    elif self._combination == self.Combination.AFFINE:
      projected_conditioning_shape = projected_conditioning_shape[1:]
      return tf.broadcast_static_shape(
          input_shape, projected_conditioning_shape
      )
    else:
      raise ValueError(f'Unsupported combination: {self._combination}')

  def _build(
      self, input_shape: tf.TensorShape, conditioning_shape: tf.TensorShape
  ):
    if self._built:
      return
    self._built = True
    if self._projection == self.Projection.LINEAR:
      self._dense_shaped = dense.DenseShaped(input_shape, name='dense')
      linear_projection = self._dense_shaped.layer
    elif self._projection == self.Projection.LINEAR_AFFINE:
      self._dense_shaped = dense.DenseShaped(
          tf.TensorShape([2]).concatenate(input_shape), name='dense'
      )
      linear_projection = self._dense_shaped.layer
    else:
      linear_projection = None
    self._projection_fn = {
        self.Projection.IDENTITY: lambda s, _: s,
        self.Projection.LINEAR: linear_projection,
        self.Projection.LINEAR_AFFINE: linear_projection,
    }[self._projection]

  def _project(
      self, conditioning: types.Sequence, training: bool
  ) -> types.Sequence:
    if self._projection_fn is None:
      raise ValueError('self._projection_fn is None, _build() not called')
    return self._projection_fn(conditioning, training)

  def _combine(
      self, x: types.Sequence, conditioning: types.Sequence
  ) -> types.Sequence:
    def _affine_fn(x, conditioning):
      scale, shift = tf.unstack(conditioning.values, axis=2)
      # Offset scale by 1.0
      scale = types.Sequence(1.0 + scale, conditioning.mask)
      shift = types.Sequence(shift, conditioning.mask)
      return utils.sequence_broadcast_affine(x, scale, shift)

    return {
        self.Combination.ADD: utils.sequence_broadcast_add,
        self.Combination.CONCAT: utils.sequence_broadcast_concat,
        self.Combination.AFFINE: _affine_fn,
    }[self._combination](x, conditioning).mask_invalid()


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
  """

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if isinstance(conditioning, types.Sequence):
      return tf.convert_to_tensor(0, dtype=tf.int32)
    else:
      return ()

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    # PyType is confused by isinstance(conditioning, tf.Tensor).
    if not isinstance(conditioning, types.Sequence):
      conditioning = _tensor_to_fake_sequence(conditioning)
    broadcasted_shape = tf.broadcast_static_shape(
        x.values.shape[:2], conditioning.values.shape[:2]
    )
    x.values.shape[:2].assert_is_compatible_with(broadcasted_shape)
    self._build(x.channel_shape, conditioning.channel_shape)
    conditioned_x = self._combine(x, self._project(conditioning, training))
    return conditioned_x

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    # PyType is confused by isinstance(conditioning, tf.Tensor).
    if not isinstance(conditioning, types.Sequence):
      conditioning_batch_shape = conditioning.shape[:1]
      conditioning_channel_shape = conditioning.shape[1:]
      conditioning = _tensor_to_fake_sequence(conditioning)
    else:
      time_index = state
      conditioning_batch_shape = conditioning.values.shape[:1]
      conditioning_channel_shape = conditioning.channel_shape
      step_size = tf.compat.dimension_value(x.values.shape[1])
      conditioning = _time_slice(
          conditioning, time_index, step_size, name='cond_slice'
      )
      state = time_index + step_size
    x.values.shape[:1].assert_is_compatible_with(conditioning_batch_shape)
    self._build(x.channel_shape, conditioning_channel_shape)
    conditioned_x = self._combine(x, self._project(conditioning, training))
    return conditioned_x, state


class UpsampleConditioning(BaseConditioning):
  """Conditions the sequence x on an upsampled conditioning sequence c.

  Conditioning is done in a time-synchronized way, where each time step t_x of x
  is conditioned on `t_c = t_x // upsample_ratio` from c.

  Conditioning is defined as Combine(x, Upsample(Project(c))), where supported
  projection and combination functions are available in Projection and
  Combination enums above.

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

  Where t_x = t_c * upsample_ratio.
  """

  def __init__(
      self,
      conditioning_name: str,
      projection: BaseConditioning.Projection,
      combination: BaseConditioning.Combination,
      upsample_ratio: int,
      name: Optional[str] = None,
  ):
    super().__init__(conditioning_name, projection, combination, name=name)
    self._upsample_ratio = upsample_ratio

  def _get_conditioning(
      self, constants: Optional[types.Constants]
  ) -> types.Sequence:
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if isinstance(conditioning, tf.Tensor):
      raise ValueError(
          f'{self} expected a conditioning Sequence to upsample, '
          f'got: {conditioning}'
      )
    return conditioning

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    batch_size = utils.smart_dimension_size(x.values, 0)
    x.channel_shape.assert_is_fully_defined()
    x.channel_shape.with_rank_at_least(1)
    conditioning = self._get_conditioning(constants)
    projected_conditioning_shape = self._projected_condition_shape(
        x.channel_shape, conditioning.channel_shape
    )
    # State is a 3-tuple of:
    # x_t: The current timestep we are processing in the overall sequence.
    # prev_cond_t: The conditioning timestep that prev_cond is computed
    #   from. Initialized to -1 to indicate that it is invalid.
    # cond_slice: The cached projection of conditioning timestep prev_cond_t.
    return (
        tf.convert_to_tensor(0, dtype=tf.int32),
        tf.convert_to_tensor(-1, dtype=tf.int32),
        types.Sequence(
            tf.zeros(
                [batch_size, 1] + projected_conditioning_shape.as_list(),
                dtype=x.values.dtype,
            ),
            tf.ones((batch_size, 1), dtype=types.MASK_DTYPE),
        ),
    )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    conditioning = self._get_conditioning(constants)
    self._build(x.channel_shape, conditioning.channel_shape)

    # Optionally project the conditioning input.
    projected_conditioning = self._project(conditioning, training)

    # Upsample the conditioning input to the input rate.
    upsampled = types.Sequence(
        _upsample_tensor(projected_conditioning.values, self._upsample_ratio),
        _upsample_tensor(projected_conditioning.mask, self._upsample_ratio),
    )

    if not x.values.shape[:2].is_compatible_with(upsampled.values.shape[:2]):
      raise ValueError(
          f'{self} received an input ({x}) and conditioning '
          f'({conditioning}) that are incompatible with upsample '
          f'ratio {self._upsample_ratio}.'
      )

    # Combine x and upsampled conditioning. self._combine masks its result.
    return self._combine(x, upsampled)

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    conditioning = self._get_conditioning(constants)
    self._build(x.channel_shape, conditioning.channel_shape)

    block_size = utils.smart_dimension_size(x.values, 1)
    x_t, prev_cond_t, prev_cond_slice = state

    if not isinstance(block_size, tf.Tensor) and block_size == 1:
      # Faster path for single timestep.
      output, cond_t, cond_slice = self._read_project_and_combine_cached(
          x_t, x, conditioning, training, prev_cond_t, prev_cond_slice
      )
      state = (x_t + 1, cond_t, cond_slice)
      return output, state

    # If block_size > 1, process (upsample_ratio - x_t % upsample_ratio) samples
    # at a time until we hit the block size.
    def cond(i, x_t, prev_cond_t, prev_cond, values_ta, mask_ta):
      del x_t
      del prev_cond_t
      del prev_cond
      del values_ta
      del mask_ta
      return i < block_size

    def body(i, x_t, prev_cond_t, prev_cond_slice, values_ta, mask_ta):
      # The number of samples to condition with cond_t.
      x_t_remaining = self._upsample_ratio - x_t % self._upsample_ratio
      block_remaining = block_size - i
      x_t_size = tf.minimum(x_t_remaining, block_remaining)

      # Read the slice of samples to transform.
      x_slice = _time_slice(x, i, x_t_size, name='x_slice')

      output, cond_t, cond_slice = self._read_project_and_combine_cached(
          x_t, x_slice, conditioning, training, prev_cond_t, prev_cond_slice
      )

      write_indices = i + tf.range(x_t_size)
      values_time_major = tf.transpose(
          output.values, [1, 0] + list(range(2, output.values.shape.rank))
      )
      mask_time_major = tf.transpose(output.mask, [1, 0])

      values_ta = values_ta.scatter(write_indices, values_time_major)
      mask_ta = mask_ta.scatter(write_indices, mask_time_major)

      return (
          i + x_t_size,
          x_t + x_t_size,
          cond_t,
          cond_slice,
          values_ta,
          mask_ta,
      )

    # TODO(rryan): Simplify with SequenceArray.
    values_ta = tf.TensorArray(x.dtype, size=block_size)
    mask_ta = tf.TensorArray(tf.float32, size=block_size)
    # TODO(rryan): Consider static unroll of this loop for small sizes.
    i = tf.constant(0, tf.int32)
    _, x_t, cond_t, cond_slice, values_ta, mask_ta = tf.while_loop(
        cond,
        body,
        (i, x_t, prev_cond_t, prev_cond_slice, values_ta, mask_ta),
        maximum_iterations=block_size,
    )
    state = (x_t, cond_t, cond_slice)
    values = values_ta.stack()
    mask = mask_ta.stack()
    output = types.Sequence(
        tf.transpose(values, [1, 0] + list(range(2, values.shape.rank))),
        tf.transpose(mask, [1, 0]),
    )
    return output.mask_invalid(), state

  def _read_project_and_combine_cached(
      self,
      x_t: tf.Tensor,
      x: types.Sequence,
      conditioning: types.Sequence,
      training: bool,
      prev_cond_t: tf.Tensor,
      prev_cond_slice: types.Sequence,
  ) -> Tuple[types.Sequence, tf.Tensor, types.Sequence]:
    cond_t = x_t // self._upsample_ratio

    def read_and_project():
      cond_slice = _time_slice(conditioning, cond_t, 1, name='cond_slice')
      cond_slice = self._project(cond_slice, training=training)

      def ensure_shape(values, mask):
        values = tf.ensure_shape(values, prev_cond_slice.values.shape)
        mask = tf.ensure_shape(mask, prev_cond_slice.mask.shape)
        return values, mask

      return cond_t, cond_slice.apply(ensure_shape)

    def reuse_cached():
      return prev_cond_t, prev_cond_slice

    cond_t, cond_slice = tf.cond(
        tf.equal(prev_cond_t, cond_t), reuse_cached, read_and_project
    )
    output = self._combine(x, cond_slice)
    return output, cond_t, cond_slice


class NoiseConditioning(BaseConditioning):
  """Conditions the sequence x on a noise sequence.

  Conditioning is done in a time-synchronized way, where each time step of x is
  conditioned on the corresponding time step of the sampled noise sequence, c.

  Conditioning is defined as Combine(x, Project(c)), where supported projection
  and combination functions are available in Projection and Combination enums
  above.

  All valid broadcasts between x and c channel_shapes are supported, e.g.:
    - Projection=IDENTITY, Combination=ADD:
      - (x: [b, t, 3], c: [b, t, 3]) -> c_x: [b, t, 3]
      - (x: [b, t, 3, 1, 5], c: [b, t, 2, 5]) -> c_x: [b, t, 3, 2, 5]
    - Projection=IDENTITY, Combination=CONCAT:
      - (x: [b, t, 3], c: [b, t, 7]) -> c_x: [b, t, 10]
      - (x: [b, t, 3, 1, 5], c: [b, t, 2, 7]) -> c_x: [b, t, 3, 2, 12]
    - Projection=LINEAR, Combination=ADD:
      - (x: [b, t, 3], c: [b, t, 7]) -> c_x: [b, t, 3]
      - (x: [b, t, 3, 1, 5], c: [b, t, 2, 7]) -> c_x: [b, t, 3, 1, 5]
    - Projection=LINEAR, Combination=CONCAT:
      - (x: [b, t, 3], c: [b, t, 7]) -> c_x: [b, t, 6]
      - (x: [b, t, 3, 1, 5], c: [b, t, 2, 7]) -> c_x: [b, t, 3, 1, 10]
  """

  def __init__(
      self,
      noise_channel_shape: tf.TensorShape | list[int] | tuple[int, ...],
      noise_sampler: simple.NoiseSampler,
      projection: BaseConditioning.Projection,
      combination: BaseConditioning.Combination,
      name: str | None = None,
  ):
    super().__init__(None, projection, combination, name=name)
    self._noise_channel_shape = tf.TensorShape(noise_channel_shape)
    self._noise_sampler = noise_sampler

  def _get_conditioning_channel_shape(
      self, constants: types.Constants
  ) -> tf.TensorShape:
    del constants
    return self._noise_channel_shape

  def _get_noise_sequence(self, x: types.Sequence) -> types.Sequence:
    batch, time = utils.smart_dimension_size(x.values, dim=[0, 1])
    noise_shape = [batch, time] + self._noise_channel_shape.as_list()
    sample_spec = simple.SampleSpec(noise_shape, x.dtype)
    samples = self._noise_sampler.sample(sample_spec)
    return types.Sequence(samples, x.mask).mask_invalid()

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants | None = None
  ) -> types.State:
    return ()

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del initial_state
    conditioning = self._get_noise_sequence(x)
    broadcasted_shape = tf.broadcast_static_shape(
        x.values.shape[:2], conditioning.values.shape[:2]
    )
    x.values.shape[:2].assert_is_compatible_with(broadcasted_shape)
    self._build(x.channel_shape, conditioning.channel_shape)
    conditioned_x = self._combine(x, self._project(conditioning, training))
    return conditioned_x

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    conditioned_x = self.layer(x, training, state, constants)
    return conditioned_x, ()
