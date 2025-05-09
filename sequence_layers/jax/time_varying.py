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
"""Time-varying layers."""

import dataclasses
from typing import Callable, Sequence as TypingSequence

import flax.linen as nn
import jax
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import utils

__all__ = (
    # go/keep-sorted start
    'MaskedDense',
    'SequenceDense',
    'SequenceEmbedding',
    # go/keep-sorted end
)


def _round_up_to_multiple_of(x: int, multiple_of: int) -> int:
  return (x + multiple_of - 1) // multiple_of * multiple_of


class SequenceEmbedding(types.SequenceLayer):
  """Computes sequence embeddings of integer input codes.

  Provides step-dependent Embedding layers up to num_steps, where the sequence
  step determines which Embedding layer is used. This layer can be used with
  fixed length sequences, where num_steps is set to the fixed length. For
  variable length sequences, set num_steps to the maximum sequence length.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for SequenceEmbedding."""

    # Dimensionality of the embedded values.
    dimension: int
    # The number of embeddings in the embedding table per step. The inputs at
    # step i are expected to be in the range [0, num_embeddings_per_step[i]). If
    # num_embeddings_per_step is an int, the inputs at all steps are expected to
    # be in [0, num_embeddings_per_step).
    num_embeddings_per_step: TypingSequence[int] | int
    # The number of valid timesteps for the SequenceEmbedding layer.
    num_steps: int
    # The dtype of the embeddings output by the layer.
    compute_dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # By default, initialize embeddings to have a norm of 1.
    embedding_init: nn.initializers.Initializer = nn.linear.default_embed_init
    # Optional sharding for the embedding table.
    embedding_sharding: types.Sharding | None = None
    # Round up the number of embeddings to a multiple of this value. If 128 or
    # 256, this leads to more TPU friendly shapes. If None, does not round.
    round_num_embeddings_to_multiple_of: int | None = None
    # Optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      # Ensure hashable sequence type.
      if not isinstance(self.num_embeddings_per_step, int):
        if len(self.num_embeddings_per_step) != self.num_steps:
          raise ValueError(
              'num_embeddings_per_step must have length num_steps. Got: '
              f'{self.num_embeddings_per_step=} vs {self.num_steps=}.'
          )
        object.__setattr__(
            self,
            'num_embeddings_per_step',
            tuple(self.num_embeddings_per_step),
        )

    def make(self) -> 'SequenceEmbedding':
      return SequenceEmbedding(self, name=self.name)

  config: Config

  def setup(self):

    if isinstance(self.config.num_embeddings_per_step, int):
      num_embeddings = (
          self.config.num_embeddings_per_step * self.config.num_steps
      )
    else:
      num_embeddings = sum(self.config.num_embeddings_per_step)

    if num_embeddings <= 0:
      raise ValueError(
          f'{self.config.num_embeddings_per_step=} must be positive.'
      )

    if self.config.round_num_embeddings_to_multiple_of:
      num_embeddings = _round_up_to_multiple_of(
          num_embeddings,
          self.config.round_num_embeddings_to_multiple_of
      )

    self.embedding = self.param(
        'embedding',
        utils.shard_initializer(
            self.config.embedding_init, self.config.embedding_sharding
        ),
        (
            num_embeddings,
            self.config.dimension,
        ),
        self.config.param_dtype,
    )

  @nn.nowrap
  def _validate_input_dtype(self, dtype: types.DType):
    if not jnp.issubdtype(dtype, jnp.integer):
      raise ValueError(
          'Input to Embedding must be an integer or unsigned integer, got:'
          f' {dtype}'
      )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return tuple(input_shape) + (self.config.dimension,)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    self._validate_input_dtype(input_dtype)
    if self.config.compute_dtype is None:
      return self.config.param_dtype
    return self.config.compute_dtype

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    y, _ = self.step(x, 0, training=training, constants=constants)
    return y

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    del batch_size
    del input_spec
    del constants
    # TODO(b/338656992): Support continuous batching by removing the unbatched
    # state.
    return jnp.array(0, dtype=jnp.int32)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    del training
    del constants
    self._validate_input_dtype(x.dtype)
    num_steps = self.config.num_steps
    start_time = state
    time_delta = x.shape[1]
    end_time = start_time + time_delta
    _check_step_bounds(start_time, end_time, num_steps)

    embedding = self.embedding
    (embedding,) = nn.dtypes.promote_dtype(
        embedding, dtype=self.config.compute_dtype, inexact=False
    )

    def broadcast_to_x(a: jax.Array) -> jax.Array:
      """Reshapes (time_delta) to (1, time_delta, ...)."""
      assert a.ndim == 1 and a.shape[0] == time_delta
      return a.reshape((1,) + a.shape + (1,) * (x.ndim - 2))

    # Input is codebook indices shaped [b, time_delta, ...].

    # Ensure steps are in range.
    steps = start_time + jnp.arange(time_delta)
    steps_valid = broadcast_to_x(
        jnp.logical_and(steps >= 0, steps < self.config.num_steps)
    )

    if isinstance(self.config.num_embeddings_per_step, int):
      offsets = broadcast_to_x(self.config.num_embeddings_per_step * steps)
      # No broadcast reshaping needed for scalar.
      limits = self.config.num_embeddings_per_step
    else:
      offsets = np.cumsum(
          (0,) + tuple(self.config.num_embeddings_per_step[:-1])
      )
      offsets = broadcast_to_x(jnp.take(offsets, steps))
      limits = np.array(self.config.num_embeddings_per_step, np.int32)
      limits = broadcast_to_x(jnp.take(limits, steps))

    # Ensure per-step indices are in range. [batch, num_steps, ...].
    indices_valid = jnp.logical_and(x.values >= 0, x.values < limits)
    indices = x.values + offsets

    # Replace invalid indices with the an out-of-range value to ensure they are
    # replaced with NaN in the gather.
    indices = jnp.where(
        steps_valid & indices_valid, indices, embedding.shape[0]
    )

    embedded = x.apply_values(lambda v: jnp.take(embedding, indices, axis=0))

    return embedded, end_time


class SequenceDense(types.SequenceLayer):
  """Step-dependent Dense layer.

  Provides step-dependent Dense layers up to num_steps, where the sequence
  step determines which Dense layer is used. This layer can be used with fixed
  length sequences, where num_steps is set to the fixed length. For variable
  length sequences, set num_steps to the maximum sequence length.

  Each step-dependent Dense layer only operates on the inputs at the associated
  timestep, i.e., y[t] = Dense_t(x[t]), where Dense_t is the Dense layer
  used for timestep t.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for SequenceDense."""

    # The number of output features for the dense layer.
    features: int
    # The number of valid timesteps for the SequenceDense layer.
    num_steps: int
    # Whether to use a bias.
    use_bias: bool = True
    # An optional activation to apply after the dense layer.
    activation: Callable[[jax.Array], jax.Array] | None = None
    # The dtype to use for layer compute.
    compute_dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # Initializer for the kernel.
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    # Optional sharding for the kernel. Any axes that are present in the input
    # spec are marked as FANIN.
    kernel_sharding: types.Sharding | None = None
    # Initializer for the bias, if used and not gated by another config option.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding | None = None
    # Optional name for the layer.
    name: str | None = None

    def make(self) -> 'SequenceDense':
      return SequenceDense(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    # TODO(rryan): Support rank > 3 inputs.
    if len(input_shape) != 1:
      raise ValueError(
          f'SequenceDense requires rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.config.features,)

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    y, _ = self.step(x, 0, training=training, constants=constants)
    return y

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    del batch_size
    del input_spec
    del constants
    # TODO(b/338656992): Support continuous batching by removing the unbatched
    # state.
    return jnp.array(0, dtype=jnp.int32)

  @types.check_step
  @nn.compact
  def step(
      self,
      x: types.Sequence,
      step: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    del training
    del constants

    num_steps = self.config.num_steps
    start_time = step
    time_delta = x.shape[1]
    end_time = start_time + time_delta
    _check_step_bounds(start_time, end_time, num_steps)

    # TODO(rryan): Support rank > 3 inputs.
    if x.ndim != 3:
      raise ValueError(f'SequenceDense requires rank 3 input. Got: {x.shape=}')

    kernel_shape = (num_steps, x.shape[-1], self.config.features)

    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, meta.AxisType.FANIN, None),
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_bias:
      bias_shape = (num_steps, self.config.features)
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param('bias', bias_init, bias_shape, self.config.param_dtype)
    else:
      bias = None

    inputs, kernel, bias = nn.dtypes.promote_dtype(
        x.values, kernel, bias, dtype=self.config.compute_dtype
    )

    # Slice [T, I, O] kernel to be used with current timesteps.
    slice_size = min(self.config.num_steps, time_delta)
    step_kernel = jax.lax.dynamic_slice_in_dim(
        kernel, start_index=start_time, slice_size=slice_size, axis=0
    )

    if pad_amount := max(0, time_delta - step_kernel.shape[0]):
      step_kernel = jnp.pad(
          step_kernel,
          ((0, pad_amount), (0, 0), (0, 0)),
      )

    ret = jnp.einsum(
        '...ti,tio->...to', inputs, step_kernel, precision=self.config.precision
    )
    if bias is not None:
      step_bias = jax.lax.dynamic_slice_in_dim(
          bias, start_index=start_time, slice_size=slice_size, axis=0
      )
      if pad_amount:
        step_bias = jnp.pad(step_bias, ((0, pad_amount), (0, 0)))
      ret = utils.bias_add(ret, step_bias)
    if self.config.activation is not None:
      ret = self.config.activation(ret)

    steps_valid = (start_time + jnp.arange(time_delta)) < self.config.num_steps
    steps_valid = steps_valid.reshape(
        (1,) + steps_valid.shape + (1,) * (ret.ndim - 2)
    )
    ret = jnp.where(steps_valid, ret, jnp.full_like(ret, jnp.nan))

    # Preserve masked state if no bias or activation are in use.
    apply_fn = (
        x.apply_values
        if self.config.use_bias or self.config.activation is not None
        else x.apply_values_masked
    )
    return apply_fn(lambda v: ret), start_time + time_delta


class MaskedDense(types.SequenceLayer):
  """Step-dependent causal masked Dense layers.

  Provides step-dependent masked Dense layers up to num_steps, where the
  sequence step determines which masked Dense layer is used. This layer can be
  used with fixed length sequences, where num_steps is set to the fixed length.
  For variable length sequences, set num_steps to the maximum sequence length.

  Masked Dense layers are causally masked so that y[t] (the output at time=t)
  is a Dense transformation of x[0:t+1] (the input from time=0 up to time=t,
  inclusive.); i.e., y[t] = Dense_t(x[0,...,t]), where Dense_t is the Dense
  layer used for timestep t.

  Note:
  This layer uses a masked kernel variable in order to maintain the
  autoregressive property, and therefore, half of the weights are unused and
  have zero gradient. This could affect the behavior of certain optimizers or
  normalization schemes (e.g., AdaFactor).
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for MaskedDense."""

    features: int
    num_steps: int
    # Whether to use a bias.
    use_bias: bool = True
    # An optional activation to apply after the dense layer.
    activation: Callable[[jax.Array], jax.Array] | None = None
    # The dtype to use for layer compute.
    compute_dtype: types.DType | None = None
    # The dtype to use for layer parameters.
    param_dtype: types.DType = jnp.float32
    # An optional precision to use for the einsum.
    precision: nn.linear.PrecisionLike = None
    # Initializer for the kernel.
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    # Optional sharding for the kernel. Any axes that are present in the input
    # spec are marked as FANIN.
    kernel_sharding: types.Sharding | None = None
    # Initializer for the bias, if used and not gated by another config option.
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # Optional sharding for the bias.
    bias_sharding: types.Sharding | None = None
    # Optional name for the layer.
    name: str | None = None

    def make(self) -> 'MaskedDense':
      return MaskedDense(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    del constants
    # TODO(rryan): Support rank > 3 inputs.
    if len(input_shape) != 1:
      raise ValueError(
          f'MaskedDense requires rank 3 input. Got: {input_shape=}'
      )
    return tuple(input_shape[:-1]) + (self.config.features,)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    del constants
    # TODO(rryan): Support rank > 3 inputs.
    if len(input_spec.shape) != 1:
      raise ValueError(f'MaskedDense requires rank 3 input. Got: {input_spec=}')

    input_dim = input_spec.shape[0]

    # When executing step-by-step, we need a buffer for tracking past inputs.
    # TODO(b/338656992): Support continuous batching by removing the unbatched
    # state.
    t0 = jnp.array(0, dtype=jnp.int32)
    input_buffer = types.MaskedSequence(
        jnp.zeros(
            (batch_size, self.config.num_steps, input_dim),
            dtype=input_spec.dtype,
        ),
        jnp.zeros((batch_size, self.config.num_steps), dtype=types.MASK_DTYPE),
    )
    return t0, input_buffer

  @types.check_step
  @nn.compact
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    """Steps forward, input shape: [..., block_size, num_groups]."""
    del training, constants

    # TODO(rryan): Support rank > 3 inputs.
    if x.ndim != 3:
      raise ValueError(f'MaskedDense requires rank 3 input. Got: {x.shape=}')

    num_steps = self.config.num_steps
    start_time, input_buffer = state
    time_delta = x.shape[1]
    end_time = start_time + time_delta
    _check_step_bounds(start_time, end_time, num_steps)

    # Append new input to end of buffer and truncate at the front:
    # e.g., [0, 0, x0, x1] -> [0, x0, x1, x2], where new input is [x2].
    input_buffer = input_buffer.concatenate(x.mask_invalid())
    input_buffer = input_buffer[:, -num_steps:]
    # Roll buffer so padding is at the end:
    # e.g., [0, x0, x1, x2] -> [x0, x1, x2, 0]
    step_input = jnp.roll(input_buffer.values, shift=end_time, axis=1)

    # TODO(ebattenberg): Consider ways to eliminate unused (masked off)
    # weights (e.g., construct a masked kernel Tensor from smaller
    # sub-variables.)

    # Construct a [T, T', I, O] kernel tensor to project [B, T, I] input to
    # [B, T', O] output, where T represents the input time dimension and T'
    # represents the output time dimension, and T = T' = num_steps.
    kernel_shape = (
        num_steps,
        num_steps,
        x.shape[-1],
        self.config.features,
    )
    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(
            meta.AxisType.FANIN,
            meta.AxisType.FANIN,
            meta.AxisType.FANIN,
            None,
        ),
    )

    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )

    if self.config.use_bias:
      bias_shape = (num_steps, self.config.features)
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param('bias', bias_init, bias_shape, self.config.param_dtype)
    else:
      bias = None

    step_input, kernel, bias = nn.dtypes.promote_dtype(
        step_input, kernel, bias, dtype=self.config.compute_dtype
    )

    # Construct a [T, T', 1, 1] upper triangular autoregressive mask matrix,
    # where T represents the input time dimension and T' represents the output
    # time dimension, and T = T' = num_steps.
    mask = utils.ones_matrix_band_part(
        num_steps,
        num_steps,
        num_lower=0,
        num_upper=-1,
        out_dtype=kernel.dtype,
        out_shape=(num_steps, num_steps, 1, 1),
    )
    masked_kernel = kernel * mask

    # Slice kernel and mask to produce current output timesteps.
    # Kernel is [T, T', I, O], mask is [T, T', 1, 1], where T represents the
    # input time dimension and T' represents the output time dimension,
    # and T = T' = num_steps.
    # step_kernel = self._kernel[:, start_time:end_time]
    masked_step_kernel = jax.lax.dynamic_slice_in_dim(
        masked_kernel, start_time, time_delta, axis=1
    )

    ret = jnp.einsum(
        'BTI,TtIO->BtO',
        step_input,
        masked_step_kernel,
        precision=self.config.precision,
    )
    if self.config.use_bias:
      # Bias is [T', O].  Slice for current output timesteps.
      step_bias = jax.lax.dynamic_slice_in_dim(
          bias, start_time, time_delta, axis=0
      )
      ret = utils.bias_add(ret, step_bias)

    if self.config.activation is not None:
      ret = self.config.activation(ret)

    # Never preserve x's masked state, since even if the current timestep is
    # masked, the above einsum can produce non-zero values from buffered valid
    # outputs.
    y = types.Sequence(ret, x.mask)
    state = (end_time, input_buffer)
    return y, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    state = self.get_initial_state(
        x.shape[0], x.channel_spec, training=training
    )
    y, _ = self.step(x, state, training=training, constants=constants)
    return y


@checkify.checkify
def _check_step_bounds(start_time, end_time, num_steps):
  """Check that step indices are within range."""
  checkify.check(start_time >= 0, 'Out of range step index (< 0).')
  checkify.check(
      end_time <= num_steps, 'Out of range step index (> num_steps).'
  )
