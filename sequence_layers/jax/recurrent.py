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
"""Recurrent layers."""

import dataclasses
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from sequence_layers.jax import types
from sequence_layers.jax import utils


__all__ = (
    # go/keep-sorted start
    'LSTM',
    # go/keep-sorted end
)


def unit_forget_bias(key, shape, dtype) -> jax.Array:
  """An initializer for LSTM bias that sets the forget gate bias to one.

  This is recommended in Jozefowicz et al.
  https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf

  Args:
    key: Unused key array.
    shape: The shape of the bias.
    dtype: The dtype of the bias.

  Returns:
    A bias array with the forget gate component set to one.
  """
  del key
  if len(shape) != 1 or shape[0] % 4 != 0:
    raise ValueError(
        f'Expected a single dimensional shape divisible by 4, got: {shape}.'
    )
  units = shape[0] // 4
  return jnp.concatenate(
      [
          jnp.zeros([units], dtype),
          jnp.ones([units], dtype),
          jnp.zeros([2 * units], dtype),
      ],
      axis=0,
  )


class LSTM(types.SequenceLayer):
  """A Long Short-term Memory (LSTM) layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for LSTM."""

    units: int
    dtype: types.DType | None = None
    param_dtype: types.DType | None = jnp.float32
    precision: nn.linear.PrecisionLike = None
    activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    recurrent_activation: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    recurrent_kernel_init: nn.initializers.Initializer = (
        nn.initializers.orthogonal()
    )
    recurrent_kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = unit_forget_bias
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def make(self) -> 'LSTM':
      return LSTM(self, name=self.name)

  config: Config

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self.config.units,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.dtype
    )

  @nn.compact
  def _cell(
      self, x: jax.Array, state: tuple[jax.Array, jax.Array]
  ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    if x.ndim != 3 or x.shape[1] != 1:
      raise ValueError(f'Expected [b, 1, d] inputs, got: {x.shape}.')
    c_tm1, h_tm1 = state

    compute_dtype = self.get_output_dtype(x.dtype)

    # Project [b, 1, d] to [b, 1, 4 * units].
    z = utils.FlaxEinsumDense(
        '...d,dh->...h',
        output_shape=(4 * self.config.units,),
        dtype=self.config.dtype,
        param_dtype=self.config.param_dtype,
        precision=self.config.precision,
        kernel_init=self.config.kernel_init,
        kernel_sharding=self.config.kernel_sharding,
        name='kernel',
    )(x)
    # Project [b, 1, h] to [b, 1, 4 * units]
    z += utils.FlaxEinsumDense(
        '...u,uh->...h',
        output_shape=(4 * self.config.units,),
        dtype=self.config.dtype,
        param_dtype=self.config.param_dtype,
        precision=self.config.precision,
        kernel_init=self.config.recurrent_kernel_init,
        kernel_sharding=self.config.recurrent_kernel_sharding,
        name='recurrent_kernel',
    )(h_tm1)

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (4 * self.config.units,),
          self.config.param_dtype,
      )
      z, bias = nn.dtypes.promote_dtype(z, bias, dtype=compute_dtype)
      z = utils.bias_add(z, bias)

    z0, z1, z2, z3 = jnp.split(
        z,
        [
            self.config.units,
            2 * self.config.units,
            3 * self.config.units,
        ],
        axis=-1,
    )

    i = self.config.recurrent_activation(z0)
    f = self.config.recurrent_activation(z1)
    c = f * c_tm1 + i * self.config.activation(z2)
    o = self.config.recurrent_activation(z3)
    h = o * self.config.activation(c)
    return h, (c, h)

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if x.shape[1] == 1:

      y, new_state = self._cell(x.values, state)

      def copy_state_through(new_a: jax.Array, a: jax.Array) -> jax.Array:
        assert new_a.ndim >= 2, (new_a.shape, new_a.dtype)
        mask = x.mask.reshape(x.mask.shape + (1,) * (new_a.ndim - 2))
        return jnp.where(mask, new_a, a)

      # Don't update state on invalid timesteps.
      state = jax.tree.map(copy_state_through, new_state, state)

      return types.Sequence(y, x.mask), state

    # If we received multiple timesteps, unroll them statically (assuming step
    # inputs are usually small).
    output, state, _ = utils.step_by_step_static(
        self,
        x,
        training=training,
        initial_state=state,
        constants=constants,
        with_emits=False,
    )
    return output, state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Use dynamic unroll for layer-wise application.
    output, _, _ = utils.step_by_step_dynamic(
        self, x, training=training, constants=constants, with_emits=False
    )
    return output

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # A time axis of 1 is included in the state arrays for ease of broadcasting
    # with the input.
    compute_dtype = self.get_output_dtype(input_spec.dtype)
    c = jnp.zeros((batch_size, 1, self.config.units), dtype=compute_dtype)
    h = jnp.zeros((batch_size, 1, self.config.units), dtype=compute_dtype)
    return (c, h)
