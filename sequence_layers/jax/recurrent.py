# Copyright 2025 Google LLC
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
from typing import Callable, Literal

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
import recurrentgemma
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils


__all__ = (
    # go/keep-sorted start
    'LSTM',
    'RGLRU',
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


def orthogonal_init(scale: float = 1.0, column_axis: int = -1):
  def init(key: jax.Array, shape: types.Shape, dtype: types.DType) -> jax.Array:
    # Flax's orthogonal initializer uses jax.lax.qr internally, which only
    # supports float32. Compute the initializer in float32 then cast to the
    # desired dtype.
    value = nn.initializers.orthogonal(scale=scale, column_axis=column_axis)(
        key, shape, jnp.float32
    )
    return value.astype(dtype)

  return init


class LSTM(types.SequenceLayer):
  """A Long Short-term Memory (LSTM) layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for LSTM."""

    units: int
    compute_dtype: types.DType | None = None
    param_dtype: types.DType | None = jnp.float32
    precision: nn.linear.PrecisionLike = None
    activation: Callable[[jax.Array], jax.Array] = jax.nn.tanh
    recurrent_activation: Callable[[jax.Array], jax.Array] = jax.nn.sigmoid
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    recurrent_kernel_init: nn.initializers.Initializer = orthogonal_init()
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
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
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
        compute_dtype=self.config.compute_dtype,
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
        compute_dtype=self.config.compute_dtype,
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


def _rnn_real_param_init(
    min_rad: float,
    max_rad: float,
    transform: str = 'softplus',
    eps: float = 1e-8,
) -> nn.initializers.Initializer:
  """Initializes the `A` real parameter of the RG-LRU uniformly on a ring."""

  def init(
      key: jax.Array,
      shape: types.Shape,
      dtype: types.DType = jnp.float32,
  ) -> jt.Float[jt.ArrayT, 'e']:
    unif = jax.random.uniform(key, shape=shape)
    # Proportional to area in a ring.
    a_real = 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + eps)

    if transform == 'softplus':
      # Inverse transform.
      return jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
    else:
      raise NotImplementedError()

  return init


def _rnn_imag_param_init(
    max_rad: float,
) -> nn.initializers.Initializer:
  """Initializes the `A` imag parameter of the RG-LRU uniformly on a ring."""

  def init(
      key: jax.Array,
      shape: types.ShapeLike,
      dtype: types.DType = jnp.float32,
  ) -> jt.Float[jt.ArrayT, 'e']:
    unif = jax.random.uniform(key, shape=shape)
    return (jnp.pi * max_rad * unif).astype(dtype)

  return init


class RGLRU(types.SequenceLayer):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer.

  From the Griffin architecture: https://arxiv.org/abs/2402.19427

  Implementation follows https://github.com/google-deepmind/recurrentgemma.

  WARNING: The current implementation is not able to work with anything but
    left-aligned ragged masks. Non-contiguous masks will incorrectly compute
    results on the mask=False timesteps.
  """

  ScanType = Literal[  # pylint: disable=invalid-name
      'auto', 'linear_native', 'associative_native', 'linear_pallas'
  ]

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for RG-LRU."""

    units: int
    num_heads: int
    scan_type: 'RGLRU.ScanType' = 'auto'
    scan_sharding: types.Sharding | None = None
    only_real: bool = True
    min_rad: float = 0.9
    # Sharding for the `A` real and imaginary parameters.
    # Shape is [units] if only_real or [units // 2] if complex.
    a_sharding: types.Sharding | None = None
    gate_kernel_variance_scale: float = 1.0
    # Sharding for the kernel matrices in the gate operations. Shape is
    # [num_heads, units // units_per_head, units // units_per_head] if only_real
    # or [num_heads, units // units_per_head, units // (2 * units_per_head)] if
    # complex.
    gate_kernel_sharding: types.Sharding | None = None
    # Sharding for the bias matrices in the gate operations. Shape is
    # [num_heads, units // units_per_head] if only_real or [num_heads, units //
    # (2 * units_per_head)] if complex.
    gate_bias_sharding: types.Sharding | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType | None = jnp.float32
    # An optional name for the layer.
    name: str | None = None

    def __post_init__(self):
      if not self.only_real:
        if self.units % 2 != 0:
          raise ValueError(
              'If `only_real=False`, `units` must be even, but got'
              f' {self.units=}.'
          )
        if self.min_rad >= 0.999:
          raise ValueError(
              'If `only_real=False`, `min_rad` must be less than 0.999, but got'
              f' {self.min_rad=}.'
          )

      complex_units = self.units if self.only_real else self.units // 2
      if complex_units % self.num_heads != 0:
        raise ValueError(
            'The number of heads must divide the number of complex units, but'
            f' got {self.num_heads=} and {complex_units=}.'
        )

    def make(self) -> 'RGLRU':
      return RGLRU(self, name=self.name)

  config: Config

  @property
  def _scan_type(self) -> recurrentgemma.common.ScanType:
    match self.config.scan_type:
      case 'auto':
        return recurrentgemma.common.ScanType.AUTO
      case 'linear_native':
        return recurrentgemma.common.ScanType.LINEAR_NATIVE
      case 'associative_native':
        return recurrentgemma.common.ScanType.ASSOCIATIVE_NATIVE
      case 'linear_pallas':
        return recurrentgemma.common.ScanType.LINEAR_PALLAS
      case _:
        raise ValueError(f'Unknown scan type: {self.config.scan_type}')

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
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @jt.typed
  def merged_to_complex(
      self,
      x: jt.Float[jt.ArrayT, '*b'],
  ) -> recurrentgemma.complex_lib.RealOrComplex:
    """Returns a (complex) array from a merged array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The merged array.

    Returns:
      A (complex) array represented by `x`.
    """
    if self.config.only_real:
      return x

    assert x.shape[-1] % 2 == 0
    return self.real_imag_complex(*jnp.split(x, 2, axis=-1))

  @jt.typed
  def real_imag_complex(
      self,
      real: jt.Float[jt.ArrayT, '*b'],
      imag: jt.Float[jt.ArrayT, '*b'] | None,
  ) -> recurrentgemma.complex_lib.RealOrComplex:
    """Based on the settings, creates a (complex) number in the correct format.

    Args:
      real: The real part of the complex number.
      imag: The imaginary part of the complex number.

    Returns:
      The correct representation for a complex number. If `only_real=True`
      the function expects that `imag` is None and will directly return `real`.
      When using `bfloat16` or Pallas a `complex_lib.Complex` is returned,
      otherwise a native jax array with a complex type.
    """
    if self.config.only_real:
      assert imag is None
      return real

    if self.use_custom_complex(real.dtype):
      return recurrentgemma.complex_lib.Complex(real, imag)
    else:
      return real + 1j * imag

  def use_custom_complex(self, real_dtype: jnp.dtype) -> bool:
    return (
        real_dtype in (jnp.bfloat16, jnp.float16)
        or self._scan_type == recurrentgemma.common.ScanType.LINEAR_PALLAS
    )

  @jt.typed
  def complex_to_merged(
      self,
      x: recurrentgemma.complex_lib.RealOrComplex,
  ) -> jt.Float[jt.ArrayT, '*b']:
    """Returns a merged array from a (complex) array.

    A merged array is one where the first half over the last axis represents the
    real part of a complex array, while the second part represents the
    imaginary.

    Args:
      x: The (complex) array.

    Returns:
      A merged array represented by `x`.
    """
    if self.config.only_real:
      assert not isinstance(
          x, recurrentgemma.complex_lib.Complex
      ) and not jnp.iscomplexobj(x)
      return x
    else:
      return jnp.concatenate([x.real, x.imag], axis=-1)

  @nn.compact
  def _cell(
      self,
      x: types.Sequence,
      h: jax.Array,
      segment_pos: jax.Array,
  ) -> tuple[types.Sequence, jax.Array]:
    mask = x.mask
    x = x.values

    width_output = (
        self.config.units if self.config.only_real else self.config.units // 2
    )
    a_real_param = self.param(
        'a_param',
        utils.shard_initializer(
            _rnn_real_param_init(min_rad=self.config.min_rad, max_rad=0.999),
            self.config.a_sharding,
        ),
        [width_output],
        self.config.param_dtype,
    )

    a_imag_param = None
    if not self.config.only_real:
      a_imag_param = self.param(
          'a_imag_param',
          utils.shard_initializer(
              _rnn_imag_param_init(max_rad=0.1), self.config.a_sharding
          ),
          [width_output],
          self.config.param_dtype,
      )

    if width_output % self.config.num_heads != 0:
      raise ValueError(
          'The number of heads must divide the width of the output, but got'
          f' {self.config.num_heads=} and {width_output=}.'
      )
    units_per_head = width_output // self.config.num_heads

    input_gate = utils.FlaxEinsumDense(
        '...hi,hij->...hj',
        output_shape=(self.config.num_heads, units_per_head),
        bias_axes='hj',
        kernel_init=nn.initializers.variance_scaling(
            scale=self.config.gate_kernel_variance_scale,
            mode='fan_in',
            distribution='normal',
        ),
        kernel_sharding=self.config.gate_kernel_sharding,
        bias_sharding=self.config.gate_bias_sharding,
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        name='input_gate',
    )
    a_gate = utils.FlaxEinsumDense(
        '...hi,hij->...hj',
        output_shape=(self.config.num_heads, units_per_head),
        bias_axes='hj',
        kernel_init=nn.initializers.variance_scaling(
            scale=self.config.gate_kernel_variance_scale,
            mode='fan_in',
            distribution='normal',
        ),
        kernel_sharding=self.config.gate_kernel_sharding,
        bias_sharding=self.config.gate_bias_sharding,
        compute_dtype=self.config.compute_dtype,
        param_dtype=self.config.param_dtype,
        name='a_gate',
    )

    x, a_real_param, a_imag_param = nn.dtypes.promote_dtype(
        x,
        a_real_param,
        a_imag_param,
        dtype=self.config.compute_dtype,
    )

    # Group x into heads.
    x_heads = einops.rearrange(
        x, '... (h i) -> ... h i', h=self.config.num_heads
    )

    # Compute input and `A` gates.
    gate_x = recurrentgemma.complex_lib.sigmoid(input_gate(x_heads))
    gate_x = einops.rearrange(
        gate_x, '... h j -> ... (h j)', h=self.config.num_heads
    )

    gate_a = recurrentgemma.complex_lib.sigmoid(a_gate(x_heads))
    gate_a = einops.rearrange(
        gate_a, '... h j -> ... (h j)', h=self.config.num_heads
    )

    # Compute the parameter `A` of the recurrence.
    log_a_real = (
        -8.0 * gate_a * recurrentgemma.complex_lib.softplus(a_real_param)
    )

    if self.config.only_real:
      a = recurrentgemma.complex_lib.exp(log_a_real)
    else:
      log_a_imag = a_imag_param * gate_a
      log_a_complex = self.real_imag_complex(log_a_real, log_a_imag)
      a = recurrentgemma.complex_lib.exp(log_a_complex)

    # Since A = |A| e^(i*θ), log A = log |A| + i*θ.
    # Real(log A) = log |A| therefore |A|^2 = e^(2*Real(log A))
    mag_a_squared = recurrentgemma.complex_lib.exp(2 * log_a_real)

    x = self.merged_to_complex(x)

    assert h.dtype == jnp.float32, h.dtype
    h = self.merged_to_complex(h)

    # Gate the input.
    gated_x = x * gate_x

    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    reset = (segment_pos == 0).astype(a)
    multiplier = recurrentgemma.layers.sqrt_bound_derivative(
        1 - mag_a_squared, max_gradient=1000
    )
    multiplier = (
        reset[..., jnp.newaxis] + (1 - reset)[..., jnp.newaxis] * multiplier
    )
    normalized_x = gated_x * multiplier.astype(gated_x.dtype)

    # TODO(b/398200724): Add masking support to the scan and skip state updates
    # on invalid timesteps.
    y, h = recurrentgemma.scan.linear_scan(
        x=normalized_x,
        a=a * (1 - reset[..., jnp.newaxis]),
        h0=h,
        scan_type=self._scan_type,
        sharding_spec=self.config.scan_sharding,
        unroll=128,
    )

    y = self.complex_to_merged(y)
    h = self.complex_to_merged(h)
    assert h.dtype == jnp.float32, h.dtype
    return types.Sequence(y, mask), h

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    h, start_index = state
    segment_pos = (
        start_index[:, jnp.newaxis] + jnp.arange(x.shape[1])[jnp.newaxis, :]
    )
    y, h = self._cell(x, h, segment_pos)
    return y, (h, start_index + x.shape[1])

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    segment_pos = jnp.arange(x.shape[1])[jnp.newaxis, :]
    h, _ = self.get_initial_state(
        x.shape[0], x.channel_spec, training=training, constants=constants
    )
    y, _ = self._cell(x, h, segment_pos)
    return y

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Always use float32 for the state, regardless of the input dtype.
    h = jnp.zeros((batch_size, self.config.units), dtype=jnp.float32)
    start_index = jnp.zeros([batch_size], jnp.int32)
    return h, start_index
