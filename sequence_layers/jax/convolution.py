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
"""Convolution layers."""

import abc
import dataclasses
import fractions
import typing
from typing import Callable, Sequence as TypingSequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.learning.deepmind.jax.typing import typing as jt
from google3.learning.gemini.gemax.core.models import meta

__all__ = (
    # go/keep-sorted start
    'Conv1D',
    'Conv1DTranspose',
    'Conv2D',
    'Conv2DTranspose',
    'Conv3D',
    'DepthwiseConv1D',
    # go/keep-sorted end
)


def _compute_conv_mask_logical(
    mask: types.MaskT,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    use_logical_or: bool,
) -> types.MaskT:
  """Computes the mask resulting from applying a "logical AND" masking rule.

  Args:
    mask: The input mask.
    kernel_size: The kernel size in the time dimension.
    stride: The stride in the time dimension.
    dilation_rate: The dilation rate in the time dimension.
    use_logical_or: When False (default), an AND mask is computed. When True, an
      OR mask is used instead.

  Returns:
    The mask resulting from applying a VALID-like "logical AND" rule. If any
    input timestep in the receptive field of the convolution is invalid, then
    the output is considered invalid.
  """
  # TODO(b/273591446): Workaround for VMEM OOMs when applying convolutions or
  # framing to long mask sequences. Remove when fixed.
  # Discussion: https://chat.google.com/room/AAAAfoV82X8/o2AWMjKjdwY
  if dilation_rate == 1 and kernel_size % stride == 0:
    num_frames = mask.shape[1] // stride
    mask = mask[:, : num_frames * stride]
    mask = jnp.reshape(mask, [mask.shape[0], num_frames, stride])

    if use_logical_or:
      mask = jnp.max(mask, axis=-1)
    else:
      mask = jnp.min(mask, axis=-1)

    kernel_size = kernel_size // stride
    stride = 1

  if use_logical_or:
    # Compute an OR operation over windows so any window overlapping with at
    # least some valid samples is considered valid.
    computation_fn = jax.lax.max
    init_value = False
  else:
    # Compute an AND operation over windows so any window with invalid samples
    # is considered invalid.
    computation_fn = jax.lax.min
    init_value = True

  mask = jax.lax.reduce_window(
      mask,
      init_value=init_value,
      computation=computation_fn,
      window_dimensions=(1, kernel_size),
      base_dilation=(1, 1),
      window_dilation=(1, dilation_rate),
      window_strides=(1, stride),
      padding='VALID',
  )
  return mask


def compute_conv_mask(
    mask: types.MaskT,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: tuple[int, int] | types.PaddingModeString,
    is_step: bool,
) -> types.MaskT:
  """Computes the output mask for a convolution-like operation.

  The formula for the output time dimension for a [b, t] mask is:

    effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
    output_frames = ceil((t - effective_kernel_size + 1) / stride)

  If padding is 'causal_valid' or 'reverse_causal_valid', we assume that the
  mask has padding of `effective_kernel_size - 1` applied which ensures that the
  above formula is at least zero, because `t >= effective_kernel_size - 1`.

  Args:
    mask: The input [b, t] mask.
    kernel_size: The kernel size in the time dimension.
    stride: The stride in the time dimension.
    dilation_rate: The dilation rate in the time dimension.
    padding: The padding mode or a tuple of explicit padding amounts.
    is_step: If true, we assume effective_kernel_size - 1 timesteps of padding
      have been prepended to the mask.

  Returns:
    The output mask. [b, ceil((t - effective_kernel_size + 1) / stride)]
  """
  assert kernel_size >= 1
  assert stride >= 1
  assert dilation_rate >= 1

  explicit_padding = not isinstance(padding, str)
  if explicit_padding:
    if dilation_rate != 1:
      raise ValueError('Dilation and explicit padding is not supported.')
  else:
    padding = types.validate_padding(padding)

  # If we are stepping, the caller has prepended buffer_size frames of padding
  # to the mask.
  #
  # TODO(rryan): How does stride offset factor into this?
  #
  # To simulate the convolution this mask is associated with, we perform a valid
  # convolution on the mask with a kernel that selects the correct starting
  # timestep.
  if is_step:
    if explicit_padding or padding in (
        types.PaddingMode.SAME.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    ):
      effective_kernel_size = utils.convolution_effective_kernel_size(
          kernel_size, dilation_rate
      )
      past_pad, future_pad = utils.convolution_explicit_padding(
          padding,
          kernel_size,
          stride=stride,
          dilation_rate=dilation_rate,
      )
      if past_pad + future_pad != effective_kernel_size - 1:
        raise ValueError(f'Padding {padding=} must sum to {kernel_size - 1}.')

      # TODO(rryan): Replace with a (presumably more efficient) strided slice.
      # It's tricky to get the details right with stride and dilation.
      kernel = jnp.array([0] * past_pad + [1] + [0] * future_pad, jnp.float32)
      return jnp.squeeze(
          jax.lax.conv_general_dilated(
              mask[:, :, jnp.newaxis].astype(jnp.float32),
              kernel[:, jnp.newaxis, jnp.newaxis],
              window_strides=[stride],
              padding='VALID',
              rhs_dilation=[1],
              dimension_numbers=('NHC', 'HIO', 'NHC'),
          ),
          axis=2,
      ).astype(jnp.bool_)
    else:
      assert padding in (
          types.PaddingMode.VALID.value,
          types.PaddingMode.CAUSAL_VALID.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
      ), padding
      return _compute_conv_mask_logical(
          mask, kernel_size, stride, dilation_rate, use_logical_or=False
      )

  # All logic below concerns layer-wise mask calculation.
  assert not is_step

  if explicit_padding or padding in (
      types.PaddingMode.SAME.value,
      types.PaddingMode.CAUSAL.value,
      types.PaddingMode.REVERSE_CAUSAL.value,
      types.PaddingMode.SEMICAUSAL.value,
  ):
    if stride > 1:
      mask = mask[:, ::stride]
    return mask

  assert padding in (
      types.PaddingMode.VALID.value,
      types.PaddingMode.CAUSAL_VALID.value,
      types.PaddingMode.REVERSE_CAUSAL_VALID.value,
      types.PaddingMode.SEMICAUSAL_FULL.value,
  ), padding

  past_pad, future_pad = utils.convolution_explicit_padding(
      padding, kernel_size, stride, dilation_rate
  )
  # TODO(rryan): Fold this padding into the below reduce_window. Not sure if
  # this would introduce a bug in the workaround for b/273591446 below.
  mask = jnp.pad(
      mask,
      [(0, 0), (past_pad, future_pad)],
      mode='constant',
      constant_values=padding == types.PaddingMode.CAUSAL_VALID.value,
  )

  return _compute_conv_mask_logical(
      mask,
      kernel_size,
      stride,
      dilation_rate,
      use_logical_or=(padding == types.PaddingMode.SEMICAUSAL_FULL.value),
  )


def _compute_conv_transpose_output_length(
    time: int,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: types.PaddingModeString,
):
  """Returns the output time for a transpose convolution, matching Keras."""
  # Based on google3/third_party/py/tf_keras/utils/conv_utils.py.
  padding = types.validate_padding(padding)

  effective_kernel_size = utils.convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  match padding:
    case (
        types.PaddingMode.SAME.value
        | types.PaddingMode.CAUSAL.value
        | types.PaddingMode.SEMICAUSAL_FULL.value
    ):
      output_time = time * stride
    case types.PaddingMode.VALID.value:
      output_time = time * stride + max(effective_kernel_size - stride, 0)
    case _:
      raise ValueError(f'Unsupported padding mode: {padding}')

  return output_time


def _transpose_conv_explicit_padding(
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: types.PaddingModeString,
) -> tuple[int, int]:
  """Returns the explicit padding for the desired transpose convolution mode."""
  effective_kernel_size = utils.convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  match padding:
    case types.PaddingMode.VALID.value:
      pad_amount = (
          effective_kernel_size
          + stride
          - 2
          + max(effective_kernel_size - stride, 0)
      )
      pad_left = effective_kernel_size - 1
      pad_right = pad_amount - pad_left
    case types.PaddingMode.CAUSAL.value:
      pad_amount = effective_kernel_size + stride - 2
      pad_left = effective_kernel_size - 1
      pad_right = pad_amount - pad_left
    case types.PaddingMode.SAME.value:
      pad_amount = effective_kernel_size + stride - 2
      if stride > effective_kernel_size - 1:
        pad_left = effective_kernel_size - 1
      else:
        pad_left = int(np.ceil(pad_amount / 2))
      pad_right = pad_amount - pad_left
    case types.PaddingMode.SEMICAUSAL_FULL.value:
      pad_left = effective_kernel_size - stride
      pad_right = effective_kernel_size - 1
    case _:
      raise ValueError(f'Unsupported padding: {padding}')

  return pad_left, pad_right


def compute_conv_transpose_mask(
    mask: jax.Array,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: types.PaddingModeString,
) -> jax.Array:
  """Given an input mask, computes the output mask for a transpose convolution.

  Args:
    mask: The input [b, t] mask.
    kernel_size: The kernel size in the time dimension.
    stride: The stride in the time dimension.
    dilation_rate: The dilation rate of the transpose convolution.
    padding: The padding mode. One of 'valid', 'same', or 'causal'.

  Returns:
    The output mask. [b, t * stride + max(kernel_size - stride, 0)].
  """
  padding = types.validate_padding(padding)

  if padding in (
      types.PaddingMode.REVERSE_CAUSAL.value,
      types.PaddingMode.REVERSE_CAUSAL_VALID.value,
      types.PaddingMode.CAUSAL_VALID.value,
  ):
    raise ValueError(f'Unsupported padding mode: {padding}')

  effective_kernel_size = utils.convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  if effective_kernel_size <= stride or padding in (
      types.PaddingMode.SAME.value,
      types.PaddingMode.CAUSAL.value,
  ):
    return jnp.repeat(mask, stride, axis=1)

  # If effective_kernel_size > stride, use an actual transpose convolution to
  # compute the mask.
  explicit_padding = _transpose_conv_explicit_padding(
      kernel_size,
      stride,
      dilation_rate,
      padding,
  )

  # Any non-zero values in mask have been corrupted by invalid timesteps.
  if padding == types.PaddingMode.SEMICAUSAL_FULL.value:
    # The mask will result in an OR mask.
    test_signal = mask
    test_fn = jnp.greater
  else:
    # The invalid mask will give an AND mask.
    test_signal = jnp.logical_not(mask)
    test_fn = jnp.equal

  mask = jax.lax.conv_general_dilated(
      test_signal.astype(jnp.float32)[:, :, jnp.newaxis],
      jnp.ones((kernel_size, 1, 1)),
      window_strides=(1,),
      padding=(explicit_padding,),
      lhs_dilation=(stride,),
      rhs_dilation=(dilation_rate,),
      dimension_numbers=('NHC', 'HIO', 'NHC'),
      feature_group_count=1,
      batch_group_count=1,
  )
  return jnp.squeeze(test_fn(mask, 0.0), -1)


class BaseConv(types.SequenceLayer, metaclass=abc.ABCMeta):
  """Shared base logic for convolution layers."""

  @property
  @abc.abstractmethod
  def _kernel_size(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _strides(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _dilation_rate(self) -> tuple[int, ...]:
    pass

  @property
  @abc.abstractmethod
  def _paddings(self) -> tuple[types.PaddingModeString | tuple[int, int], ...]:
    pass

  @property
  def supports_step(self) -> bool:
    return self._paddings[0] in (
        types.PaddingMode.CAUSAL_VALID.value,
        types.PaddingMode.REVERSE_CAUSAL_VALID.value,
        types.PaddingMode.CAUSAL.value,
        types.PaddingMode.REVERSE_CAUSAL.value,
        types.PaddingMode.SEMICAUSAL.value,
    )

  @property
  def block_size(self) -> int:
    return self._strides[0]

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1, self._strides[0])

  @property
  def input_latency(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )

    match self._paddings[0]:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        # Causal padding eliminates latency.
        return 0
      case (
          types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.SEMICAUSAL_FULL.value
      ):
        # Reverse causal introduces no past padding in layer-wise mode, so we
        # need a full effective_kernel_size kernel to compute the first output
        # layer-wise processing would produce. Since we do not count the current
        # input as part of the latency, the input latency is one smaller than
        # the effective kernel size.
        # Semicausal-full will pad zeros at the end of the sequence until the
        # kernel no longer overlaps with the sequence. This results in at most
        # the effective kernel size minus 1 zeros padded.
        return effective_kernel_size - 1
      case _:
        # Unsupported.
        return 0

  @property
  def output_latency(self) -> fractions.Fraction:
    """Returns the output latency of this layer.

    Output latency is defined as the number of output timesteps before the
    step-wise output of the layer matches its layer-wise output.
    """
    match self._paddings[0]:
      case types.PaddingMode.SEMICAUSAL_FULL.value:
        # The semicausal-full padding has both left and right padding, which
        # requires a different output_latency for correct operations.
        # It only requires to wait for strides[0] - 1 samples, and the
        # output_ratio is also strides[0], which means the latency is 0.
        return fractions.Fraction(0)
      case _:
        # Other cases are handled in the parent class.
        return super().output_latency

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )

    match self._paddings[0]:
      case types.PaddingMode.SEMICAUSAL.value:
        return max(effective_kernel_size - self._strides[0], 0)
      case (
          types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
      ):
        return (
            (effective_kernel_size - 1) // self._strides[0] * self._strides[0]
        )
      case (
          types.PaddingMode.CAUSAL.value | types.PaddingMode.CAUSAL_VALID.value
      ):
        return effective_kernel_size - 1
      case _:
        raise NotImplementedError(
            f'Unsupported padding mode: {self._paddings[0]}'
        )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    # Special case kernel_size 1 since it is stateless.
    if not (buffer_width := self._buffer_width):
      return ()

    # Input spec should be the non-time conv dimensions plus a channels
    # dimension.
    if len(input_spec.shape) != len(self._kernel_size):
      raise ValueError(
          f'{type(self).__name__} requires a rank 2 input spec, got:'
          f' {input_spec}.'
      )

    match self._paddings[0]:
      case (
          types.PaddingMode.CAUSAL_VALID.value
          | types.PaddingMode.REVERSE_CAUSAL_VALID.value
          | types.PaddingMode.SEMICAUSAL_FULL.value
      ):
        mask = jnp.ones((batch_size, buffer_width), dtype=types.MASK_DTYPE)
      case (
          types.PaddingMode.CAUSAL.value
          | types.PaddingMode.REVERSE_CAUSAL.value
          | types.PaddingMode.SEMICAUSAL.value
      ):
        mask = jnp.zeros((batch_size, buffer_width), dtype=types.MASK_DTYPE)
      case _:
        raise ValueError(
            'Stepwise processing is not supported with padding:'
            f' {self._paddings[0]}'
        )

    return types.MaskedSequence(
        jnp.zeros(
            (batch_size, buffer_width) + input_spec.shape,
            dtype=input_spec.dtype,
        ),
        mask,
    )

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # In step mode, causal padding is handled by the state that is concatenated
    # below.
    explicit_paddings = ((0, 0),) + tuple(
        utils.convolution_explicit_padding(
            padding, kernel_size, stride, dilation_rate
        )
        for padding, kernel_size, stride, dilation_rate in zip(
            self._paddings[1:],
            self._kernel_size[1:],
            self._strides[1:],
            self._dilation_rate[1:],
            strict=True,
        )
    )

    effective_kernel_size = utils.convolution_effective_kernel_size(
        self._kernel_size[0], self._dilation_rate[0]
    )
    # Mask the input if the effective kernel size is greater than 1.
    if effective_kernel_size > 1:
      x = x.mask_invalid()

    if buffer_width := self._buffer_width:
      # Concatenate the new frames with the previous buffer_width frames.
      state = state.concatenate(x)
    else:
      state = x

    # Compute the output for the current timestep.
    values = self._layer(state.values, padding=explicit_paddings)
    mask = compute_conv_mask(
        state.mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
        is_step=True,
    )

    # Keep the trailing buffer_width samples for the next step.
    if buffer_width:
      state = state[:, -buffer_width:]
    else:
      state = ()

    # Convolution can leave unmasked values with non-zero values.
    return types.Sequence(values, mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None = None,
  ):
    # Mask inputs if receptive field is greater than 1.
    if self._kernel_size[0] > 1:
      x = x.mask_invalid()

    explicit_paddings = tuple(
        utils.convolution_explicit_padding(p, k, s, d)
        for p, k, s, d in zip(
            self._paddings,
            self._kernel_size,
            self._strides,
            self._dilation_rate,
            strict=True,
        )
    )
    values = self._layer(x.values, padding=explicit_paddings)
    mask = compute_conv_mask(
        x.mask,
        self._kernel_size[0],
        self._strides[0],
        self._dilation_rate[0],
        self._paddings[0],
        is_step=False,
    )
    # If the convolution has a receptive field of 1 and no bias then the output
    # preserves the input mask state.
    result_type = (
        types.Sequence
        if self.config.use_bias or self._kernel_size[0] > 1
        else type(x)
    )
    return result_type(values, mask)

  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T *S D'],
      padding: tuple[tuple[int, int], ...],
  ) -> jt.Float[jt.ArrayT, 'B T *S D']:
    raise NotImplementedError()


def _weight_norm(module: nn.Module, weight: jax.Array, name: str) -> jax.Array:
  """Normalizes weight along the last dim https://arxiv.org/abs/1602.07868."""
  # TODO: b/330719129 - EMA(WN(w)) support similar to TF2 seanet.
  # http://google3/audio/ears/minimodal/seanet/layers/wrappers.py;l=36;rcl=530722248
  if module.is_initializing():
    # Based on tensorflow addon, which preserves the original scale.
    # http://google3/third_party/py/tensorflow_addons/layers/wrappers.py;l=181;rcl=509843210
    scale = module.param(
        name,
        lambda unused_key: jnp.linalg.norm(
            weight.reshape([-1, weight.shape[-1]]), axis=0
        ),
    )
  else:
    # Dummy initializer to avoid flax's eval_shape cost.
    scale = module.param(
        name, nn.initializers.zeros_init(), [weight.shape[-1]], weight.dtype
    )
  axis = list(range(weight.ndim - 1))
  scale = jnp.expand_dims(scale, axis)
  return scale * _l2_normalize(weight, axis)


def _l2_normalize(
    x: jax.Array, axis: int | TypingSequence[int] | None, eps: float = 1e-12
) -> jax.Array:
  """L2-normalize a Jax tensor along certain dimension."""
  return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class Conv1D(BaseConv):
  """A 1D strided or dilated convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Conv1D."""

    filters: int
    kernel_size: int
    strides: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    groups: int = 1
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: nn.linear.PrecisionLike = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

    def make(self) -> 'Conv1D':
      return Conv1D(self, name=self.name)

  config: Config

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return (self.config.kernel_size,)

  @property
  def _strides(self) -> tuple[int, ...]:
    return (self.config.strides,)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return (self.config.dilation_rate,)

  @property
  def _paddings(self) -> tuple[types.PaddingModeString | tuple[int, int], ...]:
    return (self.config.padding,)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'Conv1D requires rank 3 input got: %s'
          % ([None, None] + list(input_shape))
      )
    return (self.config.filters,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.compact
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T D'],
      padding: tuple[tuple[int, int], ...],
  ) -> jt.Float[jt.ArrayT, 'B T D']:
    assert len(padding) == 1, padding
    in_features = jnp.shape(x)[-1]
    if in_features % self.config.groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({self.config.groups}).'
      )
    kernel_shape = (
        self.config.kernel_size,
        in_features // self.config.groups,
        self.config.filters,
    )
    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, meta.AxisType.FANIN, None),
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')
    # One bias weight per output channel, shared between pixels.
    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.config.filters,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(self.config.strides,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(self.config.dilation_rate,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=self.config.groups,
        precision=self.config.precision,
    )

    if bias is not None:
      y = utils.bias_add(y, bias)

    if self.config.activation:
      y = self.config.activation(y)

    return y


class DepthwiseConv1D(BaseConv):
  """A 1D depthwise strided or dilated convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for DepthwiseConv1D."""

    kernel_size: int
    strides: int = 1
    depth_multiplier: int = 1
    dilation_rate: int = 1
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: nn.linear.PrecisionLike = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

    def make(self) -> 'DepthwiseConv1D':
      return DepthwiseConv1D(self, name=self.name)

  config: Config

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    return (self.config.kernel_size,)

  @property
  def _strides(self) -> tuple[int, ...]:
    return (self.config.strides,)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    return (self.config.dilation_rate,)

  @property
  def _paddings(self) -> tuple[types.PaddingModeString | tuple[int, int], ...]:
    return (self.config.padding,)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'Conv1D requires rank 3 input got: %s'
          % ([None, None] + list(input_shape))
      )
    return (input_shape[0] * self.config.depth_multiplier,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.compact
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T D'],
      padding: tuple[tuple[int, int], ...],
  ) -> jt.Float[jt.ArrayT, 'B T D']:
    assert len(padding) == 1, padding
    in_features = jnp.shape(x)[-1]
    out_features = in_features * self.config.depth_multiplier
    kernel_shape = (self.config.kernel_size, 1, out_features)
    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, meta.AxisType.FANIN, None),
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')
    # One bias weight per output channel, shared between pixels.
    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (out_features,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(self.config.strides,),
        padding=padding,
        lhs_dilation=(1,),
        rhs_dilation=(self.config.dilation_rate,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=in_features,
        precision=self.config.precision,
    )

    if bias is not None:
      y = utils.bias_add(y, bias)

    if self.config.activation:
      y = self.config.activation(y)

    return y


class Conv2D(BaseConv):
  """A 2D strided or dilated convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Conv2D."""

    filters: int
    kernel_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # convolution. Conv2D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', or 'reverse_causal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial dimension of the convolution.
    spatial_padding: types.PaddingModeString | tuple[int, int] = (
        types.PaddingMode.SAME.value
    )
    groups: int = 1
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: nn.linear.PrecisionLike = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if isinstance(self.spatial_padding, str):
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_padding(self.spatial_padding),
        )
      else:
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_explicit_padding(self.spatial_padding),
        )
      object.__setattr__(
          self, 'kernel_size', utils.normalize_2tuple(self.kernel_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_2tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_2tuple(self.dilation_rate)
      )

    def make(self) -> 'Conv2D':
      return Conv2D(self, name=self.name)

  config: Config

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.kernel_size, tuple)
        and len(self.config.kernel_size) == 2
    ), self.config.kernel_size
    return typing.cast(tuple[int, ...], self.config.kernel_size)

  @property
  def _strides(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.strides, tuple) and len(self.config.strides) == 2
    ), self.config.strides
    return typing.cast(tuple[int, ...], self.config.strides)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.dilation_rate, tuple)
        and len(self.config.dilation_rate) == 2
    ), self.config.dilation_rate
    return typing.cast(tuple[int, ...], self.config.dilation_rate)

  @property
  def _paddings(self) -> tuple[types.PaddingModeString | tuple[int, int], ...]:
    return (self.config.time_padding, self.config.spatial_padding)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Conv2D requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_size = utils.convolution_padding_output_size(
        input_shape[0],
        self.config.spatial_padding,
        kernel_size=self.config.kernel_size[1],
        stride=self.config.strides[1],
        dilation_rate=self.config.dilation_rate[1],
    )
    return (
        spatial_output_size,
        self.config.filters,
    )

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.compact
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H D'],
      padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H D']:
    assert len(padding) == 2, padding
    in_features = jnp.shape(x)[-1]
    if in_features % self.config.groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({self.config.groups}).'
      )

    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.kernel_size, tuple)
        and len(self.config.kernel_size) == 2
    )
    assert (
        isinstance(self.config.strides, tuple) and len(self.config.strides) == 2
    )
    assert (
        isinstance(self.config.dilation_rate, tuple)
        and len(self.config.dilation_rate) == 2
    )

    kernel_shape = self.config.kernel_size + (
        in_features // self.config.groups,
        self.config.filters,
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
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')
    # One bias weight per output channel, shared between pixels.
    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.config.filters,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=self.config.strides,
        padding=padding,
        lhs_dilation=(1, 1),
        rhs_dilation=self.config.dilation_rate,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=self.config.groups,
        precision=self.config.precision,
    )

    if bias is not None:
      y = utils.bias_add(y, bias)

    if self.config.activation:
      y = self.config.activation(y)

    return y


class Conv3D(BaseConv):
  """A 3D strided or dilated convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Conv3D."""

    filters: int
    kernel_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # A padding mode string determining how to pad the time dimension of the
    # convolution. Conv3D is only streamable if time_padding is 'causal_valid'
    # 'reverse_causal_valid', 'causal', or 'reverse_causal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # A padding mode string or explicit padding values determining how to pad
    # the spatial dimensions of the convolution.
    spatial_padding: tuple[
        types.PaddingModeString | tuple[int, int],
        types.PaddingModeString | tuple[int, int],
    ] = (types.PaddingMode.SAME.value, types.PaddingMode.SAME.value)
    groups: int = 1
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: nn.linear.PrecisionLike = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )
      if len(self.spatial_padding) != 2:
        raise ValueError(
            'Conv3D expects 2 spatial padding modes got:'
            f' {self.spatial_padding}'
        )

      object.__setattr__(
          self,
          'spatial_padding',
          tuple(
              types.validate_padding(s) if isinstance(s, str) else s
              for s in self.spatial_padding
          ),
      )
      object.__setattr__(
          self, 'kernel_size', utils.normalize_3tuple(self.kernel_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_3tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_3tuple(self.dilation_rate)
      )

    def make(self) -> 'Conv3D':
      return Conv3D(self, name=self.name)

  config: Config

  @property
  def _kernel_size(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.kernel_size, tuple)
        and len(self.config.kernel_size) == 3
    ), self.config.kernel_size
    return typing.cast(tuple[int, ...], self.config.kernel_size)

  @property
  def _strides(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.strides, tuple) and len(self.config.strides) == 3
    ), self.config.strides
    return typing.cast(tuple[int, ...], self.config.strides)

  @property
  def _dilation_rate(self) -> tuple[int, ...]:
    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.dilation_rate, tuple)
        and len(self.config.dilation_rate) == 3
    ), self.config.dilation_rate
    return typing.cast(tuple[int, ...], self.config.dilation_rate)

  @property
  def _paddings(self) -> tuple[types.PaddingModeString | tuple[int, int], ...]:
    return (self.config.time_padding,) + self.config.spatial_padding

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 3:
      raise ValueError(
          'Conv3D requires rank 5 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_sizes = [
        utils.convolution_padding_output_size(
            input_shape[i],
            self.config.spatial_padding[i],
            kernel_size=self.config.kernel_size[1 + i],
            stride=self.config.strides[1 + i],
            dilation_rate=self.config.dilation_rate[1 + i],
        )
        for i in range(2)
    ]

    return tuple(spatial_output_sizes) + (self.config.filters,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @nn.compact
  def _layer(
      self,
      x: jt.Float[jt.ArrayT, 'B T H W D'],
      padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
  ) -> jt.Float[jt.ArrayT, 'B T H W D']:
    assert len(padding) == 3, padding
    in_features = jnp.shape(x)[-1]
    if in_features % self.config.groups != 0:
      raise ValueError(
          f'Input features ({in_features}) must be divisible by groups'
          f' ({self.config.groups}).'
      )

    # Config normalizes these for us in __post_init__.
    assert (
        isinstance(self.config.kernel_size, tuple)
        and len(self.config.kernel_size) == 3
    )
    assert (
        isinstance(self.config.strides, tuple) and len(self.config.strides) == 3
    )
    assert (
        isinstance(self.config.dilation_rate, tuple)
        and len(self.config.dilation_rate) == 3
    )

    kernel_shape = self.config.kernel_size + (
        in_features // self.config.groups,
        self.config.filters,
    )
    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(
            meta.AxisType.FANIN,
            meta.AxisType.FANIN,
            meta.AxisType.FANIN,
            meta.AxisType.FANIN,
            None,
        ),
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')
    # One bias weight per output channel, shared between channels.
    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.config.filters,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=self.config.strides,
        padding=padding,
        lhs_dilation=(1, 1, 1),
        rhs_dilation=self.config.dilation_rate,
        dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'),
        feature_group_count=self.config.groups,
        precision=self.config.precision,
    )

    if bias is not None:
      y = utils.bias_add(y, bias)

    if self.config.activation:
      y = self.config.activation(y)

    return y


class Conv1DTranspose(types.SequenceLayer):
  """A 1D transpose convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Conv1DTranspose."""

    filters: int
    kernel_size: int
    strides: int = 1
    dilation_rate: int = 1
    # The padding mode for the time dimension. Only 'valid', 'causal', or 'same'
    # is supported. Only streamable when using 'causal'.
    padding: types.PaddingModeString = types.PaddingMode.VALID.value
    groups: int = 1
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: str | None = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'padding', types.validate_padding(self.padding))

      if self.padding in (
          types.PaddingMode.REVERSE_CAUSAL.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
          types.PaddingMode.CAUSAL_VALID.value,
      ):
        raise ValueError(f'Unsupported padding mode: {self.padding}')

    def make(self) -> 'Conv1DTranspose':
      return Conv1DTranspose(self, name=self.name)

  config: Config

  @property
  def supports_step(self) -> bool:
    return self.config.padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.config.strides, 1)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return (self.config.filters,)

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self.config.kernel_size, self.config.dilation_rate
    )
    return max(0, effective_kernel_size - self.config.strides)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if buffer_width := self._buffer_width:
      output_spec = self.get_output_spec(input_spec, constants=constants)
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  @nn.compact
  def _layer(
      self,
      x: jax.Array,
      explicit_padding: tuple[int, int],
  ) -> tuple[jax.Array, jax.Array | None]:
    if x.ndim != 3:
      raise ValueError(f'Expected 3 dimension input. Got: {x.shape}')
    input_channels = x.shape[2]

    if input_channels % self.config.groups != 0:
      raise ValueError(
          f'Input features ({input_channels}) must be divisible by groups'
          f' ({self.config.groups}).'
      )

    kernel_init = utils.shard_initializer(
        self.config.kernel_init,
        self.config.kernel_sharding,
        projectable=True,
        axes_types=(meta.AxisType.FANIN, meta.AxisType.FANIN, None),
    )

    kernel_shape = (
        self.config.kernel_size,
        input_channels // self.config.groups,
        self.config.filters,
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.config.filters,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1,),
        padding=(explicit_padding,),
        lhs_dilation=(self.config.strides,),
        rhs_dilation=(self.config.dilation_rate,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
        feature_group_count=self.config.groups,
        batch_group_count=1,
        precision=self.config.precision,
    )

    # We apply bias and activation after overlap-adding with the state buffer.
    return y, bias

  @nn.nowrap
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Mask inputs if receptive field is greater than 1.
    if self.config.kernel_size > 1:
      x = x.mask_invalid()

    # Trim the last effective_kernel_size - stride samples since we don't
    # produce these in step mode (we can only produce stride samples at a time
    # in step mode, so we can't produce the final effective_kernel_size - stride
    # samples).

    explicit_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
    )

    values, bias = self._layer(x.values, explicit_padding)
    mask = compute_conv_transpose_mask(
        x.mask,
        self.config.kernel_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
    )

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.config.activation:
      values = self.config.activation(values)

    # Transpose convolution leaves padding with nonzero values.
    y = types.Sequence(values, mask)

    return y

  @types.check_step
  @nn.nowrap
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # Mask inputs if receptive field is greater than 1.
    if self.config.kernel_size > 1:
      x = x.mask_invalid()

    # Compute valid padding for step.
    explicit_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size,
        self.config.strides,
        self.config.dilation_rate,
        types.PaddingMode.VALID.value,
    )

    values, bias = self._layer(x.values, explicit_padding)
    mask = compute_conv_transpose_mask(
        x.mask,
        self.config.kernel_size,
        self.config.strides,
        self.config.dilation_rate,
        self.config.padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      # Pad the state to extend it to the length of the layer output.
      # output_time is at least kernel_size and buffer_width is at most
      # kernel_size - 1, so output_time - buffer_width is positive.
      state = jnp.pad(
          state, [[0, 0], [0, values.shape[1] - self._buffer_width], [0, 0]]
      )

      # Overlap-add outputs from previous timesteps into values.
      values = values + state

      # Stride samples are "ready" for output after one timestep, so the number
      # of output samples for the block is stride * time.
      output_samples = self.config.strides * time

      # We need to store (effective_kernel_size - stride) samples for future
      # steps, since their value depends on future inputs.
      values, state = jnp.split(values, [output_samples], axis=1)

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.config.activation:
      values = self.config.activation(values)

    # Transpose convolution leaves padding with nonzero values.
    return types.Sequence(values, mask), state


class Conv2DTranspose(types.SequenceLayer):
  """A 2D transpose convolution layer."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Conv2DTranspose."""

    filters: int
    kernel_size: int | TypingSequence[int]
    strides: int | TypingSequence[int] = 1
    dilation_rate: int | TypingSequence[int] = 1
    # The padding mode for the time dimension. Only 'valid', 'causal', or 'same'
    # is supported. Only streamable when using 'causal'.
    time_padding: types.PaddingModeString = types.PaddingMode.VALID.value
    # The padding mode for the spatial dimension. Only 'valid', 'causal',
    # 'same', or explicit padding is supported.
    spatial_padding: types.PaddingModeString | tuple[int, int] = (
        types.PaddingMode.SAME.value
    )
    groups: int = 1
    use_bias: bool = True
    use_weight_norm: bool = False
    activation: Callable[[jax.Array], jax.Array] | None = None
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = jnp.float32
    precision: str | None = None
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
    kernel_sharding: types.Sharding | None = None
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    bias_sharding: types.Sharding | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(
          self, 'time_padding', types.validate_padding(self.time_padding)
      )

      if self.time_padding in (
          types.PaddingMode.REVERSE_CAUSAL.value,
          types.PaddingMode.REVERSE_CAUSAL_VALID.value,
          types.PaddingMode.CAUSAL_VALID.value,
      ):
        raise ValueError(f'Unsupported padding mode: {self.time_padding}')
      if isinstance(self.spatial_padding, str):
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_padding(self.spatial_padding),
        )
        if self.spatial_padding in (
            types.PaddingMode.REVERSE_CAUSAL.value,
            types.PaddingMode.REVERSE_CAUSAL_VALID.value,
            types.PaddingMode.CAUSAL_VALID.value,
        ):
          raise ValueError(f'Unsupported padding mode: {self.spatial_padding}')
      else:
        object.__setattr__(
            self,
            'spatial_padding',
            types.validate_explicit_padding(self.spatial_padding),
        )
      object.__setattr__(
          self, 'kernel_size', utils.normalize_2tuple(self.kernel_size)
      )
      object.__setattr__(self, 'strides', utils.normalize_2tuple(self.strides))
      object.__setattr__(
          self, 'dilation_rate', utils.normalize_2tuple(self.dilation_rate)
      )

    def make(self) -> 'Conv2DTranspose':
      return Conv2DTranspose(self, name=self.name)

  config: Config

  @property
  def supports_step(self) -> bool:
    return self.config.time_padding == types.PaddingMode.CAUSAL.value

  @property
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(self.config.strides[0], 1)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 2:
      raise ValueError(
          'Conv2DTranspose requires rank 4 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    spatial_output_size = _compute_conv_transpose_output_length(
        input_shape[0],
        self.config.kernel_size[1],
        self.config.strides[1],
        dilation_rate=self.config.dilation_rate[1],
        padding=self.config.spatial_padding,
    )
    return (
        spatial_output_size,
        self.config.filters,
    )

  @nn.nowrap
  def get_output_dtype(
      self,
      input_dtype: types.DType,
  ) -> types.DType:
    return utils.get_promoted_dtype(
        input_dtype, self.config.param_dtype, dtype=self.config.compute_dtype
    )

  @property
  def _buffer_width(self) -> int:
    effective_kernel_size = utils.convolution_effective_kernel_size(
        self.config.kernel_size[0], self.config.dilation_rate[0]
    )
    return max(0, effective_kernel_size - self.config.strides[0])

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    output_spec = self.get_output_spec(input_spec, constants=constants)
    if buffer_width := self._buffer_width:
      return jnp.zeros(
          (batch_size, buffer_width) + output_spec.shape,
          dtype=output_spec.dtype,
      )
    else:
      return ()

  @nn.compact
  def _layer(
      self,
      x: jax.Array,
      explicit_padding: tuple[tuple[int, int], tuple[int, int]],
  ) -> tuple[jax.Array, jax.Array | None]:
    if x.ndim != 4:
      raise ValueError(f'Expected 4 dimension input. Got: {x.shape}')
    input_channels = x.shape[3]

    if input_channels % self.config.groups != 0:
      raise ValueError(
          f'Input features ({input_channels}) must be divisible by groups'
          f' ({self.config.groups}).'
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

    kernel_shape = tuple(self.config.kernel_size) + (
        input_channels // self.config.groups,
        self.config.filters,
    )
    kernel = self.param(
        'kernel', kernel_init, kernel_shape, self.config.param_dtype
    )
    if self.config.use_weight_norm:
      kernel = _weight_norm(self, kernel, 'scale')

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.bias_sharding
      )
      bias = self.param(
          'bias',
          bias_init,
          (self.config.filters,),
          self.config.param_dtype,
      )
    else:
      bias = None

    x, kernel, bias = nn.dtypes.promote_dtype(
        x, kernel, bias, dtype=self.config.compute_dtype
    )

    y = jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1),
        padding=explicit_padding,
        lhs_dilation=self.config.strides,
        rhs_dilation=self.config.dilation_rate,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=self.config.groups,
        batch_group_count=1,
        precision=self.config.precision,
    )

    # We apply bias and activation after overlap-adding with the state buffer.
    return y, bias

  @nn.nowrap
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Mask inputs if time receptive field is greater than 1.
    if self.config.kernel_size[0] > 1:
      x = x.mask_invalid()

    explicit_time_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        self.config.time_padding,
    )

    explicit_spatial_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size[1],
        self.config.strides[1],
        self.config.dilation_rate[1],
        self.config.spatial_padding,
    )

    # Trim the last effective_kernel_size - stride samples since we don't
    # produce these in step mode (we can only produce stride samples at a time
    # in step mode, so we can't produce the final effective_kernel_size - stride
    # samples).
    values, bias = self._layer(
        x.values, (explicit_time_padding, explicit_spatial_padding)
    )
    mask = compute_conv_transpose_mask(
        x.mask,
        self.config.kernel_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        self.config.time_padding,
    )

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.config.activation:
      values = self.config.activation(values)

    # Transpose convolution leaves padding with nonzero values.
    y = types.Sequence(values, mask)

    return y

  @types.check_step
  @nn.nowrap
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # Mask inputs if time receptive field is greater than 1.
    if self.config.kernel_size[0] > 1:
      x = x.mask_invalid()

    # Compute valid padding for step.
    explicit_time_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        types.PaddingMode.VALID.value,
    )

    explicit_spatial_padding = _transpose_conv_explicit_padding(
        self.config.kernel_size[1],
        self.config.strides[1],
        self.config.dilation_rate[1],
        self.config.spatial_padding,
    )

    values, bias = self._layer(
        x.values, (explicit_time_padding, explicit_spatial_padding)
    )
    mask = compute_conv_transpose_mask(
        x.mask,
        self.config.kernel_size[0],
        self.config.strides[0],
        self.config.dilation_rate[0],
        self.config.time_padding,
    )

    if self._buffer_width:
      time = x.shape[1]

      # Pad the state to extend it to the length of the layer output.
      # output_time is at least kernel_size and buffer_width is at most
      # kernel_size - 1, so output_time - buffer_width is positive.
      state = jnp.pad(
          state,
          [[0, 0], [0, values.shape[1] - self._buffer_width], [0, 0], [0, 0]],
      )

      # Overlap-add outputs from previous timesteps into values.
      values = values + state

      # Stride samples are "ready" for output after one timestep, so the number
      # of output samples for the block is stride * time.
      output_samples = self.config.strides[0] * time

      # We need to store (effective_kernel_size - stride) samples for future
      # steps, since their value depends on future inputs.
      values, state = jnp.split(values, [output_samples], axis=1)

    if bias is not None:
      values = utils.bias_add(values, bias)

    if self.config.activation:
      values = self.config.activation(values)

    # Transpose convolution leaves padding with nonzero values.
    return types.Sequence(values, mask), state
