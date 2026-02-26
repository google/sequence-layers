"""Conditioning layers for MLX."""

import dataclasses
import enum
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx.init_mapping import _to_mx_dtype
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------


def _broadcast_shapes(shape1, shape2):
  """Compute the broadcast shape of two shapes (numpy-style)."""
  s1 = list(shape1)
  s2 = list(shape2)
  while len(s1) < len(s2):
    s1.insert(0, 1)
  while len(s2) < len(s1):
    s2.insert(0, 1)
  result = []
  for a, b in zip(s1, s2):
    if a == 1:
      result.append(b)
    elif b == 1:
      result.append(a)
    elif a == b:
      result.append(a)
    else:
      raise ValueError(f'Shapes {shape1} and {shape2} are not broadcastable')
  return tuple(result)


def _reshape_for_broadcast(*seqs):
  """Reshape channel dims of many sequences to be broadcastable."""
  max_dims = max(x.ndim for x in seqs)

  def _maybe_reshape(values):
    extra_dims = max_dims - values.ndim
    if extra_dims == 0:
      return values
    batch_size, time = values.shape[:2]
    shape = (batch_size, time) + (1,) * extra_dims + values.shape[2:]
    return mx.reshape(values, shape)

  return tuple(x.apply_values(_maybe_reshape) for x in seqs)


def _combine_mask(*masks):
  """AND together multiple masks."""
  result = masks[0]
  for m in masks[1:]:
    if m is not result:
      result = mx.logical_and(result, m)
  return result


def _sequence_broadcast_add(x, y):
  """Broadcast-add two sequences."""
  x, y = _reshape_for_broadcast(x, y)
  return Sequence(x.values + y.values, _combine_mask(x.mask, y.mask))


def _sequence_broadcast_product(x, y):
  """Broadcast-multiply two sequences."""
  x, y = _reshape_for_broadcast(x, y)
  return Sequence(x.values * y.values, _combine_mask(x.mask, y.mask))


def _sequence_broadcast_concat(x, y):
  """Broadcast-concat on last axis."""
  x, y = _reshape_for_broadcast(x, y)
  x_shape = x.values.shape
  y_shape = y.values.shape
  # Broadcast all dims except the last.
  target_outer = []
  for i in range(len(x_shape) - 1):
    target_outer.append(max(x_shape[i], y_shape[i]))
  x_vals = mx.broadcast_to(x.values, tuple(target_outer) + (x_shape[-1],))
  y_vals = mx.broadcast_to(y.values, tuple(target_outer) + (y_shape[-1],))
  return Sequence(
      mx.concatenate([x_vals, y_vals], axis=-1),
      _combine_mask(x.mask, y.mask),
  )


def _sequence_unstack(seq, axis):
  """Unstack a sequence along a channel axis."""
  if axis < 0:
    axis += seq.ndim
  if axis <= 1 or axis >= seq.ndim:
    raise ValueError(f'Invalid axis: {axis=} {seq.ndim=}')
  n = seq.values.shape[axis]
  splits = []
  for i in range(n):
    v = mx.take(seq.values, mx.array([i]), axis=axis)
    v = mx.squeeze(v, axis=axis)
    splits.append(v)
  return [type(seq)(v, seq.mask) for v in splits]


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------


def _get_conditioning(layer, conditioning_name, constants):
  """Gets the conditioning from constants."""
  if constants is None:
    raise ValueError(
        f'{layer} requires conditioning via constants, got: {constants}'
    )
  conditioning = constants.get(conditioning_name)
  if conditioning is None:
    raise ValueError(
        f'{layer} expected {conditioning_name!r} in constants,'
        f' got keys: {list(constants.keys())}'
    )
  return conditioning


def _tensor_to_fake_sequence(t):
  """Wrap a [B, ...] tensor as a [B, 1, ...] MaskedSequence."""
  batch_size = t.shape[0]
  return MaskedSequence(
      mx.expand_dims(t, axis=1),
      mx.ones((batch_size, 1), dtype=mx.bool_),
  )


# ---------------------------------------------------------------------------
# Conditioning layer
# ---------------------------------------------------------------------------


class Conditioning(types.SequenceLayer):
  """Conditions x on a conditioning signal from constants.

  Conditioning is done time-synchronized: each timestep of x is conditioned
  on the corresponding timestep of c.

  Conditioning = Combine(x, Project(c)).
  """

  class Projection(enum.Enum):
    IDENTITY = 1
    LINEAR = 2
    LINEAR_AFFINE = 3

  class Combination(enum.Enum):
    ADD = 1
    CONCAT = 2
    AFFINE = 3
    AFFINE_SHIFT = 4
    AFFINE_SCALE = 5
    MUL = 6
    CONCAT_BEFORE = 7

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    conditioning_name: str = ''
    projection: 'Conditioning.Projection' = None
    combination: 'Conditioning.Combination' = None
    projection_channel_shape: tuple[int, ...] | None = None
    streaming: bool = False
    affine_scale_offset: complex = 1.0
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    def make(self) -> 'Conditioning':
      return Conditioning.from_config(self)

  def __init__(
      self,
      *,
      conditioning_name,
      projection,
      combination,
      projection_channel_shape=None,
      streaming=False,
      affine_scale_offset=1.0,
      compute_dtype=None,
      param_dtype=mx.float32,
  ):
    super().__init__()
    self._conditioning_name = conditioning_name
    self._projection = projection
    self._combination = combination
    self._projection_channel_shape = projection_channel_shape
    self._streaming = streaming
    self._affine_scale_offset = affine_scale_offset
    self._compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    self._validate()

    # Projection kernel/bias (deferred until first use).
    self.kernel = None
    self.bias = None
    self._equation = None
    self._proj_initialized = False

  def _validate(self):
    if (
        self._combination == self.Combination.AFFINE
        and self._projection != self.Projection.LINEAR_AFFINE
    ):
      raise ValueError('AFFINE combination requires LINEAR_AFFINE projection.')
    if (
        self._combination == self.Combination.AFFINE_SHIFT
        and self._projection != self.Projection.LINEAR
    ):
      raise ValueError('AFFINE_SHIFT combination requires LINEAR projection.')
    if (
        self._combination == self.Combination.AFFINE_SCALE
        and self._projection != self.Projection.LINEAR
    ):
      raise ValueError('AFFINE_SCALE combination requires LINEAR projection.')
    if (
        self._combination != self.Combination.AFFINE
        and self._projection == self.Projection.LINEAR_AFFINE
    ):
      raise ValueError('LINEAR_AFFINE projection requires AFFINE combination.')

  def _ensure_projection_initialized(self, x_channel_shape, cond_channel_shape):
    """Initialize projection kernel/bias on first use."""
    if self._proj_initialized:
      return
    if self._projection == self.Projection.IDENTITY:
      self._proj_initialized = True
      return

    proj_shape = self._projection_channel_shape
    if proj_shape is None:
      proj_shape = x_channel_shape

    if self._projection == self.Projection.LINEAR_AFFINE:
      output_shape = (2,) + tuple(proj_shape)
    else:
      output_shape = tuple(proj_shape)

    # Build einsum equation matching Linen DenseShaped.
    input_dims = ''.join(
        chr(ord('a') + i) for i in range(len(cond_channel_shape))
    )
    output_dims = ''.join(
        chr(ord('a') + i + len(cond_channel_shape))
        for i in range(len(output_shape))
    )

    input_weight_dims = input_dims if input_dims else 'I'
    output_weight_dims = output_dims if output_dims else 'O'
    input_kernel_shape = cond_channel_shape if cond_channel_shape else (1,)
    output_kernel_shape = output_shape if output_shape else (1,)

    self._equation = (
        f'...{input_dims},{input_weight_dims}{output_weight_dims}'
        f'->...{output_dims}'
    )
    kernel_shape = input_kernel_shape + output_kernel_shape
    self.kernel = mx.zeros(kernel_shape, dtype=self._param_dtype)
    self.bias = mx.zeros(output_kernel_shape, dtype=self._param_dtype)
    self._proj_initialized = True

  def _projected_condition_shape(self, input_shape, condition_shape):
    """Compute the channel shape after projection."""
    proj_shape = self._projection_channel_shape
    if proj_shape is None:
      proj_shape = input_shape
    if self._projection == self.Projection.IDENTITY:
      return condition_shape
    elif self._projection == self.Projection.LINEAR:
      return tuple(proj_shape)
    elif self._projection == self.Projection.LINEAR_AFFINE:
      return (2,) + tuple(proj_shape)
    else:
      raise ValueError(f'Unsupported projection: {self._projection}')

  def get_output_shape(self, input_shape, *, constants=None):
    self._validate()
    cond = _get_conditioning(self, self._conditioning_name, constants)
    if isinstance(cond, (Sequence, MaskedSequence)):
      cond_shape = cond.channel_shape
    else:
      cond_shape = cond.shape[1:]
    proj_shape = self._projected_condition_shape(input_shape, cond_shape)

    if self._combination in (
        self.Combination.ADD,
        self.Combination.MUL,
        self.Combination.AFFINE_SHIFT,
        self.Combination.AFFINE_SCALE,
    ):
      return _broadcast_shapes(input_shape, proj_shape)
    elif self._combination in (
        self.Combination.CONCAT,
        self.Combination.CONCAT_BEFORE,
    ):
      input_inner = input_shape[-1] if input_shape else 1
      proj_inner = proj_shape[-1] if proj_shape else 1
      outer = _broadcast_shapes(input_shape[:-1], proj_shape[:-1])
      return outer + (input_inner + proj_inner,)
    elif self._combination == self.Combination.AFFINE:
      proj_shape = proj_shape[1:]  # Remove the '2' dim.
      return _broadcast_shapes(input_shape, proj_shape)
    else:
      raise ValueError(f'Unsupported combination: {self._combination}')

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self._compute_dtype is not None:
      return self._compute_dtype
    return self._param_dtype

  def _project(self, x, conditioning):
    """Apply projection to conditioning."""
    if self._projection == self.Projection.IDENTITY:
      return conditioning

    self._ensure_projection_initialized(
        x.channel_shape, conditioning.channel_shape
    )
    compute_dtype = self._compute_dtype or self._param_dtype

    def project_fn(v):
      y = mx.einsum(self._equation, v.astype(compute_dtype), self.kernel)
      y = y + self.bias
      return y

    return conditioning.apply_values(project_fn)

  def _combine(self, x, conditioning):
    """Combine projected conditioning with input."""
    if self._combination == self.Combination.ADD:
      return _sequence_broadcast_add(x, conditioning)
    elif self._combination == self.Combination.CONCAT:
      return _sequence_broadcast_concat(x, conditioning)
    elif self._combination == self.Combination.CONCAT_BEFORE:
      return _sequence_broadcast_concat(conditioning, x)
    elif self._combination == self.Combination.AFFINE:
      scale, shift = _sequence_unstack(conditioning, axis=2)
      scale = scale.apply_values(lambda v: v + self._affine_scale_offset)
      x_s, scale_s = _reshape_for_broadcast(x, scale)
      x_s2, shift_s = _reshape_for_broadcast(x, shift)
      values = x_s.values * scale_s.values + shift_s.values
      mask = _combine_mask(x.mask, scale.mask, shift.mask)
      return Sequence(values, mask)
    elif self._combination == self.Combination.AFFINE_SHIFT:
      return _sequence_broadcast_add(x, conditioning)
    elif self._combination == self.Combination.AFFINE_SCALE:
      conditioning = conditioning.apply_values(
          lambda v: v + self._affine_scale_offset
      )
      return _sequence_broadcast_product(x, conditioning)
    elif self._combination == self.Combination.MUL:
      return _sequence_broadcast_product(x, conditioning)
    else:
      raise ValueError(f'Unsupported combination: {self._combination}')

  @types.check_layer
  def layer(self, x, *, constants=None):
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if not isinstance(conditioning, (Sequence, MaskedSequence)):
      conditioning = _tensor_to_fake_sequence(conditioning)
    projected = self._project(x, conditioning)
    return self._combine(x, projected)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if (
        isinstance(conditioning, (Sequence, MaskedSequence))
        and not self._streaming
    ):
      return mx.zeros((batch_size,), mx.int32)
    return ()

  @types.check_step
  def step(self, x, state, *, constants=None):
    conditioning = _get_conditioning(self, self._conditioning_name, constants)
    if not isinstance(conditioning, (Sequence, MaskedSequence)):
      conditioning = _tensor_to_fake_sequence(conditioning)
    elif not self._streaming:
      time_index = state
      step_size = x.shape[1]
      idx = int(time_index[0])
      conditioning = type(conditioning)(
          conditioning.values[:, idx : idx + step_size],
          conditioning.mask[:, idx : idx + step_size],
      )
      state = time_index + step_size
    projected = self._project(x, conditioning)
    result = self._combine(x, projected)
    return result, state

  @classmethod
  def from_config(cls, config):
    """Create from a JAX Conditioning.Config."""
    compute_dtype = getattr(config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = _to_mx_dtype(compute_dtype)
    # Map JAX enum values to MLX enum values.
    projection = cls.Projection(config.projection.value)
    combination = cls.Combination(config.combination.value)
    return cls(
        conditioning_name=config.conditioning_name,
        projection=projection,
        combination=combination,
        projection_channel_shape=config.projection_channel_shape,
        streaming=config.streaming,
        affine_scale_offset=config.affine_scale_offset,
        compute_dtype=compute_dtype,
        param_dtype=_to_mx_dtype(config.param_dtype),
    )
