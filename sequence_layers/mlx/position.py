"""Position embeddings for MLX."""

import dataclasses

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence


class ApplyRotaryPositionalEncoding(
    types.PreservesType,
    types.PreservesShape,
    types.SequenceLayer,
):
  """Applies Rotary Positional Encodings (RoPE) to the sequence."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    max_wavelength: float = 10000.0
    axis: int = -1
    only_advance_position_for_valid_timesteps: bool = True
    positions_in_at_least_fp32: bool = True
    positions_name: str | None = None
    name: str | None = None

    def make(self) -> 'ApplyRotaryPositionalEncoding':
      return ApplyRotaryPositionalEncoding.from_config(self)

  def __init__(
      self,
      *,
      max_wavelength: float,
      axis: int = -1,
      only_advance_position_for_valid_timesteps: bool = True,
  ):
    super().__init__()
    self.max_wavelength = max_wavelength
    self._axis = axis
    self.only_advance_position_for_valid_timesteps = (
        only_advance_position_for_valid_timesteps
    )

  def _apply_rope(self, x, offset):
    """Applies rotary position encoding to x with a given temporal offset.
    
    If the rotation axis is the last dimension (the default for most models),
    this method leverages the highly optimized `mx.fast.rope` native C++ operation.
    Since `mx.fast.rope` strictly expects the sequence length (time) to be the
    second-to-last dimension, we transpose the tensor, apply the rotation, and 
    transpose it back. If rotation is on an inner axis, it falls back to a 
    manual trig-based computation.
    """
    axis = self._axis + x.ndim if self._axis < 0 else self._axis
    
    if axis != x.ndim - 1:
        channel_ndim = x.ndim - 2
        axis_dim = x.shape[axis]

        freq_exponents = (
            2.0 * mx.arange(axis_dim // 2).astype(mx.float32) / axis_dim
        )
        timescale = self.max_wavelength**freq_exponents

        broadcast_shape = [1] * x.ndim
        broadcast_shape[axis] = axis_dim // 2

        # Compute position angles using offset
        positions = mx.arange(x.shape[1])[None, :] + offset[:, None]
        positions_f = positions.astype(mx.float32)
        radians = positions_f.reshape(
            positions_f.shape + (1,) * channel_ndim
        ) / timescale.reshape(broadcast_shape)
        sin_r = mx.sin(radians)
        cos_r = mx.cos(radians)

        splits = mx.split(x, 2, axis=axis)
        x1, x2 = splits[0], splits[1]
        result = mx.concatenate(
            [x1 * cos_r - x2 * sin_r, x2 * cos_r + x1 * sin_r],
            axis=axis,
        )
        return result.astype(x.dtype)

    original_axes = list(range(x.ndim))
    if x.ndim >= 3:
        transpose_axes = original_axes.copy()
        transpose_axes.pop(1)
        transpose_axes.insert(-1, 1)
        x_t = mx.transpose(x, transpose_axes)
    else:
        x_t = x
        
    y_t = mx.fast.rope(x_t, dims=x.shape[-1], traditional=False, base=self.max_wavelength, scale=1.0, offset=offset)
    
    if x.ndim >= 3:
        inv_axes = original_axes.copy()
        inv_axes.pop(-2)
        inv_axes.insert(1, x.ndim - 2)
        y = mx.transpose(y_t, inv_axes)
    else:
        y = y_t
    return y.astype(x.dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    if self.only_advance_position_for_valid_timesteps:
      return mx.full((batch_size, 1), -1, dtype=mx.int32)
    else:
      return mx.zeros((batch_size, 1), dtype=mx.int32)

  @types.check_step
  def step(self, x, state, *, constants=None):
    x_time = x.shape[1]
    
    if self.only_advance_position_for_valid_timesteps:
      # The state stores the last valid position. If initialized to -1, the next
      # valid position starts at 0.
      offset = mx.maximum(0, state[:, 0] + 1)
      
      # Update the state to hold the maximum position reached after this step.
      positions = state + mx.cumsum(x.mask.astype(mx.int32), axis=1)
      state = positions[:, -1:]
    else:
      offset = state[:, 0]
      state = state + x_time
      
    y = x.apply_values(self._apply_rope, offset)
    return y, state

  @types.check_layer
  def layer(self, x, *, constants=None):
    # In layer mode, processing starts from time step 0 for all batch elements.
    offset = mx.zeros((x.shape[0],), dtype=mx.int32)
    return x.apply_values(self._apply_rope, offset)

  @classmethod
  def from_config(cls, config):
    return cls(
        max_wavelength=config.max_wavelength,
        axis=config.axis,
        only_advance_position_for_valid_timesteps=(
            config.only_advance_position_for_valid_timesteps
        ),
    )
