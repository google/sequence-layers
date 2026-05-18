"""Combinators (Serial, Residual, Repeat, Parallel) for MLX."""

import dataclasses
import enum
from functools import reduce
from math import lcm

import mlx.core as mx

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import simple as simple_lib
from sequence_layers.mlx import types
from sequence_layers.jax.types import SequenceLayerConfig as _SequenceLayerConfig

Sequence = bt.Sequence


class CombinationMode(enum.IntEnum):
  """How parallel outputs are combined."""

  STACK = 1
  CONCAT = 2
  ADD = 3
  MEAN = 4
  PRODUCT = 5


def _broadcast_shapes(*shapes):
  """Numpy-style shape broadcasting."""
  if not shapes:
    return ()
  max_dims = max(len(s) for s in shapes)
  if max_dims == 0:
    return ()
  padded = [(1,) * (max_dims - len(s)) + tuple(s) for s in shapes]
  result = []
  for dims in zip(*padded):
    max_dim = max(dims)
    for d in dims:
      if d != 1 and d != max_dim:
        raise ValueError(f'Shapes not broadcastable: {shapes}')
    result.append(max_dim)
  return tuple(result)


def _combine_output_channel_shape(mode, *channel_shapes):
  """Compute the output channel shape for a combination mode."""
  max_dims = max(len(x) for x in channel_shapes)
  padded = tuple((1,) * (max_dims - len(x)) + tuple(x) for x in channel_shapes)

  if mode == CombinationMode.STACK:
    bcast = _broadcast_shapes(*padded)
    return (len(channel_shapes),) + bcast
  elif mode == CombinationMode.CONCAT:
    if max_dims == 0:
      # All scalar → treat as (1,) each.
      padded = tuple((1,) for _ in channel_shapes)
    prefixes = tuple(x[:-1] for x in padded)
    bcast_prefix = _broadcast_shapes(*prefixes)
    final_dim = sum(x[-1] for x in padded)
    return bcast_prefix + (final_dim,)
  else:  # ADD, MEAN, PRODUCT
    return _broadcast_shapes(*padded)


def _combine_sequences(mode, sequences):
  """Combine parallel output sequences."""
  values_list = [s.values for s in sequences]
  masks = [s.mask for s in sequences]
  mask = masks[0]
  for m in masks[1:]:
    mask = mask & m

  if mode == CombinationMode.STACK:
    values = mx.stack(values_list, axis=2)
  elif mode == CombinationMode.CONCAT:
    values = mx.concatenate(values_list, axis=-1)
  elif mode == CombinationMode.ADD:
    values = values_list[0]
    for v in values_list[1:]:
      values = values + v
  elif mode == CombinationMode.MEAN:
    values = values_list[0]
    for v in values_list[1:]:
      values = values + v
    values = values / len(values_list)
  elif mode == CombinationMode.PRODUCT:
    values = values_list[0]
    for v in values_list[1:]:
      values = values * v
  else:
    raise ValueError(f'Unknown combination mode: {mode}')

  return Sequence(values, mask)


class SerialCombinatorMixin:
  """Mixin for Serial logic.

  Provides serial processing (layer, step, initial state) for classes that
  define a ``layers`` attribute containing a sequence of SequenceLayers.
  """

  layers: list[types.SequenceLayer]

  @property
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  def block_size(self):
    return reduce(lcm, (l.block_size for l in self.layers), 1)

  @property
  def output_ratio(self):
    r = self.layers[0].output_ratio if self.layers else 1
    for l in self.layers[1:]:
      r = r * l.output_ratio
    return r

  @property
  def input_latency(self):
    latency = 0
    for l in self.layers:
      latency = l.get_accumulated_input_latency(latency)
    return latency

  @property
  def output_latency(self):
    return int(self.input_latency * self.output_ratio)

  def get_output_shape(self, input_shape, *, constants=None):
    shape = input_shape
    for l in self.layers:
      shape = l.get_output_shape(shape, constants=constants)
    return shape

  def get_output_dtype(self, input_dtype, *, constants=None):
    dtype = input_dtype
    for l in self.layers:
      dtype = l.get_output_dtype(dtype, constants=constants)
    return dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None, **kwargs):
    spec = input_spec
    states = []
    for l in self.layers:
      states.append(l.get_initial_state(batch_size, spec, constants=constants))
      spec = l.get_output_spec(spec, constants=constants)
    return tuple(states)

  def layer_with_emits(self, x, *, constants=None, **kwargs):
    emits = {}
    for i, l in enumerate(self.layers):
      x, e = l.layer_with_emits(x, constants=constants)
      emits[f'layer_{i}'] = e
    return x, emits

  def step_with_emits(self, x, state, *, constants=None, **kwargs):
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      x, s, e = l.step_with_emits(x, s, constants=constants)
      new_state.append(s)
      emits[f'layer_{i}'] = e
    return x, tuple(new_state), emits


class SerialModules(SerialCombinatorMixin, types.Emitting):
  """A Serial combinator that wraps pre-existing SequenceLayers.

  Unlike Serial (which owns its layers), SerialModules references
  pre-constructed modules parented elsewhere. This avoids duplication
  when a module graph shares sub-layers across different combinators.
  """

  def __init__(self, layers):
    super().__init__()
    self.layers = list(layers)


class Serial(types.Emitting):
  """Processes SequenceLayers serially."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    layers: tuple[_SequenceLayerConfig, ...] = ()
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))

    def make(self, backend='mlx') -> 'Serial':
      return Serial.from_config(self, backend=backend)

  def __init__(self, layers: list[types.SequenceLayer]):
    super().__init__()
    self.layers = list(layers)

  @property
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  def block_size(self):
    from functools import reduce
    from math import lcm

    return reduce(lcm, (l.block_size for l in self.layers), 1)

  @property
  def output_ratio(self):
    r = self.layers[0].output_ratio if self.layers else 1
    for l in self.layers[1:]:
      r = r * l.output_ratio
    return r

  @property
  def input_latency(self):
    latency = 0
    for l in self.layers:
      latency = l.get_accumulated_input_latency(latency)
    return latency

  def get_output_shape(self, input_shape, *, constants=None):
    shape = input_shape
    for l in self.layers:
      shape = l.get_output_shape(shape, constants=constants)
    return shape

  def get_output_dtype(self, input_dtype, *, constants=None):
    dtype = input_dtype
    for l in self.layers:
      dtype = l.get_output_dtype(dtype, constants=constants)
    return dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None, **kwargs):
    spec = input_spec
    states = []
    for l in self.layers:
      states.append(l.get_initial_state(batch_size, spec, constants=constants))
      spec = l.get_output_spec(spec, constants=constants)
    return tuple(states)

  def layer_with_emits(self, x, *, constants=None, **kwargs):
    emits = {}
    for i, l in enumerate(self.layers):
      x, e = l.layer_with_emits(x, constants=constants)
      emits[f'layer_{i}'] = e
    return x, emits

  def step_with_emits(self, x, state, *, constants=None, **kwargs):
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      x, s, e = l.step_with_emits(x, s, constants=constants)
      new_state.append(s)
      emits[f'layer_{i}'] = e
    return x, tuple(new_state), emits

  @classmethod
  def from_config(cls, config, backend='mlx'):
    layers = [c.make(backend=backend) for c in config.layers]
    return cls(layers)


class Residual(types.Emitting):
  """Residual wrapper: y = body(x) + shortcut(x)."""

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    layers: tuple[_SequenceLayerConfig, ...] = ()
    shortcut_layers: tuple[_SequenceLayerConfig, ...] | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))
      if self.shortcut_layers is not None:
        object.__setattr__(self, 'shortcut_layers', tuple(self.shortcut_layers))

    def make(self, backend='mlx') -> 'Residual':
      return Residual.from_config(self, backend=backend)

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      *,
      shortcut: types.SequenceLayer | None = None,
  ):
    super().__init__()
    self.body = Serial(layers)
    self.shortcut = shortcut if shortcut is not None else simple_lib.Identity()

  @property
  def supports_step(self):
    return self.body.supports_step and self.shortcut.supports_step

  @property
  def block_size(self):
    from math import lcm

    return lcm(self.body.block_size, self.shortcut.block_size)

  @property
  def output_ratio(self):
    return self.body.output_ratio

  @property
  def input_latency(self):
    return self.body.input_latency

  def get_output_shape(self, input_shape, *, constants=None):
    return self.body.get_output_shape(input_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.body.get_output_dtype(input_dtype, constants=constants)

  def get_initial_state(self, batch_size, input_spec, *, constants=None, **kwargs):
    body_state = self.body.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    shortcut_state = self.shortcut.get_initial_state(
        batch_size, input_spec, constants=constants
    )
    return (body_state, shortcut_state)

  def _residual_fn(self, y_body, y_shortcut):
    y_values = y_body.values + y_shortcut.values
    y_mask = y_body.mask & y_shortcut.mask
    return Sequence(y_values, y_mask)

  def layer_with_emits(self, x, *, constants=None, **kwargs):
    y_body, body_emits = self.body.layer_with_emits(x, constants=constants)
    y_shortcut, shortcut_emits = self.shortcut.layer_with_emits(
        x, constants=constants
    )
    y = self._residual_fn(y_body, y_shortcut)
    return y, (body_emits, shortcut_emits)

  def step_with_emits(self, x, state, *, constants=None, **kwargs):
    body_state, shortcut_state = state
    y_body, body_state, body_emits = self.body.step_with_emits(
        x, body_state, constants=constants
    )
    y_shortcut, shortcut_state, shortcut_emits = self.shortcut.step_with_emits(
        x, shortcut_state, constants=constants
    )
    y = self._residual_fn(y_body, y_shortcut)
    return (
        y,
        (body_state, shortcut_state),
        (body_emits, shortcut_emits),
    )

  @classmethod
  def from_config(cls, config, backend='mlx'):
    layers = [c.make(backend=backend) for c in config.layers]
    shortcut = None
    if hasattr(config, 'shortcut_layers') and config.shortcut_layers:
      shortcut_layers = [
          c.make(backend=backend) for c in config.shortcut_layers
      ]
      if len(shortcut_layers) == 1:
        shortcut = shortcut_layers[0]
      else:
        shortcut = Serial(shortcut_layers)
    return cls(layers, shortcut=shortcut)


class Repeat(types.Emitting):
  """Repeats a SequenceLayer N times.

  Unlike Linen/NNX which use scan/vmap to share stacked params,
  MLX Repeat creates N independent copies of the child layer.
  Each copy has its own parameters.
  """

  def __init__(
      self,
      layers: list[types.SequenceLayer],
  ):
    super().__init__()
    if not layers:
      raise ValueError('Repeat requires at least one layer.')
    self.layers = list(layers)
    self.num_repeats = len(layers)

  @property
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  def block_size(self):
    return self.layers[0].block_size

  @property
  def output_ratio(self):
    return self.layers[0].output_ratio

  @property
  def input_latency(self):
    latency = 0
    for l in self.layers:
      latency = l.get_accumulated_input_latency(latency)
    return latency

  def get_output_shape(self, input_shape, *, constants=None):
    return self.layers[0].get_output_shape(input_shape, constants=constants)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.layers[0].get_output_dtype(input_dtype, constants=constants)

  def get_initial_state(self, batch_size, input_spec, *, constants=None, **kwargs):
    states = []
    spec = input_spec
    for l in self.layers:
      states.append(l.get_initial_state(batch_size, spec, constants=constants))
      # All repeats have the same output spec.
    return tuple(states)

  def layer_with_emits(self, x, *, constants=None, **kwargs):
    emits = {}
    for i, l in enumerate(self.layers):
      x, e = l.layer_with_emits(x, constants=constants)
      emits[f'repeat_{i}'] = e
    return x, emits

  def step_with_emits(self, x, state, *, constants=None, **kwargs):
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      x, s, e = l.step_with_emits(x, s, constants=constants)
      new_state.append(s)
      emits[f'repeat_{i}'] = e
    return x, tuple(new_state), emits

  @classmethod
  def from_config(cls, config, backend='mlx'):
    layers = [
        config.layer.make(backend=backend) for _ in range(config.num_repeats)
    ]
    return cls(layers)


class Parallel(types.Emitting):
  """Runs N children on the same input and combines outputs.

  All children must have equal output_ratio and block_size.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(_SequenceLayerConfig):
    layers: tuple[_SequenceLayerConfig, ...] = ()
    combination: CombinationMode = CombinationMode.STACK
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))

    def make(self, backend='mlx') -> 'Parallel':
      return Parallel.from_config(self, backend=backend)

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      *,
      combination: CombinationMode = CombinationMode.STACK,
  ):
    super().__init__()
    if not layers:
      raise ValueError('Parallel requires at least one layer.')
    self.layers = list(layers)
    self.combination = combination

    # Validate constraints.
    ratios = {l.output_ratio for l in self.layers}
    if len(ratios) > 1:
      raise ValueError(
          f'All Parallel children must have equal output_ratio, got {ratios}.'
      )
    blocks = {l.block_size for l in self.layers}
    if len(blocks) > 1:
      raise ValueError(
          f'All Parallel children must have equal block_size, got {blocks}.'
      )

  @property
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  def block_size(self):
    return reduce(lcm, (l.block_size for l in self.layers), 1)

  @property
  def output_ratio(self):
    return self.layers[0].output_ratio

  @property
  def input_latency(self):
    return self.layers[0].input_latency

  def get_output_shape(self, input_shape, *, constants=None):
    shapes = tuple(
        l.get_output_shape(input_shape, constants=constants)
        for l in self.layers
    )
    return _combine_output_channel_shape(self.combination, *shapes)

  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.layers[0].get_output_dtype(input_dtype, constants=constants)

  def get_initial_state(self, batch_size, input_spec, *, constants=None, **kwargs):
    states = []
    for l in self.layers:
      states.append(
          l.get_initial_state(batch_size, input_spec, constants=constants)
      )
    return tuple(states)

  def layer_with_emits(self, x, *, constants=None, **kwargs):
    outputs = []
    emits = {}
    for i, l in enumerate(self.layers):
      y, e = l.layer_with_emits(x, constants=constants)
      outputs.append(y)
      emits[f'parallel_{i}'] = e
    combined = _combine_sequences(self.combination, outputs)
    return combined, emits

  def step_with_emits(self, x, state, *, constants=None, **kwargs):
    outputs = []
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      y, s, e = l.step_with_emits(x, s, constants=constants)
      outputs.append(y)
      new_state.append(s)
      emits[f'parallel_{i}'] = e
    combined = _combine_sequences(self.combination, outputs)
    return combined, tuple(new_state), emits

  @classmethod
  def from_config(cls, config, backend='mlx'):
    layers = [c.make(backend=backend) for c in config.layers]
    combination = CombinationMode(config.combination.value)
    return cls(layers, combination=combination)
