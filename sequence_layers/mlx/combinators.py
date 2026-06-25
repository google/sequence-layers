"""Combinators (Serial, Residual, Repeat, Parallel) for MLX."""

import dataclasses
from fractions import Fraction
from functools import reduce
from math import lcm
from typing import Any, Callable, override
from typing import Sequence as _Sequence

import mlx.core as mx

from sequence_layers.mlx import simple as simple_lib
from sequence_layers.mlx import types
from sequence_layers.mlx import utils as mlx_utils
from sequence_layers.specs import combinators as spec

from sequence_layers.mlx import types as bt

Sequence = bt.Sequence
CombinationMode = spec.CombinationMode


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
      if d not in (1, max_dim):
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
  if mode == CombinationMode.CONCAT:
    if max_dims == 0:
      # All scalar → treat as (1,) each.
      padded = tuple((1,) for _ in channel_shapes)
    prefixes = tuple(x[:-1] for x in padded)
    bcast_prefix = _broadcast_shapes(*prefixes)
    final_dim = sum(x[-1] for x in padded)
    return bcast_prefix + (final_dim,)
  # ADD, MEAN, PRODUCT
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


  @property
  def supports_step(self):
    """Returns whether all layers support step-wise execution."""
    return all(l.supports_step for l in self.layers)

  @property
  def block_size(self):
    """Returns the accumulated block size of the layers."""
    return reduce(lcm, (l.block_size for l in self.layers), 1)

  @property
  def output_ratio(self):
    """Returns the accumulated output ratio of the layers."""
    r = self.layers[0].output_ratio if self.layers else Fraction(1)
    for l in self.layers[1:]:
      r = r * l.output_ratio
    return r

  @property
  def input_latency(self):
    """Returns the accumulated input latency of the layers."""
    latency = 0
    for l in self.layers:
      latency = l.get_accumulated_input_latency(latency)
    return latency

  @property
  def output_latency(self):
    """Returns the accumulated output latency of the layers."""
    return int(self.input_latency * self.output_ratio)

  def get_output_shape(self, input_shape, *, constants=None):
    """Returns the output shape of the serial combination."""
    shape = input_shape
    for l in self.layers:
      shape = l.get_output_shape(shape, constants=constants)
    return shape

  def get_output_dtype(self, input_dtype, *, constants=None):
    """Returns the output dtype of the serial combination."""
    dtype = input_dtype
    for l in self.layers:
      dtype = l.get_output_dtype(dtype, constants=constants)
    return dtype

  def get_initial_state(
      self,
      batch_size,
      input_spec,
      *,
      training: bool = False,
      constants=None,
      **kwargs,
  ):
    """Returns the initial state for all layers in the serial combination."""
    curr_spec = input_spec
    states = []
    for l in self.layers:
      states.append(
          l.get_initial_state(
              batch_size,
              curr_spec,
              training=training,
              constants=constants,
              **kwargs,
          )
      )
      curr_spec = l.get_output_spec(curr_spec, constants=constants)
    return tuple(states)

  def layer_with_emits(
      self, x, *, training: bool = False, constants=None, **kwargs
  ):
    """Process layer-wise through all child layers, accumulating emits."""
    emits = {}
    for i, l in enumerate(self.layers):
      x, e = l.layer_with_emits(
          x, training=training, constants=constants, **kwargs
      )
      emits[f'layer_{i}'] = e
    return x, emits

  def step_with_emits(
      self, x, state, *, training: bool = False, constants=None, **kwargs
  ):
    """Process step-wise through all child layers, accumulating emits."""
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      x, s, e = l.step_with_emits(
          x, s, training=training, constants=constants, **kwargs
      )
      new_state.append(s)
      emits[f'layer_{i}'] = e
    return x, tuple(new_state), emits


class SerialModules(
    SerialCombinatorMixin,
    types.Emitting,
    spec.SerialModules[types.Sequence, types.ShapeDType],
):
  """A Serial combinator that wraps pre-existing SequenceLayers.

  Unlike Serial (which owns its layers), SerialModules references
  pre-constructed modules parented elsewhere. This avoids duplication
  when a module graph shares sub-layers across different combinators.
  """

  def __init__(self, layers: _Sequence[types.SequenceLayer]):
    super().__init__()
    self.layers = list(layers)


class Serial(
    SerialCombinatorMixin,
    types.Emitting,
    spec.Serial[types.Sequence, types.ShapeDType],
):
  """Processes SequenceLayers serially."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Serial.Config):
    """Configuration for Serial."""

    layers: _Sequence[types.SequenceLayerConfig] = ()
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))

    @override
    def make(self) -> 'Serial':
      return Serial.from_config(self)

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      names: list[str | None] | None = None,
  ):
    super().__init__()
    self.config = None
    self._layer_names = []
    for i, l in enumerate(layers):
      name = f'layers_{i}'
      if names is not None:
        name_opt = names[i]
        if isinstance(name_opt, str):
          name = name_opt
      self._layer_names.append(name)
    self.layers = layers



  @classmethod
  def from_config(cls, config, backend='mlx'):
    """Creates a Serial layer from a configuration object."""
    layers = [mlx_utils.make_layer(c, backend=backend) for c in config.layers]
    names = [getattr(c, 'name', None) for c in config.layers]
    instance = cls(layers, names=names)
    instance.config = config
    return instance


class Residual(types.Emitting, spec.Residual[types.Sequence, types.ShapeDType]):
  """Residual wrapper: y = body(x) + shortcut(x)."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Residual.Config):
    """Configuration for Residual."""

    layers: _Sequence[types.SequenceLayerConfig] = ()
    shortcut_layers: _Sequence[types.SequenceLayerConfig] | None = None
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))
      if self.shortcut_layers is not None:
        object.__setattr__(self, 'shortcut_layers', tuple(self.shortcut_layers))

    @override
    def make(self) -> 'Residual':
      return Residual.from_config(self)

  def __init__(
      self,
      layers: list[types.SequenceLayer],
      *,
      names: list[str | None] | None = None,
      shortcut: types.SequenceLayer | None = None,
  ):
    super().__init__()
    self.config = None
    self.body = Serial(layers, names=names)
    self.shortcut = (
        shortcut
        if shortcut is not None
        else simple_lib.Identity(simple_lib.Identity.Config())
    )

  @property
  @override
  def supports_step(self):
    return self.body.supports_step and self.shortcut.supports_step

  @property
  @override
  def block_size(self):
    return lcm(self.body.block_size, self.shortcut.block_size)

  @property
  @override
  def output_ratio(self):
    return self.body.output_ratio

  @property
  @override
  def input_latency(self):
    return self.body.input_latency

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return self.body.get_output_shape(input_shape, constants=constants)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.body.get_output_dtype(input_dtype, constants=constants)

  @override
  def get_initial_state(
      self,
      batch_size,
      input_spec,
      *,
      training: bool = False,
      constants=None,
      **kwargs,
  ):
    body_state = self.body.get_initial_state(
        batch_size,
        input_spec,
        training=training,
        constants=constants,
        **kwargs,
    )
    shortcut_state = self.shortcut.get_initial_state(
        batch_size,
        input_spec,
        training=training,
        constants=constants,
        **kwargs,
    )
    return (body_state, shortcut_state)

  def _residual_fn(self, y_body, y_shortcut):
    """Combines output of body and shortcut layers residuals."""
    y_values = y_body.values + y_shortcut.values
    y_mask = y_body.mask & y_shortcut.mask
    return Sequence(y_values, y_mask)

  @override
  def layer_with_emits(
      self, x, *, training: bool = False, constants=None, **kwargs
  ):
    y_body, body_emits = self.body.layer_with_emits(
        x, training=training, constants=constants, **kwargs
    )
    y_shortcut, shortcut_emits = self.shortcut.layer_with_emits(
        x, training=training, constants=constants, **kwargs
    )
    y = self._residual_fn(y_body, y_shortcut)
    return y, (body_emits, shortcut_emits)

  @override
  def step_with_emits(
      self, x, state: Any, *, training: bool = False, constants=None, **kwargs
  ):
    body_state, shortcut_state = state
    y_body, body_state, body_emits = self.body.step_with_emits(
        x,
        body_state,
        training=training,
        constants=constants,
        **kwargs,
    )
    y_shortcut, shortcut_state, shortcut_emits = self.shortcut.step_with_emits(
        x,
        shortcut_state,
        training=training,
        constants=constants,
        **kwargs,
    )
    y = self._residual_fn(y_body, y_shortcut)
    return (
        y,
        (body_state, shortcut_state),
        (body_emits, shortcut_emits),
    )

  @classmethod
  def from_config(cls, config, backend='mlx'):
    """Creates a Residual layer from a configuration object."""
    layers = [mlx_utils.make_layer(c, backend=backend) for c in config.layers]
    names = [getattr(c, 'name', None) for c in config.layers]
    shortcut = None
    if hasattr(config, 'shortcut_layers') and config.shortcut_layers:
      shortcut_layers = [
          mlx_utils.make_layer(c, backend=backend)
          for c in config.shortcut_layers
      ]
      shortcut_names = [
          getattr(c, 'name', None) for c in config.shortcut_layers
      ]
      if len(shortcut_layers) == 1:
        shortcut = shortcut_layers[0]
      else:
        shortcut = Serial(shortcut_layers, names=shortcut_names)
    instance = cls(layers, names=names, shortcut=shortcut)
    instance.config = config
    return instance


class Repeat(types.Emitting, spec.Repeat[types.Sequence, types.ShapeDType]):
  """Repeats a SequenceLayer N times.

  Unlike Linen/NNX which use scan/vmap to share stacked params,
  MLX Repeat creates N independent copies of the child layer.
  Each copy has its own parameters.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Repeat.Config):
    """Configuration for Repeat."""

    layer: types.SequenceLayerConfig
    num_repeats: int
    remat: bool = False
    prevent_cse: bool = False
    policy: Callable[..., bool] | None = None
    unroll_layer: bool = False
    unroll_step: bool = False
    name: str | None = None

    @override
    def make(self) -> 'Repeat':
      return Repeat.from_config(self)

  def __init__(
      self,
      layers: _Sequence[types.SequenceLayer],
  ):
    super().__init__()
    self.config = None
    if not layers:
      raise ValueError('Repeat requires at least one layer.')
    self.layers = list(layers)
    self.num_repeats = len(layers)

  @property
  @override
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  @override
  def block_size(self):
    return self.layers[0].block_size

  @property
  @override
  def output_ratio(self):
    return self.layers[0].output_ratio

  @property
  @override
  def input_latency(self):
    latency = 0
    for l in self.layers:
      latency = l.get_accumulated_input_latency(latency)
    return latency

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    return self.layers[0].get_output_shape(input_shape, constants=constants)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.layers[0].get_output_dtype(input_dtype, constants=constants)

  @override
  def get_initial_state(
      self,
      batch_size,
      input_spec,
      *,
      training: bool = False,
      constants=None,
      **kwargs,
  ):
    states = []
    curr_spec = input_spec
    for l in self.layers:
      states.append(
          l.get_initial_state(
              batch_size,
              curr_spec,
              training=training,
              constants=constants,
              **kwargs,
          )
      )
    return tuple(states)

  @override
  def layer_with_emits(
      self, x, *, training: bool = False, constants=None, **kwargs
  ):
    emits = {}
    for i, l in enumerate(self.layers):
      x, e = l.layer_with_emits(
          x, training=training, constants=constants, **kwargs
      )
      emits[f'repeat_{i}'] = e
    return x, emits

  @override
  def step_with_emits(
      self, x, state: Any, *, training: bool = False, constants=None, **kwargs
  ):
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      x, s, e = l.step_with_emits(
          x, s, training=training, constants=constants, **kwargs
      )
      new_state.append(s)
      emits[f'repeat_{i}'] = e
    return x, tuple(new_state), emits

  @classmethod
  def from_config(cls, config, backend='mlx'):
    """Creates a Repeat layer from a configuration object."""
    layers = [
        mlx_utils.make_layer(config.layer, backend=backend)
        for _ in range(config.num_repeats)
    ]
    instance = cls(layers)
    instance.config = config
    return instance


class Parallel(types.Emitting, spec.Parallel[types.Sequence, types.ShapeDType]):
  """Runs N children on the same input and combines outputs.

  All children must have equal output_ratio and block_size.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig, spec.Parallel.Config):
    """Configuration for Parallel."""

    layers: _Sequence[types.SequenceLayerConfig]
    combination: CombinationMode = CombinationMode.STACK
    share_scope: bool | _Sequence[bool] = False
    name: str | None = None

    def __post_init__(self):
      object.__setattr__(self, 'layers', tuple(self.layers))

    @override
    def make(self) -> 'Parallel':
      return Parallel.from_config(self)

  def __init__(
      self,
      layers: _Sequence[types.SequenceLayer],
      *,
      combination: CombinationMode = CombinationMode.STACK,
  ):
    super().__init__()
    self.config = None
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
  @override
  def supports_step(self):
    return all(l.supports_step for l in self.layers)

  @property
  @override
  def block_size(self):
    return reduce(lcm, (l.block_size for l in self.layers), 1)

  @property
  @override
  def output_ratio(self):
    return self.layers[0].output_ratio

  @property
  @override
  def input_latency(self):
    return self.layers[0].input_latency

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    shapes = tuple(
        l.get_output_shape(input_shape, constants=constants)
        for l in self.layers
    )
    return _combine_output_channel_shape(self.combination, *shapes)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    return self.layers[0].get_output_dtype(input_dtype, constants=constants)

  @override
  def get_initial_state(
      self,
      batch_size,
      input_spec,
      *,
      training: bool = False,
      constants=None,
      **kwargs,
  ):
    states = []
    for l in self.layers:
      states.append(
          l.get_initial_state(
              batch_size,
              input_spec,
              training=training,
              constants=constants,
              **kwargs,
          )
      )
    return tuple(states)

  @override
  def layer_with_emits(
      self, x, *, training: bool = False, constants=None, **kwargs
  ):
    outputs = []
    emits = {}
    for i, l in enumerate(self.layers):
      y, e = l.layer_with_emits(
          x, training=training, constants=constants, **kwargs
      )
      outputs.append(y)
      emits[f'parallel_{i}'] = e
    combined = _combine_sequences(self.combination, outputs)
    return combined, emits

  @override
  def step_with_emits(
      self, x, state: Any, *, training: bool = False, constants=None, **kwargs
  ):
    outputs = []
    new_state = []
    emits = {}
    for i, (l, s) in enumerate(zip(self.layers, state)):
      y, s, e = l.step_with_emits(
          x, s, training=training, constants=constants, **kwargs
      )
      outputs.append(y)
      new_state.append(s)
      emits[f'parallel_{i}'] = e
    combined = _combine_sequences(self.combination, outputs)
    return combined, tuple(new_state), emits

  @classmethod
  def from_config(cls, config, backend='mlx'):
    """Creates a Parallel layer from a configuration object."""
    layers = [mlx_utils.make_layer(c, backend=backend) for c in config.layers]
    combination = CombinationMode(config.combination.value)
    instance = cls(layers, combination=combination)
    instance.config = config
    return instance
