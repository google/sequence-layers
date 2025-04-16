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
"""Combinators."""

import dataclasses
import fractions
import functools
from typing import Any, Callable, Sequence as TypingSequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import simple
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.learning.gemini.gemax.core.models import meta


CombinationMode = utils.CombinationMode

__all__ = (
    # go/keep-sorted start
    'Bidirectional',
    'Blockwise',
    'CheckpointGradient',
    'CombinationMode',
    'Parallel',
    'ParallelChannels',
    'Repeat',
    'Residual',
    'Serial',
    'SerialCombinatorMixin',
    'SerialModules',
    'WrapperMixin',
    # go/keep-sorted end
)


SequenceLayerConfigOrList = (
    types.SequenceLayerConfig | TypingSequence[types.SequenceLayerConfig]
)


def _wrap_layers(layers: SequenceLayerConfigOrList) -> types.SequenceLayer:
  if isinstance(layers, types.SequenceLayerConfig):
    return layers.make()
  else:
    return Serial.Config(layers=tuple(layers)).make()


class WrapperMixin:
  """A wrapper mixin where the wrapped layer properties are unchanged.

  Used for layers that modify the wrapped layer's behavior in a way that does
  not change any of the layer properties or utility functions.
  """

  child_layer: types.SequenceLayer

  @property
  def supports_step(self) -> bool:
    return self.child_layer.supports_step

  @property
  def input_latency(self) -> int:
    return self.child_layer.input_latency

  @property
  def output_latency(self) -> fractions.Fraction:
    return self.child_layer.output_latency

  @property
  def block_size(self) -> int:
    return self.child_layer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self.child_layer.output_ratio

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    return self.child_layer.get_output_dtype(input_dtype)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    return self.child_layer.get_output_shape(input_shape, constants=constants)

  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    return self.child_layer.layer(x, training=training, constants=constants)

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    return self.child_layer.layer_with_emits(
        x, training=training, constants=constants
    )

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    return self.child_layer.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )

  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    return self.child_layer.step(
        x, state, training=training, constants=constants
    )

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    return self.child_layer.step_with_emits(
        x, state, training=training, constants=constants
    )


class SerialCombinatorMixin:
  """Mixin for Serial logic."""

  layers: tuple[types.SequenceLayer, ...]
  # Provided by nn.Module.path.
  path: tuple[str, ...]

  @property
  def supports_step(self) -> bool:
    return all(child_layer.supports_step for child_layer in self.layers)

  @property
  def input_latency(self) -> int:
    """Latency of a serial is a sum of the child latencies."""
    return sum(child_layer.input_latency for child_layer in self.layers)

  @property
  def output_latency(self) -> fractions.Fraction:
    """Accumulates output latencies for layers, accounting for output ratios."""
    output_latency = fractions.Fraction(0)
    for child_layer in self.layers:
      output_latency = (
          output_latency * child_layer.output_ratio + child_layer.output_latency
      )
    return output_latency

  @property
  def block_size(self) -> int:
    """The block size of the layer."""
    block_size = fractions.Fraction(1)
    output_ratio = fractions.Fraction(1)

    for child_layer in self.layers:
      layer_output_ratio = child_layer.output_ratio
      layer_block_size = child_layer.block_size
      block_size = (
          np.lcm(block_size * output_ratio, layer_block_size) / output_ratio
      )
      output_ratio *= layer_output_ratio

    assert block_size.denominator == 1
    return block_size.numerator

  @property
  def output_ratio(self) -> fractions.Fraction:
    output_ratio = fractions.Fraction(1)
    for child_layer in self.layers:
      output_ratio *= child_layer.output_ratio
    return output_ratio

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    """Returns initial state for the layer."""
    spec = input_spec
    states = []
    for child_layer in self.layers:
      states.append(
          child_layer.get_initial_state(
              batch_size, spec, training=training, constants=constants
          )
      )
      spec = child_layer.get_output_spec(spec, constants=constants)
    return tuple(states)

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    dtype = input_dtype
    for child_layer in self.layers:
      dtype = child_layer.get_output_dtype(dtype)
    return dtype

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    shape = tuple(input_shape)
    for child_layer in self.layers:
      shape = child_layer.get_output_shape(shape, constants=constants)
    return shape

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    """Applies the layer to a block of inputs."""
    new_state = []
    emits = {}
    if len(self.layers) != len(state):
      raise ValueError(
          f'{type(self)=} {self.path=} received unexpected state structure:'
          f' {len(self.layers)=} != {len(state)=}:'
          f' {state=} {[type(l) for l in self.layers]=}'
      )

    for child_layer, state_i in zip(self.layers, state):
      x, state_i, emits_i = child_layer.step_with_emits(
          x, state_i, training=training, constants=constants
      )
      new_state.append(state_i)
      utils.insert_with_unique_key(emits, child_layer.name, emits_i)
    return x, tuple(new_state), emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    """Applies the layer to the input sequence."""
    emits = {}
    for child_layer in self.layers:
      x, emits_i = child_layer.layer_with_emits(
          x, training=training, constants=constants
      )
      utils.insert_with_unique_key(emits, child_layer.name, emits_i)
    return x, emits


class Serial(SerialCombinatorMixin, types.Emitting):
  """A combinator that processes SequenceLayers serially."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Serial."""

    layers: TypingSequence[types.SequenceLayerConfig]
    # If true, a list of boolean values for each layer in `layers` indicating
    # whether to share this Serial's Flax parameter scope with that layer. This
    # is useful to avoid representing the Serial layer in the parameter tree. If
    # child layers have conflicting parameter names, a flax.errors.NameInUsError
    # is thrown. No parameter sharing will occur.
    share_scope: bool | TypingSequence[bool] = False
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'layers', tuple(self.layers))

    def make(self) -> 'Serial':
      return Serial(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.layers = [l.make() for l in self.config.layers]

    if self.config.share_scope:
      utils.setup_shared_scope(self, self.layers, self.config.share_scope)


class SerialModules(SerialCombinatorMixin, types.Emitting):
  """A Serial combinator that processes pre-existing SequenceLayers serially.

  Passing pre-constructed modules into another nn.Module can have unintended
  effects with respect to variable trees. The intended use of SerialModules is
  to create a Serial from pre-constructed SequenceLayers when those modules are
  parented to another nn.Module. For example:

  class MyModule(nn.Module):
    class Config:
      def make(self) -> 'MyModule':
        return MyModule(self)

    config: Config

    def setup(self) -> None:
      self.encoder = self.config.encoder.make()
      self.decoder = self.config.decoder.make()

      # Sampler is a SequenceLayer that runs encoder and decoder.get_sampler()
      # but does not own encoder or decoder.
      self.sampler = sl.SerialModules(
        [self.encoder, self.decoder.get_sampler()])

    def get_sampler(self) -> sl.SequenceLayer:
      return self.sampler
  """

  layers: tuple[types.SequenceLayer, ...]


class Parallel(types.Emitting):
  """Applies a sequence of layers in parallel.

  Outputs are broadcasted and combined together.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Parallel."""

    layers: TypingSequence[types.SequenceLayerConfig]
    combination: utils.CombinationMode = utils.CombinationMode.STACK
    # If true, a list of boolean values for each layer in `layers` indicating
    # whether to share this Serial's Flax parameter scope with that layer. This
    # is useful to avoid representing the Serial layer in the parameter tree. If
    # child layers have conflicting parameter names, a flax.errors.NameInUsError
    # is thrown. No parameter sharing will occur.
    share_scope: bool | TypingSequence[bool] = False
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'layers', tuple(self.layers))

    def make(self) -> 'Parallel':
      return Parallel(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.layers = tuple(l.make() for l in self.config.layers)

    if self.config.share_scope:
      utils.setup_shared_scope(self, self.layers, self.config.share_scope)

    if not self.layers:
      self._output_ratio = fractions.Fraction(1)
    else:
      self._output_ratio = self.layers[0].output_ratio
      for child_layer in self.layers:
        if child_layer.output_ratio != self._output_ratio:
          raise ValueError(
              'Output ratios must be equal for all layers:'
              f' {self._output_ratio} != {child_layer.output_ratio} for'
              f' {child_layer}'
          )

  @property
  def supports_step(self) -> bool:
    return all(child_layer.supports_step for child_layer in self.layers)

  @property
  def input_latency(self) -> int:
    """Latency of a parallel is the maximum of the child latencies."""
    if not self.layers:
      return 0
    return max(child_layer.input_latency for child_layer in self.layers)

  @property
  def block_size(self) -> int:
    block_size = 1
    for child_layer in self.layers:
      block_size = np.lcm(block_size, child_layer.block_size)
    return int(block_size)

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._output_ratio

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    """Returns initial state for the layer."""
    states = []
    for child_layer in self.layers:
      states.append(
          child_layer.get_initial_state(
              batch_size, input_spec, training=training, constants=constants
          )
      )
    return tuple(states)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if not self.layers:
      return tuple(input_shape)

    output_shapes = [
        child_layer.get_output_shape(input_shape, constants=constants)
        for child_layer in self.layers
    ]
    return utils.sequence_broadcast_combine_output_channel_shape(
        self.config.combination, *output_shapes
    )

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    if not self.layers:
      return input_dtype

    dtype = self.layers[0].get_output_dtype(input_dtype)
    for child_layer in self.layers[1:]:
      # TODO(rryan): We should probably disallow automatic type promotion
      # between integer and float types, since this is usually a sign of a bug.
      dtype = jnp.result_type(dtype, child_layer.get_output_dtype(input_dtype))
    return dtype

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    new_state = []
    emits = {}
    if len(self.layers) != len(state):
      raise ValueError(
          f'{type(self)=} {self.path=} received unexpected state structure:'
          f' {len(self.layers)=} != {len(state)=}:'
          f' {state=} {[type(l) for l in self.layers]=}'
      )
    if not self.layers:
      return x, state, emits

    ys = []
    for child_layer, layer_state in zip(self.layers, state):
      y, layer_state, layer_emits = child_layer.step_with_emits(
          x, layer_state, training=training, constants=constants
      )
      ys.append(y)
      new_state.append(layer_state)
      utils.insert_with_unique_key(emits, child_layer.name, layer_emits)
    y = utils.sequence_broadcast_combine(self.config.combination, *ys)
    return y, tuple(new_state), emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    emits = {}

    if not self.layers:
      return x, emits

    ys = []
    for child_layer in self.layers:
      y, layer_emits = child_layer.layer_with_emits(
          x, training=training, constants=constants
      )
      ys.append(y)
      utils.insert_with_unique_key(emits, child_layer.name, layer_emits)
    y = utils.sequence_broadcast_combine(self.config.combination, *ys)
    return y, emits


class ParallelChannels(WrapperMixin, types.SequenceLayer):
  """Applies a layer with shared parameters to groups of input channels.

  The input sequence is split on its final channels dimension into num_groups
  separate sequences and processed with the child layer. Unlike sl.Parallel,
  parameters for the child layer are shared across all parallel invocations.

  This is like a grouped convolution, but works with an arbitrary SequenceLayer.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    child_layer: types.SequenceLayerConfig
    num_groups: int
    combination: utils.CombinationMode = utils.CombinationMode.STACK
    name: str | None = None

    def make(self) -> types.SequenceLayer:
      return ParallelChannels(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.child_layer = self.config.child_layer.make()
    nn.share_scope(self, self.child_layer)

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    input_shape = list(input_shape)
    if not input_shape:
      raise ValueError(f'Input must be at least 3D, got: {input_shape=}.')
    if input_shape[-1] % self.config.num_groups != 0:
      raise ValueError(
          f'Input channels ({input_shape[-1]}) must be divisible by'
          f' num_groups ({self.config.num_groups}).'
      )
    input_shape[-1] //= self.config.num_groups
    result_shape = self.child_layer.get_output_shape(
        input_shape, constants=constants
    )
    output_shapes = (result_shape,) * self.config.num_groups
    return utils.sequence_broadcast_combine_output_channel_shape(
        self.config.combination, *output_shapes
    )

  def _validate_inputs(self, x: types.Sequence) -> None:
    if x.ndim == 2:
      raise ValueError(f'Input must be at least 3D, got: {x.shape=}.')
    if x.shape[-1] % self.config.num_groups != 0:
      raise ValueError(
          f'Input channels ({x.shape[-1]}) must be divisible by'
          f' num_groups ({self.config.num_groups}).'
      )

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    self._validate_inputs(x)

    ys = [
        self.child_layer.layer(x_i, training=training, constants=constants)
        for x_i in utils.sequence_split(x, self.config.num_groups, axis=-1)
    ]
    return utils.sequence_broadcast_combine(self.config.combination, *ys)

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    self._validate_inputs(x)
    ys, emits = [], []
    for x_i in utils.sequence_split(x, self.config.num_groups, axis=-1):
      y_i, emits_i = self.child_layer.layer_with_emits(
          x_i, training=training, constants=constants
      )
      ys.append(y_i)
      emits.append(emits_i)
    y = utils.sequence_broadcast_combine(self.config.combination, *ys)
    return y, tuple(emits)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not input_spec.shape:
      raise ValueError(f'Input must be at least 3D, got: {input_spec.shape=}.')
    if input_spec.shape[-1] % self.config.num_groups != 0:
      raise ValueError(
          f'Input channels ({input_spec.shape[-1]}) must be divisible by'
          f' num_groups ({self.config.num_groups}).'
      )
    input_shape = list(input_spec.shape)
    input_shape[-1] //= self.config.num_groups
    input_spec = types.ChannelSpec(
        shape=tuple(input_shape),
        dtype=input_spec.dtype,
    )
    state = self.child_layer.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    return (state,) * self.config.num_groups

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    self._validate_inputs(x)

    xs = utils.sequence_split(x, self.config.num_groups, axis=-1)
    ys = []
    states = []
    for x_i, state_i in zip(xs, state, strict=True):
      y_i, state_i = self.child_layer.step(
          x_i, state_i, training=training, constants=constants
      )
      ys.append(y_i)
      states.append(state_i)
    y = utils.sequence_broadcast_combine(self.config.combination, *ys)
    return y, tuple(states)

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    self._validate_inputs(x)

    xs = utils.sequence_split(x, self.config.num_groups, axis=-1)
    ys = []
    states = []
    emits = []
    for x_i, state_i in zip(xs, state, strict=True):
      y_i, state_i, emits_i = self.child_layer.step_with_emits(
          x_i, state_i, training=training, constants=constants
      )
      ys.append(y_i)
      states.append(state_i)
      emits.append(emits_i)
    y = utils.sequence_broadcast_combine(self.config.combination, *ys)
    return y, tuple(states), tuple(emits)


class Residual(SerialCombinatorMixin, types.Emitting):
  """A residual wrapper around l that computes `y = l(x) + shortcut(x)`.

  If shortcut is not provided, it defaults to an identity or a linear projection
  to match the output shape of l(x).

  Subclasses may override the combination logic via `residual_function`.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Residual."""

    layers: TypingSequence[types.SequenceLayerConfig]
    shortcut_layers: TypingSequence[types.SequenceLayerConfig] | None = None
    # If true, a list of boolean values for each layer in `layers` indicating
    # whether to share this Residual's Flax parameter scope with that layer.
    # This is useful to avoid representing the Residual layer in the parameter
    # tree. If child layers have conflicting parameter names, a
    # flax.errors.NameInUsError is thrown. No parameter sharing will occur.
    share_scope: bool | TypingSequence[bool] = False
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      object.__setattr__(self, 'layers', tuple(self.layers))
      if self.shortcut_layers is not None:
        object.__setattr__(self, 'shortcut_layers', tuple(self.shortcut_layers))

    def make(self) -> 'Residual':
      return Residual(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.layers = tuple(l.make() for l in self.config.layers)
    if self.config.share_scope:
      utils.setup_shared_scope(self, self.layers, self.config.share_scope)

    if self.config.shortcut_layers is None:
      self.shortcut_layer = simple.Identity()
    else:
      self.shortcut_layer = _wrap_layers(self.config.shortcut_layers)

    if self.shortcut_layer.output_ratio != super().output_ratio:
      raise ValueError(
          'Residual layers and shortcut_layers must have the same output '
          f'ratio {super().output_ratio} != '
          f'{self.shortcut_layer.output_ratio}.'
      )
    if self.shortcut_layer.input_latency != super().input_latency:
      raise ValueError(
          f'{self.name}: Residual layers and shortcut_layers must have the same'
          f' input latency {super().input_latency} !='
          f' {self.shortcut_layer.input_latency}. ({self.layers=},'
          f' {self.shortcut_layer=})'
      )

  @property
  def supports_step(self) -> bool:
    return super().supports_step and self.shortcut_layer.supports_step

  @property
  def block_size(self) -> int:
    if self.shortcut_layer:
      return int(np.lcm(super().block_size, self.shortcut_layer.block_size))
    return super().block_size

  @nn.nowrap
  def _validate(
      self,
      x_spec: types.ShapeDType,
      constants: types.Constants,
  ):
    layer_shape = super().get_output_shape(x_spec.shape, constants=constants)
    shortcut_shape = self.shortcut_layer.get_output_shape(
        x_spec.shape, constants=constants
    )
    # TODO(rryan): Support broadcastable shapes.
    if layer_shape != shortcut_shape:
      raise ValueError(
          f'{self.name}: Residual connection must have same output shape: '
          f'layer: {layer_shape} shortcut: {shortcut_shape}'
      )

    layer_dtype = super().get_output_dtype(x_spec.dtype)
    shortcut_dtype = self.shortcut_layer.get_output_dtype(x_spec.dtype)
    # Make sure we can compute a result dtype.
    jnp.result_type(layer_dtype, shortcut_dtype)

  def residual_function(
      self, y_body: types.Sequence, y_shortcut: types.Sequence
  ) -> types.Sequence:
    """Combines the body and shortcut layer outputs."""
    y_values = y_body.values + y_shortcut.values
    y_mask = utils.combine_mask(y_body.mask, y_shortcut.mask)

    # Assume unmasked since masking is only preserved when both sequences are
    # masked and the body and shortcut masks are equal.
    return types.Sequence(y_values, y_mask)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    self._validate(x.channel_spec, constants)
    body_state, shortcut_state = state
    y_body, body_state, body_emits = super().step_with_emits(
        x, body_state, training=training, constants=constants
    )
    y_shortcut, shortcut_state, shortcut_emits = (
        self.shortcut_layer.step_with_emits(
            x, shortcut_state, training=training, constants=constants
        )
    )
    y = self.residual_function(y_body, y_shortcut)

    state = (body_state, shortcut_state)
    emits = (body_emits, shortcut_emits)
    return y, state, emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    self._validate(x.channel_spec, constants)
    y_body, body_emits = super().layer_with_emits(
        x, training=training, constants=constants
    )
    y_shortcut, shortcut_emits = self.shortcut_layer.layer_with_emits(
        x,
        training=training,
        constants=constants,
    )
    y = self.residual_function(y_body, y_shortcut)
    emits = (body_emits, shortcut_emits)
    return y, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    body_state = super().get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    shortcut_state = self.shortcut_layer.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    return body_state, shortcut_state

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    layer_dtype = super().get_output_dtype(input_dtype)
    shortcut_dtype = self.shortcut_layer.get_output_dtype(input_dtype)
    # TODO(rryan): We should probably disallow automatic type promotion between
    # integer and float types, since this is usually a sign of a bug.
    return jnp.result_type(layer_dtype, shortcut_dtype)


class Repeat(types.Emitting):
  """A combinator that repeats the specified SequenceLayer N times.

  Execution is performed in a loop, enabling reduced compilation times since the
  layer logic only has to be compiled once.

  Variables and layer state are represented in a stacked fashion, with a
  `num_repeats` leading dimension.

  Requires that the layer being repeated has an `output_ratio` of 1, and that
  the output dtype and shape are equal to the input dtype and shape, since the
  input and output to jax.nn.scan must be equal.

  TODO(rryan): It's possible to support `output_ratios` with padding of the
  inputs or outputs.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    layer: types.SequenceLayerConfig
    num_repeats: int
    remat: bool = False
    # Whether to unroll the repeat layer when running layer-wise.
    unroll_layer: bool = False
    # Whether to unroll the repeat layer when running step-wise.
    unroll_step: bool = False
    name: str | None = None

    def make(self) -> 'Repeat':
      return Repeat(self, name=self.name)

  config: Config

  def setup(self):
    if not self.config.num_repeats:
      raise ValueError(f'Expected {self.config.num_repeats=} > 0.')
    self.child_layer = self.config.layer.make()

  def _get_child_property(
      self, property_fn: Callable[[types.SequenceLayer], Any]
  ) -> list[Any]:
    results = []

    def scan_fn(
        child_layer: types.SequenceLayer,
        scan_carry,
        scan_input,
    ):
      del scan_carry, scan_input
      results.append(property_fn(child_layer))
      return (), ()

    repeat = nn.scan(
        scan_fn,
        variable_axes={True: 0},  # Slice all variables on axis 0.
        variable_broadcast=False,
        in_axes=0,  # No input.
        out_axes=0,  # No output.
        split_rngs={
            'params': self.is_initializing(),
            nn.DenyList(('params',)): True,
        },
        length=self.config.num_repeats,
        metadata_params={
            # For params we replicate along this scan-over-layers axis.
            meta.MESH_AXIS: None,
            # For optimizers mark this axis in params as 'independent'.
            meta.AXIS_TYPE: meta.AxisType.STACKED,
        },
    )

    # No inputs or outputs.
    repeat(self.child_layer, (), ())

    return results

  @property
  def supports_step(self) -> bool:
    return all(self._get_child_property(lambda l: l.supports_step))

  @property
  def input_latency(self) -> int:
    """Latency increases by child_layer.input_latency for each repetition."""
    return self.child_layer.input_latency * self.config.num_repeats

  @property
  def block_size(self) -> int:
    """The block size of the layer."""
    block_size = fractions.Fraction(1)
    output_ratio = fractions.Fraction(1)
    layer_output_ratio = self.child_layer.output_ratio
    layer_block_size = self.child_layer.block_size

    # TODO(rryan): Clean up logic.
    for _ in range(self.config.num_repeats):
      block_size = (
          np.lcm(block_size * output_ratio, layer_block_size) / output_ratio
      )
      output_ratio *= layer_output_ratio

    assert block_size.denominator == 1
    return block_size.numerator

  @property
  def output_ratio(self) -> fractions.Fraction:
    output_ratio = fractions.Fraction(1)
    # TODO(rryan): Clean up logic.
    layer_output_ratio = self.child_layer.output_ratio
    for _ in range(self.config.num_repeats):
      output_ratio *= layer_output_ratio
    return output_ratio

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    # _validate asserts that input_dtype equals output_dtype.
    return input_dtype

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    # _validate asserts that input shape equals output shape.
    return tuple(input_shape)

  @nn.nowrap
  def _validate(
      self, input_spec: types.ChannelSpec, constants: types.Constants | None
  ):
    del input_spec
    del constants

    child_output_ratio = self._get_child_property(lambda l: l.output_ratio)[-1]
    if child_output_ratio != 1:
      raise ValueError(
          f'Repeat layer must have output ratio of 1: {child_output_ratio=}.'
      )

    # TODO(rryan): Check that child layer's output dtype matches the input
    # dtype. We cannot accurately compute child_layer's output type using
    # SequenceLayer.get_output_dtype, since the Gemax trainer might have
    # replaced our weights with donwcasted versions when bf16_mode != NOWHERE.

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    y, _ = self._layer_internal(x, training, constants, with_emits=False)
    return y

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    return self._layer_internal(x, training, constants, with_emits=True)

  def _layer_internal(
      self,
      x: types.Sequence,
      training: bool,
      constants: types.Constants | None,
      with_emits: bool,
  ) -> tuple[types.Sequence, types.Emits]:
    self._validate(x.channel_spec, constants)

    def scan_fn(child_layer, scan_carry: types.Sequence, scan_input):
      del scan_input
      x = scan_carry
      if with_emits:
        y, emits = child_layer.layer_with_emits(
            x, training=training, constants=constants
        )
      else:
        y = child_layer.layer(x, training=training, constants=constants)
        emits = ()

      scan_output = emits
      # Unmask the sequence so that scan's type checks pass.
      # TODO(rryan): Remove the type distinction between masked and unmasked
      # sequences.
      scan_carry_output = y.unmask()
      return scan_carry_output, scan_output

    if self.config.remat:
      # TODO(rryan): Support customization (prevent_cse, policy, etc.).
      scan_fn = nn.remat(scan_fn, prevent_cse=False)

    metadata_params = {
        # For params we replicate along this scan-over-layers axis.
        meta.MESH_AXIS: None,
        # For optimizers mark this axis in params as 'independent'.
        meta.AXIS_TYPE: meta.AxisType.STACKED,
    }

    if not self.is_initializing() and self.config.unroll_layer:
      emits = []

      def trans_in_fn(tree, i):
        tree = flax.core.meta.remove_axis(tree, 0, metadata_params)
        return jax.tree.map(lambda a: a[i], tree)

      def trans_out_fn(tree):
        return flax.core.meta.add_axis(tree, 0, metadata_params)

      for i in range(self.config.num_repeats):
        x, emit_i = nn.map_variables(
            scan_fn,
            trans_in_fn=functools.partial(trans_in_fn, i=i),
            trans_out_fn=trans_out_fn,
        )(self.child_layer, x, ())
        emits.append(emit_i)

      return x, tuple(emits)

    repeat = nn.scan(
        scan_fn,
        variable_axes={True: 0},  # Slice all variables on axis 0.
        variable_broadcast=False,
        in_axes=0,  # No inputs.
        out_axes=0,  # Emits packed on axis 0.
        # Split all RNGs in our scope except for params when not initializing.
        split_rngs={
            'params': self.is_initializing(),
            nn.DenyList(('params',)): True,
        },
        length=self.config.num_repeats,
        metadata_params=metadata_params,
    )

    # repeat(module, carry, *xs)
    # Unmask the sequence so that scan's type checks pass.
    # TODO(rryan): Remove the type distinction between masked and unmasked
    # sequences.
    scan_carry = x.unmask()
    scan_input = ()
    scan_carry_output, scan_output = repeat(
        self.child_layer, scan_carry, scan_input
    )

    y = scan_carry_output
    emits = scan_output

    # Emits are stacked on axis 0. Unpack them.
    emits = utils.unstack_tree(emits)

    return y, emits

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    self._validate(input_spec, constants)

    def scan_fn(
        child_layer: types.SequenceLayer,
        scan_carry,
        scan_input,
    ):
      del scan_carry, scan_input
      state = child_layer.get_initial_state(
          batch_size, input_spec, training=training, constants=constants
      )
      scan_output = state
      scan_carry_output = ()
      return scan_carry_output, scan_output

    metadata_params = {
        # For params we replicate along this scan-over-layers axis.
        meta.MESH_AXIS: None,
        # For optimizers mark this axis in params as 'independent'.
        meta.AXIS_TYPE: meta.AxisType.STACKED,
    }

    if not self.is_initializing() and self.config.unroll_step:
      states = []

      def trans_in_fn(tree, i):
        tree = flax.core.meta.remove_axis(tree, 0, metadata_params)
        return jax.tree.map(lambda a: a[i], tree)

      def trans_out_fn(tree):
        return flax.core.meta.add_axis(tree, 0, metadata_params)

      for i in range(self.config.num_repeats):
        _, state = nn.map_variables(
            scan_fn,
            trans_in_fn=functools.partial(trans_in_fn, i=i),
            trans_out_fn=trans_out_fn,
        )(self.child_layer, (), ())
        states.append(state)
      return tuple(states)

    repeat = nn.scan(
        scan_fn,
        variable_axes={True: 0},  # Slice all variables on axis 0.
        variable_broadcast=False,
        in_axes=0,  # No input.
        out_axes=1,  # State stacked on axis 1.
        split_rngs={
            'params': self.is_initializing(),
            nn.DenyList(('params',)): True,
        },
        length=self.config.num_repeats,
        metadata_params=metadata_params,
    )

    # No carry or input.
    scan_carry = ()
    scan_input = ()
    # repeat(module, carry, *xs)
    scan_carry_output, scan_output = repeat(
        self.child_layer, scan_carry, scan_input
    )
    # No carry output.
    del scan_carry_output
    state = scan_output

    return state

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    y, state, _ = self._step_internal(
        x, state, training=training, constants=constants, with_emits=False
    )
    return y, state

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    return self._step_internal(
        x, state, training=training, constants=constants, with_emits=True
    )

  def _step_internal(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None,
      with_emits: bool,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    self._validate(x.channel_spec, constants)

    def scan_fn(
        child_layer: types.SequenceLayer,
        scan_carry: types.Sequence,
        scan_input: types.State,
    ):
      x = scan_carry
      state = scan_input
      if with_emits:
        y, state, emits = child_layer.step_with_emits(
            x, state, training=training, constants=constants
        )
      else:
        y, state = child_layer.step(
            x, state, training=training, constants=constants
        )
        emits = ()

      scan_output = (state, emits)
      # Unmask the sequence so that scan's type checks pass.
      # TODO(rryan): Remove the type distinction between masked and unmasked
      # sequences.
      scan_carry_output = y.unmask()
      return scan_carry_output, scan_output

    if self.config.remat:
      scan_fn = nn.remat(scan_fn)

    metadata_params = {
        # For params we replicate along this scan-over-layers axis.
        meta.MESH_AXIS: None,
        # For optimizers mark this axis in params as 'independent'.
        meta.AXIS_TYPE: meta.AxisType.STACKED,
    }

    if not self.is_initializing() and self.config.unroll_step:
      states = []
      emits = []

      def trans_in_fn(tree, i):
        tree = flax.core.meta.remove_axis(tree, 0, metadata_params)
        return jax.tree.map(lambda a: a[i], tree)

      def trans_out_fn(tree):
        return flax.core.meta.add_axis(tree, 0, metadata_params)

      for i, state_i in enumerate(state):
        x, (state_i, emit_i) = nn.map_variables(
            scan_fn,
            trans_in_fn=functools.partial(trans_in_fn, i=i),
            trans_out_fn=trans_out_fn,
        )(self.child_layer, x, state_i)
        states.append(state_i)
        emits.append(emit_i)

      return x, tuple(states), tuple(emits)

    repeat = nn.scan(
        scan_fn,
        variable_axes={True: 0},  # Slice all variables on axis 0.
        variable_broadcast=False,
        in_axes=1,  # State packed on axis 1.
        out_axes=1,  # State and emits stacked on axis 1.
        split_rngs={
            'params': self.is_initializing(),
            nn.DenyList(('params',)): True,
        },
        length=self.config.num_repeats,
        metadata_params=metadata_params,
    )

    # Unmask the sequence so that scan's type checks pass.
    # TODO(rryan): Remove the type distinction between masked and unmasked
    # sequences.
    scan_carry = x.unmask()
    scan_input = state
    scan_carry_output, scan_output = repeat(
        self.child_layer, scan_carry, scan_input
    )

    y = scan_carry_output
    state, emits = scan_output

    # Emits are stacked on axis 1. Unpack them.
    emits = utils.unstack_tree(emits, axis=1)

    return y, state, emits


class CheckpointGradient(WrapperMixin, types.Emitting):
  """A layer that wraps the specified SequenceLayer with a gradient checkpoint.

  This enables a memory/compute trade-off during gradient calculations by
  discarding intermediate activations within the wrapped SequenceLayer in the
  forward pass, and recalculating the forward activations in the backward pass.

  This layer is a no-op when not computing gradients, however it will introduce
  an additional layer in the variable tree due to how Flax variable tracking
  works, so wrapping a layer with CheckpointGradient is not checkpoint
  compatible with a non-wrapped layer.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for CheckpointGradient."""

    layer: types.SequenceLayerConfig
    # Whether to prevent common subexpression elimination (CSE). Preventing CSE
    # comes with a potentially high cost performance-wise, but is true by
    # default since it can prevent the intended behavior of gradient
    # checkpointing. If used inside of an nn.scan, this is unnecessary and can
    # be set to false.
    prevent_cse: bool = True
    # JAX checkpoint policy. By default, recalculates everything.
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html
    # TODO(rryan): Do we need a layer vs. step policy?
    policy: Callable[..., bool] | None = None
    name: str | None = None

    def make(self) -> 'CheckpointGradient':
      return CheckpointGradient(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.child_layer = self.config.layer.make()

  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    @functools.partial(
        nn.checkpoint,
        prevent_cse=self.config.prevent_cse,
        policy=self.config.policy,
    )
    def checkpoint_fn(
        child_layer: types.SequenceLayer,
        x: types.Sequence,
        constants: types.Constants | None,
    ) -> types.Sequence:
      return child_layer.layer(x, training=training, constants=constants)

    return checkpoint_fn(self.child_layer, x, constants)

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    @functools.partial(
        nn.checkpoint,
        prevent_cse=self.config.prevent_cse,
        policy=self.config.policy,
    )
    def checkpoint_fn(
        child_layer: types.SequenceLayer,
        x: types.Sequence,
        constants: types.Constants | None,
    ) -> tuple[types.Sequence, types.Emits]:
      return child_layer.layer_with_emits(
          x, training=training, constants=constants
      )

    return checkpoint_fn(self.child_layer, x, constants)

  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    @functools.partial(
        nn.checkpoint,
        prevent_cse=self.config.prevent_cse,
        policy=self.config.policy,
    )
    def checkpoint_fn(
        child_layer: types.SequenceLayer,
        x: types.Sequence,
        state: types.State,
        constants: types.Constants | None,
    ) -> tuple[types.Sequence, types.State]:
      return child_layer.step(x, state, training=training, constants=constants)

    return checkpoint_fn(self.child_layer, x, state, constants)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    @functools.partial(
        nn.checkpoint,
        prevent_cse=self.config.prevent_cse,
        policy=self.config.policy,
    )
    def checkpoint_fn(
        child_layer: types.SequenceLayer,
        x: types.Sequence,
        state: types.State,
        constants: types.Constants | None,
    ) -> tuple[types.Sequence, types.State, types.Emits]:
      return child_layer.step_with_emits(
          x, state, training=training, constants=constants
      )

    return checkpoint_fn(self.child_layer, x, state, constants)


class Bidirectional(types.Emitting):
  """Performs bidirectional processing of a sequence.

  Input sequences are processed by a "forward" and "backward" layer. The forward
  layer processes the sequence unmodified, while the input sequence is reversed
  and processed with the backward layer. Finally, it is reversed again and then
  combined with the forward layer's output according to a combination mode.

  Sequence reversal is performed by reversing the physical time dimension of the
  sequence without making any assumptions about the layout or valid timesteps in
  the sequence.

  For example a sequence with 3 padding timesteps:

  A B C D E X X X

  Is processed by the forward layer to produce:

  FA FB FC FD FE FX FX FX

  The backward layer processes the sequence:

  X X X E D C B A

  To produce:

  BX BX BX BE BD BC BB BA

  Which is then reversed and combined with the forward layer outputs according
  to the combination mode. For example, with combination = STACK, the result is
  a [2, output_shape...] sequence:

  [FA BA] [FB BB] [FC BC] [FD BD] [FE BE] [XX] [XX] [XX]

  The resulting mask from processesing the forward and backward is combined via
  boolean AND.

  WARNING: Causal convolution layers do not work well in backward mode due to
  how causal padding is applied and masked. The physical reversal of the
  sequence combined with causal padding leads to improper masking like:

  For kernel_size = 3:

  P P X X X E D C B A

  Where P is the causal padding applied to the sequence. The correct behavior
  would be to pad like:

  X X X P P E D C B A

  The impact of this is that timestep E and D (for this kernel_size 3 example)
  become invalid, incorrectly shrinking the sequence length.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for Bidirectional."""

    # The forward layer definition.
    forward: types.SequenceLayerConfig
    # The backward layer definition.
    backward: types.SequenceLayerConfig
    # The combination mode to use for combining the forward and backward layer
    # outputs.
    combination: utils.CombinationMode = utils.CombinationMode.STACK
    # An optional name for the layer.
    name: str | None = None

    def make(self) -> 'Bidirectional':
      return Bidirectional(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.forward = self.config.forward.make()
    self.backward = self.config.backward.make()

    if self.forward.output_ratio != self.backward.output_ratio:
      raise ValueError(
          f'{self}: output ratios for the forward '
          'and backward direction must be equal. '
          f'forward={self.forward.output_ratio} '
          f'backward={self.backward.output_ratio}.'
      )

  @property
  def supports_step(self) -> bool:
    return False

  @property
  def block_size(self) -> int:
    # Bidirectional is not steppable, so block size does not matter.
    return 1

  @property
  def output_ratio(self) -> fractions.Fraction:
    # setup() checks forward and backward output ratio is equal.
    return self.forward.output_ratio

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    forward_output_shape = self.forward.get_output_shape(
        input_shape, constants=constants
    )
    backward_output_shape = self.backward.get_output_shape(
        input_shape,
        constants=constants,
    )
    return utils.sequence_broadcast_combine_output_channel_shape(
        self.config.combination, forward_output_shape, backward_output_shape
    )

  @nn.nowrap
  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    forward_dtype = self.forward.get_output_dtype(input_dtype)
    backward_dtype = self.backward.get_output_dtype(input_dtype)
    # TODO(rryan): We should probably disallow automatic type promotion between
    # integer and float types, since this is usually a sign of a bug.
    return jnp.result_type(forward_dtype, backward_dtype)

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    forward_state = self.forward.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    backward_state = self.backward.get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )
    return (forward_state, backward_state)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    raise ValueError('Bidirectional does not support step-wise processing.')

  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    y_forward = self.forward.layer(x, training=training, constants=constants)

    x_reverse = x.reverse_time()
    y_backward = self.backward.layer(
        x_reverse,
        training=training,
        constants=constants,
    )
    y_backward = y_backward.reverse_time()

    return utils.sequence_broadcast_combine(
        self.config.combination, y_forward, y_backward
    )

  def layer_with_emits(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    y_forward, forward_emits = self.forward.layer_with_emits(
        x, training=training, constants=constants
    )

    x_reverse = x.reverse_time()
    y_backward, backward_emits = self.backward.layer_with_emits(
        x_reverse,
        training=training,
        constants=constants,
    )
    y_backward = y_backward.reverse_time()
    y = utils.sequence_broadcast_combine(
        self.config.combination, y_forward, y_backward
    )
    emits = (forward_emits, backward_emits)
    return y, emits


class Blockwise(WrapperMixin, types.SequenceLayer):
  """Processes the provided layer in blocks of a given size."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for Blockwise."""

    child_layer: types.SequenceLayerConfig
    block_size: int
    name: str | None = None

    def make(self) -> 'Blockwise':
      return Blockwise(self, name=self.name)

  config: Config

  def setup(self) -> None:
    self.child_layer = self.config.child_layer.make()
    nn.share_scope(self, self.child_layer)

    if self.config.block_size % self.child_layer.block_size != 0:
      raise ValueError(
          'Block size must be a multiple of the child layer block size.'
          f' {self.config.block_size=} % {self.child_layer.block_size=} =='
          f' {self.config.block_size % self.child_layer.block_size}'
      )

  @property
  def block_size(self) -> int:
    return self.config.block_size
