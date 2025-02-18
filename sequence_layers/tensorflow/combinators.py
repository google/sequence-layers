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
"""Combinator layers."""

import collections
import enum
import fractions
from typing import Callable, Generator, List, Optional, Sequence, Tuple

from absl import logging
import numpy as np
from sequence_layers.tensorflow import dense
from sequence_layers.tensorflow import simple
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf

# Either a SequenceLayer, a list of SequenceLayers, or a callable that returns
# a SequenceLayer or a list of SequenceLayers.
SequenceLayerOrList = types.SequenceLayer | Sequence[types.SequenceLayer]
SequenceLayerListOrCallable = (
    SequenceLayerOrList | Callable[[], SequenceLayerOrList]
)


def _maybe_wrap_layers(
    module: tf.Module, layers: SequenceLayerListOrCallable
) -> types.SequenceLayer:
  if callable(layers):
    with module.name_scope:
      layers = layers()

  if isinstance(layers, types.SequenceLayer):
    return layers
  else:
    return _SerialWithoutNameScopes(layers)


def _insert_with_unique_key(
    emits: collections.OrderedDict[str, types.Emits], key: str, value
) -> str:
  """Inserts value into emits with a unique name prefixed by key."""
  if key not in emits:
    emits[key] = value
    return key
  i = 1
  while True:
    unique_key = f'{key}_{i}'
    if unique_key not in emits:
      emits[unique_key] = value
      return unique_key
    i += 1


class _SerialWithoutNameScopes(types.Emitting):
  """Applies a sequence of layers in series.

  Private implementation of Serial which doesn't introduce new name_scopes.
  This is only intended to be used as an implementation detail within other
  SequenceLayers.
  """

  @classmethod
  def _default_name(cls):
    # TF1 does not allow _serial_without_name_scopes.
    return 'serial_without_name_scopes'

  def __init__(
      self,
      layers: Sequence[types.SequenceLayer]
      | Callable[[], Sequence[types.SequenceLayer]],
      debug=False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if callable(layers):
      # This is _SerialWithoutNameScopes, but the only way a callable gets
      # passed here is if we are a Serial, so call the callable in our name
      # scope.
      with self.name_scope:
        layers = layers()
    self._layers = layers[:]  # Copy the list.
    self._debug = debug

  @property
  def supports_step(self) -> bool:
    return all(layer.supports_step for layer in self._layers)

  @property
  def block_size(self) -> int:
    block_size = fractions.Fraction(1)
    output_ratio = fractions.Fraction(1)

    for layer in self._layers:
      layer_output_ratio = layer.output_ratio
      layer_block_size = layer.block_size
      block_size = (
          np.lcm(block_size * output_ratio, layer_block_size) / output_ratio
      )
      output_ratio *= layer_output_ratio

    assert block_size.denominator == 1
    return block_size.numerator

  @property
  def output_ratio(self) -> fractions.Fraction:
    output_ratio = fractions.Fraction(1)
    for layer in self._layers:
      output_ratio *= layer.output_ratio
    return output_ratio

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    states = []
    for layer_i, layer in enumerate(self._layers):
      # get_initial_state can create variables (it shouldn't, but it does).
      with tf.name_scope('layer%d' % layer_i):
        state = layer.get_initial_state(x, constants)
        states.append(state)
        output_shape = layer.get_output_shape_for_sequence(x, constants)
        output_dtype = layer.get_output_dtype(x.dtype)
        batch_size = utils.smart_dimension_size(x.values, 0)
        x = types.Sequence(
            tf.zeros(
                [batch_size, 0] + output_shape.as_list(), dtype=output_dtype
            ),
            tf.ones([batch_size, 0], dtype=x.mask.dtype),
        )
    return states

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    output_shape = input_shape
    for layer in self._layers:
      output_shape = layer.get_output_shape(output_shape, constants)
    return output_shape

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.EmitSpecs:
    output_spec = input_spec
    emit_specs = collections.OrderedDict()
    for layer in self._layers:
      layer_emit_specs = layer.get_emit_specs(output_spec, constants)
      _insert_with_unique_key(emit_specs, layer.name, layer_emit_specs)
      output_spec = layer.get_output_spec(output_spec, constants)
    return emit_specs

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    output_dtype = input_dtype
    for layer in self._layers:
      output_dtype = layer.get_output_dtype(output_dtype)
    return output_dtype

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    layer_states = []
    emits = collections.OrderedDict()
    assert len(self._layers) == len(state)
    for layer_i, (layer, layer_state) in enumerate(zip(self._layers, state)):
      with tf.name_scope('layer%d' % layer_i):
        if self._debug:
          logging.debug('step layer_i=%d x=%s', layer_i, x)
        old_x, old_layer_state = x, layer_state
        x, layer_state, layer_emits = layer.step_with_emits(
            x, layer_state, training=training, constants=constants
        )
        if not x.channel_shape.is_fully_defined():
          raise ValueError(
              '%s caused input=%s state=%s to become undefined: %s'
              % (layer, old_x, old_layer_state, x)
          )
        if self._debug:
          logging.debug('step layer_i=%d -> x=%s', layer_i, x)
        layer_states.append(layer_state)
        _insert_with_unique_key(emits, layer.name, layer_emits)
    return x, layer_states, emits

  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = [None] * len(self._layers)

    emits = collections.OrderedDict()
    assert len(self._layers) == len(initial_state)
    for layer_i, (layer, layer_initial_state) in enumerate(
        zip(self._layers, initial_state)
    ):
      with tf.name_scope('layer%d' % layer_i):
        if self._debug:
          logging.debug('layer layer_i=%d x=%s', layer_i, x)
        x_old = x
        x, layer_emits = layer.layer_with_emits(
            x,
            initial_state=layer_initial_state,
            training=training,
            constants=constants,
        )
        if not x.channel_shape.is_fully_defined():
          raise ValueError(
              '%s caused input=%s to become undefined: %s' % (layer, x_old, x)
          )
        if self._debug:
          logging.debug('layer layer_i=%d -> x=%s', layer_i, x)
        _insert_with_unique_key(emits, layer.name, layer_emits)
    return x, emits

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    yield from super()._yield_emits(emits)
    key_map = collections.OrderedDict()
    for layer in self._layers:
      # Figure out the key we inserted the emit with.
      key = _insert_with_unique_key(key_map, layer.name, None)
      yield from layer._yield_emits(emits[key])  # pylint: disable=protected-access


class Serial(_SerialWithoutNameScopes):
  """Applies a sequence of layers in series.

  The emit structure for a Serial layer is an OrderedDict of the emits for each
  of the layers in the Serial whose keys are the layer names and the order is
  the order of layers in the Serial. When a duplicate name exists, a "_X" suffix
  is appended to the layer's name to form the key, where X is an increasing
  integer starting with 1.
  """

  @classmethod
  def _default_name(cls):
    # Override _SerialWithoutNameScopes._default_name
    return 'serial'

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    return super().step_with_emits(x, state, training, constants)

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    return super().layer_with_emits(x, training, initial_state, constants)


class Parallel(types.Emitting):
  """Applies a sequence of layers in parallel.

  Outputs are broadcasted and stacked to produce a [num_layers, ...] output
  shape.
  """

  @enum.unique
  class Combination(enum.Enum):
    """The type of combination to perform."""

    # Stack output of each parallel layer.
    STACK = 1
    # Broadcast-add the output of each parallel layer.
    ADD = 2
    # Broadcast-mean the output of each parallel layer.
    MEAN = 3

  def __init__(
      self,
      layers: Sequence[SequenceLayerListOrCallable],
      debug=False,
      combination: Combination = Combination.STACK,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._layers = []
    for layer in layers:
      self._layers.append(_maybe_wrap_layers(self, layer))
    self._debug = debug
    self._combination = combination

    output_ratios = [l.output_ratio for l in self._layers]
    if output_ratios:
      for layer, output_ratio in zip(self._layers, output_ratios):
        if output_ratio != output_ratios[0]:
          raise ValueError(
              'Output ratios must be equal for all layers: '
              f'{output_ratios[0]} != {output_ratio} for {layer}'
          )
      self._output_ratio = output_ratios[0]
    else:
      self._output_ratio = fractions.Fraction(1)

  @property
  def supports_step(self) -> bool:
    return all(layer.supports_step for layer in self._layers)

  @property
  def block_size(self) -> int:
    block_size = 1
    for layer in self._layers:
      block_size = np.lcm(block_size, layer.block_size)
    return int(block_size)

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._output_ratio

  @tf.Module.with_name_scope
  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    states = []
    for layer_i, layer in enumerate(self._layers):
      # get_initial_state can create variables (it shouldn't, but it does).
      with tf.name_scope('layer%d' % layer_i):
        states.append(layer.get_initial_state(x, constants))
    return states

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not self._layers:
      return input_shape

    output_shape = self._layers[0].get_output_shape(input_shape, constants)
    for layer in self._layers[1:]:
      layer_output_shape = layer.get_output_shape(input_shape, constants)
      try:
        output_shape = tf.broadcast_static_shape(
            output_shape, layer_output_shape
        )
      except ValueError as e:
        raise ValueError(
            'Layer output shapes are not broadcastable: '
            f'{layer_output_shape} != {output_shape}.'
        ) from e

    if self._combination == Parallel.Combination.STACK:
      return tf.TensorShape([len(self._layers)]).concatenate(output_shape)
    else:
      return output_shape

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.EmitSpecs:
    emit_specs = collections.OrderedDict()
    for layer in self._layers:
      layer_emit_specs = layer.get_emit_specs(input_spec, constants)
      _insert_with_unique_key(emit_specs, layer.name, layer_emit_specs)
    return emit_specs

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    output_dtypes = [l.get_output_dtype(input_dtype) for l in self._layers]
    output_dtype = output_dtypes[0] if output_dtypes else input_dtype
    for output_dtype, layer in zip(output_dtypes, self._layers):
      if output_dtype != output_dtypes[0]:
        raise ValueError(f'Incompatible dtypes: {output_dtype} for {layer}.')
    return output_dtype

  def _combine_outputs(self, ys: List[types.Sequence]) -> types.Sequence:
    assert ys
    channel_shape = ys[0].channel_shape
    mask = ys[0].mask
    for y in ys[1:]:
      channel_shape = tf.broadcast_static_shape(channel_shape, y.channel_shape)
      mask = utils.combine_mask(mask, y.mask)

    values = []
    for y in ys:
      batch, time = utils.smart_dimension_size(y.values, [0, 1])
      values.append(
          tf.broadcast_to(
              types.expand_to_rank(y.values, 2 + len(channel_shape)),
              [batch, time] + channel_shape.as_list(),
          )
      )
    if self._combination == Parallel.Combination.STACK:
      values = tf.stack(values, axis=2, name='values')
    elif self._combination == Parallel.Combination.ADD:
      values = tf.add_n(values, name='values')
    else:
      assert self._combination == Parallel.Combination.MEAN
      values = tf.identity(tf.add_n(values) / len(values), name='values')

    mask = tf.identity(mask, name='mask')
    return types.Sequence(values, mask)

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    layer_states = []
    emits = collections.OrderedDict()
    assert len(self._layers) == len(state)

    if not self._layers:
      return x, state, emits

    ys = []
    for layer_i, (layer, layer_state) in enumerate(zip(self._layers, state)):
      with tf.name_scope('layer%d' % layer_i):
        if self._debug:
          logging.debug('step layer_i=%d x=%s', layer_i, x)
        old_layer_state = layer_state
        y_i, layer_state, layer_emits = layer.step_with_emits(
            x, layer_state, training=training, constants=constants
        )
        ys.append(y_i)
        if not y_i.channel_shape.is_fully_defined():
          raise ValueError(
              '%s caused input=%s state=%s to become undefined: %s'
              % (layer, x, old_layer_state, y_i)
          )
        if self._debug:
          logging.debug('step layer_i=%d -> y_i=%s', layer_i, y_i)
        layer_states.append(layer_state)
        _insert_with_unique_key(emits, layer.name, layer_emits)
    y = self._combine_outputs(ys).mask_invalid()
    return y, layer_states, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = [None] * len(self._layers)

    emits = collections.OrderedDict()
    assert len(self._layers) == len(initial_state)

    if not self._layers:
      return x, emits
    ys = []
    for layer_i, (layer, layer_initial_state) in enumerate(
        zip(self._layers, initial_state)
    ):
      with tf.name_scope('layer%d' % layer_i):
        if self._debug:
          logging.debug('layer layer_i=%d x=%s', layer_i, x)
        x_old = x
        y_i, layer_emits = layer.layer_with_emits(
            x,
            initial_state=layer_initial_state,
            training=training,
            constants=constants,
        )
        ys.append(y_i)
        if not y_i.channel_shape.is_fully_defined():
          raise ValueError(
              '%s caused input=%s to become undefined: %s' % (layer, x_old, y_i)
          )
        if self._debug:
          logging.debug('layer layer_i=%d -> y_i=%s', layer_i, y_i)
        _insert_with_unique_key(emits, layer.name, layer_emits)
    y = self._combine_outputs(ys).mask_invalid()
    return y, emits

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    yield from super()._yield_emits(emits)
    key_map = collections.OrderedDict()
    for layer in self._layers:
      # Figure out the key we inserted the emit with.
      key = _insert_with_unique_key(key_map, layer.name, None)
      yield from layer._yield_emits(emits[key])  # pylint: disable=protected-access


class Residual(types.Emitting):
  """A residual wrapper around l that computes `y = l(x) + shortcut(x)`.

  If shortcut is not provided, it defaults to an identity or a linear projection
  to match the output shape of l(x).
  """

  def __init__(
      self,
      layer: SequenceLayerListOrCallable,
      shortcut_layer: Optional[SequenceLayerListOrCallable] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._layer = _maybe_wrap_layers(self, layer)
    # If shortcut_layer is not None, we build a projection layer that depends
    # on input size, so we have to lazily build.
    if shortcut_layer is None:
      self._shortcut_layer = shortcut_layer
    else:
      self._shortcut_layer = _maybe_wrap_layers(self, shortcut_layer)

    if (
        self._shortcut_layer
        and self._shortcut_layer.output_ratio != self._layer.output_ratio
    ):
      raise ValueError(
          'layer and shortcut_layer must have the same output '
          f'ratio {self._layer.output_ratio} != '
          f'{self._shortcut_layer.output_ratio}.'
      )
    elif self._shortcut_layer is None and self._layer.output_ratio != 1:
      raise ValueError(
          'Residual requires output_ratio 1, got: %s, output_ratio=%s'
          % (self._layer, self._layer.output_ratio)
      )

  @property
  def supports_step(self) -> bool:
    supports_step = self._layer.supports_step
    if self._shortcut_layer:
      supports_step = supports_step and self._shortcut_layer.supports_step
    return supports_step  # pytype: disable=bad-return-type  # bind-properties

  @property
  def block_size(self) -> int:
    if self._shortcut_layer:
      return int(
          np.lcm(self._layer.block_size, self._shortcut_layer.block_size)
      )
    return self._layer.block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._layer.output_ratio

  def build(self, x, constants: Optional[types.Constants] = None):
    if self._shortcut_layer is None:
      input_shape = x.channel_shape
      output_shape = self._layer.get_output_shape(input_shape, constants)
      if input_shape != output_shape:
        # Linear projection to make shapes match.
        self._shortcut_layer = dense.DenseShaped(
            output_shape, name='residual_projection'
        )
      else:
        self._shortcut_layer = simple.Identity()

    layer_dtype = self._layer.get_output_dtype(x.dtype)
    shortcut_dtype = self._shortcut_layer.get_output_dtype(x.dtype)
    if shortcut_dtype != layer_dtype:
      raise ValueError(
          'Residual connection must have same dtypes: '
          f'layer: {layer_dtype} shortcut: {shortcut_dtype}'
      )

  @tf.Module.with_name_scope
  def step_with_emits(
      self,
      x,
      state,
      training=False,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    self.build(x, constants)
    layer_state, shortcut_state = state
    y, layer_state, emits = self._layer.step_with_emits(
        x, layer_state, training=training, constants=constants
    )
    emits = collections.OrderedDict([[self._layer.name, emits]])
    y_shortcut, shortcut_state = self._shortcut_layer.step(
        x, shortcut_state, training=training, constants=constants
    )

    y_mask = utils.combine_mask(y_shortcut.mask, y.mask)
    y = y.apply(lambda v, m: (v + y_shortcut.values, y_mask))
    state = (layer_state, shortcut_state)
    return y.mask_invalid(), state, emits

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    self.build(x, constants)
    if initial_state is None:
      initial_shortcut_state = None
    else:
      initial_state, initial_shortcut_state = initial_state
    y, emits = self._layer.layer_with_emits(
        x, initial_state=initial_state, training=training, constants=constants
    )
    emits = collections.OrderedDict([[self._layer.name, emits]])
    y_shortcut = self._shortcut_layer.layer(
        x,
        initial_state=initial_shortcut_state,
        training=training,
        constants=constants,
    )
    y_mask = utils.combine_mask(y_shortcut.mask, y.mask)
    y = y.apply(lambda v, m: (v + y_shortcut.values, y_mask))
    return y.mask_invalid(), emits

  @tf.Module.with_name_scope
  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    self.build(x, constants)
    state = self._layer.get_initial_state(x, constants)
    shortcut_state = self._shortcut_layer.get_initial_state(x, constants)
    return state, shortcut_state

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return self._layer.get_output_shape(input_shape, constants)

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.EmitSpecs:
    return collections.OrderedDict(
        [[self._layer.name, self._layer.get_emit_specs(input_spec, constants)]]
    )

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    yield from super()._yield_emits(emits)
    yield from self._layer._yield_emits(emits[self._layer.name])  # pylint: disable=protected-access


class Skip(Parallel):
  """A skip wrapper around l that computes `y = concat([x, l(x)])`."""

  def __init__(
      self,
      layer: SequenceLayerListOrCallable,
      name: Optional[str] = None,
  ):
    super().__init__([simple.Identity(), layer], name=name)

  def _combine_outputs(self, ys: List[types.Sequence]) -> types.Sequence:
    assert len(ys) == 2
    if ys[0].values.dtype != ys[1].values.dtype:
      raise ValueError(
          'Skip connection must have same dtypes: '
          f'input_dtype:{ys[0].values.dtype} output_dtype:{ys[1].values.dtype}'
      )

    # Assumption: x is masked and y is masked, so no masking necessary.
    # TODO(rryan): Assumes x and y mask are identical. Merge masks? If changed,
    # we need to re-mask after merging.
    return ys[0].apply_values(lambda v: tf.concat([v, ys[1].values], axis=2))

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    """Output size is input-dependent."""
    assert len(self._layers) == 2
    output_shape = self._layers[1].get_output_shape(input_shape, constants)
    if input_shape.rank != output_shape.rank:
      raise ValueError(
          'Skip connection inner shapes must have same '
          'rank: input_shape=%s output_shape=%s' % (input_shape, output_shape)
      )
    if input_shape[1:] != output_shape[1:]:
      raise ValueError(
          'Skip connection requires input and output shape to '
          'match after the first dimension: input_shape=%s output_shape=%s'
          % (input_shape, output_shape)
      )
    return tf.TensorShape(
        [input_shape.dims[0].value + output_shape.dims[0].value]
    ).concatenate(output_shape[1:])


class Blockwise(types.Emitting):
  """Runs the specified layer block_size steps at a time."""

  def __init__(
      self,
      layer: SequenceLayerListOrCallable,
      block_size: int,
      name=None,
  ):
    super().__init__(name=name)
    self._layer = _maybe_wrap_layers(self, layer)
    if not block_size:
      raise ValueError(f'{self} block_size must be positive, got: {block_size}')
    self._block_size = block_size

  @property
  def supports_step(self) -> bool:
    return self._layer.supports_step  # pytype: disable=bad-return-type  # bind-properties

  @property
  def block_size(self) -> int:
    return self._block_size

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._layer.output_ratio

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return self._layer.get_output_shape(input_shape, constants)

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    return self._layer.get_output_dtype(input_dtype)

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    return self._layer.get_initial_state(x, constants)

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.EmitSpecs:
    return self._layer.get_emit_specs(input_spec, constants)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    return self._layer.step_with_emits(x, state, training, constants)

  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    y, _, emits = utils.step_by_step_dynamic(
        self,
        x,
        training=training,
        initial_state=initial_state,
        constants=constants,
    )
    return y, emits

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    yield from super()._yield_emits(emits)
    yield from self._layer._yield_emits(emits)  # pylint: disable=protected-access


class Bidirectional(types.Emitting):
  """Performs bidirectional processing on a sequence.

  TODO(rryan): Support more methods of merging the two directions beyond
  depthwise concat.
  """

  def __init__(
      self,
      forward: SequenceLayerListOrCallable,
      backward: SequenceLayerListOrCallable,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._forward = _maybe_wrap_layers(self, forward)
    self._backward = _maybe_wrap_layers(self, backward)

    if self._forward.output_ratio != self._backward.output_ratio:
      raise ValueError(
          f'{self}: output ratios for the forward '
          'and backward direction must be equal. '
          f'forward={self._forward.output_ratio} '
          f'backward={self._backward.output_ratio}.'
      )

  @property
  def supports_step(self) -> bool:
    return False

  @property
  def block_size(self) -> int:
    # Bidirectional is not steppable, so block size does not matter.
    return 0

  @property
  def output_ratio(self) -> fractions.Fraction:
    return self._forward.output_ratio

  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: types.Constants = None
  ) -> tf.TensorShape:
    forward_output_shape = self._forward.get_output_shape(
        input_shape, constants
    )
    backward_output_shape = self._backward.get_output_shape(
        input_shape, constants
    )

    if not forward_output_shape[:-1].is_compatible_with(
        backward_output_shape[:-1]
    ):
      raise ValueError(
          f'{self}: Forward and backward output shapes are not '
          f'compatible. forward={forward_output_shape} '
          f'backward={backward_output_shape}'
      )

    output_shape = forward_output_shape[:-1].merge_with(
        backward_output_shape[:-1]
    )
    final_dim = (
        forward_output_shape.dims[-1].value
        + backward_output_shape.dims[-1].value
    )
    return output_shape.concatenate(tf.TensorShape([final_dim]))

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    forward_dtype = self._forward.get_output_dtype(input_dtype)
    backward_dtype = self._backward.get_output_dtype(input_dtype)
    if forward_dtype != backward_dtype:
      raise ValueError(
          f'{self}: forward and backward dtypes are different. '
          f'forward={forward_dtype} backward={backward_dtype}'
      )
    return forward_dtype

  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants = None
  ) -> types.State:
    forward_state = self._forward.get_initial_state(x, constants)
    backward_state = self._backward.get_initial_state(x, constants)
    return (forward_state, backward_state)

  def get_emit_specs(
      self, input_spec: tf.TensorSpec, constants: types.Constants = None
  ) -> types.EmitSpecs:
    forward_emit_spec = self._forward.get_emit_specs(input_spec, constants)
    backward_emit_spec = self._backward.get_emit_specs(input_spec, constants)
    return (forward_emit_spec, backward_emit_spec)

  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants = None,
  ) -> Tuple[types.Sequence, types.State, types.Emits]:
    raise ValueError('Bidirectional does not support step-wise processing.')

  @tf.Module.with_name_scope
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State = None,
      constants: types.Constants = None,
  ) -> Tuple[types.Sequence, types.Emits]:
    if initial_state is None:
      initial_state = (None, None)
    forward_state, backward_state = initial_state

    y_forward, forward_emits = self._forward.layer_with_emits(
        x, training=training, initial_state=forward_state, constants=constants
    )

    x_reverse = x.reverse_time()
    y_backward, backward_emits = self._backward.layer_with_emits(
        x_reverse,
        training=training,
        initial_state=backward_state,
        constants=constants,
    )
    y_backward = y_backward.reverse_time()

    def concat(values, mask):
      values = tf.concat([values, y_backward.values], axis=-1)
      mask = utils.combine_mask(mask, y_backward.mask)
      return values, mask

    y = y_forward.apply(concat).mask_invalid()
    return y, (forward_emits, backward_emits)

  def _yield_emits(
      self, emits: types.Emits
  ) -> Generator[tuple[types.SequenceLayer, types.Emits], None, None]:
    forward_emits, backward_emits = emits
    yield from super()._yield_emits(emits)
    yield from self._forward._yield_emits(forward_emits)  # pylint: disable=protected-access
    yield from self._backward._yield_emits(backward_emits)  # pylint: disable=protected-access
