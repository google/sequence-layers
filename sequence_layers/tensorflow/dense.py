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
"""Dense layers."""

from typing import List, Optional, Union

from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class Dense(types.Stateless):
  """A simple dense layer."""

  def __init__(
      self,
      units: int,
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name=None,
  ):
    super().__init__(name=name)
    self._units = units
    with self.name_scope as name_scope:
      # Keras layers do not follow standard TF name scoping rules. Provide it a
      # fully scoped name.
      self._layer = tf.keras.layers.Dense(
          units,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    x.channel_shape.with_rank_at_least(1)
    return x.apply_values(self._layer).mask_invalid()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    input_shape.with_rank_at_least(1)
    return input_shape[:-1].concatenate(self._units)


class DenseShaped(types.Stateless):
  """A dense layer to project to a specific shape.

  Dense connection between every element in input_shape to
  every element in output_shape, at each time step.
  """

  def __init__(
      self,
      output_shape: Union[tf.TensorShape, List[int], tuple[int, ...]],
      activation=None,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name=None,
  ):
    super().__init__(name=name)
    self._output_shape = tf.TensorShape(output_shape)
    with self.name_scope as name_scope:
      # Keras layers do not follow standard TF name scoping rules. Provide it a
      # fully scoped name.
      self._layer = tf.keras.layers.Dense(
          self._output_shape.num_elements(),
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name=name_scope,
      )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    batch_size, time = utils.smart_dimension_size(x.values, [0, 1])

    def shaped_projection(values):
      if values.shape.rank != 3:
        values = tf.reshape(
            values, [batch_size, time, values.shape[2:].num_elements()]
        )
      values_projected = self._layer(values)
      if self._output_shape.rank == 1:
        return values_projected
      else:
        return tf.reshape(
            values_projected, [batch_size, time] + self._output_shape.as_list()
        )

    return x.apply_values(shaped_projection).mask_invalid()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return self._output_shape


class EinsumDense(types.Stateless):
  """Applies an EinsumDense layer to the channels dimension of each timestep.

  Differences from tf.keras.layers.EinsumDense:
  - Equation input and output specs must have leading ellipses to broadcast over
    the batch and time dimension.

  Example:

  Input sequence: [b, t, c1, c2, c3]
  - equation = '...abc,bd->...bd'
  - output_shape = [None, c4]
  - bias_axes = 'd'
  Output sequence: [b, t, c2, c4]

  Kernel shape: [c2, c4]
  Bias shape: [c4]

  Interpretation: Every [c1, c2, c3] tensor per timestep is transformed with the
  einsum formula abc,bd->bd.
  """

  def __init__(
      self,
      equation: str,
      output_shape: list[Optional[int]],
      activation=None,
      bias_axes=None,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name)
    with self.name_scope:
      self._layer = tf.keras.layers.EinsumDense(
          equation=equation,
          output_shape=output_shape,
          activation=activation,
          bias_axes=bias_axes,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          trainable=trainable,
          name='einsum_dense',
      )
      self._parse_and_validate_equation(equation)
      if not output_shape:
        raise ValueError(f'output_shape is required, got: {output_shape}')

  def _parse_and_validate_equation(self, equation) -> tuple[str, str, str]:
    if '->' not in equation:
      raise ValueError(f'equation is not valid for EinsumDense: {equation}')
    left, output_spec = equation.split('->')
    input_spec, kernel_spec = left.split(',')
    if not input_spec.startswith('...') or not output_spec.startswith('...'):
      raise ValueError('Equation must be of the form "...X,Y->...Z".')
    return input_spec, kernel_spec, output_spec

  def _get_and_validate_output_shape(
      self, input_shape: tf.TensorShape
  ) -> tf.TensorShape:
    equation = self._layer.equation
    input_spec, _, output_spec = self._parse_and_validate_equation(equation)
    assert input_spec.startswith('...')
    assert output_spec.startswith('...')
    # Trim '...' off.
    input_spec, output_spec = input_spec[3:], output_spec[3:]

    if len(input_spec) != len(input_shape):
      raise ValueError(
          f'Equation {input_spec=} does not match {input_shape=} rank.'
      )

    input_dims = {
        d: input_shape.dims[i].value for i, d in enumerate(input_spec)
    }
    assert len(input_dims) == len(input_spec)

    output_shape = tf.TensorShape(self._layer.partial_output_shape).as_list()
    if len(output_spec) != len(output_shape):
      raise ValueError(
          f'Equation {output_spec=} does not match {output_shape=}.'
      )

    for i, d in enumerate(output_spec):
      if output_shape[i] is None:
        output_shape[i] = input_dims[d]
      elif d in input_dims and output_shape[i] != input_dims[d]:
        raise ValueError(
            'Input shape and output shape inconsistent for dimension '
            f'{d=}. {output_shape=} {input_shape=}'
        )
    output_shape = tf.TensorShape(output_shape)
    if not output_shape.is_fully_defined():
      raise ValueError(
          f'Could not fully determine {output_shape=} from {input_shape=}. '
          f'{equation=} {self._layer.partial_output_shape=}'
      )
    return output_shape

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return self._get_and_validate_output_shape(input_shape)

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del initial_state
    self._get_and_validate_output_shape(x.channel_shape)
    return x.apply_values(
        lambda v: self._layer(v, training=training)
    ).mask_invalid()
