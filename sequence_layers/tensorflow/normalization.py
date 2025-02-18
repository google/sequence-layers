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
"""Normalization layers."""

from typing import List, Optional, Sequence, Tuple, Union

from sequence_layers.tensorflow import types
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

# TODO(dthkao): Since this is only needed for tf1 tpu context checking,
# consider removing TF1 support?
from tensorflow.python.tpu.tpu_function import get_tpu_context


def _is_tf1_tpu():
  """Returns True if we are in an XLA context."""
  if tf.__internal__.tf2.enabled():
    return False
  return get_tpu_context().number_of_shards is not None


def _validate_and_normalize_axes(
    axes: Union[int, List[int]], input_shape: tf.TensorShape
) -> List[int]:
  """Normalizes user-provided axes and checks batch/time are not specified."""
  if isinstance(axes, int):
    axes = [axes]
  else:
    axes = list(axes)
  for axis in axes:
    if axis < 0:
      axis += input_shape.rank
    if axis in (0, 1):
      raise ValueError(
          'Normalizing over the batch or time dimension is '
          f'not allowed. Got: {axes}'
      )
  return axes


class LayerNormalization(types.PreservesShape, types.Stateless):
  """Applies layer normalization to input sequences."""

  def __init__(
      self,
      axis=-1,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable=True,
      name=None,
  ):
    super().__init__(name=name)
    self._layer = None
    self._layer_kwargs = {
        'axis': axis,
        'epsilon': epsilon,
        'center': center,
        'scale': scale,
        'beta_initializer': beta_initializer,
        'gamma_initializer': gamma_initializer,
        'beta_regularizer': beta_regularizer,
        'gamma_regularizer': gamma_regularizer,
        'beta_constraint': beta_constraint,
        'gamma_constraint': gamma_constraint,
        'trainable': trainable,
    }

  def build(self, input_shape: tf.TensorShape):
    if self._layer is None:
      self._layer_kwargs['axis'] = _validate_and_normalize_axes(
          self._layer_kwargs['axis'], input_shape
      )
      with self.name_scope as name_scope:
        self._layer = tf.keras.layers.LayerNormalization(
            **self._layer_kwargs, name=name_scope
        )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State = None,
      constants: types.Constants = None,
  ) -> types.Sequence:
    self.build(x.values.shape)
    # TODO(rryan): Investigate precision of Keras LayerNormalization. It
    # requires dropping layer/step equivalence tests to to 1e-5 tolerance.
    values = self._layer(x.values, training=training)
    return types.Sequence(values, x.mask).mask_invalid()


class RMSNormalization(types.PreservesShape, types.Stateless):
  """A simplified version of LayerNormalization used in T5.

  No mean statistics or offset terms are included.

  Implementation follows:
  https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/t5.py
  """

  def __init__(
      self,
      axis: Union[int, tuple[int]] = -1,
      epsilon: float = 0.001,
      scale: bool = True,
      gamma_initializer: tf.keras.initializers.Initializer = 'ones',
      trainable: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._epsilon = epsilon
    self._axis = axis
    self._gamma_initializer = tf.keras.initializers.get(gamma_initializer)
    self._trainable = trainable
    self._built = False
    self._should_scale = scale

  def build(self, input_shape: tf.TensorShape):
    if self._built:
      return
    with self.name_scope:
      self._axis = _validate_and_normalize_axes(self._axis, input_shape)
      param_shape = [input_shape[dim] for dim in self._axis]

      if self._should_scale:
        self.scale = tf.Variable(
            lambda: self._gamma_initializer(param_shape),
            trainable=self._trainable,
            name='gamma',
        )
      self._built = True

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    x_dtype = x.values.dtype
    if not x_dtype.is_floating:
      raise ValueError(f'{self} requires floating point inputs.')
    self.build(x.values.shape)

    # Compute variance in fp32.
    variance_dtype = tf.float32 if x_dtype.size < tf.float32.size else x_dtype
    values = tf.cast(x.values, variance_dtype)
    variance = tf.math.reduce_mean(
        tf.math.square(values), axis=self._axis, keepdims=True
    )
    values = values * tf.math.rsqrt(variance + self._epsilon)
    values = tf.cast(values, x_dtype)

    if self._should_scale:
      scale = self.scale
      # If simple broadcasting doesn't work, reshape
      if self._axis != [values.shape.rank - 1]:
        # Broadcasting only necessary for norm when the axis is not just
        # the last dimension
        broadcast_shape = [1] * values.shape.rank
        for dim in self._axis:
          broadcast_shape[dim] = values.shape.dims[dim].value
        scale = tf.reshape(scale, broadcast_shape)

      values = values * tf.cast(scale, values.dtype)

    return types.Sequence(values, x.mask).mask_invalid()


class _SequenceBatchNormalization(tf.keras.layers.Layer):
  """Like Keras BatchNormalization but sequence and multi-replica aware.

  Only valid timesteps of input sequences are considered for computing moments,
  and they are computed across replicas with DistributionStrategy
  ReplicaContext.all_reduce, or in TF1 mode on TPU with
  tf.tpu.cross_replica_sum.
  """

  def __init__(
      self,
      axis=-1,
      momentum=0.99,
      epsilon=1e-3,
      center=True,
      scale=True,
      beta_initializer=tf.zeros_initializer(),
      gamma_initializer=tf.ones_initializer(),
      moving_mean_initializer=tf.zeros_initializer(),
      moving_variance_initializer=tf.ones_initializer(),
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable=True,
      name=None,
      use_cross_replica_sum=True,
      mask_output=True,
      **kwargs,
  ):
    super().__init__(name=name, **kwargs)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('axis must be int or tuple/list, got: %s' % axis)
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
    self.moving_mean_initializer = tf.keras.initializers.get(
        moving_mean_initializer
    )
    self.moving_variance_initializer = tf.keras.initializers.get(
        moving_variance_initializer
    )
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)
    self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
    self.supports_masking = True
    self._trainable = trainable
    self._use_cross_replica_sum = use_cross_replica_sum
    self._mask_output = mask_output

  @property
  def trainable(self):
    return self._trainable

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
      return tf.float32
    else:
      return self.dtype or tf.float32

  def _support_zero_size_input(self):
    return tf.distribute.has_strategy() and getattr(
        tf.distribute.get_strategy().extended,
        'enable_partial_batch_handling',
        False,
    )

  def build(self, input_and_mask_shape):
    input_shape, _ = input_and_mask_shape
    input_shape = tf.TensorShape(input_shape)
    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)
    if 0 in self.axis or 1 in self.axis:
      raise ValueError(
          'This implementation assumes axis 0 is batch and 1 is '
          'time. Computing BatchNormalization on either of these '
          'dimensions is not supported. Got axis=%s'
          % self.axis
      )

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError(
            'Input has undefined `axis` dimension. Received input '
            'with shape %s. Axis value: %s' % (tuple(input_shape), self.axis)
        )
    self.input_spec = (
        tf.keras.layers.InputSpec(ndim=ndims, axes=axis_to_dim),
        tf.keras.layers.InputSpec(shape=[None, None], dtype=types.MASK_DTYPE),
    )

    if len(axis_to_dim) == 1:
      # Single axis batch norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [
          axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
      ]

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=self.trainable,
          experimental_autocast=False,
      )
    else:
      self.gamma = None

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=self.trainable,
          experimental_autocast=False,
      )
    else:
      self.beta = None

    # Disable variable partitioning when creating the moving mean and variance
    if hasattr(self, '_scope') and self._scope:
      partitioner = self._scope.partitioner
      self._scope.set_partitioner(None)
    else:
      partitioner = None
    try:
      self.moving_mean = self.add_weight(
          name='moving_mean',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.moving_mean_initializer,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN,
          experimental_autocast=False,
      )

      self.moving_variance = self.add_weight(
          name='moving_variance',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.moving_variance_initializer,
          synchronization=tf.VariableSynchronization.ON_READ,
          trainable=False,
          aggregation=tf.VariableAggregation.MEAN,
          experimental_autocast=False,
      )
    finally:
      if partitioner:
        self._scope.set_partitioner(partitioner)
    self.built = True

  def _assign_moving_average(self, variable, value, momentum, inputs_size):
    with tf.name_scope('AssignMovingAvg') as scope:
      with tf1.colocate_with(variable):
        decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
        if decay.dtype != variable.dtype.base_dtype:
          decay = tf.cast(decay, variable.dtype.base_dtype)
        update_delta = (variable - tf.cast(value, variable.dtype)) * decay
        if inputs_size is not None:
          update_delta = tf.where(inputs_size > 0, update_delta, 0)
        return tf1.assign_sub(variable, update_delta, name=scope)

  def _assign_new_value(self, variable, value):
    with tf.name_scope('AssignNewValue') as scope:
      with tf1.colocate_with(variable):
        return tf.assign(variable, value, name=scope)

  def _maybe_cross_replica_sum(self, values):
    # Don't sum across replicas if disabled by the caller.
    if not self._use_cross_replica_sum:
      return values

    # In TF1 mode on a TPU, use tf1.tpu.cross_replica_sum.
    if _is_tf1_tpu():
      return tf1.tpu.cross_replica_sum(values)

    # If we're in a DistributionStrategy replica context, use all_reduce.
    replica_context = tf.distribute.get_replica_context()
    if replica_context is not None:
      return replica_context.all_reduce(tf.distribute.ReduceOp.SUM, values)

    # Otherwise leave values unchanged.
    return values

  def _moments(self, inputs, mask, reduction_axes, keep_dims):
    """Compute the moments of inputs/input_lengths over reduction_axes.

    This implementation follows the version in:
    https://github.com/tensorflow/lingvo/blob/master/lingvo/core/bn_layers.py

    Args:
      inputs: A [batch_size, time, ...] tensor of sequences.
      mask: A [batch_size, time, 1...] mask tensor containing 1.0 for valid
        timesteps.
      reduction_axes: The axes to reduce over for computing the moments.
      keep_dims: If true, preserve reduction_axes in the output moments.

    Returns:
      means: The means of inputs along the axes indicated in
        reduction_axes. For example, if the input is [a, b, c, d] and
        reduction_axes is [0, 1, 3], means's shape is [c] if keep_dims is
        False, and [1, 1, c, 1] if keep_dims is True.
      variances: The variance of inputs along the axes indicated in
        reduction_axes. For example, if the input is [a, b, c, d] and
        reduction_axes is [0, 1, 3], variance's shape is [c] if keep_dims is
        False, and [1, 1, c, 1] if keep_dims is True.
    """
    # This method assumes we are reducing over the batch and time dimension of
    # the sequence.
    assert 0 in reduction_axes and 1 in reduction_axes

    # TODO(rryan): We should be able to assume inputs are masked.
    sum_v = tf.reduce_sum(inputs * mask, reduction_axes, keepdims=True)
    count_v = tf.reduce_sum(mask, reduction_axes, keepdims=True)

    # Multiply counts by non-batch/time reduction dimensions.
    if len(reduction_axes) > 2:
      counts_multiplier = tf.gather(
          tf.shape(inputs), [a for a in reduction_axes if a not in (0, 1)]
      )
      count_v *= tf.cast(tf.reduce_prod(counts_multiplier), count_v.dtype)

    # Aggregate sums and counts across replicas if enabled.
    sum_v = self._maybe_cross_replica_sum(sum_v)
    count_v = self._maybe_cross_replica_sum(count_v)

    count_v = tf.maximum(count_v, 1.0)
    mean = sum_v / count_v
    sum_vv = tf.reduce_sum(
        tf.square(inputs - mean) * mask, reduction_axes, keepdims=keep_dims
    )

    # Aggregate summed squares across replicas if enabled.
    sum_vv = self._maybe_cross_replica_sum(sum_vv)

    if not keep_dims:
      count_v = tf.squeeze(count_v, reduction_axes)
      mean = tf.squeeze(mean, reduction_axes)

    variance = sum_vv / count_v
    # TODO(b/129279393): Support zero batch input in non DistributionStrategy
    # code as well.
    if self._support_zero_size_input():
      input_batch_size = tf.size(inputs)
      mean = tf.where(input_batch_size > 0, mean, 0)
      variance = tf.where(input_batch_size > 0, variance, 0)
    return mean, variance

  def _get_training_value(self, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    return training

  def call(self, inputs_and_mask, training=None):
    inputs, mask = inputs_and_mask
    expanded_mask = types.Sequence(inputs, mask).expanded_mask()
    training = self._get_training_value(training)

    inputs_dtype = inputs.dtype.base_dtype
    if inputs_dtype in (tf.float16, tf.bfloat16):
      # Do all math in float32 if given 16-bit inputs for numeric stability.
      # In particular, it's very easy for variance to overflow in float16 and
      # for safety we also choose to cast bfloat16 to float32.
      inputs = tf.cast(inputs, tf.float32)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

    def _broadcast(v):
      if (
          v is not None
          and len(v.shape) != ndims
          and reduction_axes != list(range(ndims - 1))
      ):
        return tf.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf.get_static_value(training)
    if training_value == False:  # pylint: disable=singleton-comparison,g-explicit-bool-comparison
      mean, variance = self.moving_mean, self.moving_variance
    else:
      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = len(self.axis) > 1
      mean, variance = self._moments(
          tf.cast(inputs, self._param_dtype),
          expanded_mask,
          reduction_axes,
          keep_dims=keep_dims,
      )

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf.__internal__.smart_cond.smart_cond(
          training, lambda: mean, lambda: tf.convert_to_tensor(moving_mean)
      )
      variance = tf.__internal__.smart_cond.smart_cond(
          training,
          lambda: variance,
          lambda: tf.convert_to_tensor(moving_variance),
      )

      new_mean, new_variance = mean, variance

      if self._support_zero_size_input():
        # Keras assumes that batch dimension is the first dimension for Batch
        # Normalization.
        input_batch_size = tf.shape(inputs)[0]
      else:
        input_batch_size = None

      def _do_update(var, value):
        """Compute the updates for mean and variance."""
        return self._assign_moving_average(
            var, value, self.momentum, input_batch_size
        )

      mean_update_op = [None]
      def mean_update():
        true_branch = lambda: _do_update(self.moving_mean, new_mean)
        false_branch = lambda: self.moving_mean
        mean_update = mean_update_op[0] = tf.__internal__.smart_cond.smart_cond(
            training, true_branch, false_branch
        )
        return mean_update

      variance_update_op = [None]
      def variance_update():
        """Update the moving variance."""
        true_branch = lambda: _do_update(self.moving_variance, new_variance)
        false_branch = lambda: self.moving_variance
        variance_update = variance_update_op[0] = (
            tf.__internal__.smart_cond.smart_cond(
                training, true_branch, false_branch
            )
        )
        return variance_update

      # Keras stopped populating tf1.GraphKeys.UPDATE_OPS at some point.
      # TODO(b/287668425): Remove when we delete TF1 support.
      self.add_update(mean_update)
      self.add_update(variance_update)
      if not tf1.executing_eagerly_outside_functions():
        tf1.add_to_collection(tf1.GraphKeys.UPDATE_OPS, mean_update_op[0])
        tf1.add_to_collection(tf1.GraphKeys.UPDATE_OPS, variance_update_op[0])

    mean = tf.cast(mean, inputs.dtype)
    variance = tf.cast(variance, inputs.dtype)
    if offset is not None:
      offset = tf.cast(offset, inputs.dtype)
    if scale is not None:
      scale = tf.cast(scale, inputs.dtype)
    outputs = tf.nn.batch_normalization(
        inputs,
        _broadcast(mean),
        _broadcast(variance),
        offset,
        scale,
        self.epsilon,
    )
    if inputs_dtype in (tf.float16, tf.bfloat16):
      outputs = tf.cast(outputs, inputs_dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self._mask_output:
      outputs *= expanded_mask

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': tf.keras.initializers.serialize(
            self.beta_initializer
        ),
        'gamma_initializer': tf.keras.initializers.serialize(
            self.gamma_initializer
        ),
        'moving_mean_initializer': tf.keras.initializers.serialize(
            self.moving_mean_initializer
        ),
        'moving_variance_initializer': tf.keras.initializers.serialize(
            self.moving_variance_initializer
        ),
        'beta_regularizer': tf.keras.regularizers.serialize(
            self.beta_regularizer
        ),
        'gamma_regularizer': tf.keras.regularizers.serialize(
            self.gamma_regularizer
        ),
        'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
        'gamma_constraint': tf.keras.constraints.serialize(
            self.gamma_constraint
        ),
        'use_cross_replica_sum': self._use_cross_replica_sum,
        'mask_output': self._mask_output,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchNormalization(types.PreservesShape, types.Stateless):
  """Applies batch normalization to the channels dimensions of input sequences.

  In training mode this layer computes statistics from valid sequence timesteps
  and uses a cross-replica sum for aggregating moments across replicas if
  tf.distribute.get_replica_context() is non-None. Neither of these are the
  case for tf.keras.layers.BatchNormalization, so it is not a drop-in
  replacement.

  Step-wise training is not supported, since it cannot be made identical to
  layer-wise training (it's anti-causal, since it relies on statistics of future
  timesteps). When not training, the calculation performed by this layer is
  causal, since it only relies on statistics learned in training.
  """

  def __init__(
      self,
      axis=-1,
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable=True,
      name=None,
  ):
    super().__init__(name=name)
    self._layer = None
    self._layer_kwargs = {
        'axis': axis,
        'momentum': momentum,
        'epsilon': epsilon,
        'center': center,
        'scale': scale,
        'beta_initializer': beta_initializer,
        'gamma_initializer': gamma_initializer,
        'moving_mean_initializer': moving_mean_initializer,
        'moving_variance_initializer': moving_variance_initializer,
        'beta_regularizer': beta_regularizer,
        'gamma_regularizer': gamma_regularizer,
        'beta_constraint': beta_constraint,
        'gamma_constraint': gamma_constraint,
        'trainable': trainable,
        'use_cross_replica_sum': True,
        'mask_output': True,
    }

  def build(self, input_shape: tf.TensorShape):
    if self._layer is None:
      self._layer_kwargs['axis'] = _validate_and_normalize_axes(
          self._layer_kwargs['axis'], input_shape
      )
      with self.name_scope as name_scope:
        self._layer = _SequenceBatchNormalization(
            **self._layer_kwargs, name=name_scope
        )

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants = None,
  ) -> Tuple[types.Sequence, types.State]:
    # There are too many caveats that prevent this from being correct. Just
    # disallow it.
    if training:
      raise ValueError(
          'Step-wise training is not supported for BatchNormalization.'
      )
    return self.layer(x, training, initial_state=state), state

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State = None,
      constants: types.Constants = None,
  ) -> types.Sequence:
    self.build(x.values.shape)
    # _SequenceBatchNormalization masks the output.
    values = self._layer((x.values, x.mask), training=training)
    return types.Sequence(values, x.mask)


class _SequenceInstanceNormalization(tf.keras.layers.Layer):
  r"""A sequence-aware instance normalization layer.

  Ulyanov, Veldadi and Lemitsky: https://arxiv.org/abs/1607.08022

  Normalizes and scales each channel across all but the batch and axis
  dimensions, accounting for padding of the sequence dimension.

  For an input x_{ntsc}, axis={c} with lengths l_t instance normalization
  computes:

  y_{ntsc} = (x_{ntsc} - \mu_{nc}) / sqrt(\sigma_{nc}^2 + eps)

  Where:
    T = \sum_t l_t
    \mu_{nc} = (\sum_{ts} x_{ntsc}) / (T * S)
    \sigma_{nc}^2 = (\sum_{ts} (x_{ntsc} - \mu_{nc})^2) / (T * S)

  In other words, the normalization mu and sigma are specific to each batch
  element and channel c. To contrast with batch normalization, the normalization
  mu and sigma would be computed across batch items in training (and at test
  time, a moving average would be used). Since this layer is sequence-aware,
  only valid timesteps in x contribute to \mu_{nc} and \sigma_{nc}.
  """

  def __init__(
      self,
      axis: Union[int, Sequence[int]] = -1,
      epsilon: float = 0.001,
      center: bool = True,
      scale: bool = True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable: bool = True,
      mask_output: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if isinstance(axis, (list, tuple)):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('axis must be int or tuple/list, got: %s' % axis)
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)
    self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
    self.supports_masking = True
    self._trainable = trainable
    self._mask_output = mask_output

  @property
  def trainable(self):
    return self._trainable

  @property
  def _param_dtype(self):
    # Raise parameters of fp16 batch norm to fp32
    if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
      return tf.float32
    else:
      return self.dtype or tf.float32

  def _moments(self, inputs, mask, reduction_axes):
    # This method assumes we are reducing over the time dimension of the
    # sequence.
    assert 0 not in reduction_axes
    assert 1 in reduction_axes

    # TODO(rryan): We should be able to assume inputs are masked.
    sum_v = tf.reduce_sum(inputs * mask, reduction_axes, keepdims=True)
    count_v = tf.reduce_sum(mask, reduction_axes, keepdims=True)

    # Multiply counts by non-batch/time reduction dimensions.
    if len(reduction_axes) > 1:
      counts_multiplier = tf.gather(
          tf.shape(inputs), [a for a in reduction_axes if a != 1]
      )
      count_v *= tf.cast(tf.reduce_prod(counts_multiplier), count_v.dtype)

    count_v = tf.maximum(count_v, 1.0)
    mean = sum_v / count_v
    sum_vv = tf.reduce_sum(
        tf.square(inputs - mean) * mask, reduction_axes, keepdims=True
    )
    variance = sum_vv / count_v
    return mean, variance

  def build(self, input_and_mask_shape):
    input_shape, _ = input_and_mask_shape
    input_shape = tf.TensorShape(input_shape)

    if not input_shape.ndims:
      raise ValueError('Input has undefined rank:', input_shape)
    ndims = len(input_shape)

    # Convert axis to list and resolve negatives
    if isinstance(self.axis, int):
      self.axis = [self.axis]

    for idx, x in enumerate(self.axis):
      if x < 0:
        self.axis[idx] = ndims + x

    # Validate axes
    for x in self.axis:
      if x < 0 or x >= ndims:
        raise ValueError('Invalid axis: %d' % x)
    if len(self.axis) != len(set(self.axis)):
      raise ValueError('Duplicate axis: %s' % self.axis)
    if 0 in self.axis or 1 in self.axis:
      raise ValueError(
          'This implementation assumes axis 0 is batch and 1 is '
          'time. Computing InstanceNormalization on either of '
          'these dimensions is not supported. Got axis=%s'
          % self.axis
      )

    axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
    for x in axis_to_dim:
      if axis_to_dim[x] is None:
        raise ValueError(
            'Input has undefined `axis` dimension. Received input '
            'with shape %s. Axis value: %s' % (tuple(input_shape), self.axis)
        )
    self.input_spec = (
        tf.keras.layers.InputSpec(ndim=ndims, axes=axis_to_dim),
        tf.keras.layers.InputSpec(shape=[None, None], dtype=types.MASK_DTYPE),
    )

    if len(axis_to_dim) == 1:
      # Single axis instance norm (most common/default use-case)
      param_shape = (list(axis_to_dim.values())[0],)
    else:
      # Parameter shape is the original shape but with 1 in all non-axis dims
      param_shape = [
          axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)
      ]

    if self.scale:
      self.gamma = self.add_weight(
          name='gamma',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.gamma_initializer,
          regularizer=self.gamma_regularizer,
          constraint=self.gamma_constraint,
          trainable=self.trainable,
          experimental_autocast=False,
      )
    else:
      self.gamma = None

    if self.center:
      self.beta = self.add_weight(
          name='beta',
          shape=param_shape,
          dtype=self._param_dtype,
          initializer=self.beta_initializer,
          regularizer=self.beta_regularizer,
          constraint=self.beta_constraint,
          trainable=self.trainable,
          experimental_autocast=False,
      )
    else:
      self.beta = None
    self.built = True

  def call(self, inputs_and_mask):
    inputs, mask = inputs_and_mask
    expanded_mask = types.Sequence(inputs, mask).expanded_mask()

    inputs_dtype = inputs.dtype.base_dtype
    if inputs_dtype in (tf.float16, tf.bfloat16):
      # Do all math in float32 if given 16-bit inputs for numeric stability.
      # In particular, it's very easy for variance to overflow in float16 and
      # for safety we also choose to cast bfloat16 to float32.
      inputs = tf.cast(inputs, tf.float32)

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)
    # Don't reduce over batch or axis dimensions.
    reduction_axes = [i for i in range(1, ndims) if i not in self.axis]

    # Broadcasting only necessary for single-axis instance norm where the axis
    # is not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

    def _broadcast(v):
      if (
          v is not None
          and len(v.shape) != ndims
          and reduction_axes != list(range(ndims - 1))
      ):
        return tf.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    mean, variance = self._moments(
        tf.cast(inputs, self._param_dtype), expanded_mask, reduction_axes
    )

    mean = tf.cast(mean, inputs.dtype)
    variance = tf.cast(variance, inputs.dtype)
    if offset is not None:
      offset = tf.cast(offset, inputs.dtype)
    if scale is not None:
      scale = tf.cast(scale, inputs.dtype)
    outputs = tf.nn.batch_normalization(
        inputs,
        _broadcast(mean),
        _broadcast(variance),
        offset,
        scale,
        self.epsilon,
    )
    if inputs_dtype in (tf.float16, tf.bfloat16):
      outputs = tf.cast(outputs, inputs_dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self._mask_output:
      outputs *= expanded_mask

    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'axis': self.axis,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': tf.keras.initializers.serialize(
            self.beta_initializer
        ),
        'gamma_initializer': tf.keras.initializers.serialize(
            self.gamma_initializer
        ),
        'beta_regularizer': tf.keras.regularizers.serialize(
            self.beta_regularizer
        ),
        'gamma_regularizer': tf.keras.regularizers.serialize(
            self.gamma_regularizer
        ),
        'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
        'gamma_constraint': tf.keras.constraints.serialize(
            self.gamma_constraint
        ),
        'mask_output': self._mask_output,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class InstanceNormalization(types.PreservesShape, types.Stateless):
  r"""Applies instance normalization to the specified dimensions of inputs.

  Ulyanov, Veldadi and Lemitsky: https://arxiv.org/abs/1607.08022

  Normalizes and scales each channel across all but the batch and axis
  dimensions, accounting for padding of the sequence dimension.

  For an input x_{ntsc}, axis={c} with lengths l_t instance normalization
  computes:

  y_{ntsc} = (x_{ntsc} - \mu_{nc}) / sqrt(\sigma_{nc}^2 + eps)

  Where:
    T = \sum_t l_t
    \mu_{nc} = (\sum_{ts} x_{ntsc}) / (T * S)
    \sigma_{nc}^2 = (\sum_{ts} (x_{ntsc} - \mu_{nc})^2) / (T * S)

  In other words, the normalization mu and sigma are specific to each batch
  element and channel c. To contrast with batch normalization, the normalization
  mu and sigma would be computed across batch items in training (and at test
  time, a moving average would be used). Since this layer is sequence-aware,
  only valid timesteps in x contribute to \mu_{nc} and \sigma_{nc}.
  """

  def __init__(
      self,
      axis: Union[int, Sequence[int]] = -1,
      epsilon: float = 0.001,
      center: bool = True,
      scale: bool = True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      trainable: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._layer = None
    self._layer_kwargs = {
        'axis': axis,
        'epsilon': epsilon,
        'center': center,
        'scale': scale,
        'beta_initializer': beta_initializer,
        'gamma_initializer': gamma_initializer,
        'beta_regularizer': beta_regularizer,
        'gamma_regularizer': gamma_regularizer,
        'beta_constraint': beta_constraint,
        'gamma_constraint': gamma_constraint,
        'trainable': trainable,
        'mask_output': True,
    }

  def build(self, input_shape: tf.TensorShape):
    if self._layer is None:
      self._layer_kwargs['axis'] = _validate_and_normalize_axes(
          self._layer_kwargs['axis'], input_shape
      )
      with self.name_scope as name_scope:
        self._layer = _SequenceInstanceNormalization(
            **self._layer_kwargs, name=name_scope
        )

  @property
  def supports_step(self) -> bool:
    return False

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants = None,
  ) -> Tuple[types.Sequence, types.State]:
    raise ValueError(
        'Step-wise processing is not supported for InstanceNormalization.'
    )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: types.State = None,
      constants: types.Constants = None,
  ) -> types.Sequence:
    self.build(x.values.shape)
    # _SequenceInstanceNormalization masks the output.
    values = self._layer((x.values, x.mask), training=training)
    return types.Sequence(values, x.mask)


class L2Normalize(types.PreservesShape, types.Stateless):
  """L2 Normalize."""

  def __init__(self, axis: Union[int, List[int]], name: Optional[str] = None):
    super().__init__(name=name)
    self._axis = axis

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    _validate_and_normalize_axes(self._axis, x.values.shape)
    return x.apply_values(lambda v: tf.nn.l2_normalize(v, axis=self._axis))
