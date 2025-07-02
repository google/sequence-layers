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
"""Normalization layers."""

import dataclasses
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt
from sequence_layers.jax import utils


__all__ = (
    # go/keep-sorted start
    'BatchNormalization',
    'GroupNormalization',
    'LayerNormalization',
    'RMSNormalization',
    # go/keep-sorted end
)


def _guard_against_excess_precision(x: jax.Array) -> jax.Array:
  """Guards against "excess precision" introduced by XLA fusions."""
  finfo = jnp.finfo(x.dtype)  # jnp important!
  return jax.lax.reduce_precision(x, finfo.nexp, finfo.nmant)


def _validate_and_normalize_axes(
    axes: int | list[int] | tuple[int, ...],
    input_shape: types.Shape,
) -> tuple[int, ...]:
  """Normalizes user-provided axes and checks batch/time are not specified."""
  # If axis is left unspecified, normalize all dimensions.
  if isinstance(axes, int):
    axes = (axes,)
  else:
    normalized_axes = set()
    for axis in axes:
      if axis < 0:
        axis += len(input_shape)
      if axis < 0 or axis > len(input_shape) - 1:
        raise ValueError(
            f'Axis out of range {axis=} for {axes=} with {input_shape=}.'
        )
      normalized_axes.add(axis)
    axes = tuple(sorted(normalized_axes))
  for axis in axes:
    if axis in (0, 1):
      raise ValueError(
          'Normalizing over the batch or time dimension is '
          f'not allowed. Got: {axes}'
      )
  return axes


def _zero_gradient_helper(
    forward_fn: Callable[..., tuple[jt.AnyPyTree, jt.AnyPyTree]],
    *args: jt.AnyPyTree,
) -> jt.AnyPyTree:
  """Wrap forward_fn's gradient with safety checks that zero gradient.

  Args:
    forward_fn: The function to be wrapped with a custom gradient. The function
      returns a two-tuple of PyTrees (output, should_zero_gradient), where
      should_zero_gradient is a PyTree matching the structure of the *args
      array, indicating whether the pointwise gradient flowing to that argument
      should be zeroed.
    *args: The arguments to pass to forward_fn.

  Returns:
    Returns the first tuple element in the two-tuple return value of
    forward_fn(*args).
  """

  @jax.custom_gradient
  def forward_fn_custom_gradient(*args):
    values, vjp_fn, should_zero_gradients = jax.vjp(
        forward_fn, *args, has_aux=True
    )

    def custom_gradient(input_gradients: jt.AnyPyTree) -> jt.AnyPyTree:
      grads = vjp_fn(input_gradients)

      def maybe_zero_gradient(
          grad: jax.Array | types.Sequence, should_zero_gradient: jax.Array
      ):
        jnp.broadcast_shapes(grad.shape, should_zero_gradient.shape)
        if should_zero_gradient.ndim != grad.ndim:
          raise ValueError(
              f'Expected {should_zero_gradient.shape=} to be the same size is'
              f' {grad.shape=}.'
          )
        if isinstance(grad, types.Sequence):
          grad = grad.apply_values_masked(
              lambda g: jnp.where(should_zero_gradient, jnp.zeros_like(g), g)
          )
        else:
          grad = jnp.where(should_zero_gradient, jnp.zeros_like(grad), grad)
        return grad

      return jax.tree.map(
          maybe_zero_gradient,
          grads,
          should_zero_gradients,
          is_leaf=lambda x: isinstance(x, types.Sequence),
      )

    return values, custom_gradient

  return forward_fn_custom_gradient(*args)


class LayerNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies layer normalization to input sequences."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for LayerNormalization."""

    axis: int | types.ShapeLike = -1
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    scale_init: nn.initializers.Initializer = nn.initializers.ones_init()
    sharding: types.Sharding | None = None
    reductions_in_at_least_fp32: bool = True
    param_dtype: types.DType = jnp.float32
    # If true, guards against "excess precision" introduced by XLA fusions in
    # the backward pass. Equivalent to --noxla_allow_excess_precision but only
    # for the input to this layer.
    guard_against_excess_precision: bool = False
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'LayerNormalization':
      return LayerNormalization(self, name=self.name)

  config: Config

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    reduction_axes = _validate_and_normalize_axes(self.config.axis, x.shape)

    if self.config.guard_against_excess_precision:
      x = x.apply_values_masked(_guard_against_excess_precision)

    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in reduction_axes:
      feature_shape[ax] = x.shape[ax]
      reduced_feature_shape.append(x.shape[ax])

    @utils.maybe_in_at_least_fp32(self.config.reductions_in_at_least_fp32)
    def reductions(values: jax.Array) -> jax.Array:

      def forward_fn(values: jax.Array) -> tuple[jt.AnyPyTree, jt.AnyPyTree]:
        # Since layer normalization does not operate over batch and time, we can
        # skip masking the moment calculation.
        mean = jnp.mean(values, axis=reduction_axes, keepdims=True)
        variance = jnp.mean(
            jnp.square(values - mean), axis=reduction_axes, keepdims=True
        )
        # TODO(rryan): Evaluate the flax approach of applying scale before
        # broadcasting.
        normed = (values - mean) * jax.lax.rsqrt(variance + self.config.epsilon)

        # Guard against blowup from all-zero inputs.
        should_zero_gradient = variance == 0

        return normed, (should_zero_gradient,)

      return _zero_gradient_helper(forward_fn, values)

    y = reductions(x.values)

    # Apply scale and bias.
    if self.config.use_scale:
      scale_init = utils.shard_initializer(
          self.config.scale_init,
          self.config.sharding,
          labels=[meta.IS_NORMALIZER],
      )
      scale = self.param(
          'scale',
          scale_init,
          reduced_feature_shape,
          self.config.param_dtype,
      ).astype(y.dtype)
      y *= jnp.reshape(scale, feature_shape)

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.sharding
      )
      y += jnp.reshape(
          self.param(
              'bias',
              bias_init,
              reduced_feature_shape,
              self.config.param_dtype,
          ).astype(y.dtype),
          feature_shape,
      )

    # Normalization leaves padded regions unmasked.
    return types.Sequence(y, x.mask)


class RMSNormalization(types.PreservesType, types.StatelessPointwise):
  """A simplified version of LayerNormalization used in T5.

  No mean statistics or offset terms are included.

  Implementation follows:
  google3/third_party/tensorflow_models/official/nlp/modeling/models/t5.py

  As in flax.linen.RMSNorm, reductions are performed in at least float32 unless
  reductions_in_at_least_fp32=False. This is a difference with respect to many
  common RMSNorm implementations (e.g. Praxis). If interoperability is required,
  consider turning this off.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for RMSNormalization."""

    axis: int | types.ShapeLike = -1
    epsilon: float = 1e-6
    use_scale: bool = True
    scale_init: nn.initializers.Initializer = nn.initializers.ones_init()
    sharding: types.Sharding | None = None
    reductions_in_at_least_fp32: bool = True
    param_dtype: types.DType = jnp.float32
    # If true, guards against "excess precision" introduced by XLA fusions in
    # the backward pass. Equivalent to --noxla_allow_excess_precision but only
    # for the input to this layer.
    guard_against_excess_precision: bool = False
    name: str | None = None

    def __post_init__(self):
      # Use hashable types for sequences.
      if not isinstance(self.axis, int):
        object.__setattr__(self, 'axis', tuple(self.axis))

    def make(self) -> 'RMSNormalization':
      return RMSNormalization(self, name=self.name)

  config: Config

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    reduction_axes = _validate_and_normalize_axes(
        self.config.axis, x.values.shape
    )

    feature_shape = [1] * x.values.ndim
    reduced_feature_shape = []
    for ax in reduction_axes:
      feature_shape[ax] = x.values.shape[ax]
      reduced_feature_shape.append(x.values.shape[ax])

    if self.config.guard_against_excess_precision:
      x = x.apply_values_masked(_guard_against_excess_precision)

    @utils.maybe_in_at_least_fp32(self.config.reductions_in_at_least_fp32)
    def reductions(values: jax.Array) -> jax.Array:

      def forward_fn(values: jax.Array) -> tuple[jt.AnyPyTree, jt.AnyPyTree]:
        # Since RMS normalization does not operate over batch and time, we can
        # skip masking the moment calculation.
        mean_square = jnp.mean(
            jnp.square(values), axis=reduction_axes, keepdims=True
        )
        # TODO(rryan): Evaluate the flax approach of applying scale before
        # broadcasting.
        normed = values * jax.lax.rsqrt(mean_square + self.config.epsilon)

        # Guard against blowup from all-zero inputs.
        should_zero_gradient = mean_square == 0

        return normed, (should_zero_gradient,)

      return _zero_gradient_helper(forward_fn, values)

    y = reductions(x.values)

    # Apply scale and bias.
    if self.config.use_scale:
      scale_init = utils.shard_initializer(
          self.config.scale_init,
          self.config.sharding,
          labels=[meta.IS_NORMALIZER],
      )
      y *= jnp.reshape(
          self.param(
              'scale',
              scale_init,
              reduced_feature_shape,
              self.config.param_dtype,
          ).astype(y.dtype),
          feature_shape,
      )

    # Normalization leaves padded regions unmasked.
    return types.Sequence(y, x.mask)


class BatchNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies batch normalization to the channels dimensions of input sequences.

  In training mode this layer computes statistics from valid sequence timesteps
  and uses a cross-replica sum when running under pmap.

  Step-wise training is not supported, since it cannot be made identical to
  layer-wise training (it's not causal, since it relies on statistics of future
  timesteps). When not training, the calculation performed by this layer is
  causal, since it only relies on statistics learned in training.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Batch normalization config."""

    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    use_bias: bool = True
    use_scale: bool = True
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    scale_init: nn.initializers.Initializer = nn.initializers.ones_init()
    sharding: types.Sharding | None = None
    use_fast_variance: bool = True
    param_dtype: types.DType = jnp.float32
    # If true, guards against "excess precision" introduced by XLA fusions in
    # the backward pass. Equivalent to --noxla_allow_excess_precision but only
    # for the input to this layer.
    guard_against_excess_precision: bool = False
    name: str | None = None

    def make(self) -> 'BatchNormalization':
      return BatchNormalization(self, name=self.name)

  config: Config

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    # There are too many caveats that prevent this from being correct. Just
    # disallow it.
    if training:
      raise ValueError(
          'Step-wise training is not supported for BatchNormalization.'
      )
    return self.layer(x, training=training, constants=constants), state

  @nn.compact
  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    (axis,) = _validate_and_normalize_axes([self.config.axis], x.shape)

    scale_init = utils.shard_initializer(
        self.config.scale_init,
        self.config.sharding,
        labels=[meta.IS_NORMALIZER],
    )
    bias_init = utils.shard_initializer(
        self.config.bias_init, self.config.sharding
    )

    if self.config.guard_against_excess_precision:
      x = x.apply_values_masked(_guard_against_excess_precision)

    bn = nn.BatchNorm(
        use_running_average=None,
        axis=axis,
        momentum=self.config.momentum,
        epsilon=self.config.epsilon,
        dtype=x.dtype,
        param_dtype=self.config.param_dtype,
        use_bias=self.config.use_bias,
        use_scale=self.config.use_scale,
        bias_init=bias_init,
        scale_init=scale_init,
        use_fast_variance=self.config.use_fast_variance,
        name=self.config.name,
    )
    y = bn(x.values, use_running_average=not training, mask=x.expanded_mask())
    # Normalization leaves padded regions unmasked.
    return types.Sequence(y, x.mask)


def _cumulative_masked_moments_streaming(
    inputs: types.Sequence,
    reduction_axes: list[int],
    cumulative_axis: int,
    accum_sum_v: jax.Array,
    accum_count_v: jax.Array,
    accum_sum_vv: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Compute the moments of inputs/input_lengths over reduction_axes.

  This implementation follows Babelfish's in:
  google3/third_party/py/lingvo/core/bn_layers.py

  Args:
    inputs: A [batch_size, time, ...] sequence.
    reduction_axes: The axes to reduce over for computing the moments.
    cumulative_axis: If non-None, accumulates moments along this axis. Must not
      be an axis in reduction_axes.
    accum_sum_v: Streaming accumulator for sum_v.
    accum_count_v: Streaming accumulator for count_v.
    accum_sum_vv: Streaming accumulator for sum_vv.

  Returns:
    means: The means of inputs along the axes indicated in
      reduction_axes. For example, if the input is [a, b, c, d] and
      reduction_axes is [0, 1, 3], means's shape is [1, 1, c, 1].
    variances: The variance of inputs along the axes indicated in
      reduction_axes. For example, if the input is [a, b, c, d] and
      reduction_axes is [0, 1, 3], variance's shape is [1, 1, c, 1].
    accum_sum_v: Streaming accumulator for sum_v.
    accum_count_v: Streaming accumulator for count_v.
    accum_sum_vv: Streaming accumulator for sum_vv.
  """
  # Mask input if it isn't already.
  inputs = inputs.mask_invalid()
  mask = inputs.expanded_mask()
  assert accum_sum_v.shape[cumulative_axis] == 1
  assert accum_count_v.shape[cumulative_axis] == 1
  assert accum_sum_vv.shape[cumulative_axis] == 1
  sum_v = jnp.sum(inputs.values, reduction_axes, keepdims=True)
  count_v = jnp.sum(mask, reduction_axes, keepdims=True)

  # Multiply counts by non-batch/time reduction dimensions. This is more
  # efficient than broadcast the count_v sum above to the shape of inputs.
  non_batch_time_axes = [a for a in reduction_axes if a not in (0, 1)]
  if non_batch_time_axes:
    counts_multiplier = np.prod(
        [inputs.shape[ax] for ax in non_batch_time_axes]
    )
    count_v *= jnp.array(counts_multiplier, dtype=count_v.dtype)

  # Broadcast add the accumulators to the cumsum of the input timesteps.
  sum_v = accum_sum_v + jnp.cumsum(sum_v, axis=cumulative_axis)
  count_v = accum_count_v + jnp.cumsum(count_v, axis=cumulative_axis)

  # Keep the total sum/count up to the end of this chunk as the accumulator
  # state.
  accum_sum_v = jax.lax.slice_in_dim(
      sum_v, -1, limit_index=None, axis=cumulative_axis
  )
  accum_count_v = jax.lax.slice_in_dim(
      count_v, -1, limit_index=None, axis=cumulative_axis
  )

  count_v = jnp.maximum(count_v, 1.0)
  mean = sum_v / count_v
  sum_vv = jnp.sum(
      jnp.where(mask, jnp.square(inputs.values - mean), 0),
      reduction_axes,
      keepdims=True,
  )
  # Broadcast add the accumulators to the cumsum of the input timesteps.
  sum_vv = accum_sum_vv + jnp.cumsum(sum_vv, axis=cumulative_axis)

  # Keep the total sum/count up to the end of this chunk as the accumulator
  # state.
  accum_sum_vv = jax.lax.slice_in_dim(
      sum_vv, -1, limit_index=None, axis=cumulative_axis
  )

  variance = sum_vv / count_v
  return mean, variance, accum_sum_v, accum_count_v, accum_sum_vv


def _masked_moments(
    inputs: types.Sequence,
    reduction_axes: list[int],
    cumulative_axis: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Compute the moments of inputs/input_lengths over reduction_axes.

  This implementation follows Babelfish's in:
  google3/third_party/py/lingvo/core/bn_layers.py

  Args:
    inputs: A [batch_size, time, ...] sequence.
    reduction_axes: The axes to reduce over for computing the moments.
    cumulative_axis: If non-None, accumulates moments along this axis. Must not
      be an axis in reduction_axes.

  Returns:
    means: The means of inputs along the axes indicated in
      reduction_axes. For example, if the input is [a, b, c, d] and
      reduction_axes is [0, 1, 3], means's shape is [1, 1, c, 1].
    variances: The variance of inputs along the axes indicated in
      reduction_axes. For example, if the input is [a, b, c, d] and
      reduction_axes is [0, 1, 3], variance's shape is [1, 1, c, 1].
  """
  # Mask input if it isn't already.
  inputs = inputs.mask_invalid()
  mask = inputs.expanded_mask()
  if cumulative_axis is not None:
    assert cumulative_axis not in reduction_axes

  sum_v = jnp.sum(inputs.values, reduction_axes, keepdims=True)
  count_v = jnp.sum(mask, reduction_axes, keepdims=True)

  # Multiply counts by non-batch/time reduction dimensions. This is more
  # efficient than broadcast the count_v sum above to the shape of inputs.
  non_batch_time_axes = [a for a in reduction_axes if a not in (0, 1)]
  if non_batch_time_axes:
    counts_multiplier = np.prod(
        [inputs.shape[ax] for ax in non_batch_time_axes]
    )
    count_v *= jnp.array(counts_multiplier, dtype=count_v.dtype)

  if cumulative_axis is not None:
    sum_v = jnp.cumsum(sum_v, axis=cumulative_axis)
    count_v = jnp.cumsum(count_v, axis=cumulative_axis)

  count_v = jnp.maximum(count_v, 1.0)
  mean = sum_v / count_v
  sum_vv = jnp.sum(
      jnp.where(mask, jnp.square(inputs.values - mean), 0),
      reduction_axes,
      keepdims=True,
  )
  if cumulative_axis is not None:
    sum_vv = jnp.cumsum(sum_vv, axis=cumulative_axis)

  variance = sum_vv / count_v
  return mean, variance


class GroupNormalization(types.PreservesType, types.StatelessPointwise):
  """Applies group normalization to input sequences.

  https://arxiv.org/abs/1803.08494
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Config for GroupNormalization."""

    num_groups: int
    axis: int = -1
    epsilon: float = 1e-6
    cumulative: bool = False
    use_bias: bool = True
    use_scale: bool = True
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    scale_init: nn.initializers.Initializer = nn.initializers.ones_init()
    sharding: types.Sharding | None = None
    param_dtype: types.DType = jnp.float32
    # If true, guards against "excess precision" introduced by XLA fusions in
    # the backward pass. Equivalent to --noxla_allow_excess_precision but only
    # for the input to this layer.
    guard_against_excess_precision: bool = False
    name: str | None = None

    def make(self) -> 'GroupNormalization':
      if self.num_groups <= 0:
        raise ValueError(f'{self.num_groups=} must be positive.')
      return GroupNormalization(self, name=self.name)

  config: Config

  @property
  def supports_step(self) -> bool:
    return self.config.cumulative

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    return {0: (-np.inf, 0 if self.config.cumulative else np.inf)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if not self.supports_step:
      raise ValueError(
          'Step-wise processing requires cumulative mode for'
          ' GroupNormalization.'
      )
    assert self.config.cumulative
    (axis,) = _validate_and_normalize_axes(
        [self.config.axis], (None, None) + input_spec.shape
    )

    input_shape = (batch_size, 1) + input_spec.shape
    axis_size = input_shape[axis]
    if axis_size % self.config.num_groups != 0:
      raise ValueError(
          f'Input shape ({input_shape}) axis={axis} must be'
          f' divisible by {self.config.num_groups}'
      )

    group_size = axis_size // self.config.num_groups

    outer_indices, _, inner_indices = np.split(
        input_shape, indices_or_sections=[axis, axis + 1]
    )

    expanded_shape = (
        outer_indices.tolist()
        + [self.config.num_groups, group_size]
        + inner_indices.tolist()
    )

    dtype = jnp.promote_types(input_spec.dtype, jnp.float32)
    moments_shape = [
        expanded_shape[ax] if ax in (0, 1, axis) else 1
        for ax in range(len(expanded_shape))
    ]

    sum_v = jnp.zeros(moments_shape, dtype)
    count_v = jnp.zeros(moments_shape, dtype)
    sum_vv = jnp.zeros(moments_shape, dtype)

    return sum_v, count_v, sum_vv

  @nn.nowrap
  def _group(self, values: jax.Array, normalized_axis: int) -> jax.Array:
    axis_size = values.shape[normalized_axis]
    if axis_size % self.config.num_groups != 0:
      raise ValueError(
          f'Input shape ({values.shape}) axis={normalized_axis} must be'
          f' divisible by {self.config.num_groups}'
      )

    group_size = axis_size // self.config.num_groups

    outer_indices, _, inner_indices = np.split(
        values.shape, indices_or_sections=[normalized_axis, normalized_axis + 1]
    )

    expanded_shape = (
        outer_indices.tolist()
        + [self.config.num_groups, group_size]
        + inner_indices.tolist()
    )
    return jnp.reshape(values, expanded_shape)

  @nn.compact
  def _scale_and_shift(
      self,
      values: jax.Array,
      reduced_feature_shape: list[int],
      feature_shape: list[int],
  ) -> jax.Array:
    # Apply scale and bias.
    if self.config.use_scale:
      scale_init = utils.shard_initializer(
          self.config.scale_init,
          self.config.sharding,
          labels=[meta.IS_NORMALIZER],
      )
      values *= jnp.reshape(
          self.param(
              'scale',
              scale_init,
              reduced_feature_shape,
              self.config.param_dtype,
          ),
          feature_shape,
      )

    if self.config.use_bias:
      bias_init = utils.shard_initializer(
          self.config.bias_init, self.config.sharding
      )
      values += jnp.reshape(
          self.param(
              'bias',
              bias_init,
              reduced_feature_shape,
              self.config.param_dtype,
          ),
          feature_shape,
      )

    return values

  @types.check_step
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State]:
    if not self.supports_step:
      raise ValueError(
          'Step-wise processing requires cumulative mode for'
          ' GroupNormalization.'
      )
    assert self.config.cumulative

    (axis,) = _validate_and_normalize_axes([self.config.axis], x.shape)

    axis_size = x.shape[axis]
    feature_shape = [1] * x.ndim
    feature_shape[axis] = axis_size
    reduced_feature_shape = [axis_size]

    if self.config.guard_against_excess_precision:
      x = x.apply_values_masked(_guard_against_excess_precision)

    # GroupNorm's gradients are particularly sensitive, so we do most
    # operations (inc. scale and shift in higher precision).
    @utils.run_in_at_least_fp32
    def reductions(x: types.Sequence) -> jax.Array:
      nonlocal state

      state_sum_v, state_count_v, state_sum_vv = state
      # Grouping leaves the mask status of the input unchanged, since
      # axis > 1.
      grouped_x = x.apply_values_masked(self._group, axis)
      expanded_rank = x.values.ndim + 1

      # Reduce over all dimension but batch, time, and num_groups.
      reduction_axes = [
          ax for ax in range(expanded_rank) if ax not in (0, 1, axis)
      ]

      def forward_fn(
          grouped_x: types.Sequence,
          state_sum_v: jax.Array,
          state_count_v: jax.Array,
          state_sum_vv: jax.Array,
      ) -> tuple[jt.AnyPyTree, jt.AnyPyTree]:
        group_mean, group_variance, state_sum_v, state_count_v, state_sum_vv = (
            _cumulative_masked_moments_streaming(
                grouped_x,
                reduction_axes,
                cumulative_axis=1,
                accum_sum_v=state_sum_v,
                accum_count_v=state_count_v,
                accum_sum_vv=state_sum_vv,
            )
        )

        group_stddev_inv = jax.lax.rsqrt(group_variance + self.config.epsilon)
        grouped_values = (grouped_x.values - group_mean) * group_stddev_inv
        should_zero_gradient = (
            group_variance == 0.0,
            jnp.zeros(state_sum_v.shape, jnp.bool_),
            jnp.zeros(state_count_v.shape, jnp.bool_),
            jnp.zeros(state_sum_vv.shape, jnp.bool_),
        )
        return (
            (grouped_values, state_sum_v, state_count_v, state_sum_vv),
            should_zero_gradient,
        )

      grouped_values, state_sum_v, state_count_v, state_sum_vv = (  # pylint: disable=unbalanced-tuple-unpacking
          _zero_gradient_helper(
              forward_fn, grouped_x, state_sum_v, state_count_v, state_sum_vv
          )
      )

      state = (state_sum_v, state_count_v, state_sum_vv)
      # Combine num_groups and group_size.
      values = jnp.reshape(grouped_values, x.values.shape)
      return self._scale_and_shift(values, reduced_feature_shape, feature_shape)

    # Normalization leaves padded regions unmasked.
    return types.Sequence(reductions(x), x.mask), state

  @types.check_layer
  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    # Only one axis in [2, ndim) is supported.
    (axis,) = _validate_and_normalize_axes([self.config.axis], x.shape)

    axis_size = x.shape[axis]
    feature_shape = [1] * x.ndim
    feature_shape[axis] = axis_size
    reduced_feature_shape = [axis_size]

    if self.config.guard_against_excess_precision:
      x = x.apply_values_masked(_guard_against_excess_precision)

    # GroupNorm's gradients are particularly sensitive, so we do most
    # operations (inc. scale and shift in higher precision).
    @utils.run_in_at_least_fp32
    def reductions(x: types.Sequence) -> jax.Array:
      # Grouping leaves the mask status of the input unchanged, since
      # axis > 1.
      grouped_x = x.apply_values_masked(self._group, axis)
      expanded_rank = x.values.ndim + 1

      if self.config.cumulative:
        # Reduce over all dimension but batch, time, and num_groups.
        reduction_axes = [
            ax for ax in range(expanded_rank) if ax not in (0, 1, axis)
        ]
      else:
        # Reduce over all dimension but batch and num_groups.
        reduction_axes = [
            ax for ax in range(expanded_rank) if ax not in (0, axis)
        ]

      def forward_fn(
          grouped_x: types.Sequence,
      ) -> tuple[jt.AnyPyTree, jt.AnyPyTree]:
        group_mean, group_variance = _masked_moments(
            # Grouping leaves the mask status of the input unchanged, since axis
            # > 1.
            grouped_x,
            reduction_axes,
            cumulative_axis=1 if self.config.cumulative else None,
        )

        group_stddev_inv = jax.lax.rsqrt(group_variance + self.config.epsilon)
        grouped_values = (grouped_x.values - group_mean) * group_stddev_inv

        # Guard against blowup from all-zero inputs.
        should_zero_gradient = group_variance == 0.0

        return grouped_values, (should_zero_gradient,)

      grouped_values = _zero_gradient_helper(forward_fn, grouped_x)

      # Combine num_groups and group_size.
      values = jnp.reshape(grouped_values, x.values.shape)
      return self._scale_and_shift(values, reduced_feature_shape, feature_shape)

    # Normalization leaves padded regions unmasked.
    return types.Sequence(reductions(x), x.mask)
