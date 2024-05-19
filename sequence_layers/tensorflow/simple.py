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
"""Simple layers."""

import abc
import dataclasses
import fractions
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sequence_layers.tensorflow import dense
from sequence_layers.tensorflow import types
from sequence_layers.tensorflow import utils
import tensorflow.compat.v2 as tf


class Scale(types.StatelessPointwise):
  """Scales the input by a provided constant."""

  # PEP 484 says to use complex when you mean any numeric type.
  def __init__(self, scale: complex, name: Optional[str] = None):
    super().__init__(name=name)
    self._scale = scale

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    # No masking required since x is already masked and we're multiplying.
    return x.apply_values(lambda v: v * tf.cast(self._scale, v.dtype))


class Translate(types.StatelessPointwise):
  """Translates the input by a provided constant."""

  # PEP 484 says to use complex when you mean any numeric type.
  def __init__(self, shift: complex, name: Optional[str] = None):
    super().__init__(name=name)
    self._shift = shift

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    x = x.apply_values(lambda v: v + tf.cast(self._shift, v.dtype))
    return x.mask_invalid()


class Abs(types.StatelessPointwiseFunctor):
  """Absolute value."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.math.abs(values), mask

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    # The absolute value of complex numbers is a real magnitude.
    return input_dtype.real_dtype


class Tanh(types.StatelessPointwiseFunctor):
  """Tanh activation."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.tanh(values), mask


class Relu(types.StatelessPointwiseFunctor):
  """Relu activation."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.relu(values), mask


class LeakyRelu(types.StatelessPointwiseFunctor):
  """Leaky Relu activation."""

  def __init__(self, alpha: complex, name: Optional[str] = None):
    super().__init__(name=name)
    self._alpha = alpha

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.leaky_relu(values, alpha=self._alpha), mask


class Elu(types.StatelessPointwiseFunctor):
  """Elu activation."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.elu(values), mask


class Exp(types.StatelessPointwiseFunctor):
  """Exp."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.math.exp(values), mask


class Log(types.StatelessPointwiseFunctor):
  """Log."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    # TODO(rryan): There has to be a better way to clear NaNs from invalid
    # regions.
    mask_expanded = types.Sequence(values, mask).expanded_mask()
    return (
        tf.where(mask_expanded > 0.0, tf.math.log(values), 0),
        mask,
    )


class Sigmoid(types.StatelessPointwiseFunctor):
  """Sigmoid."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.sigmoid(values), mask


class Softplus(types.StatelessPointwiseFunctor):
  """Softplus."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.math.softplus(values), mask


class Softmax(types.StatelessPointwiseFunctor):
  """Softmax."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.math.softmax(values), mask


class Swish(types.StatelessPointwiseFunctor):
  """Swish."""

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.swish(values), mask


class Gelu(types.StatelessPointwiseFunctor):
  """Gaussian Error Linear Unit (GELU)."""

  def __init__(self, approximate: bool = False, name: str | None = None):
    super().__init__(name=name)
    self._approximate = approximate

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.nn.gelu(values, approximate=self._approximate), mask


class Cast(types.StatelessPointwiseFunctor):
  """Cast input values to the specified type."""

  def __init__(self, dtype, name: Optional[str] = None):
    super().__init__(name=name)
    self._dtype = dtype

  def fn(
      self, values: tf.Tensor, mask: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(values, self._dtype), mask

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    return self._dtype


class GatedUnit(types.Stateless):
  """Computes a generalized Gated Unit, reducing the input channels by 2x."""

  def __init__(
      self,
      feature_activation: Callable[[tf.Tensor], tf.Tensor] | None,
      gate_activation: Callable[[tf.Tensor], tf.Tensor] | None,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._feature_activation = feature_activation
    self._gate_activation = gate_activation

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    feature, gate = tf.split(x.values, 2, axis=-1)
    if self._feature_activation:
      feature = self._feature_activation(feature)
    if self._gate_activation:
      gate = self._gate_activation(gate)
    values = feature * gate
    return types.Sequence(values, x.mask).mask_invalid()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    channels_static = input_shape.dims[-1].value
    if channels_static is None:
      raise ValueError(
          '%s depends on input shape, but input has unknown '
          'channels dimension: %s' % (self, input_shape)
      )
    if channels_static % 2 != 0:
      raise ValueError(
          'Input to %s must have an even number of channels: %s'
          % (self, input_shape)
      )
    return input_shape[:-1].concatenate(channels_static // 2)


class GatedLinearUnit(GatedUnit):
  """Computes a Gated Linear Unit, reducing the input channels by 2x."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(None, tf.nn.sigmoid, name=name)


class GatedTanhUnit(GatedUnit):
  """Computes a Gated Tanh Unit, reducing the input channels by 2x."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(tf.nn.tanh, tf.nn.sigmoid, name=name)


class Dropout(types.StatelessPointwise):
  """Computes stateful (i.e. using stateful random ops) dropout."""

  def __init__(
      self,
      rate: float = 0.5,
      noise_shape: Optional[Sequence[Optional[int]]] = None,
      seed: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    with self.name_scope as name_scope:
      self._layer = tf.keras.layers.Dropout(
          rate, noise_shape=noise_shape, seed=seed, name=name_scope
      )

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    # No masking required since dropout only zeros values.
    # PyType does not think that Dropout.__call__ exists when passing it as the
    # map_fn to apply_values.
    return x.apply_values(lambda v: self._layer(v, training=training))


class DeterministicDropout(types.SequenceLayer):
  """Computes deterministic (i.e. using stateless random ops) dropout."""

  SEED_DTYPE = tf.int32

  def __init__(
      self,
      rate: float = 0.5,
      initial_seed_name: str = '',
      noise_shape: Optional[Sequence[Optional[int]]] = None,
      always_dropout: bool = False,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if not initial_seed_name:
      raise ValueError('initial_seed_name is required.')
    if rate < 0.0 or rate >= 1.0:
      raise ValueError(f'Rate must be in [0.0, 1.0). Got: {rate}')

    self._rate = rate
    self._initial_seed_name = initial_seed_name
    self._always_dropout = always_dropout

    if noise_shape:
      if any(d is not None and d <= 0 for d in noise_shape):
        raise ValueError(
            'noise_shape values should be positive integers or '
            f'None. Got: {noise_shape}'
        )
      if noise_shape[1] is not None:
        raise ValueError(
            'Specifying noise_shape for the time dimension is unsupported. '
            f'Set it to None. Got: {noise_shape}'
        )

    self._noise_shape = None if noise_shape is None else tuple(noise_shape)

  def get_output_shape(
      self, input_shape: tf.TensorShape, constants: types.Constants = None
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s depends on input shape, but input has unknown '
          'channels dimension: %s' % (self, input_shape)
      )
    return input_shape

  @tf.Module.with_name_scope
  def get_initial_state(
      self, x: types.Sequence, constants: types.Constants = None
  ) -> types.State:
    if self._rate > 0.0:
      return tf.constant(0, self.SEED_DTYPE, name='t')
    else:
      return ()

  def _validate(self, x: types.Sequence):
    if not x.values.dtype.is_floating:
      raise ValueError('Dropout required floating point sequences.')

  def _get_initial_seed(self, constants: types.Constants):
    if constants is None:
      raise ValueError('constants is required.')
    initial_seed = constants.get(self._initial_seed_name)
    if initial_seed is None:
      raise ValueError(
          f'Expected a {self._initial_seed_name} entry in constants. '
          f'Got: {constants}'
      )
    initial_seed = tf.cast(initial_seed, self.SEED_DTYPE)
    initial_seed.shape.assert_has_rank(0)
    return initial_seed

  def _get_seed(self, t: tf.Tensor, constants: types.Constants) -> tf.Tensor:
    initial_seed = self._get_initial_seed(constants)
    initial_seed = tf.broadcast_to(initial_seed, tf.shape(t))
    return tf.stack([initial_seed, t], axis=-1)

  def _get_noise_shape(self, input_shape: Sequence[Union[tf.Tensor, int]]):
    if self._noise_shape is None:
      return input_shape

    if len(self._noise_shape) != len(input_shape):
      raise ValueError(
          'Expected noise_shape to match input shape. '
          f'{self._noise_shape} != {input_shape}.'
      )

    # Use noise_shape[i] if it's non-None, else the original input shape.
    return tuple(
        n_d if n_d else i_d for n_d, i_d in zip(self._noise_shape, input_shape)
    )

  def _dropout(
      self, x: types.Sequence, t: tf.Tensor, constants: types.Constants
  ) -> Tuple[types.Sequence, types.State]:
    t = tf.convert_to_tensor(t, dtype=self.SEED_DTYPE)
    t.shape.assert_has_rank(0)

    input_shape = utils.smart_dimension_size(x.values)
    num_steps = input_shape[1]
    noise_shape = self._get_noise_shape(input_shape)

    ts = t + tf.range(num_steps, dtype=t.dtype)
    seeds = self._get_seed(ts, constants)

    def noise_fn(seed_t):
      noise_shape_t = (noise_shape[0],) + noise_shape[2:]
      return tf.random.stateless_uniform(
          noise_shape_t, seed=seed_t, dtype=x.values.dtype
      )

    # Manually unroll small numbers of steps when num_steps is statically known.
    if isinstance(num_steps, int) and num_steps < 128:
      noise = tf.stack([noise_fn(seeds[i]) for i in range(num_steps)], axis=1)
    else:
      noise = tf.map_fn(noise_fn, seeds, dtype=x.values.dtype)
      noise = tf.transpose(noise, [1, 0] + list(range(2, len(noise_shape))))

    assert self._rate > 0.0
    keep_rate = 1.0 - self._rate
    scale = tf.cast(1.0 / keep_rate, x.values.dtype)

    # Use >= to match tf.nn.dropout.
    keep = noise >= self._rate

    # No masking required since dropout only zeros values.
    y = x.apply_values(lambda v: v * scale * tf.cast(keep, v.dtype))

    if not tf.executing_eagerly():
      y.values.set_shape(x.values.shape)

    return y, t + num_steps

  @tf.Module.with_name_scope
  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: types.Constants = None,
  ) -> Tuple[types.Sequence, types.State]:
    self._validate(x)
    if (not self._always_dropout and not training) or self._rate == 0.0:
      return x, state

    return self._dropout(x, state, constants)

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    self._validate(x)
    if (not self._always_dropout and not training) or self._rate == 0.0:
      return x

    y, _ = self._dropout(x, 0, constants)
    return y


class TacotronDropoutAtEval(Dropout):
  """Computes dropout in both training and eval/inference, as in Tacotron.

  This is a nonstandard use of dropout, which you probably don't intend to use.
  USE WITH CAUTION.
  """

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    return super().layer(
        x, training=True, initial_state=initial_state, constants=constants
    )


@dataclasses.dataclass
class SampleSpec:
  shape: Any
  dtype: Any


class NoiseSampler(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def sample(self, spec: SampleSpec) -> tf.Tensor:
    pass


@dataclasses.dataclass
class UniformSampler(NoiseSampler):
  minval: float
  maxval: float

  def sample(self, spec: SampleSpec) -> tf.Tensor:
    return tf.random.uniform(
        spec.shape,
        tf.cast(self.minval, spec.dtype),
        tf.cast(self.maxval, spec.dtype),
        dtype=spec.dtype,
    )


@dataclasses.dataclass
class NormalSampler(NoiseSampler):
  mean: float
  stddev: float

  def sample(self, spec: SampleSpec) -> tf.Tensor:
    return tf.random.normal(
        spec.shape,
        tf.cast(self.mean, spec.dtype),
        tf.cast(self.stddev, spec.dtype),
        dtype=spec.dtype,
    )


@dataclasses.dataclass
class TruncatedNormalSampler(NoiseSampler):
  mean: float
  stddev: float

  def sample(self, spec: SampleSpec) -> tf.Tensor:
    return tf.random.truncated_normal(
        spec.shape,
        tf.cast(self.mean, spec.dtype),
        tf.cast(self.stddev, spec.dtype),
        dtype=spec.dtype,
    )


class Noise(types.StatelessPointwise):
  """Adds noise to inputs from a specified distribution.

  Uses the provided NoiseSampler instance to generate noise of a specified type
  and shape.
  """

  def __init__(
      self,
      sampler: NoiseSampler,
      training_only: bool,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._sampler = sampler
    self._training_only = training_only

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    # If we're only applying noise in training then skip if we're not training.
    if self._training_only and not training:
      return x
    noise = self._sampler.sample(SampleSpec(tf.shape(x.values), x.dtype))
    return x.apply_values(lambda v: v + noise).mask_invalid()


class Slice(types.Stateless):
  """Slices the channels dimensions of input tensors."""

  def __init__(self, the_slice=None, name: Optional[str] = None):
    super().__init__(name=name)
    self._slice = the_slice

  def __getitem__(self, the_slice):
    """Convenience method for setting the layer's slice."""
    # TODO(rryan): This might be too unintuitive, but making slice() objects
    # manually is also unintuitive.
    assert self._slice is None
    if isinstance(the_slice, slice):
      self._slice = (the_slice,)
    else:
      self._slice = the_slice
    return self

  def _validate_slice_for_input_shape(self, input_shape: tf.TensorShape):
    # None (tf.newaxis) dimensions are expansions.
    non_none_dimensions = sum(1 for s in self._slice if s is not None)
    if non_none_dimensions != input_shape.rank:
      raise ValueError(
          'Slice has wrong size for input: input_shape=%s slice=%s'
          % (input_shape, self._slice)
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s requires fully defined input shape: %s' % (self, input_shape)
      )

    self._validate_slice_for_input_shape(input_shape)
    output_dims = []
    input_index = 0

    # Compute the output shape:
    # - int: Remove the current input dimension.
    # - slice: Compute the output dimension size using slice.indices.
    # - None (tf.newaxis): Add a dimension.
    for the_slice in self._slice:
      if isinstance(the_slice, slice):
        # Use slice.indices(dim_size) to figure out the new length of the
        # dimension.
        output_dims.append(
            len(range(*the_slice.indices(input_shape.dims[input_index].value)))
        )
        input_index += 1
      elif isinstance(the_slice, int):
        # Skip this input dimension.
        input_index += 1
      elif the_slice is None:
        # Add a dimension.
        output_dims.append(1)
      else:
        raise NotImplementedError(
            f'Unsupported slice type: {type(the_slice)}, {the_slice}'
        )
    return tf.TensorShape(output_dims)

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    # Slice the batch and time dimensions with [:, :].
    full_slice = (
        slice(None, None, None),
        slice(None, None, None),
    ) + self._slice
    self._validate_slice_for_input_shape(x.channel_shape)
    return x.apply_values(lambda v: v.__getitem__(full_slice))


class Flatten(types.Stateless):
  """Flattens the channel dimensions of the input sequence.

  An input sequence with shape [batch_size, time, ...] is reshaped to
  [batch_size, time, prod(...)]. The mask is unchanged.
  """

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return tf.TensorShape(input_shape.num_elements())

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    batch_size, time = utils.smart_dimension_size(x.values, [0, 1])
    num_elements = x.channel_shape.num_elements()
    return x.apply_values(
        lambda v: tf.reshape(v, [batch_size, time, num_elements])
    )


class Transpose(types.Stateless):
  """Transposes the channel dimensions of the input sequence.

  E.g., with perm = [1, 0], the effective perm is [0, 1, 3, 2] to keep the batch
  and time dimensions as is, and only transpose the channel dimensions.
  """

  def __init__(
      self,
      perm: List[int],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if any([p < 0 for p in perm]):
      raise ValueError(f'Transpose does not accept negative indices: {perm=}')
    self._channel_perm = perm

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s requires fully defined input shape: %s' % (self, input_shape)
      )
    channel_perm_rank = len(self._channel_perm)
    input_channel_rank = input_shape.rank
    if channel_perm_rank != input_channel_rank:
      raise ValueError(f'{channel_perm_rank=} != {input_channel_rank}')
    return tf.TensorShape([input_shape[d] for d in self._channel_perm])

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    # Adjust permutation to take into account batch and time dimensions.
    perm = [0, 1] + [d + 2 for d in self._channel_perm]
    return x.apply_values(lambda v: tf.transpose(v, perm=perm))


class Reshape(types.Stateless):
  """Reshapes the channel dimensions of the input sequence.

  E.g., with shape = [4, 3], tensor of shape [8, 6, 2, 6] is reshaped to
  [8, 6, 4, 3].
  """

  def __init__(
      self,
      shape: Sequence[int],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    if all([s >= 0 for s in shape]):
      pass
    elif not (
        # Only one negative dim which is set to 1.
        (sum([s < 0 for s in shape]) == sum([s == -1 for s in shape]) == 1)
        # No dim is set to 0.
        and (all([s != 0 for s in shape]))
    ):
      raise ValueError(f'Invalid target shape: {shape}')
    self._shape = list(shape)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    shape = self._shape.copy()
    if any([s < 0 for s in shape]):
      shape[shape.index(-1)] = input_shape.num_elements() // np.prod(
          [s if s != -1 else 1 for s in shape]
      )
    if input_shape.num_elements() != tf.TensorShape(shape).num_elements():
      raise ValueError(f'Shape mismatch: {input_shape=}, target_shape={shape}.')
    return tf.TensorShape(shape)

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    batch, time = utils.smart_dimension_size(x.values, [0, 1])
    shape = tf.stack([batch, time] + self._shape)
    # Preserve channels dimension shape information when batch or time are
    # unknown.
    static_shape = [None, None] + self.get_output_shape_for_sequence(
        x, constants=constants
    )
    ret = x.apply_values(lambda v: tf.reshape(v, shape=shape))
    return ret.apply_values(lambda v: tf.ensure_shape(v, static_shape))


class Emit(types.StatelessEmitting):
  """An identity layer that emits its input."""

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape

  def get_emit_specs(
      self,
      input_spec: tf.TensorSpec,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    return types.Sequence(
        tf.TensorSpec(
            self.get_output_shape(input_spec.shape, constants), input_spec.dtype
        ),
        tf.TensorSpec(tf.TensorShape([]), types.MASK_DTYPE),
    )

  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.Sequence]:
    return x, x


class OneHot(types.Stateless):
  """Computes one-hot vector of the input."""

  def __init__(self, depth: int, name: Optional[str] = None):
    super().__init__(name)
    self._depth = depth

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape.concatenate(self._depth)

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    assert input_dtype in [tf.uint8, tf.int32, tf.int64]
    # TODO(soroosho): Add remaining arguments of tf.one_hot like on/off_value
    # and set output_dtype based on that. Default values are tf.float32.
    return utils.compute_dtype()

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    # "0" is mapped to "[1] + [0] * (depth - 1)", so we mask_invalid it.
    return x.apply_values(
        lambda v: tf.one_hot(v, self._depth, dtype=utils.compute_dtype())
    ).mask_invalid()


class Embedding(types.Stateless):
  """Computes embeddings of integer input codes."""

  def __init__(
      self,
      dimension: int,
      num_embeddings: int,
      embeddings_initializer='uniform',
      embeddings_regularizer=None,
      activity_regularizer=None,
      embeddings_constraint=None,
      trainable: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name)
    with self.name_scope:
      self._embedding = tf.keras.layers.Embedding(
          input_dim=num_embeddings,
          output_dim=dimension,
          embeddings_initializer=embeddings_initializer,
          embeddings_regularizer=embeddings_regularizer,
          activity_regularizer=activity_regularizer,
          embeddings_constraint=embeddings_constraint,
          trainable=trainable,
          name='embedding',
      )

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape.concatenate(self._embedding.output_dim)

  def get_output_dtype(self, input_dtype: tf.DType) -> tf.DType:
    assert input_dtype in [tf.uint8, tf.int32, tf.int64]
    return utils.compute_dtype()

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del initial_state
    return x.apply_values(
        lambda v: self._embedding(v, training=training)
    ).mask_invalid()


class StyleToken(types.Stateless):
  """Projects input onto a "style token" bottleneck.

  The inputs are used to predict a soft-weighting of a set of N style token
  vectors. This is equivalent to projecting the input onto an N-dimensional
  simplex formed by the style token vectors.

  Based on: https://arxiv.org/abs/1803.09017
  """

  def __init__(
      self,
      num_style_tokens: int,
      num_heads: int,
      units_per_head: int,
      name=None,
  ):
    super().__init__(name=name)
    self._num_heads = num_heads
    self._units_per_head = units_per_head
    with self.name_scope:
      # The original style tokens implementation used glorot_uniform
      # initialization, which initializes with U[-limit, limit],
      # limit = sqrt(6 / (num_style_tokens + units_per_head)).
      # The scale shouldn't depend on the number of tokens so we use fan_out
      # here to initialize with limit = sqrt(6 / units_per_head).
      self._style_tokens = tf.Variable(
          tf.keras.initializers.variance_scaling(
              mode='fan_out', distribution='uniform'
          )((num_style_tokens, units_per_head)),
          name='style_tokens',
      )
      # Traditionally the key vectors are a projection from the style tokens
      # themselves, but making the keys a disconnected variable is
      # equally (potentially more?) flexible and more efficient.
      # Since this variable represents a batch of projection matrices,
      # compute the limit as glorot_uniform for a
      # [num_style_tokens, units_per_head] matrix.
      limit = math.sqrt(6.0 / (num_style_tokens + units_per_head))
      self._style_token_keys = tf.Variable(
          tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)(
              (1, 1, num_heads, num_style_tokens, units_per_head)
          ),
          name='style_token_keys',
      )
      self._q_proj = dense.DenseShaped(
          [num_heads, 1, units_per_head], name='query_projection'
      )
      self._to_logits = dense.Dense(1, name='to_logits')

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return tf.TensorShape([self._num_heads, self._units_per_head])

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del initial_state
    del constants

    # Terminology for shapes below:
    # b: Batch size.
    # t: Query time.
    # n: Number of heads.
    # s: Number of style tokens.
    # h: Units per head.

    # Project x to per-head queries: [b, t, n, s=1, h]
    q = self._q_proj.layer(x, training)

    # Broadcast-add the key vector across batch, query time, and style tokens
    # and apply tanh nonlinearity. [b, t, n, s, h]
    q = q.apply_values(lambda v: tf.tanh(v + self._style_token_keys))

    # Project the inner h dimension to a single dimension, then squeeze.
    # [b, t, n, s]
    logits = self._to_logits.layer(q, training).apply_values(tf.squeeze, -1)

    # Normalize the logits to probabilities with softmax.
    # [b, t, n, s]
    probabilities = logits.apply_values(tf.nn.softmax)

    # Compute weighted sums of the style tokens.
    context_vector = probabilities.apply_values(
        lambda v: tf.einsum('BTNS,SH->BTNH', v, self._style_tokens)
    )
    return context_vector.mask_invalid()


class Identity(types.StatelessPointwise):
  """Identity pass-through of the input."""

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    return x


class ReverseSequence(types.SequenceLayer):
  """Reverses the sequence.

  Note: this layer does not support running step by step.
  """

  @property
  def block_size(self) -> int:
    raise NotImplementedError(
        'The block_size property is undefined for ReverseSequence since it '
        'does not support running step by step.'
    )

  def step(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      constants: Optional[types.Constants] = None,
  ) -> Tuple[types.Sequence, types.State]:
    raise NotImplementedError(
        'ReverseSequence does not support running step by step.'
    )

  def get_initial_state(
      self, x: types.Sequence, constants: Optional[types.Constants] = None
  ) -> types.State:
    return ()

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    if not input_shape.is_fully_defined():
      raise ValueError(
          '%s depends on input shape, but input has unknown '
          'channels dimension: %s' % (self, input_shape)
      )
    return input_shape

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    return x.apply_values(
        lambda v: tf.reverse_sequence(v, x.lengths(), seq_axis=1, batch_axis=0)
    )


class Upsample1D(types.Stateless):
  """Upsamples input by repeating each frame 'rate' times."""

  @classmethod
  def _default_name(cls):
    """The default module name for this class."""
    return 'upsample1d'

  def __init__(self, rate: int, name=None):
    super().__init__(name=name)
    self._rate = rate

  @property
  def output_ratio(self) -> fractions.Fraction:
    """The number of output frames for one input frame."""
    return fractions.Fraction(self._rate)

  def get_output_shape(
      self,
      input_shape: tf.TensorShape,
      constants: Optional[types.Constants] = None,
  ) -> tf.TensorShape:
    return input_shape

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    def _repeat(x):
      return tf.repeat(x, self._rate, axis=1)

    return x.apply(lambda x, mask: (_repeat(x), _repeat(mask)))


class Snake(types.StatelessPointwise):
  """The "snake" activation, i.e.

  x + 1/beta * sin^2(x*alpha) activation.

  Originally proposed in: https://arxiv.org/abs/2006.08195

  Follows the extension in BigVGAN: https://arxiv.org/abs/2206.04658
  which uses a separate beta, models a parameter per channel dimension, and
  learns the parameters in log scale.
  """

  def __init__(self, separate_beta: bool, name: Optional[str] = None):
    super().__init__(name=name)
    self._built = False
    self._separate_beta = separate_beta
    self._alpha = None
    self._beta = None

  def _build(self, channels_shape: tf.TensorShape):
    if self._built:
      return
    with self.name_scope:
      self._alpha = tf.Variable(lambda: tf.zeros(channels_shape), name='alpha')
      if self._separate_beta:
        self._beta = tf.Variable(lambda: tf.zeros(channels_shape), name='beta')
      self._built = True

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state
    self._build(x.channel_shape)

    alpha = tf.math.exp(self._alpha[tf.newaxis, tf.newaxis, ...])
    if self._separate_beta:
      beta = tf.math.exp(self._beta[tf.newaxis, tf.newaxis, ...])
    else:
      beta = alpha

    sin2_xa = tf.math.square(tf.math.sin(x.values * alpha))
    values = tf.math.divide(sin2_xa, beta + 1e-12)
    x = x.apply_values(lambda v: v + values).mask_invalid()
    return x


class AssertChannelSpec(types.StatelessPointwise):
  """Assert that the channel dimension has a particular shape and dtype."""

  def __init__(
      self,
      expected_channel_shape: tf.TensorShape | None = None,
      expected_dtype: tf.DType | None = None,
      name: str | None = None,
  ):
    super().__init__(name=name)
    self._expected_channel_shape = expected_channel_shape
    self._expected_dtype = expected_dtype

  @tf.Module.with_name_scope
  def layer(
      self,
      x: types.Sequence,
      training: bool,
      initial_state: Optional[types.State] = None,
      constants: Optional[types.Constants] = None,
  ) -> types.Sequence:
    del training
    del initial_state

    if (
        self._expected_channel_shape
        and x.channel_shape != self._expected_channel_shape
    ):
      raise ValueError(f'{x} does not have {self._expected_channel_shape=}')

    if self._expected_dtype and x.dtype != self._expected_dtype:
      raise ValueError(f'{x} does not have {self._expected_dtype=}')

    return x
