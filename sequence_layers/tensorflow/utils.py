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
"""Helper functions."""

import collections
import functools
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
from sequence_layers.tensorflow import types
import tensorflow.compat.v2 as tf


def squeeze_mask(mask: tf.Tensor, factor: int) -> tf.Tensor:
  with tf.name_scope('squeeze_mask'):
    batch_size, time = smart_dimension_size(mask, [0, 1])
    num_frames = (time + factor - 1) // factor
    pad_amount = num_frames * factor - time
    if isinstance(pad_amount, tf.Tensor) or pad_amount > 0:
      mask = tf.pad(mask, [[0, 0], [0, pad_amount]])
    mask = tf.reshape(mask, [batch_size, num_frames, factor])
    return tf.reduce_min(mask, axis=2)


def unsqueeze_mask(mask: tf.Tensor, factor: int) -> tf.Tensor:
  with tf.name_scope('unsqueeze_mask'):
    batch_size, time = smart_dimension_size(mask)
    return tf.reshape(
        tf.tile(mask[:, :, tf.newaxis], [1, 1, factor]),
        [batch_size, time * factor],
    )


def convolution_effective_kernel_size(
    kernel_size: int, dilation_rate: int
) -> int:
  """Returns kernel_size with dilation_rate holes inserted."""
  return (kernel_size - 1) * dilation_rate + 1


def convolution_padding_output_size(
    input_size: int,
    padding: str,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
) -> int:
  """Returns the output size for a convolution over input_size."""
  # Formula from: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
  # SAME: ceil(input_shape / stride)
  # VALID: ceil((input_shape - (kernel_size - 1) * dilation_rate) / stride)
  if padding in ('same', 'causal', 'reverse_causal'):
    # Ceiling division.
    return (input_size + stride - 1) // stride
  else:
    assert padding == 'valid'
    output_size = input_size - (kernel_size - 1) * dilation_rate
    # Ceiling division.
    output_size = (output_size + stride - 1) // stride
    return output_size


def convolution_explicit_padding(
    padding: str, kernel_size: int, dilation_rate: int
) -> tuple[int, int]:
  """Returns explicit padding amounts to achieve the desired pad mode."""
  effective_kernel_size = convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  if padding == types.PaddingMode.CAUSAL.value:
    return effective_kernel_size - 1, 0
  elif padding == types.PaddingMode.REVERSE_CAUSAL.value:
    return 0, effective_kernel_size - 1
  elif padding == types.PaddingMode.SAME.value:
    pad_amount = effective_kernel_size - 1
    left_padding = pad_amount // 2
    right_padding = pad_amount - left_padding
    return left_padding, right_padding
  elif padding == types.PaddingMode.VALID.value:
    return 0, 0
  else:
    raise ValueError(f'Unknown padding mode: {padding}')


def smart_dimension_size(
    tensor: tf.Tensor, dim: Optional[Union[int, Iterable[int]]] = None
):
  """Return the sizes of one or more dimensions.

  If the dimension size is known at graph construction time, then that value is
  returned as an integer. Otherwise, the dynamic run-time size is returned as a
  Tensor.

  Args:
    tensor: Tensor for which dimension sizes are to be computed.
    dim: A Python int or iterable of ints, the dimensions to get the size of.
      Integers may be negative to index from the back.  If `None`, then the size
      of all dimensions is returned.

  Returns:
    output: If dim is an int, a single value (int or scalar Tensor) is returned.
      If dim is an iterable of ints, a tuple is returned containing the size
      value for all dimensions in dim. If dim is `None`, then a tuple is
      returned containing the size value for all dimensions.

  Raises:
    ValueError: If dim=None and tensor.shape.rank is None.
  """
  shape_tensor = tf.shape(tensor)

  def get_dim(dim):
    if tensor.shape.rank is not None:
      dim_value = tf.compat.dimension_value(tensor.shape[dim])
      if dim_value is not None:
        return dim_value
    return shape_tensor[dim]

  if dim is None:
    if tensor.shape.rank is None:
      raise ValueError(
          f'{tensor} rank must be known to extract all dimensions.'
      )
    return tuple(get_dim(d) for d in range(tensor.shape.rank))

  # PyType is confused by using try/except TypeError to attempt to iterate dim.
  if isinstance(dim, collections.abc.Iterable):
    return tuple(get_dim(d) for d in dim)
  else:
    return get_dim(dim)


def step_by_step_fn_static(
    transition_fn,
    num_blocks: int,
    input_block_size: int,
    x: types.Sequence,
    state: types.State,
) -> Tuple[types.Sequence, types.State, types.Emits]:
  """Sequentially applies transition_fn to blocks of x with static control flow.

  Args:
    transition_fn: A callable that takes a (Sequence, State) and returns a
      (Sequence, State, Emits).
    num_blocks: The number of blocks to split x into.
    input_block_size: The size of each block.
    x: The input (batch major) sequence to process.
    state: The initial state.

  Returns:
    y: The output (batch major) sequence.
    state: The state produced from the last block.
    emits: The emits produced by transition_fn stacked across time.
  """
  # If static unrolling, run a for loop over num_blocks, strided slice
  # the block from x, and run transition_fn to get the output block
  # and next state.
  output_blocks = []
  output_emits = []

  def read_block(b: int) -> types.Sequence:
    start = b * input_block_size
    end = start + input_block_size
    return x[:, start:end]

  for b in range(num_blocks):
    x_block = read_block(b)
    y_block, state, y_emit = transition_fn(x_block, state)
    output_blocks.append(y_block)
    output_emits.append(y_emit)

  # Concatenate all timesteps.
  output = types.Sequence.concatenate_sequences(output_blocks)
  output_emits = tf.nest.map_structure(
      lambda *ts: tf.concat(ts, axis=1), *output_emits
  )
  return output, state, output_emits


def _transpose_batch_and_time(
    values: tf.Tensor, indices: List[int]
) -> tf.Tensor:
  """Swaps the batch and time dimension of values."""
  return tf.transpose(values, [1, 0] + indices)


def _ta_scatter(ta, indices, values) -> tf.TensorArray:
  """TODO(b/153504706): Replace with ta.scatter(indices, values)."""
  for i in range(int(indices.shape[0])):
    ta = ta.write(tf.gather(indices, i), tf.gather(values, i))
  return ta


def _ta_gather(ta, indices) -> tf.Tensor:
  """TODO(b/153504706): Replace with ta.gather(indices)."""
  reads = []
  for i in range(int(indices.shape[0])):
    reads.append(ta.read(tf.gather(indices, i)))
  return tf.stack(reads)


def step_by_step_fn_dynamic(
    transition_fn,
    input_time: Union[tf.Tensor, int],
    input_shape: tf.TensorShape,
    output_spec: tf.TensorSpec,
    emit_specs: types.EmitSpecs,
    num_blocks: Union[tf.Tensor, int],
    input_block_size: int,
    output_block_size: int,
    x: types.Sequence,
    state: types.State,
) -> Tuple[types.Sequence, types.State, types.Emits]:
  """Sequentially applies transition_fn to blocks of x with dynamic controlflow.

  Args:
    transition_fn: A callable that takes a (Sequence, State) and returns a
      (Sequence, State, Emits).
    input_time: The length of the input sequence.
    input_shape: The shape of the channel dimensions of the input.
    output_spec: A tf.TensorSpec for the channel dimensions of the output.
    emit_specs: A nested structure of tf.TensorSpec produced by transition_fn.
    num_blocks: The number of blocks to split x into.
    input_block_size: The size of each input block.
    output_block_size: The size of each output block.
    x: The input (batch major) sequence to process.
    state: The initial state.

  Returns:
    y: The output (batch major) sequence.
    state: The state produced from the last block.
    emits: The emits produced by the transition_fn stacked across time.
  """
  # Create a time-major TensorArray to store x's values and mask.
  # TODO(rryan): Benchmark with and without a TensorArray for reads.
  input_values_ta = tf.TensorArray(dtype=x.values.dtype, size=input_time)
  input_mask_ta = tf.TensorArray(dtype=x.mask.dtype, size=input_time)

  # The indices of the inner-dimensions of the input and output used for
  # transposing batch and time dimensions of the input and output below.
  input_shape_indices = list(range(2, 2 + input_shape.rank))
  output_shape_indices = list(range(2, 2 + output_spec.shape.rank))
  # Use TensorShape for the indices since nest doesn't flatten it.
  emit_shapes_indices = tf.nest.map_structure(
      lambda s: tf.TensorShape(range(2, 2 + s.shape.rank)), emit_specs
  )

  # Transpose values and mask to time-major and unstack into the TensorArrays.
  input_values_time_major = _transpose_batch_and_time(
      x.values, input_shape_indices
  )
  input_mask_time_major = _transpose_batch_and_time(x.mask, [])
  input_values_ta = input_values_ta.unstack(input_values_time_major)
  input_mask_ta = input_mask_ta.unstack(input_mask_time_major)

  # Create time-major output TensorArrays to store blocks of the output
  # sequence. The below while loop writes output_block_size entries at a time.
  output_time = num_blocks * output_block_size
  static_batch_size = x.values.shape.dims[0].value
  output_values_ta = tf.TensorArray(
      dtype=output_spec.dtype,
      size=output_time,
      element_shape=tf.TensorShape([static_batch_size]).concatenate(
          output_spec.shape
      ),
  )
  output_mask_ta = tf.TensorArray(
      dtype=x.mask.dtype,
      size=output_time,
      element_shape=tf.TensorShape([static_batch_size]),
  )

  def _make_ta(s: tf.TensorSpec):
    return tf.TensorArray(
        dtype=s.dtype,
        size=output_time,
        element_shape=tf.TensorShape([static_batch_size]).concatenate(s.shape),
    )

  emit_tas = tf.nest.map_structure(_make_ta, emit_specs)

  def read_block(b: tf.Tensor) -> types.Sequence:
    """Reads batch-major Sequence block b from time-major input TensorArrays."""
    start = b * input_block_size
    # TPU compatible range.
    input_indices = start + tf.range(input_block_size)
    x_block_values_time_major = _ta_gather(input_values_ta, input_indices)
    x_block_mask_time_major = _ta_gather(input_mask_ta, input_indices)
    # Build a Sequence of batch-major values/mask for passing to transition_fn.
    x_block = types.Sequence(
        _transpose_batch_and_time(
            x_block_values_time_major, input_shape_indices
        ),
        _transpose_batch_and_time(x_block_mask_time_major, []),
    )

    # TensorArray.gather can't tell that we're reading input_block_size samples.
    def set_shape(v, m):
      v.set_shape([None, input_block_size] + [None] * input_shape.rank)
      m.set_shape([None, input_block_size])
      return v, m

    return x_block.apply(set_shape)

  def write_block(
      b: tf.Tensor,
      y_block: types.Sequence,
      emits: types.Emits,
      values_ta: tf.TensorArray,
      mask_ta: tf.TensorArray,
      emit_tas: Union[tf.TensorArray, Any],
  ) -> Tuple[tf.TensorArray, tf.TensorArray, Union[tf.TensorArray, Any]]:
    """Writes batch-major Sequence block b to time-major values_ta / mask_ta."""
    start = b * output_block_size
    # TPU compatible range.
    output_indices = start + tf.range(output_block_size)

    y_block_values_time_major = _transpose_batch_and_time(
        y_block.values, output_shape_indices
    )
    values_ta = _ta_scatter(
        values_ta, output_indices, y_block_values_time_major
    )

    y_block_mask_time_major = _transpose_batch_and_time(y_block.mask, [])
    mask_ta = _ta_scatter(mask_ta, output_indices, y_block_mask_time_major)

    emits_time_major = tf.nest.map_structure(
        lambda e, i: _transpose_batch_and_time(e, i.as_list()),
        emits,
        emit_shapes_indices,
    )

    def write_emit(emit_ta, emit_time_major):
      return _ta_scatter(emit_ta, output_indices, emit_time_major)

    emit_tas = tf.nest.map_structure(write_emit, emit_tas, emits_time_major)

    return values_ta, mask_ta, emit_tas

  def cond(b: tf.Tensor, *unused_args):
    return b < num_blocks

  def body(
      b: tf.Tensor,
      output_values_ta: tf.TensorArray,
      output_mask_ta: tf.TensorArray,
      emit_tas: Union[tf.TensorArray, Any],
      state: types.State,
  ):
    """Performs the autoregressive loop."""
    x_block = read_block(b)
    y_block, state, y_emits = transition_fn(x_block, state)
    output_values_ta, output_mask_ta, emit_tas = write_block(
        b, y_block, y_emits, output_values_ta, output_mask_ta, emit_tas
    )
    return b + 1, output_values_ta, output_mask_ta, emit_tas, state

  # Run a while loop for num_blocks timesteps. Each iteration:
  # - reads input_block_size frames from input_values_ta / input_mask_ta
  # - calls transition_fn(sequence, state)
  # - writes the resulting output_block_size frames to
  #     output_values_ta / output_mask_ta
  # - carries state over to the next iteration
  _, output_values_ta, output_mask_ta, emit_tas, state = tf.while_loop(
      cond,
      body,
      (0, output_values_ta, output_mask_ta, emit_tas, state),
      maximum_iterations=num_blocks,
  )

  y = types.Sequence(
      _transpose_batch_and_time(output_values_ta.stack(), output_shape_indices),
      _transpose_batch_and_time(output_mask_ta.stack(), []),
  )

  emits_time_major = tf.nest.map_structure(
      lambda emit_ta: emit_ta.stack(), emit_tas
  )
  emits = tf.nest.map_structure(
      lambda e, i: _transpose_batch_and_time(e, i.as_list()),
      emits_time_major,
      emit_shapes_indices,
  )
  return y, state, emits


def _pad_to_multiple(
    x: types.Sequence, block_size: int
) -> Union[types.Sequence, tf.Tensor, tf.Tensor]:
  """Pads x to multiple of block_size."""
  time = smart_dimension_size(x.values, 1)
  num_blocks = (time + block_size - 1) // block_size
  padded_time = num_blocks * block_size
  pad_amount = padded_time - time
  x = x.pad_time(0, pad_amount, valid=False)
  return x, padded_time, num_blocks


def step_by_step_static(
    l: types.SequenceLayer,
    x: types.Sequence,
    training: bool,
    initial_state: types.State = None,
    blocks_per_step: int = 1,
    constants: types.Constants = None,
) -> Tuple[types.Sequence, types.State, types.Emits]:
  """Executes a SequenceLayer timestep-by-timestep statically.

  If the sequence length is not a multiple of l.block_size * blocks_per_step,
  the sequence is padded to that length with invalid timesteps.

  Args:
    l: The SequenceLayer to invoke step-by-step.
    x: The input Sequence to process step-by-step.
    training: Whether we are in training mode.
    initial_state: The initial state to use. If None, the initial state returned
      by l.get_initial_state is used.
    blocks_per_step: Runs the layer with this multiple of l.block_size timesteps
      per step.
    constants: The constants dictionary to provide to the step function.

  Returns:
    The resulting Sequence and emits.
  """
  if not l.supports_step:
    raise ValueError(f'{l} cannot be stepped.')
  if initial_state is None:
    initial_state = l.get_initial_state(x, constants)
  input_block_size = l.block_size * blocks_per_step
  x, _, num_blocks = _pad_to_multiple(x, input_block_size)
  transition_fn = functools.partial(
      l.step_with_emits, training=training, constants=constants
  )
  outputs, state, emits = step_by_step_fn_static(
      transition_fn, num_blocks, input_block_size, x, initial_state
  )
  return outputs, state, emits


def step_by_step_dynamic(
    l: types.SequenceLayer,
    x: types.Sequence,
    training: bool,
    initial_state: types.State = None,
    blocks_per_step: int = 1,
    constants: types.Constants = None,
) -> Tuple[types.Sequence, types.State, types.Emits]:
  """Executes a SequenceLayer timestep-by-timestep dynamically.

  If the sequence length is not a multiple of l.block_size * blocks_per_step,
  the sequence is padded to that length with invalid timesteps.

  Args:
    l: The SequenceLayer to invoke step-by-step.
    x: The input Sequence to process step-by-step.
    training: Whether we are in training mode.
    initial_state: The initial state to use. If None, the initial state returned
      by l.get_initial_state is used.
    blocks_per_step: Runs the layer with this multiple of l.block_size timesteps
      per step.
    constants: The constants dictionary to provide to the step function.

  Returns:
    The resulting Sequence.
  """
  if not l.supports_step:
    raise ValueError(f'{l} cannot be stepped.')

  if initial_state is None:
    initial_state = l.get_initial_state(x, constants)

  output_spec = l.get_output_spec_for_sequence(x, constants)
  emit_specs = l.get_emit_specs_for_sequence(x, constants)

  input_block_size = l.block_size * blocks_per_step
  output_block_size = int(input_block_size * l.output_ratio)
  # Pad to a multiple of block_size timesteps.
  x, time, num_blocks = _pad_to_multiple(x, input_block_size)

  transition_fn = functools.partial(
      l.step_with_emits, training=training, constants=constants
  )

  outputs, state, emits = step_by_step_fn_dynamic(
      transition_fn,
      time,
      x.channel_shape,
      output_spec,
      emit_specs,
      num_blocks,
      input_block_size,
      output_block_size,
      x,
      initial_state,
  )
  return outputs, state, emits


def _reshape_for_broadcast(
    x: types.Sequence, y: types.Sequence
) -> Tuple[types.Sequence, types.Sequence]:
  """Reshapes channel dims of x and y to be broadcastable to each other."""
  # The time dimensions may not match if we are broadcasting over time.
  x.values.shape[:1].assert_is_compatible_with(y.values.shape[:1])
  x.channel_shape.assert_is_fully_defined()
  y.channel_shape.assert_is_fully_defined()
  extra_dims = abs(x.channel_shape.rank - y.channel_shape.rank)

  def _reshape(values: tf.Tensor) -> tf.Tensor:
    batch_size, time = smart_dimension_size(values, [0, 1])
    shape = [batch_size, time] + [1] * extra_dims + values.shape[2:].as_list()
    return tf.reshape(values, shape)

  if x.channel_shape.rank > y.channel_shape.rank:
    y = y.apply_values(_reshape)
  elif x.channel_shape.rank < y.channel_shape.rank:
    x = x.apply_values(_reshape)
  assert x.channel_shape.rank == y.channel_shape.rank
  return x, y


def sequence_broadcast_add(
    x: types.Sequence, y: types.Sequence
) -> types.Sequence:
  """Broadcast-add x and y."""
  x, y = _reshape_for_broadcast(x, y)
  return types.Sequence(x.values + y.values, combine_mask(x.mask, y.mask))


def sequence_broadcast_concat(
    x: types.Sequence, y: types.Sequence
) -> types.Sequence:
  """Broadcast-concatenate x and y."""
  x, y = _reshape_for_broadcast(x, y)
  channel_outer_dims_broadcast_shape = tf.broadcast_static_shape(
      x.channel_shape[:-1], y.channel_shape[:-1]
  )

  batch_size, time = smart_dimension_size(x.values, [0, 1])

  def _broadcast_channel_outer_dims(values):
    # Expand dim 2D tensors for concatenation.
    if values.shape.rank == 2 and channel_outer_dims_broadcast_shape.rank == 0:
      values = values[..., tf.newaxis]
    shape = (
        [batch_size, time]
        + channel_outer_dims_broadcast_shape.as_list()
        + [values.shape[-1]]
    )
    return tf.broadcast_to(values, shape)

  x_values = _broadcast_channel_outer_dims(x.values)
  y_values = _broadcast_channel_outer_dims(y.values)
  return types.Sequence(
      tf.concat([x_values, y_values], -1), combine_mask(x.mask, y.mask)
  )


def sequence_broadcast_affine(
    x: types.Sequence, scale: types.Sequence, shift: types.Sequence
) -> types.Sequence:
  """Affine transform of x."""
  x, scale = _reshape_for_broadcast(x, scale)
  x, shift = _reshape_for_broadcast(x, shift)
  values = x.values * scale.values + shift.values
  # Scale and shift are often derived from the same sequence so combine their
  # mask first as an optimization.
  mask = combine_mask(x.mask, combine_mask(scale.mask, shift.mask))
  return types.Sequence(values, mask)


def dynamic_filter_conv1d(inputs, filters, stride=1, padding='SAME'):
  """Apply different 1d convolution filters to each batch member.

  The `filters` tensor has a batch dimension that determines which filters
  are applied to the input from each batch member.

  Args:
    inputs: A [batch, in_width, in_channels] input tensor.
    filters: A [batch, filt_width, in_channels, out_channels] filter tensor.
    stride: The convolution stride as an int (default=1).
    padding: A str that determines the padding mode (default='SAME').

  Returns:
    outputs: A [batch, out_width, out_channels] output tensor.
  """
  with tf.name_scope('dynamic_filter_conv1d'):
    # Group the batch and in_chan dimensions together for the filters.
    # [B, Wf, Ci, Co]
    batch, filt_width, in_chan, out_chan = smart_dimension_size(filters)
    # [B, Wf, Ci, Co] -> [Wf, B, Ci, Co]
    filters = tf.transpose(filters, [1, 0, 2, 3])
    # [W, B, Ci, Co] -> [H=1, Wf, B*Ci, Co]
    filters = tf.reshape(filters, [1, filt_width, batch * in_chan, out_chan])

    # Group the batch and in_chan dimensions together for the input.
    # [B, Wi, Ci]
    batch, in_width, in_chan = smart_dimension_size(inputs)
    # [B, Wi, Ci] -> [W, B, Ci]
    inputs = tf.transpose(inputs, [1, 0, 2])
    # [Wi, B, Ci] -> [1, H=1, Wi, B*Ci] (Singleton leading batch dimension).
    inputs = tf.reshape(inputs, [1, 1, in_width, batch * in_chan])

    # Depthwise convolution with grouped batch and channel dims.
    # [1, H=1, Wi, B*Ci] -> [1, H=1, Wo, B*Ci*Co]
    outputs = tf.nn.depthwise_conv2d(
        inputs, filters, strides=(1, 1, stride, 1), padding=padding
    )

    # Ungroup the batch and channel dimensions.
    # [1, H=1, Wo, B*Ci*Co]
    _, _, out_width, _ = smart_dimension_size(outputs)
    # [1, H=1, Wo, B*Ci*Co] -> [Wo, B, Ci, Co]
    outputs = tf.reshape(outputs, [out_width, batch, in_chan, out_chan])
    # [Wo, B, Ci, Co] -> [B, Wo, Ci, Co]
    outputs = tf.transpose(outputs, [1, 0, 2, 3])

    # Sum across input channels after separating from batch dim.
    # [B, Wo, Ci, Co] -> [B, Wo, Co]
    outputs = tf.reduce_sum(outputs, axis=2)

  return outputs


def log_without_underflow(
    inputs: tf.Tensor, min_output: float, min_input: float | None = None
) -> tf.Tensor:
  """Natural logarithm preventing underflow.

  Args:
    inputs: Input tensor.
    min_output: Minimum output value.
    min_input: Minimum value that should be passed through the log function (a
      positive number that won't lead to log underflow).

  Returns:
    outputs: The element-wise log of inputs. Outputs for input values less than
      min_input are set to min_output.

  Raises:
    ValueError: If log(min_input) is less than min_output.
  """
  with tf.name_scope('log_without_underflow'):
    if min_input is None:
      # Smallest value that doesn't lead to log underflow for float32.
      min_input = 1.775e-38

    if np.log(min_input) < min_output:
      raise ValueError('log(min_input) must not be less than min_output.')

    outputs = tf.math.log(tf.maximum(inputs, min_input))
    outputs = tf.where(inputs >= min_input, outputs, min_output)

  return outputs


def combine_mask(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Combines mask x and y (float32, shaped [b, t]) in a logical AND fashion."""
  if x.dtype != types.MASK_DTYPE or x.shape.rank != 2:
    raise ValueError(f'Expected x as a [b, t] float32 tensor, got: {x}')
  if y.dtype != types.MASK_DTYPE or y.shape.rank != 2:
    raise ValueError(f'Expected y as a [b, t] float32 tensor, got: {y}')
  if x is y:
    return x
  return x * y


class LambdaTensor:

  def __init__(self, value_fn: Callable[[], tf.Tensor]):
    self._value_fn = value_fn

  def to_tensor(self) -> tf.Tensor:
    return self._value_fn()


def _lambdatensor_to_tensor(
    value: LambdaTensor,
    dtype: Optional[tf.DType] = None,
    name: Optional[str] = None,
    as_ref: bool = False,
) -> tf.Tensor:
  del as_ref
  return tf.convert_to_tensor(value.to_tensor(), dtype=dtype, name=name)


tf.register_tensor_conversion_function(LambdaTensor, _lambdatensor_to_tensor)


class WeightNorm(tf.keras.layers.Wrapper):
  """Decouple weight magnitude and direction.

  Modified from tensor2tensor/tensor2tensor/layers/common_layers.py.

  This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem.

  Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
  Tim Salimans, Diederik P. Kingma (2016)

  Raises:
    ValueError: If not initialized with a `Layer` instance.
    ValueError: If `Layer` does not contain a `kernel` of weights
  """

  def __init__(self, layer: tf.keras.layers.Layer, **kwargs):
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          'Please initialize `WeightNorm` layer with a '
          '`Layer` instance. You passed: {input}'.format(input=layer)
      )
    super().__init__(layer, **kwargs)
    self._track_trackable(layer, name='layer')

  def _compute_weights(self) -> tf.Tensor:
    """Generate weights with normalization."""
    with tf.name_scope('compute_weights'):
      return (
          tf.nn.l2_normalize(self.layer.v, axis=self.norm_axes) * self.layer.g
      )

  def build(self, input_shape: Optional[tf.TensorShape] = None) -> None:
    """Build `Layer`."""
    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = False

      if not hasattr(self.layer, 'kernel'):
        raise ValueError(
            '`WeightNorm` must wrap a layer that'
            ' contains a `kernel` for weights'
        )

      # The kernel's filter or unit dimension is -1
      self.layer_depth = int(self.layer.kernel.shape[-1])
      self.norm_axes = list(range(self.layer.kernel.shape.rank - 1))

      self.layer.v = self.layer.kernel
      self.layer.g = self.layer.add_weight(
          name='g',
          shape=(self.layer_depth,),
          initializer=tf.ones_initializer,
          dtype=self.layer.kernel.dtype,
          trainable=True,
      )

      # Replace the kernel with a LambdaTensor, which is converted to a tensor
      # on the fly when evaluated with tf.convert_to_tensor, but does not store
      # graph-specific tensors as member variables.
      self.layer.kernel = LambdaTensor(self._compute_weights)

      self.layer.built = True
    self.input_spec = self.layer.input_spec

    super().build()
    assert self.built

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    return self.layer.call(inputs)

  def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list()
    )


def compute_dtype() -> tf.DType:
  """Returns the global Keras precision policy's compute dtype."""
  # In TF1, compute_dtype is None.
  return tf.as_dtype(
      tf.keras.mixed_precision.global_policy().compute_dtype or 'float32'
  )


def variable_dtype() -> tf.DType:
  """Returns the global Keras precision policy's variable dtype."""
  # In TF1, variable_dtype is None.
  return tf.as_dtype(
      tf.keras.mixed_precision.global_policy().variable_dtype or 'float32'
  )


def slice_and_pad_tensor(
    tensor: tf.Tensor,
    slices: list[tuple[int, tf.Tensor | int, int]],
    pad_value: float,
    tensor_is_pre_padded: bool,
    name: str,
) -> tf.Tensor:
  """Slices tensor on multiple axes, padding to a specified size.

  TODO(b/298748783): This function assumes tensor is of at least length (start +
  length) on all axes to slice.

  Args:
    tensor: The tensor to slice and pad.
    slices: A list of (axis, start, length) of slices to perform.
    pad_value: A value to pad with when the tensor slice is incomplete.
    tensor_is_pre_padded: Workaround for b/298748783. If true, avoids a dynamic
      slice.
    name: A name for this slice.

  Returns:
    A tensor with all axes in slices sliced and padded to the specified length.
  """
  with tf.name_scope(name):
    rank = tensor.shape.rank
    begin = [0] * rank
    size = [-1] * rank
    paddings = [[0, 0]] * rank
    shape = smart_dimension_size(tensor)
    static_shape = tensor.shape.as_list()

    for slice_axis, slice_start, slice_length in slices:
      assert isinstance(slice_axis, int)
      assert isinstance(slice_length, int)
      axis_size = shape[slice_axis]
      begin[slice_axis] = slice_start
      if (
          isinstance(slice_start, int)
          and isinstance(slice_length, int)
          and isinstance(axis_size, int)
      ):
        size[slice_axis] = min(axis_size - slice_start, slice_length)
      else:
        # TODO(b/298748783): The slice will fail if slice_length elements are
        # not available.
        if tensor_is_pre_padded:
          size[slice_axis] = slice_length
        else:
          size[slice_axis] = tf.minimum(axis_size - slice_start, slice_length)

      paddings[slice_axis] = [0, slice_length - size[slice_axis]]
      static_shape[slice_axis] = slice_length

    chunk = tf.slice(tensor, begin, size)
    chunk = tf.pad(
        chunk, paddings, constant_values=tf.constant(pad_value, tensor.dtype)
    )

    # Check that the slice succeeds.
    chunk = tf.ensure_shape(chunk, static_shape)

    return chunk
