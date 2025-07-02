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
"""Utilities."""

import enum
import fractions
import functools
import math
import operator
import pprint
import re
import typing
from typing import Any, Callable, Protocol, Self, Sequence as TypingSequence, TypeVar

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import typing as jt


@jt.typed
def combine_mask(
    *masks: TypingSequence[jt.Bool[jt.ArrayT, '#B #T']]
) -> jt.Bool[jt.ArrayT, '#B #T']:
  """Combines masks with logical AND or a shortcut when reference equal."""
  if all(x is masks[0] for x in masks[1:]):
    return masks[0]
  return functools.reduce(jnp.logical_and, masks)


def convolution_effective_kernel_size(
    kernel_size: int, dilation_rate: int
) -> int:
  """Returns kernel_size with dilation_rate holes inserted."""
  return (kernel_size - 1) * dilation_rate + 1


def convolution_padding_output_size(
    input_size: int,
    padding: types.PaddingModeString | tuple[int, int],
    kernel_size: int,
    stride: int,
    dilation_rate: int,
) -> int:
  """Returns the output size for a convolution over input_size."""
  # Formula from: https://www.tensorflow.org/api_docs/python/tf/nn/convolution
  # SAME: ceil(input_shape / stride)
  # VALID: ceil((input_shape - (kernel_size - 1) * dilation_rate) / stride)
  pad_amount = sum(
      convolution_explicit_padding(padding, kernel_size, stride, dilation_rate)
  )
  effective_kernel_size = convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )
  output_size = max(0, input_size + pad_amount - (effective_kernel_size - 1))
  # Ceiling division.
  return (output_size + stride - 1) // stride


def convolution_explicit_padding(
    padding: types.PaddingModeString | tuple[int, int],
    kernel_size: int,
    stride: int,
    dilation_rate: int,
) -> tuple[int, int]:
  """Returns explicit padding amounts to achieve the desired pad mode."""
  if not isinstance(padding, str):
    return padding

  effective_kernel_size = convolution_effective_kernel_size(
      kernel_size, dilation_rate
  )

  match padding:
    case types.PaddingMode.CAUSAL_VALID.value | types.PaddingMode.CAUSAL.value:
      return (effective_kernel_size - 1, 0)
    case types.PaddingMode.SEMICAUSAL.value:
      pad_amount = effective_kernel_size - 1
      pad_left = max(effective_kernel_size - stride, 0)
      pad_right = pad_amount - pad_left
      return pad_left, pad_right
    case (
        types.PaddingMode.REVERSE_CAUSAL_VALID.value
        | types.PaddingMode.REVERSE_CAUSAL.value
    ):
      return (0, effective_kernel_size - 1)
    case types.PaddingMode.SAME.value:
      pad_amount = effective_kernel_size - 1
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
      return pad_left, pad_right
    case types.PaddingMode.VALID.value:
      return 0, 0
    case types.PaddingMode.SEMICAUSAL_FULL.value:
      pad_left = max(effective_kernel_size - stride, 0)
      pad_right = effective_kernel_size - 1
      return pad_left, pad_right
    case _:
      raise ValueError(f'Unsupported padding: {padding}')


def assert_is_compatible_with(a: types.ShapeLike, b: types.ShapeLike):
  """TODO(rryan): Replace with JAX asserts module."""
  a, b = tuple(a), tuple(b)
  if len(a) != len(b):
    raise ValueError(f'Shapes differ in rank: {a=} vs. {b=}')
  for i, (d1, d2) in enumerate(zip(a, b)):
    if d2 is None or d2 == 1:
      continue
    if d1 != d2:
      raise ValueError(f'Incompatible dimension {i} {a=} {b=}.')


def assert_has_rank(shape: types.ShapeLike, rank: int) -> None:
  """TODO(rryan): Replace with JAX asserts module."""
  if len(shape) != rank:
    raise ValueError(f'Rank of {shape=} is not {rank}.')


def _pad_to_multiple(
    x: types.Sequence, block_size: int
) -> tuple[types.Sequence, int, int]:
  """Pads x to multiple of block_size."""
  time = x.shape[1]
  num_blocks = (time + block_size - 1) // block_size
  padded_time = num_blocks * block_size
  pad_amount = padded_time - time
  x = x.pad_time(0, pad_amount, valid=False)
  return x, padded_time, num_blocks


def _extract_blocks(x: types.Sequence, block_size: int) -> types.Sequence:
  """Pads x and extract blocks of size block_size."""
  x, _, num_blocks = _pad_to_multiple(x, block_size)
  b, _, *rest = x.shape

  return type(x)(
      jnp.reshape(x.values, [b, num_blocks, block_size] + rest),
      jnp.reshape(x.mask, [b, num_blocks, block_size]),
  )


def _flatten_blocks(x: jax.Array) -> jax.Array:
  """Combines the num_blocks and block_size dimension into one."""
  b, num_blocks, block_size, *rest = x.shape
  return jnp.reshape(x, [b, num_blocks * block_size] + rest)


def insert_with_unique_key(
    emits: dict[str, types.Emits], key: str, value
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


def setup_shared_scope(
    parent: nn.Module,
    layers: TypingSequence[nn.Module],
    share_scope: bool | TypingSequence[bool],
) -> None:
  """Shares scopes between parent and layers based on share_scope."""
  if isinstance(share_scope, bool):
    share_scope = [share_scope] * len(layers)

  if len(share_scope) != len(layers):
    raise ValueError(
        'share_scope must be a bool or a sequence of the same length as'
        f' layers: {len(share_scope)} != {len(layers)}'
    )
  for layer, share_scope in zip(layers, share_scope, strict=True):
    if share_scope:
      nn.share_scope(parent, layer)


def step_by_step_static(
    l: types.Steppable,
    x: types.Sequence,
    training: bool,
    initial_state: types.State = None,
    blocks_per_step: int = 1,
    constants: types.Constants | None = None,
    with_emits: bool = True,
    stream_constants: bool = False,
) -> tuple[types.Sequence, types.State, types.Emits]:
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
    with_emits: If True, collects emits from the layer for each step and returns
      them. If False, returns an empty tuple for emits. If emits are not used it
      is more efficient to disable this.
    stream_constants: If True, stream Sequences present in constants at the same
      block size as x.

  Returns:
    The resulting sequence, final state, and emits stacked over time (if
    with_emits is True).
  """
  if not l.supports_step:
    raise ValueError(f'{l} cannot be stepped.')
  if initial_state is None:
    initial_state = l.get_initial_state(
        batch_size=x.shape[0],
        input_spec=x.channel_spec,
        training=training,
        constants=constants,
    )
  input_block_size = l.block_size * blocks_per_step
  x, _, num_blocks = _pad_to_multiple(x, input_block_size)

  if constants is None:
    constants = {}

  if stream_constants:

    def pad_constant(x):
      if isinstance(x, types.Sequence):
        return _pad_to_multiple(x, input_block_size)[0]
      else:
        return x

    constants = jax.tree_util.tree_map(
        pad_constant, constants, is_leaf=lambda x: isinstance(x, types.Sequence)
    )

  # If static unrolling, run a for loop over num_blocks, strided slice
  # the block from x, and run transition_fn to get the output block
  # and next state.
  output_blocks = []
  output_emits = []

  def read_block(x: types.Sequence, b: int) -> types.Sequence:
    start = b * input_block_size
    end = start + input_block_size
    x_b = x[:, start:end]
    pad_amount = input_block_size - x_b.shape[1]
    if pad_amount:
      x_b = x_b.pad_time(0, pad_amount, valid=False)
    return x_b

  state = initial_state
  for b in range(num_blocks):
    x_block = read_block(x, b)

    if stream_constants:
      step_constants = jax.tree_util.tree_map(
          lambda x: read_block(x, b) if isinstance(x, types.Sequence) else x,  # pylint: disable=cell-var-from-loop
          constants,
          is_leaf=lambda x: isinstance(x, types.Sequence),
      )
    else:
      step_constants = constants

    if with_emits:
      y_block, state, y_emit = l.step_with_emits(
          x_block, state, training=training, constants=step_constants
      )
    else:
      y_block, state = l.step(
          x_block, state, training=training, constants=step_constants
      )
      y_emit = ()

    output_blocks.append(y_block)
    output_emits.append(y_emit)

  # Concatenate all timesteps.
  output = types.Sequence.concatenate_sequences(output_blocks)

  def concatenate(*ts):
    if isinstance(ts[0], types.Sequence):
      return types.Sequence.concatenate_sequences(ts)
    else:
      return jnp.concatenate(ts, axis=1)

  output_emits = jax.tree_util.tree_map(concatenate, *output_emits)
  return output, state, output_emits


def step_by_step_dynamic(
    l: types.Steppable,
    x: types.Sequence,
    training: bool,
    initial_state: types.State = None,
    blocks_per_step: int = 1,
    constants: types.Constants | None = None,
    with_emits: bool = True,
    unroll: int = 1,
    stream_constants: bool = False,
) -> tuple[types.Sequence, types.State, types.Emits]:
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
    with_emits: If True, collects emits from the layer for each step and returns
      them. If False, returns an empty tuple for emits. If emits are not used it
      is more efficient to disable this.
    unroll: How many scan iterations to unroll in one iteration of the dynamic
      loop. Unrolling enables more XLA fusion, but increases memory usage and
      compile times.
    stream_constants: If True, stream Sequences present in constants at the same
      block size as x.

  Returns:
    The resulting sequence, final state, and emits stacked over time (if
    with_emits is True).
  """
  if initial_state is None:
    initial_state = l.get_initial_state(
        batch_size=x.shape[0],
        input_spec=x.channel_spec,
        training=training,
        constants=constants,
    )

  input_block_size = l.block_size * blocks_per_step
  x_blocked = _extract_blocks(x, input_block_size)

  if constants is None:
    constants = {}

  if stream_constants:
    blocked_constants = {}
    static_constants = {}
    for k, v in constants.items():
      if isinstance(v, types.Sequence):
        blocked_constants[k] = _extract_blocks(v, input_block_size)
      else:
        static_constants[k] = v
  else:
    blocked_constants = {}
    static_constants = constants

  def step_fn(
      layer: types.Steppable,
      state: types.State,
      inputs: types.Sequence,
      step_constants: types.Constants,
  ) -> tuple[types.State, tuple[types.Sequence, types.Emits]]:
    # Check supports_step within the nn.scan body so that layers are bound.
    if with_emits:
      output, state, emits = layer.step_with_emits(
          inputs,
          state,
          training=training,
          constants=static_constants | step_constants,
      )
    else:
      output, state = layer.step(
          inputs,
          state,
          training=training,
          constants=static_constants | step_constants,
      )
      emits = ()

    return state, (output, emits)

  scan_fn = nn.scan(
      step_fn,
      variable_broadcast=True,
      # Split all RNGs in our scope except for params (which when split during
      # initialization leads to different parameters on each iteration of the
      # loop and causes compilation errors).
      split_rngs={'params': False, nn.DenyList(('params',)): True},
      in_axes=1,
      out_axes=1,
      length=x_blocked.shape[1],
      unroll=unroll,
  )

  state, (output, emits) = scan_fn(
      l, initial_state, x_blocked, blocked_constants
  )
  output, emits = jax.tree_util.tree_map(_flatten_blocks, (output, emits))
  return output, state, emits


def _reshape_for_broadcast(
    *seqs: types.Sequence,
) -> tuple[types.Sequence, ...]:
  """Reshapes channel dims of many sequences to be broadcastable to each other."""

  max_dims = max(x.ndim for x in seqs)

  def _maybe_reshape(values: jax.Array) -> jax.Array:
    extra_dims = max_dims - values.ndim
    if extra_dims == 0:
      return values

    batch_size, time = values.shape[:2]
    shape = (batch_size, time) + (1,) * extra_dims + values.shape[2:]
    return jnp.reshape(values, shape)

  reshaped = tuple(x.apply_values(_maybe_reshape) for x in seqs)
  for y in reshaped[1:]:
    assert len(reshaped[0].channel_shape) == len(y.channel_shape)
  return reshaped


def sequence_broadcast_add(*seqs: types.Sequence) -> types.Sequence:
  """Broadcast-add sequences.

  This follows NumPy "right-to-left" broadcasting semantics in the case of
  mismatched channels (i.e., assume sequences with fewer dimensions are dim=1
  in the extra dimensions). Separately, batch and time dimensions broadcast as
  usual (equal or 1 over all sequences).

          b   t   channels
  X:      2 x 3 x 4 x 1 x 5
  Y:      1 x 3 x     2 x 1
  Result: 2 x 3 x 4 x 2 x 5

  Args:
    *seqs: The sequences to broadcast.

  Returns:
    A new Sequence with the sum of the broadcasted values and the logical-and of
    the broadcasted combine masks.
  """
  seqs = _reshape_for_broadcast(*seqs)
  return types.Sequence(
      sum(x.values for x in seqs), combine_mask(*(x.mask for x in seqs))
  )


def sequence_broadcast_mean(*seqs: types.Sequence) -> types.Sequence:
  """Broadcast-average sequences.

  This follows NumPy "right-to-left" broadcasting semantics in the case of
  mismatched channels (i.e., assume sequences with fewer dimensions are dim=1
  in the extra dimensions). Separately, batch and time dimensions broadcast as
  usual (equal or 1 over all sequences).

          b   t   channels
  X:      2 x 3 x 4 x 1 x 5
  Y:      1 x 3 x     2 x 1
  Result: 2 x 3 x 4 x 2 x 5

  Args:
    *seqs: The sequences to broadcast.

  Returns:
    A new Sequence with the average of the broadcasted values and the
    logical-and of the broadcasted combine masks.
  """
  return sequence_broadcast_add(*seqs).apply_values_masked(
      lambda x: x / len(seqs)
  )


def sequence_broadcast_product(*seqs: types.Sequence) -> types.Sequence:
  """Broadcast-multiply sequences.

  This follows NumPy "right-to-left" broadcasting semantics in the case of
  mismatched channels (i.e., assume sequences with fewer dimensions are dim=1
  in the extra dimensions). Separately, batch and time dimensions broadcast as
  usual (equal or 1 over all sequences).

          b   t   channels
  X:      2 x 3 x 4 x 1 x 5
  Y:      1 x 3 x     2 x 1
  Result: 2 x 3 x 4 x 2 x 5

  Args:
    *seqs: The sequences to broadcast.

  Returns:
    A new Sequence with the product of the broadcasted values and the
    logical-and of the broadcasted combine masks.
  """
  seqs = _reshape_for_broadcast(*seqs)
  return types.Sequence(
      functools.reduce(operator.mul, (x.values for x in seqs)),
      combine_mask(*(x.mask for x in seqs)),
  )


def sequence_broadcast_stack(*seqs: types.Sequence) -> types.Sequence:
  """Broadcast-stack sequences into a new first channel.

  This follows NumPy "right-to-left" broadcasting semantics in the case of
  mismatched channels (i.e., assume sequences with fewer dimensions are dim=1
  in the extra dimensions). Separately, batch and time dimensions broadcast as
  usual (equal or 1 over all sequences).

          b   t       channels
  X:      2 x 3 x     4 x 1 x 5
  Y:      1 x 3 x         2 x 1
  Z:      2 x 1 x         1 x 1
  Result: 2 x 3 x 3 x 4 x 2 x 5

  Args:
    *seqs: The sequences to broadcast.

  Returns:
    A new Sequence stacked on the new first channel of the broadcasted values.
  """
  seqs = _reshape_for_broadcast(*seqs)
  broadcasted_channel_shape = jnp.broadcast_shapes(
      *(x.channel_shape for x in seqs)
  )

  # Also broadcast batch and time. Callers can impose further constraints.
  batch_size = max(x.shape[0] for x in seqs)
  time = max(x.shape[1] for x in seqs)

  def _broadcast_channel_shape(values):
    shape = (batch_size, time) + broadcasted_channel_shape
    return jnp.broadcast_to(values, shape)

  return types.Sequence(
      jnp.stack([_broadcast_channel_shape(x.values) for x in seqs], axis=2),
      combine_mask(*(x.mask for x in seqs)),
  )


def sequence_broadcast_concat(*seqs: types.Sequence) -> types.Sequence:
  """Broadcast-concatenate sequences on their final axis.

  This follows NumPy "right-to-left" broadcasting semantics in the case of
  mismatched channels (i.e., assume sequences with fewer dimensions are dim=1
  in the extra dimensions). Separately, batch and time dimensions broadcast as
  usual (equal or 1 over all sequences).

          b   t   channels
  X:      2 x 3 x 4 x 1 x 5
  Y:      1 x 3 x     2 x 1
  Z:      2 x 1 x     1 x 1
  Result: 2 x 3 x 4 x 2 x 7

  Args:
    *seqs: The sequences to broadcast.

  Returns:
    A new Sequence broadcasting on all but the last channel, then concatenating
    on the last channel.
  """
  seqs = _reshape_for_broadcast(*seqs)
  channel_outer_dims_broadcast_shape = jnp.broadcast_shapes(
      *(x.channel_shape[:-1] for x in seqs)
  )

  # Also broadcast batch and time. Callers can impose further constraints.
  batch_size = max(x.shape[0] for x in seqs)
  time = max(x.shape[1] for x in seqs)

  def _broadcast_channel_outer_dims(values):
    # Expand dim 2D tensors for concatenation.
    if values.ndim == 2 and not channel_outer_dims_broadcast_shape:
      values = values[..., jnp.newaxis]
    shape = (
        (batch_size, time)
        + channel_outer_dims_broadcast_shape
        + values.shape[-1:]
    )
    return jnp.broadcast_to(values, shape)

  values = [_broadcast_channel_outer_dims(x.values) for x in seqs]
  return types.Sequence(
      jnp.concatenate(values, -1), combine_mask(*(x.mask for x in seqs))
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


@enum.unique
class CombinationMode(enum.Enum):
  """The type of combination to perform."""

  # Broadcasts inputs together and stacks them on the first channel axis:
  #
  # Examples:
  # x=() y=() -> (2)
  # x=() y=(2) -> (2, 2)
  # x=(3) y=(3) -> (2, 3)
  # x=(5) y=(3, 5) -> (2, 3, 5)
  STACK = 1

  # Broadcasts inputs together and concatenates them on the final channel axis:
  #
  # Examples:
  # x=() y=() -> (2)
  # x=() y=(2) -> (3)
  # x=(3) y=(3) -> (6)
  # x=(5) y=(3, 5) -> (3, 10)
  CONCAT = 2
  # Broadcasts inputs together and adds them.
  #
  # Examples:
  # x=() y=() -> ()
  # x=() y=(2) -> (2)
  # x=(3) y=(3) -> (3)
  # x=(5) y=(3, 5) -> (3, 5)
  ADD = 3
  # Broadcasts inputs together and averages them.
  #
  # Examples:
  # x=() y=() -> ()
  # x=() y=(2) -> (2)
  # x=(3) y=(3) -> (3)
  # x=(5) y=(3, 5) -> (3, 5)
  MEAN = 4
  # Examples:
  # x=() y=() -> ()
  # x=() y=(2) -> (2)
  # x=(3) y=(3) -> (3)
  # x=(5) y=(3, 5) -> (3, 5)
  PRODUCT = 5


def sequence_broadcast_combine(
    mode: CombinationMode, *seqs: types.Sequence
) -> types.Sequence:
  """Combines sequences according to the specified CombinationMode."""
  match (mode):
    case CombinationMode.STACK:
      return sequence_broadcast_stack(*seqs)
    case CombinationMode.CONCAT:
      return sequence_broadcast_concat(*seqs)
    case CombinationMode.ADD:
      return sequence_broadcast_add(*seqs)
    case CombinationMode.MEAN:
      return sequence_broadcast_mean(*seqs)
    case CombinationMode.PRODUCT:
      return sequence_broadcast_product(*seqs)
    case _:
      raise NotImplementedError(f'Unknown combination: {mode}')


def sequence_broadcast_combine_output_channel_shape(
    mode: CombinationMode, *channel_shapes: types.Shape
) -> types.Shape:
  """Computes the output channel shape for given mode / input channel shapes."""
  max_dims = max(len(x) for x in channel_shapes)
  channel_shapes = tuple((1,) * (max_dims - len(x)) + x for x in channel_shapes)

  match (mode):
    case CombinationMode.STACK:
      channel_broadcasted_shape = jnp.broadcast_shapes(*channel_shapes)
      return (len(channel_shapes),) + channel_broadcasted_shape
    case CombinationMode.CONCAT:
      channel_broadcasted_shape = jnp.broadcast_shapes(
          *(x[:-1] for x in channel_shapes)
      )
      if not channel_broadcasted_shape:
        channel_shapes = tuple(x if x else (1,) for x in channel_shapes)
      final_dim = sum(x[-1] for x in channel_shapes)
      return channel_broadcasted_shape + (final_dim,)
    case CombinationMode.ADD:
      return jnp.broadcast_shapes(*channel_shapes)
    case CombinationMode.MEAN:
      return jnp.broadcast_shapes(*channel_shapes)
    case CombinationMode.PRODUCT:
      return jnp.broadcast_shapes(*channel_shapes)
    case _:
      raise NotImplementedError(f'Unknown combination: {mode}')


class EinsumCommon(Protocol):
  """Common base for Einsum layer users."""

  def param(
      self,
      name: str,
      init_fn: Callable[..., nn.initializers.Initializer],
      *init_args,
  ):
    """Existence of the param() function (as in an nn.Module) is assumed."""

  @nn.nowrap
  def einsum(
      self,
      inputs: jax.Array,
      equation: str,
      kernel_shape: types.ShapeLike,
      bias_shape: types.ShapeLike | None,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      compute_dtype: types.DType | None = None,
      param_dtype: types.DType = jnp.float32,
      precision: nn.linear.PrecisionLike = None,
      kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init,
      kernel_sharding: types.Sharding | None = None,
      bias_init: nn.initializers.Initializer = nn.initializers.zeros_init(),
      bias_sharding: types.Sharding | None = None,
      einsum_factory: types.EinsumFactoryT | None = None,
      **kernel_init_kwargs,
  ) -> jax.Array:
    """Compute einsum with the given shapes. Call from an @nn.compact method."""
    kernel_init = shard_initializer(
        kernel_init, kernel_sharding, **kernel_init_kwargs
    )
    kernel = self.param('kernel', kernel_init, kernel_shape, param_dtype)
    if bias_shape is not None:
      bias_init = shard_initializer(bias_init, bias_sharding)
      bias = self.param('bias', bias_init, bias_shape, param_dtype)
    else:
      bias = None

    inputs, kernel, bias = nn.dtypes.promote_dtype(
        inputs, kernel, bias, dtype=compute_dtype
    )

    if einsum_factory is None:
      ret = jnp.einsum(equation, inputs, kernel, precision=precision)
    else:
      einsum_func = einsum_factory()
      ret = einsum_func(equation, inputs, kernel)

    if bias is not None:
      ret = bias_add(ret, bias)
    if activation is not None:
      ret = activation(ret)
    return ret


class FlaxEinsumDense(nn.Module, EinsumCommon):
  """A dense layer that uses `jnp.einsum` as the backing computation.

  Copied from Keras's EinsumDense layer.
  """

  # An equation describing the einsum to perform. This equation must be a valid
  # einsum string of the form `ab,bc->ac`, `...ab,bc->...ac`, or
  # `ab...,bc->ac...` where 'ab', 'bc', and 'ac' can be any valid einsum axis
  # expression sequence.
  equation: str
  # The expected shape of the output tensor (excluding the batch dimension and
  # any dimensions represented by ellipses). You can specify None for any
  # dimension that is unknown or can be inferred from the input shape.
  output_shape: tuple[int, ...]
  # A string containing the output dimension(s) to apply a bias to. Each
  # character in the `bias_axes` string should correspond to a character in the
  # output portion of the equation string.
  bias_axes: str | None = None
  activation: Callable[[jax.Array], jax.Array] | None = None
  compute_dtype: types.DType | None = None
  param_dtype: types.DType = jnp.float32
  precision: nn.linear.PrecisionLike = None
  kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init
  kernel_sharding: types.Sharding | None = None
  bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
  bias_sharding: types.Sharding | None = None
  # Optional einsum factory to replace the default jnp.einsum callable.
  einsum_factory: types.EinsumFactoryT | None = None

  def get_output_dtype(self, input_dtype: types.DType) -> types.DType:
    """Returns the layer's output dtype for the specified input dtype."""
    if self.is_initializing():
      param_dtype = jnp.float32
    else:
      param_dtype = jax.tree_util.tree_leaves(self.scope.variables())[0].dtype
    return get_promoted_dtype(
        input_dtype, param_dtype, dtype=self.compute_dtype
    )

  def project_sequence(self, inputs: types.Sequence) -> types.Sequence:
    """Applies dense to Sequence. If no bias, mask status is preserved."""
    # TODO(rryan): If activation(0) == 0, we can use apply_values_masked.
    if not self.bias_axes and self.activation is None:
      return inputs.apply_values_masked(self)
    else:
      return inputs.apply_values(self)

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    input_shape = list(inputs.shape)
    kernel_shape, bias_shape, _ = _analyze_einsum_string(
        self.equation,
        self.bias_axes,
        input_shape,
        list(self.output_shape),
    )
    return self.einsum(
        inputs,
        self.equation,
        kernel_shape,
        bias_shape,
        activation=self.activation,
        compute_dtype=self.compute_dtype,
        param_dtype=self.param_dtype,
        precision=self.precision,
        kernel_init=self.kernel_init,
        kernel_sharding=self.kernel_sharding,
        bias_init=self.bias_init,
        bias_sharding=self.bias_sharding,
        einsum_factory=self.einsum_factory,
    )


def _analyze_einsum_string(equation, bias_axes, input_shape, output_shape):
  """Analyzes an einsum string to determine the required weight shape."""

  dot_replaced_string = re.sub(r'\.\.\.', '0', equation)

  # This is the case where no ellipses are present in the string.
  split_string = re.match(
      '([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)', dot_replaced_string
  )
  if split_string:
    return einsum_analyze_split_string(
        split_string.groups(), bias_axes, input_shape, output_shape
    )

  # This is the case where ellipses are present on the left.
  split_string = re.match(
      '0([a-zA-Z]+),([a-zA-Z]+)->0([a-zA-Z]+)', dot_replaced_string
  )
  if split_string:
    return einsum_analyze_split_string(
        split_string.groups(),
        bias_axes,
        input_shape,
        output_shape,
        left_elided=True,
    )

  # This is the case where ellipses are present on the right.
  split_string = re.match(
      '([a-zA-Z]{2,})0,([a-zA-Z]+)->([a-zA-Z]+)0', dot_replaced_string
  )
  if split_string:
    return einsum_analyze_split_string(
        split_string.groups(), bias_axes, input_shape, output_shape
    )

  raise ValueError(
      f"Invalid einsum equation '{equation}'. Equations must be in the form "
      '[X],[Y]->[Z], ...[X],[Y]->...[Z], or [X]...,[Y]->[Z]....'
  )


def einsum_analyze_split_string(
    split_string: tuple[str, str, str],
    bias_axes: str,
    input_shape: types.ShapeLike,
    output_shape: types.ShapeLike,
    left_elided: bool = False,
) -> tuple[types.ShapeLike, types.ShapeLike | None, types.ShapeLike]:
  """Analyze a pre-split einsum string to find the weight shape."""
  input_spec = split_string[0]
  weight_spec = split_string[1]
  output_spec = split_string[2]
  elided = len(input_shape) - len(input_spec)

  output_shape = list(output_shape)

  output_shape.insert(0, input_shape[0])

  if elided > 0 and left_elided:
    for i in range(1, elided):
      # We already inserted the 0th input dimension at dim 0, so we need
      # to start at location 1 here.
      output_shape.insert(1, input_shape[i])
  elif elided > 0 and not left_elided:
    for i in range(len(input_shape) - elided, len(input_shape)):
      output_shape.append(input_shape[i])

  if left_elided:
    # If we have beginning dimensions elided, we need to use negative
    # indexing to determine where in the input dimension our values are.
    input_dim_map = {
        dim: (i + elided) - len(input_shape) for i, dim in enumerate(input_spec)
    }
    # Because we've constructed the full output shape already, we don't need
    # to do negative indexing.
    output_dim_map = {dim: i + elided for i, dim in enumerate(output_spec)}
  else:
    input_dim_map = {dim: i for i, dim in enumerate(input_spec)}
    output_dim_map = {dim: i for i, dim in enumerate(output_spec)}

  for dim in input_spec:
    input_shape_at_dim = input_shape[input_dim_map[dim]]
    if dim in output_dim_map:
      output_shape_at_dim = output_shape[output_dim_map[dim]]
      if (
          output_shape_at_dim is not None
          and output_shape_at_dim != input_shape_at_dim
      ):
        raise ValueError(
            'Input shape and output shape do not match at shared '
            f"dimension '{dim}'. Input shape is {input_shape_at_dim}, "
            'and output shape '
            f'is {output_shape[output_dim_map[dim]]}.'
        )

  for dim in output_spec:
    if dim not in input_spec and dim not in weight_spec:
      raise ValueError(
          f"Dimension '{dim}' was specified in the output "
          f"'{output_spec}' but has no corresponding dim in the input "
          f"spec '{input_spec}' or weight spec '{output_spec}'"
      )

  weight_shape = []
  for dim in weight_spec:
    if dim in input_dim_map:
      weight_shape.append(input_shape[input_dim_map[dim]])
    elif dim in output_dim_map:
      weight_shape.append(output_shape[output_dim_map[dim]])
    else:
      raise ValueError(
          f"Weight dimension '{dim}' did not have a match in either "
          f"the input spec '{input_spec}' or the output "
          f"spec '{output_spec}'. For this layer, the weight must "
          'be fully specified.'
      )

  if bias_axes:
    num_left_elided = elided if left_elided else 0
    idx_map = {
        char: output_shape[i + num_left_elided]
        for i, char in enumerate(output_spec)
    }

    for char in bias_axes:
      if char not in output_spec:
        raise ValueError(
            f"Bias dimension '{char}' was requested, but is not part "
            f"of the output spec '{output_spec}'"
        )

    first_bias_location = min([output_spec.find(char) for char in bias_axes])
    bias_output_spec = output_spec[first_bias_location:]

    bias_shape = [
        idx_map[char] if char in bias_axes else 1 for char in bias_output_spec
    ]

    if not left_elided:
      for _ in range(elided):
        bias_shape.append(1)
  else:
    bias_shape = None

  return weight_shape, bias_shape, output_shape


def shard_initializer(
    initializer: nn.initializers.Initializer,
    sharding: types.Sharding | None,
    **kwargs,
) -> nn.initializers.Initializer:
  """Applies sharding to initializer (if provided)."""
  if sharding is None and not kwargs:
    return initializer
  return meta.with_meta(initializer, sharding, **kwargs)


def bias_add(y: jax.Array, bias: jax.Array) -> jax.Array:
  """Adds bias to y with appropriate broadcasting."""
  # Assert broadcastable.
  jnp.broadcast_shapes(y.shape, bias.shape)
  return y + jnp.reshape(bias, (1,) * (y.ndim - bias.ndim) + bias.shape)


def sequence_split(
    seq: types.SequenceSelf,
    indices_or_sections: int | TypingSequence[int],
    axis: int = 0,
) -> list[types.SequenceSelf]:
  """Splits an sl.Sequence on the specified axes, preserving sequence type."""

  if axis < 0 and axis >= -seq.ndim:
    axis += seq.ndim
  if axis < 0 or axis >= seq.ndim:
    raise ValueError(f'Invalid axis for sequence_split: {axis=} {seq.ndim=}')

  x_values = jnp.split(seq.values, indices_or_sections, axis=axis)
  x_masks = (
      jnp.split(seq.mask, indices_or_sections, axis=axis)
      if axis < 2
      else [seq.mask] * len(x_values)
  )

  return [type(seq)(v, m) for v, m in zip(x_values, x_masks)]


def sequence_stack(
    seqs: TypingSequence[types.SequenceSelf], axis: int
) -> types.SequenceSelf:
  """Stacks sl.Sequences on the specified axis, preserving sequence type.

  Args:
    seqs: The sl.Sequences to stack.
    axis: The axis to stack along. Can be negative.

  Returns:
    A sequence with the same type as the input sequences, where the input
    sequences are stacked along the specified axis.

  Raises:
    ValueError: If no sequences are given, the sequences have different shapes,
      or the axis is invalid, i.e., out of range or the first two (non-channel)
      axes.
  """
  if len({seq.shape for seq in seqs}) > 1:
    raise ValueError(
        'All sequences must have the same dimensions; if not, use'
        ' sequence_broadcast_stack instead.'
    )
  if not seqs:
    raise ValueError('No sequences provided.')
  seq_type, ndim = type(seqs[0]), seqs[0].ndim
  if axis < 0 and axis >= -ndim:
    axis += ndim + 1
  if axis <= 1 or axis > ndim:
    raise ValueError(f'Invalid axis for sequence_stack: {axis=} {ndim=}')
  return seq_type(
      jnp.stack([seq.values for seq in seqs], axis=axis),
      combine_mask(*(x.mask for x in seqs)),
  )


def unstack(x: jax.Array, axis: int = 0) -> list[jax.Array]:
  """Unstacks x on the specified axis. Inverse of np.stack(xs, axis=axis)."""
  return [
      jax.lax.index_in_dim(x, i, axis, keepdims=False)
      for i in range(x.shape[axis])
  ]


def sequence_unstack(
    seq: types.SequenceSelf, axis: int
) -> list[types.SequenceSelf]:
  """Unstacks an sl.Sequence on the specified axis, preserving sequence type.

  Args:
    seq: The sl.Sequence to unstack.
    axis: The axis to unstack along. Can be negative.

  Returns:
    A list of sequences with the same type as the input sequence, where the
    input sequences are unstacked along the specified axis.

  Raises:
    ValueError: If the axis is invalid, i.e., out of range or the first two
      (non-channel) axes.
  """
  if axis < 0 and axis >= -seq.ndim:
    axis += seq.ndim
  if axis <= 1 or axis >= seq.ndim:
    raise ValueError(f'Invalid axis for sequence_unstack: {axis=} {seq.ndim=}')
  return [type(seq)(v, seq.mask) for v in unstack(seq.values, axis)]


def unstack_tree(tree: jt.AnyPyTree, axis: int = 0) -> list[jt.AnyPyTree]:
  """Unstacks provided tree on axis into multiple trees with same structure."""
  leaves, treedef = jax.tree_util.tree_flatten(tree)
  leaves = [unstack(leaf, axis=axis) for leaf in leaves]
  flat_trees = list(zip(*leaves))
  return [
      jax.tree_util.tree_unflatten(treedef, tree_i) for tree_i in flat_trees
  ]


def normalize_ntuple(x: int | TypingSequence[int], n: int) -> tuple[int, ...]:
  if isinstance(x, int):
    return (x,) * n
  result = tuple(x)
  if len(result) != n:
    raise ValueError(f'Expected a {n}-element iterable, got: {x}')
  return result


def normalize_2tuple(x: int | TypingSequence[int]) -> tuple[int, int]:
  return typing.cast(tuple[int, int], normalize_ntuple(x, 2))


def normalize_3tuple(x: int | TypingSequence[int]) -> tuple[int, int, int]:
  return typing.cast(tuple[int, int, int], normalize_ntuple(x, 3))


def get_promoted_dtype(
    *args: types.DType | None,
    dtype: types.DType | None = None,
    inexact: bool = True,
) -> types.DType:
  """Returns the dtype result of nn.dtypes.promote_dtype."""
  if dtype is None:
    dtype = jnp.result_type(*(x for x in args if x is not None))
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
      dtype = jnp.promote_types(jnp.float32, dtype)
  if inexact and not jnp.issubdtype(dtype, jnp.inexact):
    raise ValueError(f'Dtype must be inexact: {dtype}')
  return dtype


def match_shape_along_axes(
    channel_shape: types.Shape,
    axes: int | TypingSequence[int] | None,
) -> types.Shape:
  """Return a shape matching channel_shape on axes and is equal to 1 elsewhere.

  Use this to make a equal-length shape that will unambiguously broadcast with
  channel_shape.

  Args:
    channel_shape: The shape to match.
    axes: The axes to match along to (int or sequence of ints). If None, all
      axes are matched. Negative indices are supported.

  Returns:
    A shape with the same number of dimensions as channel_shape, where the
    specified axes are equal to channel_shape[axis] and all other axes are 1.

  Raises:
    ValueError: If any of the given axes does not exist in channel_shape.
  """
  if axes is None:
    return tuple(channel_shape)

  target_shape = [1] * len(channel_shape)
  if isinstance(axes, int):
    axes = [axes]
  for axis in axes:
    if not -len(channel_shape) <= axis < len(channel_shape):
      raise ValueError(f'Invalid {axis=} found in {axes=}.')
    target_shape[axis] = channel_shape[axis]
  return tuple(target_shape)


@typing.runtime_checkable
class Castable(Protocol):
  """Protocol for types that have a DType and can be cast to another DType."""

  @property
  def dtype(self) -> types.DType:
    ...

  def astype(self, dtype) -> Self:
    ...


# TODO: b/339021149 - Use TypeVarTuple, ParamSpecArgs type hints once in pytype.
_CastableOrNoneT = TypeVar('_CastableOrNoneT', bound=Castable | None)
PreservesDTypeFn = Callable[..., Any]
_PreservesDTypeFnT = TypeVar('_PreservesDTypeFnT', bound=PreservesDTypeFn)
PreservesDTypeFnDecorator = Callable[..., _PreservesDTypeFnT]


def _run_with_min_dtype(
    min_dtype: types.DType | None = None,
    restore_dtypes: bool = True,
) -> PreservesDTypeFnDecorator[PreservesDTypeFn]:
  """Factory for decorators to promote inputs to a DType within a function."""

  # See run_in_at_least_fp32's docstring for the interface of this decorator
  # (where min_dtype=jnp.float32).
  def _decorator(fn: _PreservesDTypeFnT) -> _PreservesDTypeFnT:

    def _maybe_cast(
        value: _CastableOrNoneT,
        dtype: types.DType | None,
        promote: bool = False,
    ) -> _CastableOrNoneT:
      if value is not None and dtype is not None:
        if promote:
          try:
            dtype = jnp.promote_types(value.dtype, dtype)
          except Exception as e:
            raise TypeError(
                f'Cannot promote {value=} of type {type(value)} with {dtype=}.'
            ) from e
        try:
          if value.dtype != dtype:
            value = value.astype(dtype)
        except Exception as e:
          raise TypeError(
              f'Cannot cast {value=} of type {type(value)} into {dtype=}.'
          ) from e
      return value

    @functools.wraps(fn)
    def wrapped(*args: _CastableOrNoneT) -> Any:
      # Promote arguments if possible, then call the wrapped function.
      promoted_args = tuple(
          _maybe_cast(value, min_dtype, promote=True) for value in args
      )
      results = fn(*promoted_args)

      if not restore_dtypes:
        return results

      # Validate results and cast back to their respective inputs' DTypes.
      if (len(results) if isinstance(results, tuple) else 1) != len(args):
        raise TypeError(
            f'The promoted function `{wrapped.__name__}` must return a number'
            ' of outputs equal to the number of positional arguments.'
        )
      if isinstance(results, tuple):
        return tuple(
            _maybe_cast(value, arg.dtype if arg is not None else None)
            for value, arg in zip(results, args)
        )
      else:
        value, arg = results, args[0]
        return _maybe_cast(value, arg.dtype if arg is not None else None)

    return wrapped

  return _decorator


@typing.overload
def run_in_at_least_fp32(
    fn: _PreservesDTypeFnT, *, restore_dtypes: bool = True
) -> _PreservesDTypeFnT:
  ...


@typing.overload
def run_in_at_least_fp32(
    fn: None = None, *, restore_dtypes: bool = True
) -> PreservesDTypeFnDecorator[PreservesDTypeFn]:
  ...


def run_in_at_least_fp32(
    fn: _PreservesDTypeFnT | None = None, *, restore_dtypes: bool = True
) -> _PreservesDTypeFnT | PreservesDTypeFnDecorator[PreservesDTypeFn]:
  """Decorator for promoting inputs to >=float32 for the function's duration.

  Args:
    fn: Function to wrap. `fn` must return the same number of positional
      arguments as it receives. All its arguments must be Castable or None, and
      each of its outputs must be castable to its respective input's DType
      (unless that input was None).
    restore_dtypes: Whether to cast the outputs back to their respective inputs'
      DTypes.

  Returns:
    If `fn` is provided, we return a version of `fn` that promotes its
    positional arguments with float32 and casts each of its outputs to its
    respective input's DType (unless that input was None). `fn` cannot be called
    with keyword arguments. If `fn` is None, we return a decorator that can be
    applied to a function to do the same.

  Raises:
    TypeError: A keyword argument was provided, an argument could not be
      promoted with float32, a result could not be cast back to an input DType,
      or `fn` did not return the same number of arguments as it received.
  """
  decorator = _run_with_min_dtype(jnp.float32, restore_dtypes=restore_dtypes)
  return decorator(fn) if fn else decorator


def maybe_in_at_least_fp32(
    promote: bool, *, restore_dtypes: bool = True
) -> PreservesDTypeFnDecorator[PreservesDTypeFn]:
  """Variant of `run_in_at_least_fp32` that is conditional on promote=True."""
  return (
      _run_with_min_dtype(jnp.float32, restore_dtypes=restore_dtypes)
      if promote
      else lambda fn: fn
  )


def left_shift(
    x: types.Sequence,
    shifts: jt.Int[jt.ArrayT, 'B'] | TypingSequence[int] | int,
) -> types.Sequence:
  """Left shift the sequence by the provided amounts on the time axis.

  Bitmask trick taken from google3/third_party/py/t5x/decoding.py.

  TODO(rryan): Benchmark against a vmap + roll + mask.

  Args:
    x: The sequence to be left shifted on the time axis.
    shifts: A scalar integer or [batch_size] array of offsets to shift by per
      batch item.

  Returns:
    The sequence left shifted by the specified amount per batch item, with
    invalid zero padding inserted on the right. Sequences that are shifted
    outside of the physical extent of the array are truncated. The physical
    shape of the returned sequence is unchanged.
  """
  max_len = x.shape[1]
  if not max_len:
    return x

  if isinstance(shifts, int):
    if shifts <= 0:
      return x
    return x[:, shifts:].pad_time(0, shifts, valid=False)

  shifts = jnp.maximum(0, jnp.asarray(shifts))
  nbits = np.ceil(np.log2(max_len)).astype(np.int32)

  def apply_fn(
      values: jax.Array, mask: jax.Array, bitmask: int
  ) -> tuple[jax.Array, jax.Array]:
    assert bitmask > 0
    c = shifts & bitmask

    paddings = [(0, 0)] * values.ndim
    paddings[1] = (0, bitmask)

    values = jnp.where(
        c.reshape(shifts.shape + (1,) * (values.ndim - 1)),
        jnp.pad(values, paddings)[:, bitmask:, ...],
        values,
    )
    paddings = [(0, 0)] * mask.ndim
    paddings[1] = (0, bitmask)
    mask = jnp.where(
        c.reshape(shifts.shape + (1,)),
        jnp.pad(mask, paddings)[:, bitmask:],
        mask,
    )
    return values, mask

  for i in range(0, nbits + 1):
    bitmask = 2**i
    x = x.apply_masked(apply_fn, bitmask=bitmask)

  return x


def right_shift(
    x: types.Sequence,
    shifts: jt.Int[jt.ArrayT, 'B'] | TypingSequence[int] | int,
) -> types.Sequence:
  """Right shift the sequence by the provided amounts on the time axis.

  Bitmask trick taken from google3/third_party/py/t5x/decoding.py.

  TODO(rryan): Benchmark against a vmap + roll + mask.

  Args:
    x: The sequence to be right shifted on the time axis.
    shifts: A scalar integer or [batch_size] array of offsets to shift by per
      batch item.

  Returns:
    The sequence right shifted by the specified amount per batch item, with
    invalid zero padding inserted on the left. Sequences that are shifted
    outside of the physical extent of the array are truncated. The physical
    shape of the returned sequence is unchanged.
  """
  max_len = x.shape[1]
  if not max_len:
    return x

  if isinstance(shifts, int):
    if shifts <= 0:
      return x
    return x[:, :-shifts].pad_time(shifts, 0, valid=False)

  shifts = jnp.maximum(0, jnp.asarray(shifts))
  nbits = np.ceil(np.log2(max_len)).astype(np.int32)

  def apply_fn(
      values: jax.Array, mask: jax.Array, bitmask: int
  ) -> tuple[jax.Array, jax.Array]:
    assert bitmask > 0

    paddings = [(0, 0)] * values.ndim
    paddings[1] = (bitmask, 0)

    values = jnp.where(
        (shifts & bitmask).reshape(shifts.shape + (1,) * (values.ndim - 1)),
        jnp.pad(values, paddings)[:, :-bitmask, ...],
        values,
    )

    paddings = [(0, 0)] * mask.ndim
    paddings[1] = (bitmask, 0)
    mask = jnp.where(
        (shifts & bitmask).reshape(shifts.shape + (1,)),
        jnp.pad(mask, paddings)[:, :-bitmask],
        mask,
    )
    return values, mask

  for i in range(0, nbits + 1):
    bitmask = 2**i
    x = x.apply_masked(apply_fn, bitmask=bitmask)

  return x


def ragged_sequence_concat(
    x: types.Sequence, y: types.Sequence
) -> types.Sequence:
  """Concatenates left-aligned contiguous sequences logically.

  x and y are assumed to be left-aligned sequences with contiguous masks and
  different lengths per batch item. x and y must have equal batch size and
  channel spec.

  As opposed to sl.Sequence.concatenate_sequences which concatenates physical
  sequence arrays, this function concatenates sequences logically by right
  shifting y by the lengths of each sequence in x, then stitching them together.

  Both x and y are padded to the sum of their physical lengths before
  concatenation, so no truncation of y can occur.

  Args:
    x: Sequence to prepend to y.
    y: Sequence to append to x.

  Returns:
    A new sequence where each batch item is the concatenation of the logical
    sequence x[i] and y[i].
  """
  if x.shape[0] != y.shape[0] or x.channel_spec != y.channel_spec:
    raise ValueError(
        'x and y must have the same batch size and channel spec:'
        f' {x.shape[0]=} {x.channel_spec=} != {y.shape[0]} {y.channel_spec=}'
    )
  max_length = x.shape[1] + y.shape[1]
  x_lengths = x.lengths()

  # Pad both x and y to the maximum of their combined physical lengths.
  x = x.pad_time(0, max_length - x.shape[1], valid=False)
  y = y.pad_time(0, max_length - y.shape[1], valid=False)

  y_shifted = right_shift(y, x_lengths)

  values = jnp.where(x.expanded_mask(), x.values, y_shifted.values)
  mask = jnp.logical_or(x.mask, y_shifted.mask)
  return types.Sequence(values, mask)


def get_timing_signal_1d_pos(
    position: jax.Array,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    dtype: types.DType = jnp.float32,
) -> jax.Array:
  """Sinusoidal position embeddings with explicit positions.

  Copied from tensor2tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    position: A [batch, length] tensor of positions to compute timing signal
      for.
    channels: scalar, size of timing embeddings to create. The number of
      different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    dtype: Type of the returned timing signal.

  Returns:
    a Tensor of timing signals [batch, length, channels]
  """
  position = jnp.array(position, jnp.float32)
  assert_has_rank(position.shape, 2)
  num_timescales = channels // 2
  log_timescale_increment = math.log(
      float(max_timescale) / float(min_timescale)
  ) / max(num_timescales - 1, 1)
  inv_timescales = min_timescale * np.exp(
      np.arange(num_timescales, dtype=np.float32) * -log_timescale_increment
  )
  scaled_time = (
      position[:, :, jnp.newaxis] * inv_timescales[jnp.newaxis, jnp.newaxis, :]
  )
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  timing_signal = jnp.concatenate(
      [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
  )
  timing_signal = jnp.pad(
      timing_signal, [[0, 0], [0, 0], [0, np.mod(channels, 2)]]
  )
  return timing_signal.astype(dtype)


def ones_matrix_band_part(
    rows: int,
    cols: int,
    num_lower: int,
    num_upper: int,
    out_dtype: types.DType = jnp.float32,
    out_shape: types.ShapeLike | None = None,
):
  """Matrix band part of ones.

  Copied from tensor2tensor.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_dtype: Output dtype.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = jnp.asarray(band, out_dtype)
  else:
    m = jnp.arange(rows).reshape((rows, 1))
    n = jnp.arange(cols).reshape((1, cols))
    band = ((num_lower < 0) | ((m - n) <= num_lower)) & (
        (num_upper < 0) | ((n - m) <= num_upper)
    )
    if out_shape:
      band = jnp.reshape(band, out_shape)
    band = band.astype(out_dtype)

  return band


def get_input_latency(
    config: types.SequenceLayerConfig, accumulated_input_latency: int = 0
) -> int:
  """Returns the input latency of the provided SequenceLayerConfig."""

  def fn():
    layer = config.make()

    def init_fn(layer: types.SequenceLayer):
      return layer.get_accumulated_input_latency(accumulated_input_latency)

    result, _ = layer.init_with_output(jax.random.PRNGKey(0), method=init_fn)
    return jnp.zeros((result,))

  (result,) = jax.eval_shape(fn).shape
  return result


def get_output_latency(
    config: types.SequenceLayerConfig, accumulated_output_latency: int = 0
) -> int:
  """Returns the output latency of the provided SequenceLayerConfig."""

  def fn():
    layer = config.make()

    def init_fn(layer: types.SequenceLayer):
      return layer.get_accumulated_output_latency(accumulated_output_latency)

    result, _ = layer.init_with_output(jax.random.PRNGKey(0), method=init_fn)
    return jnp.zeros((result,))

  (result,) = jax.eval_shape(fn).shape
  return result


def get_required_stepwise_delay(
    output_ratio: fractions.Fraction, input_latency: int
) -> int:
  """Returns the delay required so input_latency is divisible by 1/output_ratio.

  b/372530075 - When combining upsampling and downsampling layers with latency,
  layer/step equivalence requires inserting delays. This function returns the
  correct amount of step-wise delay to insert.

  Args:
    output_ratio: The output ratio of the layer.
    input_latency: The accumulated input latency of layers preceding the layer.

  Returns:
    The amount of delay required to ensure input latency is divisible by
    output_ratio.
  """
  if 1 not in output_ratio.as_integer_ratio():
    raise NotImplementedError(
        'get_required_stepwise_delay expects integer upsampling or'
        f' downsampling, got {output_ratio=}'
    )
  return int(-input_latency % (1 / output_ratio))


def get_output_ratio(config: types.SequenceLayerConfig) -> fractions.Fraction:
  """Returns the output ratio of the provided SequenceLayerConfig."""

  def fn():
    layer = config.make()

    def init_fn(layer: types.SequenceLayer):
      return layer.output_ratio

    output, _ = layer.init_with_output(jax.random.PRNGKey(0), method=init_fn)
    return jnp.zeros((output.numerator, output.denominator))

  numerator, denominator = jax.eval_shape(fn).shape
  return fractions.Fraction(numerator, denominator)


@jt.typed
def batch_where(
    cond: jt.Bool[jt.ArrayT, 'B'], a: jt.AnyPyTree, b: jt.AnyPyTree
) -> jt.AnyPyTree:
  """A jnp.where-like operation switching on the outermost batch dimension."""

  def where_a(a: jax.Array, b: jax.Array) -> jax.Array:
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.shape[:1] == cond.shape

    c = cond.reshape(cond.shape + (1,) * (a.ndim - 1))
    return jnp.where(c, a, b)

  return jax.tree.map(where_a, a, b)


def split_dimension(
    x: jax.Array, axis: int, shape: types.ShapeLike
) -> jax.Array:
  """Splits axis in x into the provided shape."""
  axis_size = x.shape[axis]
  if axis_size != np.prod(shape):
    raise ValueError(
        f'Splitting {axis=} of {x.shape=}, incorrect number of elements for'
        f' {shape=}.'
    )

  axis = axis % x.ndim
  outer_dimensions = x.shape[:axis]
  inner_dimensions = x.shape[axis + 1 :]
  new_shape = outer_dimensions + tuple(shape) + inner_dimensions
  return x.reshape(new_shape)


def is_integral(dtype: np.dtype) -> bool:
  """Returns true if the dtype is integral."""
  # Follow .dtype if passed a jnp.bfloat16/float32/etc. constructor.
  if hasattr(dtype, 'dtype'):
    dtype = dtype.dtype

  return issubclass(dtype.type, jnp.integer)


def is_floating(dtype: np.dtype) -> bool:
  """Returns true if the dtype is floating point."""
  # Follow .dtype if passed a jnp.bfloat16/float32/etc. constructor.
  if hasattr(dtype, 'dtype'):
    dtype = dtype.dtype

  return dtype == jnp.bfloat16 or issubclass(dtype.type, jnp.floating)


def pprint_tree_shapes_types(tree: Any) -> None:
  """Pretty-prints the shapes and types of a pytree."""
  pprint.pprint(
      jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
  )


def pformat_tree_shapes_types(tree: Any) -> str:
  """Pretty-formats the shapes and types of a pytree."""
  return pprint.pformat(
      jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
  )


@jt.typed
def batched_time_slice(
    x: types.SequenceT[jt.Shaped, 'B T ...'],
    begin: jt.Int[jt.ArrayT, 'B'],
    size: int,
) -> types.SequenceT[jt.Shaped, 'B {size} ...']:
  """Gathers a slice of time dimension of the sequence."""
  time = x.shape[1]
  available = jnp.maximum(time - begin, 0)
  valid_size = jnp.minimum(size, available)

  if not time:
    # Return an empty slice if x is zero-length.
    return x.pad_time(0, size, valid=False)

  # Use jnp.arange(size) and clip indices that are out of range instead of
  # jnp.arange(valid_size) so that shapes are fixed (therefore TPU compatible).
  time_indices = jnp.minimum(
      begin[:, jnp.newaxis] + jnp.arange(size)[jnp.newaxis, :],
      jnp.maximum(0, time - 1),
  )

  # Make a mask that is zero if the index went beyond the end.
  valid_mask = types.sequence_mask(valid_size, size)

  mask = jnp.logical_and(
      valid_mask, jnp.take_along_axis(x.mask, time_indices, axis=1)
  )
  time_indices = time_indices.reshape(time_indices.shape + (1,) * (x.ndim - 2))
  values = jnp.take_along_axis(x.values, time_indices, axis=1)

  # Preserve MaskedSequence type if x is MaskedSequence.
  return type(x)(values, mask)


def _get_constant(
    layer: types.SequenceLayer,
    constants: types.Constants | None,
    name: str,
    expected_shape: types.ShapeLike | None,
    expected_dtype: types.DType | None,
    allow_broadcastable: bool,
    optional: bool,
) -> jax.Array | types.Sequence | None:
  """Returns the constant with the given name from the constants dictionary."""
  value = (constants or {}).get(name)
  if value is None:
    if optional:
      return None
    else:
      raise ValueError(
          f'{layer} requires the constant {name} to be provided via '
          f'constants, got: {constants}'
      )

  wrong_shape = False
  if expected_shape is not None:
    if allow_broadcastable:
      try:
        jnp.broadcast_shapes(value.shape, expected_shape)
      except ValueError:
        wrong_shape = True
    else:
      wrong_shape = value.shape != expected_shape
  wrong_dtype = expected_dtype is not None and value.dtype != expected_dtype
  if wrong_shape or wrong_dtype:
    raise ValueError(
        f'{layer} requires the constant {name} to have shape'
        f' {expected_shape} dtype {expected_dtype}, got:'
        f' {value.shape=} {value.dtype=}'
    )

  return value


def get_constant_array(
    layer: types.SequenceLayer,
    constants: types.Constants | None,
    name: str,
    expected_shape: types.ShapeLike | None = None,
    expected_dtype: types.DType | None = None,
    unpack_sequence: bool = False,
    allow_broadcastable: bool = False,
) -> jax.Array:
  """Returns the constant with the given name from the constants dictionary."""
  value = _get_constant(
      layer,
      constants,
      name,
      expected_shape,
      expected_dtype,
      allow_broadcastable,
      optional=False,
  )
  if isinstance(value, types.Sequence) and unpack_sequence:
    value = value.values
  if not isinstance(value, jax.Array):
    raise ValueError(
        f'{layer} requires the constant {name} to be an array, got: {value}'
    )
  return value


def get_constant_sequence(
    layer: types.SequenceLayer,
    constants: types.Constants | None,
    name: str,
    expected_shape: types.ShapeLike | None = None,
    expected_dtype: types.DType | None = None,
    allow_broadcastable: bool = False,
) -> types.Sequence:
  """Returns the constant with the given name from the constants dictionary."""
  value = _get_constant(
      layer,
      constants,
      name,
      expected_shape,
      expected_dtype,
      allow_broadcastable,
      optional=False,
  )
  if not isinstance(value, types.Sequence):
    raise ValueError(
        f'{layer} requires the constant {name} to be a sequence, got: {value}'
    )
  return value


def get_optional_constant_sequence(
    layer: types.SequenceLayer,
    constants: types.Constants | None,
    name: str,
    expected_shape: types.ShapeLike | None = None,
    expected_dtype: types.DType | None = None,
    allow_broadcastable: bool = False,
) -> types.Sequence | None:
  """Returns the constant with the given name from the constants dictionary."""
  value = _get_constant(
      layer,
      constants,
      name,
      expected_shape,
      expected_dtype,
      allow_broadcastable,
      optional=True,
  )
  if value is None:
    return None
  if not isinstance(value, types.Sequence):
    raise ValueError(
        f'{layer} requires the constant {name} to be a sequence, got: {value}'
    )
  return value


def get_step_with_emits_output_spec(
    layer: types.SequenceLayer,
    x: types.Sequence,
    state: types.State,
    *,
    training: bool,
    constants: types.Constants | None = None,
) -> tuple[types.Sequence, types.Emits]:
  """Returns the output spec of the layer."""

  def step_fn(
      layer: types.SequenceLayer,
      x: types.Sequence,
      state: types.State,
      constants: types.Constants,
  ):
    y, _, emits = layer.step_with_emits(
        x, state, training=training, constants=constants
    )
    return y, emits

  step_fn = functools.partial(nn.jit(step_fn), layer)
  return jax.eval_shape(step_fn, x, state, constants)


def serial_block_size(layers: TypingSequence[types.SequenceLayer]) -> int:
  """Returns the block size of a serial combination of layers."""
  block_size = fractions.Fraction(1)
  output_ratio = fractions.Fraction(1)

  for child_layer in layers:
    layer_output_ratio = child_layer.output_ratio
    layer_block_size = child_layer.block_size
    block_size = (
        np.lcm(block_size * output_ratio, layer_block_size) / output_ratio
    )
    assert block_size.denominator == 1
    output_ratio *= layer_output_ratio

  return block_size.numerator


def serial_output_ratio(
    layers: TypingSequence[types.SequenceLayer],
) -> fractions.Fraction:
  """Returns the output ratio of a serial combination of layers."""
  output_ratio = fractions.Fraction(1)
  for child_layer in layers:
    output_ratio *= child_layer.output_ratio
  return output_ratio


def serial_input_latency(
    layers: TypingSequence[types.SequenceLayer],
    input_latency: int = 0,
) -> int:
  """Returns the input latency of a serial combination of layers."""
  for child_layer in reversed(layers):
    input_latency = child_layer.get_accumulated_input_latency(input_latency)
  return input_latency


def serial_output_latency(
    layers: TypingSequence[types.SequenceLayer], output_latency: int = 0
) -> int:
  """Returns the output latency of a serial combination of layers."""
  for child_layer in layers:
    output_latency = child_layer.get_accumulated_output_latency(output_latency)
  return output_latency


def receptive_field_union(
    rf_a: types.ReceptiveField, rf_b: types.ReceptiveField
) -> types.ReceptiveField:
  """Returns the union of the receptive fields of the layers."""
  if rf_a is None:
    return rf_b
  if rf_b is None:
    return rf_a
  past = min(rf_a[0], rf_b[0])
  future = max(rf_a[1], rf_b[1])
  return past, future


def receptive_field_at(
    layer_receptive_field_per_step: dict[int, types.ReceptiveField],
    layer_output_ratio: fractions.Fraction,
    output_step: int,
) -> types.ReceptiveField:
  """Returns the receptive field of the layer at the given output step."""
  rf_per_step = layer_receptive_field_per_step
  types.validate_receptive_field_per_step(rf_per_step)
  # `output_step` can take any value, and we shift it to the range of values in
  # rf_per_step to compute the relative receptive field, and apply the reverse
  # shift in the input steps to get the actual input range for `output_step`.
  normalized_step = output_step % len(rf_per_step)
  shift = (output_step - normalized_step) // layer_output_ratio
  rf = rf_per_step[normalized_step]
  if rf is None:
    return None
  past, future = rf
  return past + shift, future + shift


def layer_receptive_field_at(
    layer: types.SequenceLayer,
    output_step: int,
) -> types.ReceptiveField:
  """Returns the receptive field of the layer at the given output step."""
  return receptive_field_at(
      layer.receptive_field_per_step, layer.output_ratio, output_step
  )


def reduce_receptive_field_per_step(
    rf_per_step: dict[int, types.ReceptiveField],
    output_ratio: fractions.Fraction,
) -> types.ReceptiveField:
  """Returns the union of the receptive fields in rf_per_step.

  Args:
    rf_per_step: The receptive field per step.
    output_ratio: The output ratio of the layer.

  Returns:
    The overall union of the receptive fields.
  """
  types.validate_receptive_field_per_step(rf_per_step)

  # Pick the steps that have receptive field.
  rf_per_step = {k: v for k, v in rf_per_step.items() if v is not None}
  if not rf_per_step:
    # No receptive field if all steps have no receptive field.
    return None

  # Compute the overall union of the receptive fields.
  min_past = np.inf
  max_future = -np.inf
  for step, rf in rf_per_step.items():
    past, future = rf
    past -= step // output_ratio
    future -= step // output_ratio
    min_past = min(min_past, past)
    max_future = max(max_future, future)
  return min_past, max_future


def aggregate_layers_receptive_field_per_steps(
    layers: TypingSequence[types.SequenceLayer],
) -> dict[int, types.ReceptiveField]:
  return aggregate_receptive_field_per_steps(
      [layer.receptive_field_per_step for layer in layers]
  )


def aggregate_receptive_field_per_steps(
    receptive_field_per_step_list: TypingSequence[
        dict[int, types.ReceptiveField]
    ],
) -> dict[int, types.ReceptiveField]:
  """Aggregates a list receptive field per step."""
  agg_rf_per_step = {}
  for rf_per_step in receptive_field_per_step_list:
    types.validate_receptive_field_per_step(rf_per_step)
    for step, rf in rf_per_step.items():
      agg_rf_per_step[step] = receptive_field_union(
          rf, agg_rf_per_step.get(step)
      )
  types.validate_receptive_field_per_step(agg_rf_per_step)
  return agg_rf_per_step


def propagate_receptive_field_to_prev_layer(
    layer_rf_per_step_next: dict[int, types.ReceptiveField],
    layer_rf_per_step_prev: dict[int, types.ReceptiveField],
    layer_output_ratio_prev: fractions.Fraction,
) -> dict[int, types.ReceptiveField]:
  """Propagates the receptive field of the next layer to the previous layer.

  Given the receptive field per step of next layer, this function traces back
  and expands the receptive field to the previous layer, returns the overall
  receptive per step obtained by stacking the two layers.

  Args:
    layer_rf_per_step_next: The receptive field of the next layer.
    layer_rf_per_step_prev: The receptive field of the previous layer.
    layer_output_ratio_prev: The output ratio of the previous layer.

  Returns:
    The receptive field per step of the Serial([previous, next]) layers.
  """
  expanded_rf_per_step = {}
  for step_next, rf_next in layer_rf_per_step_next.items():
    if rf_next is None:
      expanded_rf_per_step[step_next] = None
      continue
    types.validate_receptive_field(rf_next)
    past, future = rf_next

    rf_prev_list = []

    if past == -np.inf and future == np.inf:
      rf_prev_list.append((-np.inf, np.inf))
    else:
      if past == -np.inf:
        rf_prev_list.append((-np.inf, -np.inf))
      if future == np.inf:
        rf_prev_list.append((np.inf, np.inf))

      # Is +1 enough when one side is +/-inf?
      start = past if past != -np.inf else future
      end = future + 1 if future != np.inf else past + 1
      # There is a possibility of optimizing by only considering
      # the first (past) and last (future) steps in computation below, but need
      # to verify that it works for the case of layers with varying rf per step.
      rf_prev_list.extend([
          receptive_field_at(layer_rf_per_step_prev, layer_output_ratio_prev, i)
          for i in range(start, end)
      ])
      rf_prev_list = [rf for rf in rf_prev_list if rf is not None]
    if not rf_prev_list:
      expanded_rf_per_step[step_next] = None
      continue

    min_past = np.inf
    max_future = -np.inf
    for rf_prev in rf_prev_list:
      types.validate_receptive_field(rf_prev)
      past_prev, future_prev = rf_prev
      min_past = min(min_past, past_prev)
      max_future = max(max_future, future_prev)
    expanded_rf_per_step[step_next] = (min_past, max_future)
  return expanded_rf_per_step


def receptive_field_per_step_of_serial_layers(
    layers: TypingSequence[types.SequenceLayer],
) -> dict[int, types.ReceptiveField]:
  """Returns the receptive field per step of the Serial(layers)."""
  if not layers:
    return {0: (0, 0)}
  receptive_field_per_step_list = [
      layer.receptive_field_per_step for layer in layers
  ]
  output_ratio_list = [layer.output_ratio for layer in layers]
  # Note extra steps does not affect the result, and LCM is the
  # upperbound of what we need, and we can reduce it to speed up computation.
  num_steps = math.lcm(*[len(r) for r in receptive_field_per_step_list])
  rf_per_step = {k: (k, k) for k in range(num_steps)}
  # Start from the last layer and work our way to the first.
  for output_ratio_i, rf_per_step_i in zip(
      reversed(output_ratio_list), reversed(receptive_field_per_step_list)
  ):
    rf_per_step = propagate_receptive_field_to_prev_layer(
        rf_per_step, rf_per_step_i, output_ratio_i
    )
  return rf_per_step
