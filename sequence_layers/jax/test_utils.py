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
"""Test utilities."""

import dataclasses
import itertools
import random
from typing import Any, Callable, Iterable, Mapping, Sequence as TypingSequence, TypeVar

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import types
from sequence_layers.jax import utils

from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


_SequenceLayerT = TypeVar('_SequenceLayerT', bound=types.SequenceLayer)
_T = TypeVar('_T')
_TestFnT = Callable[..., None]


def random_sequence(
    *dims: int,
    dtype=jnp.float32,
    random_mask: bool = False,
    random_lengths: bool | None = None,
    low: int | None = 0,
    high: int | None = 10,
    low_length: int = 0,
    high_length: int | None = None,
) -> types.Sequence:
  """Generates a random Sequence with dims dimension.

  Each batch item has a randomly generated length in [0, dims[1]) which is used
  to compute the mask.

  Args:
    *dims: The dimensions of the sequence.
    dtype: The dtype of the sequence.
    random_mask: Whether to generate a random, non-contiguous mask.
    random_lengths: Whether to generate random lengths (i.e. a random,
      contiguous mask).
    low: For integer dtypes, the minimum value to generate.
    high: For integer dtypes, one above the maximum value to generate.
    low_length: For random_lengths, the minimum length to generate.
    high_length: For random lengths, one above the maximum length to generate.

  Returns:
    A Sequence with the specified dimensions and type. Invalid values timesteps
    are masked.
  """
  # If random_mask is disabled, default to random_lengths.
  if random_lengths is None:
    random_lengths = not random_mask
  if random_mask and random_lengths:
    raise ValueError('Must not specify random_mask and random_lengths.')
  if len(dims) < 2:
    raise ValueError(
        'random_sequence expects at least 2 dimensions, got: %s' % (dims,)
    )

  is_complex = dtype in (np.complex64, np.complex128)
  is_integer = np.issubdtype(dtype, np.integer)

  if is_complex:
    np_values = np.random.normal(size=dims) + 1j * np.random.normal(size=dims)
  elif is_integer:
    np_values = np.random.randint(low, high, size=dims)
  else:
    np_values = np.random.normal(size=dims)

  batch_size, time = dims[0], dims[1]
  values = np_values.astype(dtype)
  if random_mask:
    mask = np.random.uniform(size=(batch_size, time)) > 0.5
  else:
    if time > 0:
      if random_lengths:
        if high_length is None:
          high_length = time + 1
        lengths = np.random.randint(
            low_length, high_length, size=[batch_size]
        ).astype(np.int32)
      else:
        lengths = np.full([batch_size], time).astype(np.int32)
    else:
      lengths = np.full([batch_size], 0).astype(np.int32)

    mask = np.arange(time)[np.newaxis, :] < lengths[:, np.newaxis]
  # We cast to jax.Array to avoid being surprised by NumPy promotion rules.
  return types.Sequence(jnp.asarray(values), jnp.asarray(mask)).mask_invalid()


def cast_from_to(tree, from_dtype: types.DType, to_dtype: types.DType):
  """Casts arrays with dtype from_dtype to to_dtype."""
  return jax.tree.map(
      lambda a: a.astype(to_dtype) if a.dtype == from_dtype else a, tree
  )


def standard_dtype_configs(
    param: bool | None = None,
    input: bool | None = None,  # pylint: disable=redefined-builtin
    compute: bool | None = None,
    param_dtype: str = 'param_dtype',
    input_dtype: str = 'input_dtype',
    compute_dtype: str = 'compute_dtype',
    expected: bool = False,
    expected_dtype: str = 'expected_dtype',
    praxis_only: bool = False,
    include_default: bool = True,
    named: bool = False,
) -> list[dict[str, types.DType | None | str]]:
  """Returns a list of standard dtype configurations; restrictable via flags.

  Args:
    param: If any of these three are set, restrict the returned configs to only
      include args set to True (deduplicating if needed).
    input: (see above)
    compute: (see above)
    param_dtype: Key to use for parameter dtypes in the returned configs.
    input_dtype: Key to use for input dtypes in the returned configs.
    compute_dtype:  Key to use for compute dtypes in the returned configs.
    expected: Whether to include the expected dtype in the returned configs.
    expected_dtype:  Key to use for expected dtypes in the returned configs.
    praxis_only: Whether to only include Praxis-like configs (e.g., for
      conversion/equivalence).
    include_default: Whether to include the default (fp32, fp32, None) dtype
      config.
    named: Whether to include a `testcase_name` in the returned configs.

  Returns:
    A list of known configs mapping each class of dtype parameter to a dtype.
    Each config will be ordered (param, input, compute), for use in unlabelled
    parameter tuples. If named=True, the `testcase_name` will be prepended to
    the config dictionary, with an abbreviated name for the dtype combination.
  """
  if param is None and input is None and compute is None:
    param = True
    input = True
    compute = True

  results = []

  if include_default:
    results += [
        # Tests W32A32 (this is the default):
        {
            param_dtype: jnp.float32,
            input_dtype: jnp.float32,
            compute_dtype: None,
        },
    ]

  # Subset of configurations that we care about matching with Praxis.
  results += [
      # W16A16 (typical inference config):
      {
          param_dtype: jnp.bfloat16,
          input_dtype: jnp.bfloat16,
          compute_dtype: jnp.bfloat16,
      },
  ]

  if not praxis_only:
    results += [
        # Tests W32A16 (typical training config):
        {
            param_dtype: jnp.float32,
            input_dtype: jnp.bfloat16,
            compute_dtype: jnp.bfloat16,
        },
        # Tests W16A32 w/ automatic promotion:
        {
            param_dtype: jnp.bfloat16,
            input_dtype: jnp.float32,
            compute_dtype: None,
        },
        # Tests W32A16 when inputs are float32:
        {
            param_dtype: jnp.float32,
            input_dtype: jnp.float32,
            compute_dtype: jnp.bfloat16,
        },
    ]

  # Remove unneeded keys, deduplicate, then return:
  keep = {
      param_dtype: param,
      input_dtype: input,
      compute_dtype: compute,
  }
  keep_keys = tuple(k for k, v in keep.items() if v)

  shorthands = {
      jnp.float32: 'fp32',
      jnp.bfloat16: 'bf16',
      None: 'None',
  }

  found = set()
  reduced = []
  for result in results:
    config_keys = tuple(k for k in result if keep[k])
    assert keep_keys == config_keys, (
        'Dtype configs should be ordered params, keys, compute. Expected'
        f' {keep_keys}, got {config_keys}'
    )
    config_vals = tuple(result[k] for k in config_keys)
    if config_vals not in found:
      found.add(config_vals)
      # We still want to return the dictionary form:
      config = {k: v for k, v in result.items() if keep[k]}
      # If named=True, we want to add the testcase name first:
      if named:
        config |= {
            'testcase_name': '_'.join(
                f'{k[0]}-{shorthands[v]}' for k, v in config.items()
            )
        }

      # We still want to return the dictionary form:
      reduced.append(config)

  if expected:
    for reduced_config in reduced:
      args = []
      if param_dtype in reduced_config:
        args.append(reduced_config[param_dtype])
      if input_dtype in reduced_config:
        args.append(reduced_config[input_dtype])
      reduced_config[expected_dtype] = utils.get_promoted_dtype(
          *args,
          dtype=reduced_config.get(compute_dtype),
      )

  return reduced


def zip_longest(
    targets: Iterable[Iterable[Any]],
    sources: Iterable[_T],
) -> list[_T]:
  """Applies zip_longest, specialized to @parameterized's argument format.

  Args:
    targets: Iterable of parameterized test arguments.
    sources: Iterable of parameterized test arguments. If `targets` is a mapping
      `sources` must be a mapping as well.

  Returns:
    A list of the zipped arguments, of the type of `targets` and with each
    zipped argument internally sorted (target, source). If either input sequence
    was longer, the last element of the shorter input sequence is repeated.
  """

  results = []
  prev_source, prev_target = None, None
  for source, target in itertools.zip_longest(sources, targets):
    # If either runs out ahead-of-time, we repeat the final non-None element.
    # (This is safest as we cannot inspect the function's defaults.)
    if source is None:
      source = prev_source
    elif target is None:
      target = prev_target

    if isinstance(target, Mapping):
      assert isinstance(source, Mapping)
      results.append({**target, **source})
    elif isinstance(sources, Iterable):
      # target is a non-mapping iterable, like tuple or list.
      if isinstance(source, Mapping):
        # To match the target, we replace the source with its unlabeled values.
        source = source.values()
      results.append((*target, *source))
      prev_source, prev_target = source, target
    else:
      raise NotImplementedError(
          f'Targets of type {type(target)=} are unsupported.'
      )

  return results


def named_product(
    first: Iterable[TypingSequence[Any] | Mapping[str, Any]],
    second: Iterable[TypingSequence[Any] | Mapping[str, Any]],
) -> Callable[[_TestFnT], _TestFnT]:
  """Builds named parameters from the product of iterators of named parameters.

  As in parameterized.named_parameters, if an iterator's items are sequences,
  the first element is interpreted as the name. If an iterator's items are
  mappings, the `testcase_name` key is used.

  Args:
    first: Iterable of named parameters, whose names will be the first part of
      the named product's test names.
    second: Iterable of named parameters, whose names will be the second part of
      the named product's test names.

  Returns:
    A decorator that calls the test function with the cartesian product of the
    given iterators, whose items are named parameters with names of the form
    `{first_item_name}_{second_item_name}`. If both iterators' items are
    mappings, the product's items are mappings; otherwise they are ordered
    tuples.

    For example, if `first` is
    `[{**foo, 'testcase_name': 'foo'}, {**bar, 'testcase_name': 'bar'}]` and
    `second` is `[['baz', *baz], ['qux', *qux]]`, the items will be
    `('foo_baz', *foo.values(), *baz), ('foo_qux', *foo.values(), *qux), ...`

  Raises:
    ValueError: A testcase_name is missing; either an iterator item is empty, or
      one is a mapping without a `testcase_name` key.
  """

  results = []

  for p1, p2 in itertools.product(first, second):

    for source, parameters in enumerate([p1, p2]):
      if isinstance(parameters, Mapping):
        if 'testcase_name' not in parameters:
          raise ValueError(
              f'Mapping {parameters} from iterable #{source+1} does not have'
              ' key `testcase_name`.'
          )
      elif not parameters:
        raise ValueError(
            f'An sequence from iterable #{source+1} is empty; the first entry'
            ' is expected to be a testcase name.'
        )

    # When both are mappings, we merge by key:
    if isinstance(p1, Mapping) and isinstance(p2, Mapping):
      testcase_name = f'{p1["testcase_name"]}_{p2["testcase_name"]}'
      p1 = {k: v for k, v in p1.items() if k != 'testcase_name'}
      p2 = {k: v for k, v in p2.items() if k != 'testcase_name'}
      results.append({**p1, **p2, 'testcase_name': testcase_name})

    # Else, we return an ordered tuple based on each parameter set's order:
    else:

      if isinstance(p1, Mapping):
        p1_name = p1['testcase_name']
        p1 = tuple(v for k, v in p1.items() if k != 'testcase_name')
      else:
        p1_name = p1[0]
        p1 = p1[1:]

      if isinstance(p2, Mapping):
        p2_name = p2['testcase_name']
        p2 = tuple(v for k, v in p2.items() if k != 'testcase_name')
      else:
        p2_name = p2[0]
        p2 = p2[1:]

      testcase_name = f'{p1_name}_{p2_name}'
      results.append((testcase_name, *p1, *p2))

  return parameterized.named_parameters(*results)


def get_grad_tols(
    l: types.SequenceLayer,
    x: types.Sequence,
    param_dtype: types.DType | None,
    compute_dtype: types.DType | None = None,
) -> dict[str, float | None]:
  """Estimate reasonable tolerances given the layer, inputs, and dtypes."""
  del l, x
  supported_param_dtypes = {jnp.float32, jnp.bfloat16, jnp.float16, None}
  supported_compute_dtypes = {jnp.float32, jnp.bfloat16, jnp.float16, None}
  if param_dtype not in supported_param_dtypes:
    raise ValueError(f'Unsupported param_dtype: {param_dtype}')
  if compute_dtype not in supported_compute_dtypes:
    raise ValueError(f'Unsupported compute_dtype: {compute_dtype}')
  # If no param_dtype is specified, defer to the callee's defaults.
  if param_dtype is None:
    return {}

  if param_dtype == jnp.float32 and (
      compute_dtype is None or compute_dtype == jnp.float32
  ):
    return {'grad_rtol': 1e-5, 'grad_atol': 1e-5}
  else:
    return {'grad_rtol': 1e-1, 'grad_atol': 1e-1}


def flax_init(layer: nn.Module, *args, **kwargs):
  """Initialize a Flax Module with optional jax.jit wrapping it."""
  method = kwargs.pop('method', '__call__')
  should_jit = kwargs.pop('jit', False)

  def init_layer(*args, **kwargs):
    return layer.init(*args, **kwargs, method=method)

  if should_jit:
    init_layer = jax.jit(init_layer)
  return init_layer(*args, **kwargs)


def flax_apply(layer: nn.Module, params, *args, **kwargs):
  method = kwargs.pop('method', '__call__')
  should_jit = kwargs.pop('jit', True)

  def layer_fn(params, *args, **kwargs):
    return layer.apply(params, *args, **kwargs, method=method)

  if should_jit:
    layer_fn = jax.jit(layer_fn)
  return layer_fn(params, *args, **kwargs)


def sl_init(layer: types.SequenceLayer, *args, **kwargs):
  training = kwargs.pop('training', False)
  method = kwargs.pop('method', '__call__')
  should_jit = kwargs.pop('jit', True)

  def init_layer(*args, **kwargs):
    return layer.init(*args, **kwargs, training=training, method=method)

  if should_jit:
    init_layer = jax.jit(init_layer)
  return init_layer(*args, **kwargs)


def sl_layer(
    layer: types.SequenceLayer, params, *args, **kwargs
) -> types.MaskedSequence:
  """Runs layer, masking the output with optional jax.jit wrapping."""
  training = kwargs.pop('training', False)
  method = kwargs.pop('method', 'layer')
  should_jit = kwargs.pop('jit', True)

  def layer_fn(params, *args, **kwargs):
    return layer.apply(
        params, *args, **kwargs, training=training, method=method
    ).mask_invalid()

  if should_jit:
    layer_fn = jax.jit(layer_fn)
  return layer_fn(params, *args, **kwargs)


def sl_layer_with_emits(
    layer: types.SequenceLayer, params, *args, **kwargs
) -> tuple[types.MaskedSequence, types.Emits]:
  """Runs layer_with_emits masking the output with optional jax.jit wrapping."""
  training = kwargs.pop('training', False)
  method = kwargs.pop('method', 'layer_with_emits')
  should_jit = kwargs.pop('jit', True)

  def layer_fn(params, *args, **kwargs):
    output, emits = layer.apply(
        params, *args, **kwargs, training=training, method=method
    )
    return output.mask_invalid(), emits

  if should_jit:
    layer_fn = jax.jit(layer_fn)
  return layer_fn(params, *args, **kwargs)


def pad_batch_axis_with_garbage(tree):
  """Pads the batch dimension of all arrays in tree with garbage values."""

  def pad_with_garbage(
      x: types.Sequence | jax.Array,
  ) -> types.Sequence | jax.Array:
    paddings = [(0, 0)] * x.ndim
    paddings[0] = (1, 1)

    if issubclass(x.dtype.type, jnp.floating):
      pad_value = jnp.nan
    elif issubclass(x.dtype.type, jnp.integer):
      pad_value = jnp.iinfo(x.dtype).max
    else:
      pad_value = None

    if isinstance(x, jax.Array):
      return jnp.pad(x, paddings, constant_values=pad_value)
    else:
      return type(x)(
          jnp.pad(x.values, paddings, constant_values=pad_value),
          jnp.pad(x.mask, [(1, 1), (0, 0)], constant_values=True),
      )

  return jax.tree.map(
      pad_with_garbage, tree, is_leaf=lambda x: isinstance(x, types.Sequence)
  )


def strip_batch_axis_of_garbage(tree):
  """Strips garbage values from the batch dimension of all arrays in tree."""
  return jax.tree.map(
      lambda a: a[1:-1], tree, is_leaf=lambda x: isinstance(x, types.Sequence)
  )


def _mask_and_pad_to_max_length(
    a: types.Sequence, b: types.Sequence
) -> tuple[types.Sequence, types.Sequence]:
  # Only compare values in non-masked regions.
  a = a.mask_invalid()
  b = b.mask_invalid()
  a_time = a.values.shape[1]
  b_time = b.values.shape[1]
  max_time = max(a_time, b_time)
  a = a.pad_time(0, max_time - a_time, valid=False)
  b = b.pad_time(0, max_time - b_time, valid=False)
  return a, b


class SequenceLayerTest(parameterized.TestCase):
  """Base class for SequenceLayer tests."""

  def setUp(self):
    super().setUp()
    # To avoid flakes, fix random seeds.
    random.seed(123456789)
    np.random.seed(123456789)

  def init_and_bind_layer(
      self,
      key: jax.Array,
      layer: _SequenceLayerT,
      x: types.Sequence,
      randomize_weights: bool = False,
      constants: types.Constants | None = None,
      jit: bool = False,
  ) -> _SequenceLayerT:
    """Initializes and binds a SequenceLayer for testing.

    Args:
      key: Random key for generating weights.
      layer: Layer to initialize and bind.
      x: Example input sequence to use for initialization.
      randomize_weights: Whether to replace all variables with random normal
        generations of the same shape and dtype.
      constants: Constants to provide to layer, if needed.
      jit: Whether to jit the initialization function.

    Returns:
      The provided layer, bound with initialized weights.
    """

    def init_fn(x: types.Sequence, constants: types.Constants):
      return layer.init(key, x, training=False, constants=constants)

    if jit:
      init_fn = jax.jit(init_fn)

    variables = init_fn(x, constants)

    if randomize_weights:

      @jax.jit
      def randomize_weights_fn(variables):
        variables, variables_pytree = jax.tree.flatten(variables)
        # TODO(rryan): Use jax.random.split here.
        keys = [key] * len(variables)
        variables = [
            jax.random.normal(k, v.shape, v.dtype)
            for k, v in zip(keys, variables)
        ]
        return jax.tree.unflatten(variables_pytree, variables)

      variables = randomize_weights_fn(variables)

    return layer.bind(variables)

  def verify_masked(self, x: types.Sequence):
    """Asserts all invalid timesteps in x have values masked to zero."""
    # Manually mask even if x is a MaskedSequence.
    expected = types.Sequence(x.values, x.mask).mask_invalid()
    self.assertAllEqual(x.values, expected.values)

  def verify_contract(
      self,
      l: types.SequenceLayer,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
      stream_constants: bool = False,
      rtol: float = 1e-6,
      atol: float = 1e-6,
      test_gradients: bool = True,
      grad_rtol=None,
      grad_atol=None,
      padding_invariance_pad_value: float = jnp.nan,
      pad_constants: bool = False,
      pad_constants_ratio: int = 1,
      test_2x_step: bool = True,
      jit: bool = False,
      test_batching: bool = True,
      test_padding_invariance: bool = True,
  ) -> types.Sequence:
    """Verifies that the provided layer obeys the SequenceLayer contract.

    The contract has three main requirements:
    1. Layer-wise and step-wise equivalence of outputs and gradients.
    2. Step must support any multiple of block size.
    3. Padding invariance.

    This function tests l for all three using the provided input x, training
    mode, and constants.

    To test layer-wise and step-wise equivalence, the layer is executed
    layer-wise and step-wise on x and the valid regions are compared for
    equivalence up to rtol/atol. If test_gradients is True, the gradient of the
    input x with respect to the output y and the gradient of the layer
    parameters with respect to y are measured and checked for equivalence up to
    grad_rtol/grad_atol.

    To test that step supports sequences whose length is any multiple of its
    block size, the layer is run on step-wise on inputs of length 2 *
    l.block_size and the output is compared to the step-wise output on inputs of
    length l.block_size.

    To test padding invariance, all invalid values in the sequence are replaced
    with padding_invariance_pad_value and the sequence and constants are
    extended by 4 block_sizes worth of padding. The layer-wise output is checked
    for equivalence on this padded sequence.

    If the provided layer is not steppable, only padding invariance is tested.

    Args:
      l: The layer to test.
      x: The sequence to use as input.
      training: Whether to verify the contract in training mode.
      constants: Optional constants to provide to the layer.
      stream_constants: If True, stream Sequences present in constants at the
        same block size as x.
      rtol: The relative tolerance to test equality with.
      atol: The absolute tolerance to test equality with.
      test_gradients: Whether to compute and compare the gradients for l applied
        to x in step-wise and layer-wise mode.
      grad_rtol: Optional relative tolerance to test layer/step gradient
        equivalence to. Uses rtol if not specified.
      grad_atol: Optional absolute tolerance to test layer/step gradient
        equivalence to. Uses atol if not specified.
      padding_invariance_pad_value: The value to use for testing padding
        invariance. By default, use NaN to show that NaNs do not leak into valid
        regions of the sequence.
      pad_constants: Whether to pad sl.Sequence instances in constants along
        with padding x with padding_invariance_pad_value, when testing padding
        invariance.
      pad_constants_ratio: The ratio between the time dimension of x and
        sequences in constants.
      test_2x_step: Whether to test that the layer can be run in step-wise mode
        on sequences whose length is twice the block size.
      jit: If true, jits all functions associated with the model.
      test_batching: Whether to increase the batch size to verify batching
        works.
      test_padding_invariance: Whether to test the layer for padding invariance.

    Returns:
      The output of
        l.layer(x, training=training, constants=constants).mask_invalid().
      This is a convenience for tests that want to do correctness testing after
      verifying the contract, without repeating the work of the layer method.
    """
    if l.scope is None:
      raise ValueError(f'Expected {l=} to be bound to variables for testing.')

    if grad_rtol is None:
      grad_rtol = rtol
    if grad_atol is None:
      grad_atol = atol

    self.verify_masked(x)

    x_spec = x.channel_spec
    y_spec = l.get_output_spec(x_spec, constants=constants)
    output_latency = int(l.output_latency)
    input_latency = int(output_latency / l.output_ratio)

    def get_initial_state_fn(
        l: types.SequenceLayer, batch_size: int, constants: types.Constants
    ) -> types.State:
      return l.get_initial_state(
          batch_size, x_spec, training=training, constants=constants
      )

    def get_dy(y: types.Sequence) -> types.Sequence:
      # Do not flow gradient through masked regions by masking dy.
      dy = types.Sequence(jnp.ones_like(y.values), y.mask).mask_invalid().values
      dmask = jnp.zeros_like(y.mask)
      return type(y)(dy, dmask)

    def layer_fn(
        l: types.SequenceLayer, x: types.Sequence, constants: types.Constants
    ) -> types.Sequence:
      return l.layer(x, training=training, constants=constants).mask_invalid()

    def layer_vjp_fn(
        l: types.SequenceLayer, x: types.Sequence, constants: types.Constants
    ):
      y, layer_vjp_fn = nn.vjp(layer_fn, l, x, constants)
      params_grad, x_grad, unused_constants_grad = layer_vjp_fn(get_dy(y))
      x_grad = types.Sequence(x_grad.values, x.mask).mask_invalid()
      return y, x_grad, params_grad

    def step_fn(
        l: types.SequenceLayer,
        x: types.Sequence,
        constants: types.Constants,
        *,
        blocks_per_step: int = 1,
    ) -> tuple[types.Sequence, types.State]:
      state = get_initial_state_fn(l, x.shape[0], constants)
      y, new_state, _ = utils.step_by_step_static(
          l,
          x,
          training=training,
          initial_state=state,
          constants=constants,
          stream_constants=stream_constants,
          blocks_per_step=blocks_per_step,
      )
      chex.assert_trees_all_equal_shapes_and_dtypes(new_state, state)
      # Trim output latency. Doing so here avoids awkwardness with vjp.
      y = y[:, output_latency:]
      return y.mask_invalid(), new_state

    def step_vjp_fn(
        l: types.SequenceLayer,
        x: types.Sequence,
        constants: types.Constants,
    ):
      (y, new_state), step_vjp_fn = nn.vjp(step_fn, l, x, constants)
      dy = get_dy(y)
      dstate = jax.tree_util.tree_map(jnp.zeros_like, new_state)
      params_grad, x_grad, unused_constants_grad = step_vjp_fn((dy, dstate))
      x_grad = types.Sequence(x_grad.values, x.mask).mask_invalid()
      return y, new_state, x_grad, params_grad

    if jit:
      layer_fn = nn.jit(layer_fn)
      layer_vjp_fn = nn.jit(layer_vjp_fn)
      step_fn = nn.jit(step_fn, static_argnames=('blocks_per_step',))
      step_vjp_fn = nn.jit(step_vjp_fn)

    if test_gradients:
      y_layer, y_layer_x_grad, y_layer_params_grad = layer_vjp_fn(
          l, x, constants
      )
    else:
      y_layer = layer_fn(l, x, constants)
      y_layer_x_grad = None
      y_layer_params_grad = None

    chex.assert_equal(y_layer.channel_shape, y_spec.shape)
    if y_layer.dtype != y_spec.dtype:
      raise ValueError(
          f'{y_layer.dtype=} does not match the output spec {y_spec.dtype=}.'
      )

    if not y_layer.values.size:
      raise ValueError(
          f'Layer output is empty, probably unintended: {y_layer=}.'
      )

    # Pad with 4 blocks of invalid garbage data.
    pad_amount = 4 * l.block_size

    def _pad(x: types.Sequence, pad_back: int) -> types.Sequence:
      """Pad pad_back steps and masks with padding_invariance_pad_value."""
      return x.pad_time(0, pad_back, valid=False).mask_invalid(
          padding_invariance_pad_value
      )

    x_padded, constants_padded = x, constants
    if test_padding_invariance:
      x_padded = _pad(x, pad_amount)
      if pad_constants and constants_padded is not None:
        constants_padded = {
            k: (
                _pad(v, pad_amount * pad_constants_ratio)
                if isinstance(v, types.Sequence)
                else v
            )
            for k, v in constants.items()
        }
      y_layer_padded = layer_fn(l, x_padded, constants_padded)
      self.assertSequencesClose(y_layer, y_layer_padded, rtol=rtol, atol=atol)

    x_batch = None
    constants_batch = None
    if test_batching:
      x_batch, constants_batch = pad_batch_axis_with_garbage((x, constants))
      y_layer_batch = layer_fn(l, x_batch, constants_batch)
      y_layer_batch = strip_batch_axis_of_garbage(y_layer_batch)
      self.assertSequencesClose(y_layer, y_layer_batch, rtol=rtol, atol=atol)

    if l.supports_step:
      # Extend x by the input latency so we run the layer until all outputs are
      # flushed.
      x = x.pad_time(0, input_latency, valid=False)
      if test_padding_invariance:
        x_padded = x_padded.pad_time(0, input_latency, valid=False)

      if test_gradients:
        y_step, _, y_step_x_grad, y_step_params_grad = step_vjp_fn(
            l, x, constants
        )
      else:
        y_step, _ = step_fn(l, x, constants)
        y_step_x_grad = None
        y_step_params_grad = None

      if test_2x_step:
        y_step_2x, _ = step_fn(l, x, constants, blocks_per_step=2)
      else:
        y_step_2x = None

      if test_batching:
        assert x_batch is not None
        x_batch = x_batch.pad_time(0, input_latency, valid=False)
        y_step_batch, _ = step_fn(l, x_batch, constants_batch)
        y_step_batch = strip_batch_axis_of_garbage(y_step_batch)
        self.assertSequencesClose(y_step, y_step_batch, rtol=rtol, atol=atol)

      # Property 1: Check layer-wise and step-wise equivalence.
      self.assertSequencesClose(y_layer, y_step, rtol=rtol, atol=atol)
      if test_2x_step:
        self.assertSequencesClose(y_layer, y_step_2x, rtol=rtol, atol=atol)

      # Property 2: Padding invariance.
      if test_padding_invariance:
        y_step_padded, _ = step_fn(l, x_padded, constants_padded)
        self.assertSequencesClose(y_step, y_step_padded, rtol=rtol, atol=atol)

      if test_gradients:
        # Property 1: Check layer-wise and step-wise equivalence of gradients.
        # Skip comparison of JAX float0 types, which are used in
        # jvp/linearization of integers / booleans. This likely means the input
        # is an integer type. go/jax-integer-autodiff
        assert y_layer_x_grad is not None
        if y_layer_x_grad.dtype != jax.dtypes.float0:
          self.assertSequencesClose(
              y_layer_x_grad, y_step_x_grad, rtol=grad_rtol, atol=grad_atol
          )
        chex.assert_trees_all_close(
            y_layer_params_grad,
            y_step_params_grad,
            rtol=grad_rtol,
            atol=grad_atol,
        )

    return y_layer

  def assertEmitsCompatible(  # pylint: disable=invalid-name
      self, emit_specs: types.EmitSpecs, emits: types.Emits
  ):
    """Checks the provided emits are compatible with emit_specs."""

    def check_compatible_fn(emit_spec, emit) -> bool:
      if emit_spec.dtype != emit.dtype:
        return False
      try:
        utils.assert_is_compatible_with(emit.shape[2:], emit_spec.shape)
        return True
      except ValueError:
        return False

    def error_message_fn(emit_spec, emit) -> str:
      if isinstance(emit, types.Sequence):
        emit = emit.channel_spec
      return f'{emit_spec=} does not match {emit=}.'

    chex.assert_trees_all_equal_comparator(
        check_compatible_fn, error_message_fn, emit_specs, emits
    )

  def assertSequencesClose(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    """After padding, checks sequence values are close and masks are equal."""
    a, b = _mask_and_pad_to_max_length(a, b)
    self.assertAllClose(a.values, b.values, atol=atol, rtol=rtol)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesNotClose(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
      atol: float = 1e-6,
      rtol: float = 1e-6,
  ):
    """After padding, checks sequence values aren't close, masks are equal."""
    a, b = _mask_and_pad_to_max_length(a, b)
    self.assertNotAllClose(a.values, b.values, atol=atol, rtol=rtol)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesEqual(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
  ):
    """After padding, checks sequence values are equal and masks are equal."""
    a, b = _mask_and_pad_to_max_length(a, b)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesNotEqual(  # pylint: disable=invalid-name
      self,
      a: types.Sequence,
      b: types.Sequence,
  ):
    """After padding, checks sequence values aren't equal, masks are equal."""
    a, b = _mask_and_pad_to_max_length(a, b)
    self.assertNotAllEqual(a.values, b.values)
    self.assertAllEqual(a.mask, b.mask)

  def assertAllEqual(self, a, b):  # pylint: disable=invalid-name
    """Asserts that two arrays are equal."""
    if jnp.iscomplexobj(a) or jnp.iscomplexobj(b):
      a_real, a_imag = jnp.real(a), jnp.imag(a)
      b_real, b_imag = jnp.real(b), jnp.imag(b)
      chex.assert_trees_all_equal(a_real, b_real)
      chex.assert_trees_all_equal(a_imag, b_imag)
    else:
      chex.assert_trees_all_equal(a, b)

  def assertAllClose(self, a, b, atol: float = 1e-6, rtol: float = 1e-6):  # pylint: disable=invalid-name
    """Asserts that two arrays have close values."""
    if jnp.iscomplexobj(a) or jnp.iscomplexobj(b):
      a_real, a_imag = jnp.real(a), jnp.imag(a)
      b_real, b_imag = jnp.real(b), jnp.imag(b)
      chex.assert_trees_all_close(a_real, b_real, atol=atol, rtol=rtol)
      chex.assert_trees_all_close(a_imag, b_imag, atol=atol, rtol=rtol)
    else:
      chex.assert_trees_all_close(a, b, atol=atol, rtol=rtol)

  def assertNotAllEqual(self, a, b):  # pylint: disable=invalid-name
    """Asserts that two arrays do not have equal values."""
    try:
      chex.assert_trees_all_equal(a, b)
    except AssertionError:
      return
    raise AssertionError(
        'The two values are equal at all elements. %s %s' % (a, b)
    )

  def assertNotAllClose(self, a, b, atol: float = 1e-6, rtol: float = 1e-6):  # pylint: disable=invalid-name
    """Asserts that two arrays do not have close values."""
    try:
      self.assertAllClose(a, b, atol=atol, rtol=rtol)
    except AssertionError:
      return
    raise AssertionError(
        'The two values are close at all elements. %s %s' % (a, b)
    )


class AssertConstantsLayer(types.PreservesType, types.StatelessPointwise):
  """Identity layer that raises ValueError if 'test' is not in constants."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'AssertConstantsLayer':
      return AssertConstantsLayer(self, name=self.name)

  config: Config

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ChannelSpec,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return super().get_initial_state(
        batch_size, input_spec, training=training, constants=constants
    )

  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return super().get_output_shape(input_shape, constants=constants)

  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return x


class NonSteppableLayer(types.PreservesType, types.StatelessPointwise):
  """A test layer that does not support stepping."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    name: str | None = None

    def make(self) -> 'NonSteppableLayer':
      return NonSteppableLayer(self, name=self.name)

  config: Config

  @property
  def supports_step(self):
    return False

  def layer(
      self,
      x: types.Sequence,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.Sequence:
    del training
    del constants
    return x


# Forward main for convenience.
main = googletest.main
