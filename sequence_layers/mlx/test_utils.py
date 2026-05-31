"""Test utilities for MLX sequence layers."""

import dataclasses
from typing import Any, Callable, Iterable, Mapping, override
from typing import Sequence as TypingSequence
from typing import TypeVar

from absl.testing import absltest
import mlx.core as mx
import numpy as np

from sequence_layers import specs
from sequence_layers.mlx import types
from sequence_layers.specs import test_utils as spec

Sequence = types.Sequence
MaskedSequence = types.MaskedSequence
ShapeDType = types.ShapeDType

_T = TypeVar('_T')
_TestFnT = Callable[..., None]


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
  return spec.zip_longest(targets, sources)


def random_sequence(
    *dims: int,
    dtype=None,
    random_mask: bool = False,
    random_lengths: bool | None = None,
    low: int | None = 0,
    high: int | None = 10,
    low_length: int = 0,
    high_length: int | None = None,
) -> types.Sequence:
  """Generates a random sequence for MLX testing."""
  # pylint: disable=unused-argument
  if len(dims) < 2:
    raise ValueError('dims must be at least (batch, time)')
  batch_size = dims[0]
  time = dims[1]
  shape = dims[2:]

  if dtype is not None:
    if dtype == np.float32:
      dtype = mx.float32
    elif dtype == np.float16:
      dtype = mx.float16
    elif dtype == np.int32:
      dtype = mx.int32
    elif dtype == np.bool_:
      dtype = mx.bool_

  if dtype is not None and dtype in (
      mx.int32,
      mx.int16,
      mx.int8,
      mx.uint32,
      mx.uint16,
      mx.uint8,
  ):
    values_np = np.random.randint(
        low if low is not None else 0,
        high if high is not None else 10,
        size=(batch_size, time) + shape,
    )
  else:
    values_np = np.random.normal(size=(batch_size, time) + shape).astype(
        np.float32
    )
  values = mx.array(values_np, dtype=dtype or mx.float32)

  mask_np = np.ones((batch_size, time), dtype=bool)
  mask = mx.array(mask_np, dtype=mx.bool_)

  return types.Sequence(values, mask)


def step_by_step(
    layer: types.SequenceLayer,
    x: types.Sequence,
    *,
    block_size: int = 1,
    constants=None,
    stream_constants=None,
    stream_constants_list: list[Any] | None = None,
) -> tuple[types.Sequence, Any]:
  """Applies step-by-step processing to the sequence using MLX step functions."""
  batch = x.values.shape[0] if hasattr(x, 'values') else x.shape[0]
  time = x.values.shape[1] if hasattr(x, 'values') else x.shape[1]
  remainder = time % block_size
  if remainder != 0:
    pad_amount = block_size - remainder
    x = x.pad_time(0, pad_amount, valid=False)
    time = x.values.shape[1]

  # Pad to multiple of block_size.
  num_blocks = (time + block_size - 1) // block_size
  padded_time = num_blocks * block_size
  pad_amount = padded_time - time
  x = x.pad_time(0, pad_amount, valid=False)
  time = padded_time

  input_spec = types.ShapeDType(x.channel_shape, x.dtype)

  # Handle JAX-style stream_constants (dict of sequences or bool)
  if isinstance(stream_constants, bool) and stream_constants:
    stream_source = constants
  elif isinstance(stream_constants, dict):
    stream_source = stream_constants
  else:
    stream_source = None

  if stream_source and stream_constants_list is None:
    padded_stream_source = {}
    for name, seq in stream_source.items():
      if hasattr(seq, 'pad_time'):
        pad_back = max(0, time - seq.values.shape[1])
        padded_stream_source[name] = seq.pad_time(0, pad_back, valid=False)
      else:
        padded_stream_source[name] = seq
    stream_source = padded_stream_source

    stream_constants_list = []
    for t in range(0, time, block_size):
      step_dict = {}
      for name, seq in stream_source.items():
        if hasattr(seq, 'values') and hasattr(seq, 'mask'):
          step_dict[name] = types.Sequence(
              seq.values[:, t : t + block_size],
              seq.mask[:, t : t + block_size],
          )
      stream_constants_list.append(step_dict)

  init_constants = dict(constants) if constants else {}
  if isinstance(stream_constants, dict):
    init_constants.update(stream_constants)

  state = layer.get_initial_state(
      batch, input_spec, constants=init_constants or None, training=False
  )

  outputs_values = []
  outputs_masks = []

  for t in range(0, time, block_size):
    x_block = types.Sequence(
        x.values[:, t : t + block_size],
        x.mask[:, t : t + block_size],
    )

    step_constants = dict(constants) if constants else {}
    if stream_constants_list:
      step_idx = t // block_size
      if step_idx < len(stream_constants_list):
        step_constants.update(stream_constants_list[step_idx])

    y_block, state = layer.step(
        x_block, state, constants=step_constants or None, training=False
    )
    outputs_values.append(y_block.values)
    outputs_masks.append(y_block.mask)

  y_values = mx.concatenate(outputs_values, axis=1)
  y_mask = mx.concatenate(outputs_masks, axis=1)

  return types.Sequence(y_values, y_mask), state


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
  """
  return spec.named_product(first, second)


def _mask_and_pad_to_max_length(
    a: types.Sequence, b: types.Sequence
) -> tuple[types.Sequence, types.Sequence]:
  """Masks and pads two sequences to the same max length."""
  # Only compare values in non-masked regions.
  a = a.mask_invalid()
  b = b.mask_invalid()
  a_time = a.values.shape[1]
  b_time = b.values.shape[1]
  max_time = max(a_time, b_time)
  a = a.pad_time(0, max_time - a_time, valid=False)
  b = b.pad_time(0, max_time - b_time, valid=False)
  return a, b


class SequenceLayerTest(spec.SequenceLayerTest):
  """Base class for MLX SequenceLayer tests."""

  import sequence_layers.mlx as sl_module  # pylint: disable=import-outside-toplevel

  sl = sl_module  # pyrefly: ignore[bad-assignment]  # module-as-protocol

  @override
  def setUp(self):
    super().setUp()
    # To avoid flakes, fix random seeds.
    # MLX doesn't have a global seed, but we can set numpy seed.
    np.random.seed(123456789)

  @override
  def get_variables(self, layer: Any) -> dict[str, Any]:

    return layer.parameters()

  @override
  def init_layer(self, layer, x, bind_only=False, constants=None):
    if not bind_only:
      _ = layer.layer(x, training=False, constants=constants)
    return layer

  @override
  def random_sequence(
      self,
      *dims: int,
      dtype=None,
      random_mask: bool = False,
      random_lengths: bool | None = None,
      low: int | None = 0,
      high: int | None = 10,
      low_length: int = 0,
      high_length: int | None = None,
  ) -> types.Sequence:
    return random_sequence(
        *dims,
        dtype=dtype,
        random_mask=random_mask,
        random_lengths=random_lengths,
        low=low,
        high=high,
        low_length=low_length,
        high_length=high_length,
    )

  @override
  def assertEqual(self, first, second, msg=None):
    """Override to handle MLX vs NumPy dtypes."""
    if isinstance(first, mx.Dtype) and isinstance(second, (type, np.dtype)):
      first_str = str(first).rsplit('.', maxsplit=1)[-1]
      second_str = np.dtype(second).name
      if first_str == second_str:
        return
    super().assertEqual(first, second, msg)

  @override
  def assertAllEqual(self, x, y):
    """Asserts that two arrays are equal (or close if float)."""
    x_np = np.array(x) if isinstance(x, mx.array) else np.asarray(x)
    y_np = np.array(y) if isinstance(y, mx.array) else np.asarray(y)
    if np.issubdtype(x_np.dtype, np.floating) or np.issubdtype(
        y_np.dtype, np.floating
    ):
      np.testing.assert_allclose(x_np, y_np, rtol=1e-5, atol=1e-5)
    else:
      np.testing.assert_array_equal(x_np, y_np)

  @override
  def assertSequencesEqual(  # pyrefly: ignore[bad-override]
      self, x: types.Sequence, y: types.Sequence
  ):
    """After padding, checks sequence values are equal and masks are equal."""
    x, y = _mask_and_pad_to_max_length(x, y)
    self.assertAllEqual(x.values, y.values)
    self.assertAllEqual(x.mask, y.mask)

  @override
  # pyrefly: ignore[bad-override]
  def _step_by_step(
      self,
      layer: types.SequenceLayer,
      x: types.Sequence,
      *,
      block_size: int = 1,
      constants=None,
      stream_constants=None,
      stream_constants_list: list[Any] | None = None,
  ) -> tuple[types.Sequence, Any]:
    return step_by_step(
        layer,
        x,
        block_size=block_size,
        constants=constants,
        stream_constants=stream_constants,
        stream_constants_list=stream_constants_list,
    )

  @override
  # pyrefly: ignore[bad-override]
  def verify_contract(
      self,
      l: types.SequenceLayer,
      x: types.Sequence | tuple[int, ...],
      *,
      training: bool = False,
      constants=None,
      stream_constants: bool = False,
      stream_constants_list: list[Any] | None = None,
      atol: float = 1e-5,
      rtol: float = 1e-5,
      **kwargs,
  ) -> types.Sequence:
    if isinstance(x, tuple):
      x = self.random_sequence(2, 5, *x)

    if hasattr(x, 'channel_shape'):
      input_shape = x.channel_shape
    elif hasattr(x, 'shape'):
      input_shape = x.shape[2:]
    else:
      raise ValueError(f'Cannot determine input shape from {x}')
    dtype = x.dtype if hasattr(x, 'dtype') else self.xp.float32

    y_layer = l.layer(x, training=training, constants=constants)

    expected_shape = l.get_output_shape(input_shape, constants=constants)
    self.assertEqual(y_layer.channel_shape, expected_shape)

    expected_dtype = l.get_output_dtype(dtype, constants=constants)
    self.assertEqual(y_layer.dtype, expected_dtype)

    if not l.supports_step:
      return y_layer

    block_size = l.block_size
    x_padded = x.pad_time(0, l.input_latency, valid=False)
    y_step, _ = self._step_by_step(
        l,
        x_padded,
        block_size=block_size,
        constants=constants,
        stream_constants=stream_constants,
        stream_constants_list=stream_constants_list,
    )
    y_step = y_step[:, l.output_latency :]

    self.assertSequencesClose(y_layer, y_step, atol=atol, rtol=rtol)

    return y_layer

  @override
  def assertSequencesClose(self, x: Any, y: Any, **kwargs) -> None:
    def _to_numpy(v):
      if hasattr(v, 'dtype') and v.dtype == mx.bfloat16:
        return np.array(v.astype(mx.float32))
      return np.array(v)

    if hasattr(x, 'values') and hasattr(y, 'values'):
      x, y = _mask_and_pad_to_max_length(x, y)
    x_np = _to_numpy(x.values) if hasattr(x, 'values') else _to_numpy(x)
    y_np = _to_numpy(y.values) if hasattr(y, 'values') else _to_numpy(y)
    # No float16/bfloat16 tolerance relaxation

    np.testing.assert_allclose(x_np, y_np, **kwargs)
    if hasattr(x, 'mask') and hasattr(y, 'mask'):
      mask_x = _to_numpy(x.mask)
      mask_y = _to_numpy(y.mask)
      np.testing.assert_array_equal(mask_x, mask_y)


class ModuleSpecTest(SequenceLayerTest, spec.ModuleSpecTest):

  @override
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec):
    return {backend_sl.test_utils: spec.ModuleSpec}


# pylint: disable=abstract-method
# pylint: disable=abstract-class-instantiated
class NonSteppableLayer(types.PreservesType, types.StatelessPointwise):
  """A test layer that does not support stepping."""

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for NonSteppableLayer."""

    name: str | None = None

    @override
    def make(self) -> 'NonSteppableLayer':
      return NonSteppableLayer(self, name=self.name)

  config: Config

  def __init__(self, config: Config, *, name: str | None = None):
    # pylint: disable=unused-argument
    super().__init__()
    self.config = config

  @property
  @override
  def supports_step(self):
    return False

  @override
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


main = absltest.main
