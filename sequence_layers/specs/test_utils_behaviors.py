"""Abstract tests for test utilities."""

# pylint: disable=abstract-method

import fractions
from typing import Any, override
from unittest import mock

from absl.testing import parameterized
import numpy as np

from sequence_layers import specs
from sequence_layers.specs import test_utils as test_utils_spec
from sequence_layers.specs import types as types_spec


class ModuleSpecTest(test_utils_spec.ModuleSpecTest):

  @override
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec):
    return {backend_sl.test_utils: test_utils_spec.ModuleSpec}


class NamedProductTest(test_utils_spec.SequenceLayerTest):

  @parameterized.parameters(
      {
          'first': [('a', 'alpha'), ('b', 'beta')],
          'second': [('1', 1), ('2', 2), ('3', 3)],
          'expected': [
              ('a_1', 'alpha', 1),
              ('a_2', 'alpha', 2),
              ('a_3', 'alpha', 3),
              ('b_1', 'beta', 1),
              ('b_2', 'beta', 2),
              ('b_3', 'beta', 3),
          ],
      },
      {
          'first': [{'a': 'alpha', 'testcase_name': 'test'}],
          'second': [('1', 1), ('2', 2)],
          'expected': [
              ('test_1', 'alpha', 1),
              ('test_2', 'alpha', 2),
          ],
      },
      {
          'first': [
              {'letter': 'a', 'testcase_name': 'alpha'},
              {'testcase_name': 'beta', 'letter': 'b'},
          ],
          'second': [
              {'testcase_name': 'one', 'number': 1},
              {'number': 2, 'testcase_name': 'two'},
          ],
          'expected': [
              {'letter': 'a', 'number': 1, 'testcase_name': 'alpha_one'},
              {'letter': 'a', 'number': 2, 'testcase_name': 'alpha_two'},
              {'letter': 'b', 'number': 1, 'testcase_name': 'beta_one'},
              {'letter': 'b', 'number': 2, 'testcase_name': 'beta_two'},
          ],
      },
  )
  @mock.patch.object(parameterized, 'named_parameters', autospec=True)
  def test_builds_named_products(self, mock_fn, first, second, expected):
    self.sl.test_utils.named_product(first, second)
    self.assertSequenceEqual(mock_fn.call_args.args, expected)

  @parameterized.parameters(
      {
          'first': [{'testcase_name': 'alpha', 'letter': 'a'}, {'letter': 'b'}],
          'second': [('1', 1), ('2', 2), ('3', 3)],
          'iterator_without_testcase_name': 1,
      },
      {
          'first': [{'testcase_name': 'alpha', 'letter': 'a'}],
          'second': [('1', 1), ()],
          'iterator_without_testcase_name': 2,
      },
  )
  def test_raises_on_missing_testcase_names(
      self, first, second, iterator_without_testcase_name
  ):
    with self.assertRaisesRegex(
        ValueError, str(iterator_without_testcase_name)
    ):
      self.sl.test_utils.named_product(first, second)


class ZipLongestTest(test_utils_spec.SequenceLayerTest):

  @parameterized.parameters(
      {
          'targets': [('a',), ('b',)],
          'sources': [(1,), (2,)],
          'expected': [('a', 1), ('b', 2)],
      },
      {
          'targets': [('a',), ('b',)],
          'sources': [(1,)],
          'expected': [('a', 1), ('b', 1)],
      },
      {
          'targets': [('a',)],
          'sources': [(1,), (2,)],
          'expected': [('a', 1), ('a', 2)],
      },
      {
          'targets': [{'testcase_name': 'a'}],
          'sources': [{'val': 1}],
          'expected': [{'testcase_name': 'a', 'val': 1}],
      },
  )
  def test_zip_longest(self, targets, sources, expected):
    results = self.sl.test_utils.zip_longest(targets, sources)
    self.assertEqual(results, expected)


class GenericDummyLayer(types_spec.SequenceLayer):
  """Generic dummy layer for testing verify_contract."""

  @override
  def layer(
      self,
      x: types_spec.Sequence,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.Sequence:
    return x

  @override
  def step(
      self,
      x: types_spec.Sequence,
      state: types_spec.State,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> tuple[types_spec.Sequence, types_spec.State]:
    return x, state

  @override
  def step_with_emits(
      self,
      x: types_spec.Sequence,
      state: types_spec.State,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> tuple[types_spec.Sequence, types_spec.State, types_spec.Emits]:
    y, state = self.step(x, state, constants=constants, training=training)
    return y, state, ()

  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types_spec.ChannelSpec,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.State:
    return None

  @property
  @override
  def receptive_field(self) -> tuple[int, int]:
    return (0, 0)

  @property
  @override
  def block_size(self) -> int:
    return 1

  @property
  @override
  def output_ratio(self) -> fractions.Fraction:
    return fractions.Fraction(1)

  @property
  @override
  def input_latency(self) -> int:
    return 0

  @property
  @override
  def output_latency(self) -> int:
    return 0

  @property
  @override
  def supports_step(self) -> bool:
    return True

  @override
  def get_accumulated_input_latency(self, input_latency: int) -> int:
    return input_latency

  @override
  def get_accumulated_output_latency(self, output_latency: int) -> int:
    return output_latency

  @override
  def layer_with_emits(
      self,
      x: types_spec.Sequence,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> tuple[types_spec.Sequence, types_spec.Emits]:
    return self.layer(x, training=training, constants=constants), ()

  @override
  def get_output_shape(
      self,
      input_shape: types_spec.ShapeLike,
      *,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.Shape:
    return tuple(input_shape)

  @override
  def get_output_dtype(
      self,
      input_dtype: types_spec.DType,
      *,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.DType:
    return input_dtype

  @override
  def get_output_spec(
      self,
      input_spec: Any,
      *,
      constants: types_spec.Constants | None = None,
  ) -> Any:
    shape = self.get_output_shape(input_spec.shape, constants=constants)
    dtype = self.get_output_dtype(input_spec.dtype, constants=constants)

    class Spec:
      """Dummy spec class."""

      def __init__(self, s, d):
        self.shape = s
        self.dtype = d

    return Spec(shape, dtype)


class GenericMismatchedDummyLayer(GenericDummyLayer):
  """Dummy layer that induces a mismatch by returning zeros in layer()."""

  @override
  def layer(
      self,
      x: types_spec.Sequence,
      *,
      training: bool,
      constants: types_spec.Constants | None = None,
  ) -> types_spec.Sequence:
    return x.apply_values(lambda v: v * 0.0)


class VerifyContractTest(test_utils_spec.SequenceLayerTest):
  """Abstract tests for verify_contract."""

  def get_dummy_layer(self, mismatch: bool) -> Any:
    """Returns a dummy layer for testing."""
    backend_sl = self.sl

    if mismatch:

      class BackendMismatchedDummyLayer(
          GenericMismatchedDummyLayer, backend_sl.types.SequenceLayer
      ):
        """Mismatched dummy layer for backend."""

      return BackendMismatchedDummyLayer()

    class BackendDummyLayer(GenericDummyLayer, backend_sl.types.SequenceLayer):
      """Dummy layer for backend."""

    return BackendDummyLayer()

  def test_verify_contract_catches_step_mismatch(self):
    layer = self.get_dummy_layer(mismatch=True)

    x = self.sl.Sequence(
        self.xp.array(np.ones((2, 5, 10))),
        self.xp.array(np.ones((2, 5), dtype=bool)),
    )

    with self.assertRaises(AssertionError):
      self.verify_contract(layer, x, training=False)

  def test_verify_contract_succeeds_when_equivalent(self):
    layer = self.get_dummy_layer(mismatch=False)

    x = self.sl.Sequence(
        self.xp.array(np.ones((2, 5, 10))),
        self.xp.array(np.ones((2, 5), dtype=bool)),
    )

    self.verify_contract(layer, x, training=False)

  def test_verify_contract_handles_stream_constants(self):
    layer = self.get_dummy_layer(mismatch=False)

    x = self.sl.Sequence(
        self.xp.array(np.ones((2, 5, 10))),
        self.xp.array(np.ones((2, 5), dtype=bool)),
    )
    constants = {
        'c': self.sl.Sequence(
            self.xp.array(np.ones((2, 5, 1))),
            self.xp.array(np.ones((2, 5), dtype=bool)),
        )
    }

    self.verify_contract(
        layer, x, training=False, constants=constants, stream_constants=True
    )
