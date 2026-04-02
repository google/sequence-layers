"""Abstract tests for Sequence types."""

import abc
from types import ModuleType
from typing import Any, Callable, Sequence as TypingSequence, TYPE_CHECKING, cast, override
import dataclasses
from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import types as spec

import fractions
from absl.testing import parameterized
import numpy as np
import unittest.mock


class DefaultTestLayer(spec.SequenceLayer):

  @override
  def layer(
      self,
      x: spec.Sequence,
      *,
      training: bool,
      constants: spec.Constants | None = None,
  ) -> spec.Sequence:
    return x

  @override
  def layer_with_emits(
      self,
      x: spec.Sequence,
      *,
      training: bool,
      constants: spec.Constants | None = None,
  ) -> tuple[spec.Sequence, spec.Emits]:
    return self.layer(x, training=training, constants=constants), (
        'test_emits',
    )

  @override
  def step(
      self,
      x: spec.Sequence,
      state: spec.State,
      *,
      training: bool,
      constants: spec.Constants | None = None,
  ) -> tuple[spec.Sequence, spec.State]:
    return x, ('new_test_state',)

  @override
  def step_with_emits(
      self,
      x: spec.Sequence,
      state: spec.State,
      *,
      training: bool,
      constants: spec.Constants | None = None,
  ) -> tuple[spec.Sequence, spec.State, spec.Emits]:
    return *self.step(x, state, training=training, constants=constants), (
        'test_emits',
    )

  @override
  def get_initial_state(
      self,
      batch_size: int,
      input_spec: spec.ChannelSpec,
      *,
      training: bool,
      constants: spec.Constants | None = None,
  ) -> spec.State:
    return ('test_state',)

  @override
  def get_output_shape(
      self,
      input_shape: spec.ShapeLike,
      *,
      constants: spec.Constants | None = None,
  ) -> spec.Shape:
    return tuple(input_shape) + (1,)

  @override
  def get_output_dtype(
      self, input_dtype: spec.DType, *, constants: spec.Constants | None = None
  ) -> spec.DType:
    return np.float64


class SequenceLayerTest[SequenceT: spec.Sequence = spec.Sequence](
    parameterized.TestCase
):
  """Base test class providing common sequence testing assertions and binds a backend implementation to tests."""

  # sequence_layers.<backend> module
  sl: specs.ModuleSpec

  @property
  def xp(self) -> backend_spec.xp:
    return self.sl.backend.xp

  @abc.abstractmethod
  def assertSequencesEqual(self, x: SequenceT, y: SequenceT) -> None:
    ...

  @abc.abstractmethod
  def assertAllEqual(self, x: Any, y: Any) -> None:
    ...


class ModuleInterfaceTest(SequenceLayerTest):

  def test_backend_specific_module_has_interface(self) -> None:
    self.assertIsInstance(self.sl.types, spec.ModuleSpec)


class SequenceTest(SequenceLayerTest):
  """Abstract tests for the Sequence class."""

  @parameterized.named_parameters(
      ('mask_value=None', 0.0, None),
      ('mask_value=0.0', 0.0, 0.0),
      ('mask_value=-1.0', -1.0, -1.0),
  )
  def test_mask_invalid(
      self, mask_value: float, expected_mask_value: float | None
  ) -> None:
    values = self.xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = self.xp.array(
        [[True, True, False, False], [False, False, False, True]]
    )

    # Pass mask_value only if it is not None (to test default None behavior vs explicit value)
    if expected_mask_value is None:
      output = self.sl.Sequence(values, mask).mask_invalid()
      fill_value = 0.0
    else:
      output = self.sl.Sequence(values, mask).mask_invalid(mask_value)
      fill_value = mask_value

    expected_values = self.xp.array([
        [1.0, 2.0, fill_value, fill_value],
        [fill_value, fill_value, fill_value, 40.0],
    ])
    self.assertAllEqual(output.values, expected_values)
    self.assertAllEqual(output.mask, mask)

  def test_pad_time(self) -> None:
    values = self.xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = self.xp.array(
        [[True, True, False, False], [False, False, False, True]]
    )

    x = self.sl.Sequence(values, mask).mask_invalid()

    y = x.pad_time(0, 0, valid=False)
    self.assertAllEqual(y.values, x.values)
    self.assertAllEqual(y.mask, x.mask)

    y = x.pad_time(1, 0, valid=False)

    x_left1 = self.sl.Sequence(
        self.xp.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
        ]),
        self.xp.array([
            [False, True, True, False, False],
            [False, False, False, False, True],
        ]),
    ).mask_invalid()
    self.assertAllEqual(y.values, x_left1.values)
    self.assertAllEqual(y.mask, x_left1.mask)

  def _create_test_sequence(
      self, shape: spec.Shape
  ) -> spec.Sequence[spec.Array, spec.Array]:
    size = 1
    for d in shape:
      size *= d
    values_np = np.arange(size, dtype=np.float32).reshape(shape)
    mask_np = np.ones(shape[:2], dtype=bool)
    if shape[0] > 0 and shape[1] > 1:
      mask_np[0, 1] = False

    values = self.xp.array(values_np)
    mask = self.xp.array(mask_np)
    return self.sl.Sequence(values, mask)

  def test_slice(self) -> None:
    x = self._create_test_sequence((3, 5, 9))

    self.assertSequencesEqual(
        x[:, 1:], self.sl.Sequence(x.values[:, 1:], x.mask[:, 1:])
    )
    self.assertSequencesEqual(
        x[:, ::2], self.sl.Sequence(x.values[:, ::2], x.mask[:, ::2])
    )
    self.assertSequencesEqual(
        x[::2, ::3], self.sl.Sequence(x.values[::2, ::3], x.mask[::2, ::3])
    )

  def test_slice_can_slice_channel_dimensions(self) -> None:
    x = self._create_test_sequence((3, 5, 9, 4))

    self.assertSequencesEqual(
        x[:, 1:, :], self.sl.Sequence(x.values[:, 1:], x.mask[:, 1:])
    )
    self.assertSequencesEqual(
        x[:, ::2, :3],
        self.sl.Sequence(x.values[:, ::2, :3], x.mask[:, ::2]),
    )

  def test_apply_values(self) -> None:
    values = self.xp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = self.xp.array(
        [[True, True, False, False], [False, True, False, True]]
    )

    x = self.sl.Sequence(values, mask)
    masked = x.mask_invalid()

    # Simple abs function
    fn = abs

    y = x.apply_values(fn)
    self.assertAllEqual(y.values, fn(x.values))
    self.assertAllEqual(y.mask, x.mask)

    y = masked.apply_values(fn)
    self.assertAllEqual(y.values, fn(masked.values))
    self.assertAllEqual(y.mask, x.mask)

    y = masked.apply_values_masked(fn)
    self.assertAllEqual(y.values, fn(masked.values))
    self.assertAllEqual(y.mask, x.mask)

  def test_apply_values_args(self) -> None:
    values = self.xp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = self.xp.array(
        [[True, True, False, False], [False, True, False, True]]
    )
    x = self.sl.Sequence(values, mask)

    target_shape = (2, 4, 1)
    y = x.apply_values(lambda v, s: v.reshape(s), target_shape)
    self.assertAllEqual(y.values.shape, target_shape)
    self.assertAllEqual(y.mask.shape, (2, 4))

  def test_from_values(self) -> None:
    values_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values = self.xp.array(values_np)
    # Get the class from an instance
    seq = self.sl.Sequence(
        values, self.xp.array(np.ones(values.shape[:2], dtype=bool))
    )
    SeqClass = type(seq)

    x = SeqClass.from_values(values)  # type: ignore
    self.assertAllEqual(x.values, values)
    self.assertAllEqual(
        x.mask, self.xp.array(np.ones(values.shape[:2], dtype=bool))
    )

  def test_astype(self) -> None:
    values_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True, False], [False, True]], dtype=bool)

    values = self.xp.array(values_np)
    mask = self.xp.array(mask_np)

    x = self.sl.Sequence(values, mask)

    y = x.astype(self.xp.int32)

    # Check values match casted version
    self.assertAllEqual(y.mask, mask)
    # y.values might be mlx array, values.astype(dtype) might be numpy if values was numpy?
    # values is backend array. values.astype(dtype) should work if dtype is backend dtype.
    self.assertAllEqual(y.values, values.astype(self.xp.int32))

  def test_mask_invalid_idempotent(self) -> None:
    values = self.xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = self.xp.array(
        [[True, True, False, False], [False, False, False, True]]
    )

    x = self.sl.Sequence(values, mask)
    masked = x.mask_invalid()
    self.assertIsNot(masked, x)
    self.assertIsInstance(masked, self.sl.MaskedSequence)

    masked_again = masked.mask_invalid()
    self.assertIs(masked_again, masked)
    self.assertIsInstance(masked_again, self.sl.MaskedSequence)

    masked2 = x.mask_invalid()
    self.assertIsNot(masked2, masked)
    self.assertIsInstance(masked2, self.sl.MaskedSequence)

  def test_from_lengths(self) -> None:
    values = self.xp.array(
        np.arange(5 * 17 * 2).reshape((5, 17, 2)).astype(np.float32)
    )
    lengths_np = np.array([0, 5, 10, 17, 12], dtype=np.int32)
    mask_np = np.arange(17)[None, :] < lengths_np[:, None]
    mask = self.xp.array(mask_np)

    x_expected = self.sl.Sequence(values, mask)
    x = self.sl.Sequence.from_lengths(x_expected.values, lengths_np)
    self.assertAllEqual(x.values, x_expected.values)
    self.assertAllEqual(x.mask, x_expected.mask)

    # Out of range lengths are clipped to 0 or max.
    x = self.sl.Sequence.from_lengths(
        x_expected.values, self.xp.array([-1, 5, 10, 17, 18])
    )
    self.assertAllEqual(x.lengths(), self.xp.array([0, 5, 10, 17, 17]))
    self.assertNotIsInstance(x, self.sl.MaskedSequence)

    # Return type is MaskedSequence if is_masked=True.
    x = self.sl.Sequence.from_lengths(
        x_expected.values, [-1, 5, 10, 17, 18], is_masked=True
    )
    self.assertAllEqual(x.lengths(), self.xp.array([0, 5, 10, 17, 17]))
    self.assertIsInstance(x, self.sl.MaskedSequence)


class SteppableTest(SequenceLayerTest):
  """Abstract tests for Steppable layers."""

  def create_steppable(self) -> spec.Steppable:
    """Creates a basic Steppable instance that should have default properties."""

    class DefaultSteppable(DefaultTestLayer, self.sl.types.Steppable):  # type: ignore[name-defined, misc]

      @override
      def layer_with_emits(self, *args, **kwargs):
        return super(DefaultTestLayer, self).layer_with_emits(*args, **kwargs)  # type: ignore

      @override
      def step_with_emits(self, *args, **kwargs):
        return super(DefaultTestLayer, self).step_with_emits(*args, **kwargs)  # type: ignore

    return DefaultSteppable()  # type: ignore

  def test_steppable_defaults(self) -> None:
    layer = self.create_steppable()
    self.assertEqual(layer.block_size, 1)
    self.assertEqual(layer.output_ratio, fractions.Fraction(1))
    self.assertTrue(layer.supports_step)
    self.assertEqual(layer.input_latency, 0)
    self.assertEqual(layer.output_latency, 0)
    self.assertEqual(layer.get_accumulated_input_latency(0), 0)
    self.assertEqual(layer.get_accumulated_output_latency(0), 0)

  def create_sequence(self) -> spec.Sequence:
    return self.sl.Sequence(
        self.xp.zeros((2, 3, 5)), self.xp.zeros((2, 3), dtype=self.xp.bool_)
    )

  def test_steppable_with_emits_defaults_to_tuple_with_empty_emits(
      self,
  ) -> None:
    layer = self.create_steppable()
    seq = self.create_sequence()
    state_in = {'a': 'b'}
    state_out = {1: 2}

    with unittest.mock.patch.object(
        layer, 'layer', return_value=seq
    ) as mock_layer:
      out, emits = layer.layer_with_emits(seq, training=False, constants=None)
      self.assertEqual(out, seq)
      self.assertEqual(emits, ())
      mock_layer.assert_called_with(seq, training=False, constants=None)

    with unittest.mock.patch.object(
        layer, 'step', return_value=(seq, state_out)
    ) as mock_step:
      out, state, emits = layer.step_with_emits(
          seq, state_in, training=True, constants=None
      )
      self.assertEqual(out, seq)
      self.assertEqual(state, state_out)
      self.assertEqual(emits, ())
      mock_step.assert_called_with(seq, state_in, training=True, constants=None)


class SequenceLayerConfigTest(SequenceLayerTest):

  def test_copy(self) -> None:

    @dataclasses.dataclass(frozen=True)
    class Config(self.sl.SequenceLayerConfig):  # type: ignore[name-defined,misc]
      a: int = 1234
      b: str = 'default string'

      def make(self) -> Any:
        return 'dummy_layer'

    config = Config()  # type: ignore
    new_config = config.copy(b='new string')
    self.assertEqual(new_config.a, config.a)
    self.assertEqual(new_config.b, 'new string')

  def test_copy_raises_on_non_dataclass(self) -> None:

    class NonDataclassConfig(self.sl.SequenceLayerConfig):  # type: ignore[name-defined,misc]

      def make(self) -> Any:
        return 'dummy_layer'

    config = NonDataclassConfig()  # type: ignore
    with self.assertRaises(TypeError):
      new_config = config.copy()
      del new_config

  def test_copy_disallows_new_fields(self) -> None:

    @dataclasses.dataclass(frozen=True)
    class Config(self.sl.SequenceLayerConfig):  # type: ignore[name-defined,misc]

      def make(self) -> Any:
        return 'dummy_layer'

    config = Config()  # type: ignore
    # dataclasses.replace raises TypeError for unknown arguments
    # JAX implementation wraps it in AttributeError
    with self.assertRaises((TypeError, AttributeError)):
      new_config = config.copy(field_does_not_exist=1234)
      del new_config


class PreservesTypeTest(SequenceLayerTest):

  def create_layer(self) -> spec.PreservesType:
    class DummyLayer(DefaultTestLayer, self.sl.types.PreservesType):  # type: ignore[name-defined, misc]

      @override
      def get_output_dtype(self, *args, **kwargs):
        return super(DefaultTestLayer, self).get_output_dtype(*args, **kwargs)  # type: ignore

    return DummyLayer()  # type: ignore

  def test_preserves_dtype(self) -> None:
    layer = self.create_layer()
    self.assertEqual(layer.get_output_dtype('fake_dtype123'), 'fake_dtype123')


class PreservesShapeTest(SequenceLayerTest):

  def create_layer(self) -> spec.PreservesShape:
    class DummyLayer(DefaultTestLayer, self.sl.types.PreservesShape):  # type: ignore[name-defined, misc]

      @override
      def get_output_shape(self, *args, **kwargs):
        return super(DefaultTestLayer, self).get_output_shape(*args, **kwargs)  # type: ignore

    return DummyLayer()  # type: ignore

  def test_preserves_shape(self) -> None:
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((1, 2, 3, 5)), (1, 2, 3, 5))


class StatelessTest(SequenceLayerTest):

  def create_layer(self) -> spec.Stateless:
    class DummyLayer(DefaultTestLayer, self.sl.types.Stateless):  # type: ignore[name-defined, misc]

      @override
      def get_initial_state(self, *args, **kwargs):
        return super(DefaultTestLayer, self).get_initial_state(*args, **kwargs)  # type: ignore

      @override
      def step(self, *args, **kwargs):
        return super(DefaultTestLayer, self).step(*args, **kwargs)  # type: ignore

    return DummyLayer()  # type: ignore

  def test_stateless_behaviors(self) -> None:
    layer = self.create_layer()

    # Initial state must be empty
    self.assertEqual(
        layer.get_initial_state(32, 'fake_spec', training=False), ()
    )

    # step unconditionally delegates to layer and returns identical empty state
    with unittest.mock.patch.object(
        layer, 'layer', return_value='layer_out'
    ) as mock_layer:
      out, state = layer.step('mock_x', 'mock_state', training=True, constants={'c': 1})  # type: ignore
      self.assertEqual(out, 'layer_out')
      self.assertEqual(state, 'mock_state')
      mock_layer.assert_called_once_with(
          'mock_x', training=True, constants={'c': 1}
      )


class EmittingTest(SequenceLayerTest):

  def create_layer(self) -> spec.Emitting:
    class DummyLayer(DefaultTestLayer, self.sl.types.Emitting):  # type: ignore[name-defined, misc]

      @override
      def layer(self, *args, **kwargs):
        return super(DefaultTestLayer, self).layer(*args, **kwargs)  # type: ignore

      @override
      def step(self, *args, **kwargs):
        return super(DefaultTestLayer, self).step(*args, **kwargs)  # type: ignore

    return DummyLayer()  # type: ignore

  def test_emitting_drops_emits_on_standard_calls(self) -> None:
    layer = self.create_layer()

    with unittest.mock.patch.object(
        layer, 'layer_with_emits', return_value=('out', 'emits')
    ) as m_layer:
      self.assertEqual(layer.layer('mock_x', training=False), 'out')  # type: ignore
      m_layer.assert_called_once_with('mock_x', training=False, constants=None)

    with unittest.mock.patch.object(
        layer, 'step_with_emits', return_value=('out', 'state', 'emits')
    ) as m_step:
      out, state = layer.step('mock_x', 'state', training=True, constants={'c': 1})  # type: ignore
      self.assertEqual(out, 'out')
      self.assertEqual(state, 'state')
      m_step.assert_called_once_with(
          'mock_x', 'state', training=True, constants={'c': 1}
      )


class StatelessEmittingTest(SequenceLayerTest):

  def create_layer(self) -> spec.SequenceLayer:
    class DummyLayer(DefaultTestLayer, self.sl.types.StatelessEmitting):  # type: ignore[name-defined, misc]

      @override
      def get_initial_state(self, *args, **kwargs):
        return super(DefaultTestLayer, self).get_initial_state(*args, **kwargs)  # type: ignore

      @override
      def step_with_emits(self, *args, **kwargs):
        return super(DefaultTestLayer, self).step_with_emits(*args, **kwargs)  # type: ignore

    return DummyLayer()  # type: ignore

  def test_stateless_emitting_behaviors(self) -> None:
    layer = self.create_layer()

    self.assertEqual(
        layer.get_initial_state(32, 'fake_spec', training=False), ()
    )

    with unittest.mock.patch.object(
        layer, 'layer_with_emits', return_value=('out', 'emits')
    ) as m_layer:
      out, state, emits = layer.step_with_emits('mock_x', 'state', training=False)  # type: ignore
      self.assertEqual(out, 'out')
      self.assertEqual(state, 'state')
      self.assertEqual(emits, 'emits')
      m_layer.assert_called_once_with('mock_x', training=False, constants=None)


class StatelessPointwiseFunctorTest(SequenceLayerTest):

  def create_layer(self, is_mask_required: bool) -> spec.SequenceLayer[Any]:

    class DummyLayer(DefaultTestLayer, self.sl.types.StatelessPointwiseFunctor):  # type: ignore[name-defined, misc]

      @override
      def layer(self, *args, **kwargs):
        return super(DefaultTestLayer, self).layer(*args, **kwargs)  # type: ignore

      @override
      def get_output_shape(self, *args, **kwargs):
        return super(DefaultTestLayer, self).get_output_shape(*args, **kwargs)  # type: ignore

      @property
      @override
      def mask_required(self_inner) -> bool:
        return is_mask_required

      @override
      def fn(self_inner, values: Any, mask: Any) -> tuple[Any, Any]:
        return values, mask

    return DummyLayer()  # type: ignore

  def create_sequence(self) -> spec.Sequence[spec.Array, spec.Array]:
    return self.sl.Sequence(
        self.xp.zeros((2, 3, 5)), self.xp.zeros((2, 3), dtype=self.xp.bool_)
    )

  def test_layer_applies_fn_based_on_mask_required(self) -> None:
    for mask_required in [True, False]:
      with self.subTest(mask_required=mask_required):
        layer = self.create_layer(mask_required)
        x = self.create_sequence()
        # Mock the apply methods on the Sequence class itself so we return a valid Sequence
        # that satisfies any @check_layer decorators.
        with unittest.mock.patch.object(
            type(x), 'apply', return_value=x
        ) as mock_apply:
          with unittest.mock.patch.object(
              type(x), 'apply_masked', return_value=x
          ) as mock_apply_masked:
            layer.layer(x, training=False)

            if mask_required:
              mock_apply.assert_called_once()
              mock_apply_masked.assert_not_called()
            else:
              mock_apply_masked.assert_called_once()
              mock_apply.assert_not_called()
