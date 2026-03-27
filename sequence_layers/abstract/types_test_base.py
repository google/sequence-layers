"""Abstract tests for Sequence types."""

import abc
from typing import Any, Callable, Sequence as TypingSequence
import dataclasses
from sequence_layers.abstract import types

import fractions
from absl.testing import parameterized
import numpy as np
import unittest.mock

class SequenceLayerTest(parameterized.TestCase):
  """Base abstract test class providing common sequence testing assertions."""

  @abc.abstractmethod
  def assertSequencesClose(self, x: Any, y: Any, **kwargs):
    pass

  @abc.abstractmethod
  def assertSequencesNotClose(self, x: Any, y: Any, **kwargs):
    pass

  @abc.abstractmethod
  def assertSequencesEqual(self, x: Any, y: Any):
    pass

  @abc.abstractmethod
  def assertSequencesNotEqual(self, x: Any, y: Any):
    pass

  @abc.abstractmethod
  def assertAllEqual(self, x: Any, y: Any):
    pass

  @abc.abstractmethod
  def assertAllClose(self, x: Any, y: Any, **kwargs):
    pass

  @abc.abstractmethod
  def assertNotAllEqual(self, x: Any, y: Any):
    pass

  @abc.abstractmethod
  def assertNotAllClose(self, x: Any, y: Any, **kwargs):
    pass


class SequenceTest(SequenceLayerTest):
  """Abstract tests for the Sequence class."""

  @abc.abstractmethod
  def get_backend(self) -> Any:
    """Returns the backend module (jax.numpy or mlx.core)."""
  
  @property
  @abc.abstractmethod
  def Sequence(self) -> type[types.Sequence]:
    """Returns the Sequence class for the backend."""

  @property
  @abc.abstractmethod
  def MaskedSequence(self) -> Any:
     """Returns the MaskedSequence class for the backend."""
  
  @property
  def check_trees_all_equal(self) -> Callable[[Any, Any], None]:
    """Returns a function to check tree equality."""
    return self.assertAllEqual

  def test_mask_invalid_idempotent(self):
    xp = self.get_backend()
    values = xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    # Different backends might handle boolean creation differently, but standard numpy-like syntax usually works
    mask = xp.array([[True, True, False, False], [False, False, False, True]])

    x = self.Sequence(values, mask)
    masked = x.mask_invalid()
    self.assertIsNot(masked, x)
    # We can't easily check isinstance here without importing the concrete classes, 
    # but we can check behavior or use a property if we added one. 
    # For now, we trust the concrete tests to check types if needed, 
    # or we could add abstract methods to check types.

    masked_again = masked.mask_invalid()
    self.assertIs(masked_again, masked)

    masked2 = x.mask_invalid()
    self.assertIsNot(masked2, masked)

  @parameterized.named_parameters(
      ('mask_value=None', 0.0, None),
      ('mask_value=0.0', 0.0, 0.0),
      ('mask_value=-1.0', -1.0, -1.0),
  )
  def test_mask_invalid(self, mask_value, expected_mask_value):
    xp = self.get_backend()
    values = xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = xp.array([[True, True, False, False], [False, False, False, True]])

    # Pass mask_value only if it is not None (to test default None behavior vs explicit value)
    if expected_mask_value is None:
       output = self.Sequence(values, mask).mask_invalid()
       fill_value = 0.0
    else:
       output = self.Sequence(values, mask).mask_invalid(mask_value)
       fill_value = mask_value

    expected_values = xp.array([
        [1.0, 2.0, fill_value, fill_value],
        [fill_value, fill_value, fill_value, 40.0],
    ])
    self.check_trees_all_equal(output.values, expected_values)
    self.check_trees_all_equal(output.mask, mask)

  def test_pad_time(self):
    xp = self.get_backend()
    values = xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = xp.array([[True, True, False, False], [False, False, False, True]])
    
    x = self.Sequence(values, mask).mask_invalid()

    y = x.pad_time(0, 0, valid=False)
    self.check_trees_all_equal(y.values, x.values)
    self.check_trees_all_equal(y.mask, x.mask)

    y = x.pad_time(1, 0, valid=False)

    x_left1 = self.Sequence(
        xp.array([
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
        ]),
        xp.array([
            [False, True, True, False, False],
            [False, False, False, False, True],
        ]),
    ).mask_invalid()
    self.check_trees_all_equal(y.values, x_left1.values)
    self.check_trees_all_equal(y.mask, x_left1.mask)

  def _create_test_sequence(self, shape):
    xp = self.get_backend()
    size = 1
    for d in shape: size *= d
    values_np = np.arange(size, dtype=np.float32).reshape(shape)
    mask_np = np.ones(shape[:2], dtype=bool)
    if shape[0] > 0 and shape[1] > 1:
        mask_np[0, 1] = False
    
    values = xp.array(values_np)
    mask = xp.array(mask_np)
    return self.Sequence(values, mask)

  def test_slice(self):
    x = self._create_test_sequence((3, 5, 9))
    
    self.assertSequencesEqual(
        x[:, 1:], self.Sequence(x.values[:, 1:], x.mask[:, 1:])
    )
    self.assertSequencesEqual(
        x[:, ::2], self.Sequence(x.values[:, ::2], x.mask[:, ::2])
    )
    self.assertSequencesEqual(
        x[::2, ::3], self.Sequence(x.values[::2, ::3], x.mask[::2, ::3])
    )

  def test_slice_can_slice_channel_dimensions(self):
    x = self._create_test_sequence((3, 5, 9, 4))
    
    self.assertSequencesEqual(
        x[:, 1:, :], self.Sequence(x.values[:, 1:], x.mask[:, 1:])
    )
    self.assertSequencesEqual(
        x[:, ::2, :3],
        self.Sequence(x.values[:, ::2, :3], x.mask[:, ::2]),
    )

  def test_apply_values(self):
    xp = self.get_backend()
    values = xp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = xp.array([[True, True, False, False], [False, True, False, True]])
    
    x = self.Sequence(values, mask)
    masked = x.mask_invalid()
    
    # Simple abs function
    fn = abs
    
    y = x.apply_values(fn)
    self.check_trees_all_equal(y.values, fn(x.values))
    self.check_trees_all_equal(y.mask, x.mask)

    y = masked.apply_values(fn)
    self.check_trees_all_equal(y.values, fn(masked.values))
    self.check_trees_all_equal(y.mask, x.mask)

    y = masked.apply_values_masked(fn)
    self.check_trees_all_equal(y.values, fn(masked.values))
    self.check_trees_all_equal(y.mask, x.mask)

  def test_apply_values_args(self):
    xp = self.get_backend()
    values = xp.array([
        [-1.0, 2.0, 3.0, 4.0],
        [10.0, -20.0, 30.0, 40.0],
    ])
    mask = xp.array([[True, True, False, False], [False, True, False, True]])
    x = self.Sequence(values, mask)
    
    target_shape = (2, 4, 1)
    y = x.apply_values(lambda v, s: v.reshape(s), target_shape)
    self.check_trees_all_equal(y.values.shape, target_shape)
    self.check_trees_all_equal(y.mask.shape, (2, 4))

  def test_from_values(self):
    xp = self.get_backend()
    values_np = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ], dtype=np.float32)
    values = xp.array(values_np)
    # Get the class from an instance
    seq = self.Sequence(values, xp.array(np.ones(values.shape[:2], dtype=bool)))
    SeqClass = type(seq)
    
    x = SeqClass.from_values(values)
    self.check_trees_all_equal(x.values, values)
    self.check_trees_all_equal(x.mask, xp.array(np.ones(values.shape[:2], dtype=bool)))

  def test_astype(self):
    xp = self.get_backend()
    values_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_np = np.array([[True, False], [False, True]], dtype=bool)
    
    values = xp.array(values_np)
    mask = xp.array(mask_np)
    
    x = self.Sequence(values, mask)
    
    # We need a dtype that matches the backend
    if xp.__name__ == 'jax.numpy':
       dtype = xp.int32
    elif xp.__name__ == 'mlx.core':
       dtype = xp.int32
    else:
       dtype = np.int32

    y = x.astype(dtype)
    
    # Check values match casted version
    self.check_trees_all_equal(y.mask, mask)
    # y.values might be mlx array, values.astype(dtype) might be numpy if values was numpy?
    # values is backend array. values.astype(dtype) should work if dtype is backend dtype.
    self.check_trees_all_equal(y.values, values.astype(dtype))

  def test_mask_invalid_idempotent(self):
    xp = self.get_backend()
    values = xp.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = xp.array([[True, True, False, False], [False, False, False, True]])

    x = self.Sequence(values, mask)
    masked = x.mask_invalid()
    self.assertIsNot(masked, x)
    self.assertIsInstance(masked, self.MaskedSequence)

    masked_again = masked.mask_invalid()
    self.assertIs(masked_again, masked)
    self.assertIsInstance(masked_again, self.MaskedSequence)

    masked2 = x.mask_invalid()
    self.assertIsNot(masked2, masked)
    self.assertIsInstance(masked2, self.MaskedSequence)

  def test_from_lengths(self):
    xp = self.get_backend()
    values = xp.array(np.arange(5 * 17 * 2).reshape((5, 17, 2)).astype(np.float32))
    lengths_np = np.array([0, 5, 10, 17, 12], dtype=np.int32)
    mask_np = np.arange(17)[None, :] < lengths_np[:, None]
    mask = xp.array(mask_np)

    x_expected = self.Sequence(values, mask)
    x = self.Sequence.from_lengths(x_expected.values, lengths_np)
    self.check_trees_all_equal(x.values, x_expected.values)
    self.check_trees_all_equal(x.mask, x_expected.mask)

    # Out of range lengths are clipped to 0 or max.
    x = self.Sequence.from_lengths(x_expected.values, [-1, 5, 10, 17, 18])
    self.check_trees_all_equal(x.lengths(), xp.array([0, 5, 10, 17, 17]))
    self.assertNotIsInstance(x, self.MaskedSequence)

    # Return type is MaskedSequence if is_masked=True.
    x = self.Sequence.from_lengths(
        x_expected.values, [-1, 5, 10, 17, 18], is_masked=True
    )
    self.check_trees_all_equal(x.lengths(), xp.array([0, 5, 10, 17, 17]))
    self.assertIsInstance(x, self.MaskedSequence)



class SteppableTest(parameterized.TestCase):
  """Abstract tests for Steppable layers."""

  @abc.abstractmethod
  def create_steppable(self) -> Any:
    """Creates a basic Steppable instance that should have default properties."""

  def test_steppable_defaults(self):
    layer = self.create_steppable()
    self.assertEqual(layer.block_size, 1)
    self.assertEqual(layer.output_ratio, fractions.Fraction(1))
    self.assertTrue(layer.supports_step)
    self.assertEqual(layer.input_latency, 0)
    self.assertEqual(layer.output_latency, 0)
    self.assertEqual(layer.get_accumulated_input_latency(0), 0)
    self.assertEqual(layer.get_accumulated_output_latency(0), 0)

  def test_steppable_with_emits_defaults_to_tuple_with_empty_emits(self):
    layer = self.create_steppable()

    with unittest.mock.patch.object(layer, 'layer', return_value='mock_layer_out') as mock_layer:
      out, emits = layer.layer_with_emits('mock_x', training=False, constants=None)
      self.assertEqual(out, 'mock_layer_out')
      self.assertEqual(emits, ())
      mock_layer.assert_called_with('mock_x', training=False, constants=None)

    with unittest.mock.patch.object(layer, 'step', return_value=('step_out', 'state_out')) as mock_step:
      out, state, emits = layer.step_with_emits('mock_x', 'state_in', training=True, constants=None)
      self.assertEqual(out, 'step_out')
      self.assertEqual(state, 'state_out')
      self.assertEqual(emits, ())
      mock_step.assert_called_with('mock_x', 'state_in', training=True, constants=None)


class SequenceLayerConfigTest(SequenceLayerTest):

  @abc.abstractmethod
  def get_config_base_cls(self) -> type[types.SequenceLayerConfig]:
    """Returns the backend-specific SequenceLayerConfig class."""

  def test_copy(self):
    ConfigBase = self.get_config_base_cls()

    @dataclasses.dataclass(frozen=True)
    class Config(ConfigBase):
      a: int = 1234
      b: str = 'default string'

      def make(self) -> Any:
        return 'dummy_layer'

    config = Config()
    new_config = config.copy(b='new string')
    self.assertEqual(new_config.a, config.a)
    self.assertEqual(new_config.b, 'new string')

  def test_copy_raises_on_non_dataclass(self):
    ConfigBase = self.get_config_base_cls()

    class NonDataclassConfig(ConfigBase):

      def make(self) -> Any:
        return 'dummy_layer'

    config = NonDataclassConfig()
    with self.assertRaises(TypeError):
      new_config = config.copy()
      del new_config

  def test_copy_disallows_new_fields(self):
    ConfigBase = self.get_config_base_cls()

    @dataclasses.dataclass(frozen=True)
    class Config(ConfigBase):

      def make(self) -> Any:
        return 'dummy_layer'

    config = Config()
    # dataclasses.replace raises TypeError for unknown arguments
    # JAX implementation wraps it in AttributeError
    with self.assertRaises((TypeError, AttributeError)):
      new_config = config.copy(field_does_not_exist=1234)
      del new_config


class PreservesTypeTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self) -> types.PreservesType:
    pass

  def test_preserves_dtype(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_dtype('fake_dtype123'), 'fake_dtype123')


class PreservesShapeTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self) -> types.PreservesShape:
    pass

  def test_preserves_shape(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((1, 2, 3, 5)), (1, 2, 3, 5))


class StatelessTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self) -> types.Stateless:
    pass

  def test_stateless_behaviors(self):
    layer = self.create_layer()

    # Initial state must be empty
    self.assertEqual(
        layer.get_initial_state(32, 'fake_spec', training=False), ()
    )

    # step unconditionally delegates to layer and returns identical empty state
    with unittest.mock.patch.object(layer, 'layer', return_value='layer_out') as mock_layer:
      out, state = layer.step('mock_x', 'mock_state', training=True, constants={'c': 1})
      self.assertEqual(out, 'layer_out')
      self.assertEqual(state, 'mock_state')
      mock_layer.assert_called_once_with('mock_x', training=True, constants={'c': 1})


class EmittingTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self) -> types.Emitting:
    pass

  def test_emitting_drops_emits_on_standard_calls(self):
    layer = self.create_layer()

    with unittest.mock.patch.object(layer, 'layer_with_emits', return_value=('out', 'emits')) as m_layer:
      self.assertEqual(layer.layer('mock_x', training=False), 'out')
      m_layer.assert_called_once_with('mock_x', training=False, constants=None)

    with unittest.mock.patch.object(layer, 'step_with_emits', return_value=('out', 'state', 'emits')) as m_step:
      out, state = layer.step('mock_x', 'state', training=True, constants={'c': 1})
      self.assertEqual(out, 'out')
      self.assertEqual(state, 'state')
      m_step.assert_called_once_with('mock_x', 'state', training=True, constants={'c': 1})


class StatelessEmittingTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self) -> types.StatelessEmitting:
    pass

  def test_stateless_emitting_behaviors(self):
    layer = self.create_layer()

    self.assertEqual(
        layer.get_initial_state(32, 'fake_spec', training=False), ()
    )

    with unittest.mock.patch.object(layer, 'layer_with_emits', return_value=('out', 'emits')) as m_layer:
      out, state, emits = layer.step_with_emits('mock_x', 'state', training=False)
      self.assertEqual(out, 'out')
      self.assertEqual(state, 'state')
      self.assertEqual(emits, 'emits')
      m_layer.assert_called_once_with('mock_x', training=False, constants=None)


class StatelessPointwiseFunctorTest(parameterized.TestCase):

  @abc.abstractmethod
  def create_layer(self, mask_required: bool) -> types.StatelessPointwiseFunctor:
    pass

  @abc.abstractmethod
  def create_sequence(self) -> types.Sequence:
    pass

  def test_layer_applies_fn_based_on_mask_required(self):
    for mask_required in [True, False]:
      with self.subTest(mask_required=mask_required):
        layer = self.create_layer(mask_required)
        x = self.create_sequence()
        # Mock the apply methods on the Sequence class itself so we return a valid Sequence
        # that satisfies any @check_layer decorators.
        with unittest.mock.patch.object(type(x), 'apply', return_value=x) as mock_apply:
          with unittest.mock.patch.object(type(x), 'apply_masked', return_value=x) as mock_apply_masked:
            layer.layer(x, training=False)
            
            if mask_required:
              mock_apply.assert_called_once()
              mock_apply_masked.assert_not_called()
            else:
              mock_apply_masked.assert_called_once()
              mock_apply.assert_not_called()

