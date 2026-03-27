import mlx.core as mx
import numpy as np
from sequence_layers.abstract import types_test_base
from sequence_layers.mlx import types
from absl.testing import parameterized
from absl.testing import absltest


class SequenceTest(types_test_base.SequenceTest):

  def get_backend(self):
    return mx

  @property
  def Sequence(self):
    return types.Sequence

  @property
  def MaskedSequence(self):
    return types.MaskedSequence

  def assertAllEqual(self, a, b):
    a = np.array(a) if isinstance(a, mx.array) else a
    b = np.array(b) if isinstance(b, mx.array) else b
    np.testing.assert_array_equal(a, b)

  def assertSequencesEqual(self, a, b):
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.mask, b.mask)


class SteppableTest(types_test_base.SteppableTest):

  def create_steppable(self):

    class DefaultSteppable(types.Steppable):

      def layer(self, x, *, training: bool, constants=None):
        return x

      def step(self, x, state, *, training: bool, constants=None):
        return x, state

      def get_initial_state(self, batch_size, input_spec, *, constants=None):
        return ()

      def get_output_shape(self, input_shape, *, constants=None):
        return input_shape

      def get_output_dtype(self, input_dtype, *, constants=None):
        return input_dtype

    return DefaultSteppable()


class SequenceLayerConfigTest(types_test_base.SequenceLayerConfigTest):

  def get_config_base_cls(self):
    return types.SequenceLayerConfig


class PreservesTypeTest(types_test_base.PreservesTypeTest):
  def create_layer(self):
    class DummyLayer(types.PreservesType, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def step(self, x, state, *, training: bool, constants=None): return x, state
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
    return DummyLayer()


class PreservesShapeTest(types_test_base.PreservesShapeTest):
  def create_layer(self):
    class DummyLayer(types.PreservesShape, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def step(self, x, state, *, training: bool, constants=None): return x, state
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class StatelessTest(types_test_base.StatelessTest):
  def create_layer(self):
    class DummyLayer(types.Stateless, types.SequenceLayer):
      def layer(self, x, *, training: bool, constants=None): return x
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class EmittingTest(types_test_base.EmittingTest):
  def create_layer(self):
    class DummyLayer(types.Emitting, types.SequenceLayer):
      def get_initial_state(self, batch_size, input_spec, *, training: bool, constants=None): return ()
      def layer_with_emits(self, x, *, training: bool, constants=None): return x, ()
      def step_with_emits(self, x, state, *, training: bool, constants=None): return x, state, ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class StatelessEmittingTest(types_test_base.StatelessEmittingTest):
  def create_layer(self):
    class DummyLayer(types.StatelessEmitting, types.SequenceLayer):
      def layer_with_emits(self, x, *, training: bool, constants=None): return x, ()
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()


class StatelessPointwiseFunctorTest(types_test_base.StatelessPointwiseFunctorTest):
  def create_layer(self, is_mask_required: bool):
    class DummyLayer(types.StatelessPointwiseFunctor, types.SequenceLayer):
      @property
      def mask_required(self): return is_mask_required
      def fn(self, values, mask): return values, mask
      def get_output_shape(self, input_shape, *, constants=None): return input_shape
      def get_output_dtype(self, input_dtype, *, constants=None): return input_dtype
    return DummyLayer()

  def create_sequence(self):
    return types.Sequence(mx.zeros((2, 3, 5)), mx.zeros((2, 3), dtype=mx.bool_))


if __name__ == '__main__':
  absltest.main()
