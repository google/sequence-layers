from sequence_layers.mlx import test_utils
from sequence_layers.mlx import types
from sequence_layers.abstract import types_test_base

class SequenceTest(test_utils.SequenceLayerTest, types_test_base.SequenceTest):
    pass


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


if __name__ == '__main__':
  absltest.main()
