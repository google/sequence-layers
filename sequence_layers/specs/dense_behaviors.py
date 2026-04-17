"""Behavior tests for dense layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method

from absl.testing import parameterized

from sequence_layers.specs import test_utils


class DenseTest(test_utils.SequenceLayerTest):
  """Test behavior of Dense layer."""

  def test_rank2_unsupported(self):
    l = self.sl.Dense.Config(features=3, name='dense').make()
    x = self.random_sequence(2, 13)
    with self.assertRaises(ValueError):
      self.init_layer(l, x)

  @parameterized.parameters(((5,),), ((5, 7),))
  def test_dense(self, channels_shape):
    l = self.sl.Dense.Config(features=3, name='dense').make()
    x = self.random_sequence(2, 13, *channels_shape, random_mask=True)
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'dense')
    self.assertEqual(
        l.get_output_shape_for_sequence(x), channels_shape[:-1] + (3,)
    )
    self.verify_contract(l, x, training=False)

  @parameterized.parameters(True, False)
  def test_use_bias(self, use_bias):
    l = self.sl.Dense.Config(features=3, use_bias=use_bias).make()
    x = self.random_sequence(2, 3, 5)
    l = self.init_layer(l, x)
    self.verify_contract(l, x, training=False)


class EinsumDenseTest(test_utils.SequenceLayerTest):
  """Test behavior of EinsumDense layer."""

  @parameterized.parameters(
      (
          (2, 3, 5),
          '...a,ab->...b',
          (7,),
          '',
          (7,),
      ),
      (
          (2, 3, 5, 7),
          '...ab,ac->...cb',
          (11, 7),
          'c',
          (11, 7),
      ),
      (
          (2, 3, 5, 7),
          '...ab,b->...a',
          (None,),
          '',
          (5,),
      ),
  )
  def test_einsum_dense(
      self,
      shape,
      equation,
      output_shape,
      bias_axes,
      expected_output_shape,
  ):
    x = self.random_sequence(*shape)
    l = self.sl.EinsumDense.Config(
        equation=equation,
        output_shape=output_shape,
        bias_axes=bias_axes,
        name='einsum_dense',
    ).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'einsum_dense')
    self.assertEqual(l.get_output_shape_for_sequence(x), expected_output_shape)
    self.verify_contract(l, x, training=False)

  def test_einsum_dense_nonbroadcasting_equation(self):
    with self.assertRaises(ValueError):
      x = self.random_sequence(2, 3, 4, 5, 6)
      l = self.sl.EinsumDense.Config(
          equation='btabc,bc->btad', output_shape=[None, 2]
      ).make()
      l = self.init_layer(l, x)

  def test_einsum_dense_inconsistent_input_shape(self):
    x = self.random_sequence(2, 3, 5)
    l = self.sl.EinsumDense.Config(
        equation='...abc,bc->...ad', output_shape=[None, 2]
    ).make()
    with self.assertRaises(ValueError):
      self.init_layer(l, x)
