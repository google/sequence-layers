"""Behavior tests for simple layers.

Backend-specific test files should inherit from these tests.
"""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

from fractions import Fraction
from typing import Any, override
from unittest import mock

from absl import logging
from absl.testing import parameterized
import numpy as np

from sequence_layers.specs import simple as simple_spec
from sequence_layers.specs import test_utils


class ModuleSpecTest(test_utils.ModuleSpecTest):
  """Test that a backend-specific module implements the ModuleSpec protocol."""

  @override
  def module_spec_pairs(self, backend_sl: Any) -> dict[Any, Any]:
    return {backend_sl.simple: simple_spec.ModuleSpec}


class IdentityTest(test_utils.SequenceLayerTest):
  """Test behavior of Identity layer."""

  def test_defaults(self):
    # pyrefly: ignore [missing-attribute]
    self.assertConfigDefaults(self.sl.Identity.Config, {'name': None})

  @parameterized.parameters((((2, 3, 5)),), (((2, 3, 5, 9)),))
  def test_identity(self, shape):
    x = self.random_sequence(*shape)
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Identity.Config(name='identity').make()
    l = self.init_layer(l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.verify_contract(l, x, training=False)


class PointwiseMathTest(test_utils.SequenceLayerTest):
  """Test behavior of pointwise math layers."""

  def test_defaults(self):
    # pyrefly: ignore [missing-attribute]
    for layer_cls in [self.sl.Abs, self.sl.Exp, self.sl.Log]:
      with self.subTest(layer=layer_cls.__name__):
        self.assertConfigDefaults(layer_cls.Config, {'name': None})

  def make_layer(self, layer_name):
    """Helper to create a layer by name."""
    layer_cls = getattr(self.sl, layer_name)
    return layer_cls.Config(name=layer_name.lower()).make()

  def test_pointwise_math(self):
    params = [
        ('Relu', 'relu', False),
        ('Sigmoid', 'sigmoid', False),
        ('Tanh', 'tanh', False),
        ('Elu', 'elu', False),
        ('Softplus', 'softplus', False),
        ('Swish', 'swish', False),
        ('Gelu', 'gelu', False),
        ('Abs', 'abs', True),
        ('Exp', 'exp', True),
        ('Log', 'log', True),
        ('Softmax', 'softmax', False),
    ]
    for layer_name, method_name, is_xp in params:
      with self.subTest(layer=layer_name):
        x = self.random_sequence(2, 10, 4)
        l = self.make_layer(layer_name)
        l = self.init_layer(l, x)

        self.assertEqual(l.block_size, 1)
        self.assertEqual(l.output_ratio, 1)

        y = self.verify_contract(l, x, training=False)

        activation = getattr(
            self.sl.backend.xp if is_xp else self.nn, method_name
        )
        y_expected = x.apply_values(activation).mask_invalid()
        self.assertSequencesClose(y, y_expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(
      ('Softmax', 'softmax', -1),
      ('Softmax', 'softmax', -2),
      ('Softmax', 'softmax', 2),
      ('Softmax', 'softmax', 3),
  )
  def test_pointwise_math_axis(self, layer_name, method_name, axis):
    batch_size, time, channels, channels2 = 2, 10, 4, 3
    x = self.random_sequence(batch_size, time, channels, channels2)
    l = getattr(self.sl, layer_name).Config(name='test', axis=axis).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (channels, channels2))
    self.assertEqual(l.name, 'test')
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    activation = getattr(self.nn, method_name)
    y_expected = x.apply_values(
        lambda v: activation(v, axis=axis)
    ).mask_invalid()
    self.assertSequencesClose(y, y_expected)

  @parameterized.parameters(
      ('Softmax', (2, 10, 4), -2),
      ('Softmax', (2, 10, 4), -3),
      ('Softmax', (2, 10, 4), 0),
      ('Softmax', (2, 10, 4), 1),
      ('Softmax', (2, 10), -1),
  )
  def test_pointwise_math_axis_invalid(self, layer_name, shape, axis):
    x = self.random_sequence(*shape)
    l = getattr(self.sl, layer_name).Config(name='test', axis=axis).make()

    with self.assertRaises(ValueError):
      l = self.init_layer(l, x)
      l.layer(x, training=False)


class Downsample1DTest(test_utils.SequenceLayerTest):
  """Test behavior of Downsample1D layer."""

  @parameterized.parameters(((2, 3, 5), 2), ((2, 3, 5, 9), 3))
  def test_downsample1d(self, shape, rate):
    x = self.random_sequence(*shape)
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Downsample1D.Config(rate=rate, name='downsample_1d').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, rate)
    self.assertEqual(l.output_ratio, Fraction(1, rate))

    self.assertEqual(l.get_output_shape_for_sequence(x), x.channel_shape)
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    np.testing.assert_array_equal(y.values, x.values[:, ::rate])
    np.testing.assert_array_equal(y.mask, x.mask[:, ::rate])


class Upsample1DTest(test_utils.SequenceLayerTest):
  """Test behavior of Upsample1D layer."""

  @parameterized.parameters(((2, 3, 5), 2), ((2, 3, 5, 9), 3))
  def test_upsample1d(self, shape, rate):
    x = self.random_sequence(*shape)
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Upsample1D.Config(rate=rate, name='upsample_1d').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, rate)

    self.assertEqual(l.get_output_shape_for_sequence(x), x.channel_shape)
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    for i in range(rate):
      np.testing.assert_array_equal(x.values, y.values[:, i::rate])
      np.testing.assert_array_equal(x.mask, y.mask[:, i::rate])


class TransposeTest(test_utils.SequenceLayerTest):
  """Test behavior of Transpose layer."""

  @parameterized.parameters(
      ((2, 3, 4, 5), (2, 3), (4, 5)),
      ((2, 3, 4, 5, 6), (4, 2, 3), (6, 4, 5)),
      ((2, 3), None, ()),
  )
  def test_transpose(self, input_shape, axes, _output_shape):
    x = self.random_sequence(*input_shape)
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Transpose.Config(axes=axes, name='transpose').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), _output_shape)
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify shape and values
    if axes is not None:
      y_expected = x.apply_values(np.transpose, (0, 1) + axes)
    else:
      axes_seq = (0, 1) + tuple(range(2, x.ndim))[::-1]
      y_expected = x.apply_values(np.transpose, axes_seq)

    self.assertSequencesEqual(y, y_expected)


class DropoutTest(test_utils.SequenceLayerTest):
  """Test behavior of Dropout layer."""

  def test_defaults(self):
    self.assertConfigDefaults(
        # pyrefly: ignore [missing-attribute]
        self.sl.Dropout.Config,
        {'rate': 0.0, 'name': None},
    )

  def test_dropout_inference(self):
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Dropout.Config(rate=0.5, name='dropout').make()
    x = self.random_sequence(2, 3, 5)
    l = self.init_layer(l, x)
    y = l.layer(x, training=False)
    # In inference, dropout should be identity
    np.testing.assert_allclose(y.values, x.values)


class FlattenTest(test_utils.SequenceLayerTest):
  """Test behavior of Flatten layer."""

  @parameterized.parameters(
      (((2, 3, 5)),), (((2, 3, 5, 9)),), (((2, 3, 5, 9, 2)),)
  )
  def test_flatten(self, shape):
    x = self.random_sequence(*shape)
    # pyrefly: ignore [missing-attribute]
    l = self.sl.Flatten.Config(name='flatten').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    num_elements = np.prod(shape[2:])

    self.assertEqual(l.get_output_shape_for_sequence(x), (num_elements,))
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify shape
    expected_shape = shape[:2] + (num_elements,)
    self.assertEqual(y.values.shape, expected_shape)

    # Verify values
    y_expected = x.apply_values(np.reshape, shape[:2] + (num_elements,))
    self.assertSequencesEqual(y, y_expected)


class ReshapeTest(test_utils.SequenceLayerTest):
  """Test behavior of Reshape layer."""

  @parameterized.parameters(
      ((2, 3, 5), (1, 5, 1)),
      ((2, 3, 5, 9), (3, 3, 5)),
      ((2, 3, 1), ()),
      ((2, 3), (1,)),
  )
  def test_reshape(self, shape, output_shape):
    x = self.random_sequence(*shape)
    l = self.sl.Reshape.Config(output_shape, name='reshape').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), output_shape)
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify shape
    expected_shape = shape[:2] + output_shape
    self.assertEqual(y.values.shape, expected_shape)

    # Verify values
    y_expected = x.apply_values(np.reshape, shape[:2] + output_shape)
    self.assertSequencesEqual(y, y_expected)


class ExpandDimsTest(test_utils.SequenceLayerTest):
  """Test behavior of ExpandDims layer."""

  def test_basic(self):
    x = self.random_sequence(2, 3, 4)
    l = self.sl.ExpandDims.Config(axis=-1, name='expand_dims').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    y = self.verify_contract(l, x, training=False)
    self.assertEqual(y.values.shape, (2, 3, 4, 1))

  def test_output_shape(self):
    l = self.sl.ExpandDims.Config(axis=0, name='expand_dims').make()
    self.assertEqual(l.get_output_shape((4, 8)), (1, 4, 8))


class SqueezeTest(test_utils.SequenceLayerTest):
  """Test behavior of Squeeze layer."""

  @parameterized.named_parameters(
      {
          'testcase_name': 'float_input',
          'input_array': np.array([[[3]]], dtype=np.float32),
          'expected_output': np.array([[3]]),
      },
      {
          'testcase_name': 'int_input',
          'input_array': np.array([[[3]]], dtype=np.int32),
          'expected_output': np.array([[3]], dtype=np.int32),
      },
      {
          'testcase_name': 'no_op_input',
          'input_array': np.array([[3]], dtype=np.float32),
          'expected_output': np.array([[3]]),
      },
      {
          'testcase_name': 'input_with_extra_dims',
          'input_array': np.array([[[[[3], [4]]]]], dtype=np.float32),
          'expected_output': np.array([[[3, 4]]]),
      },
  )
  def test_squeeze(self, input_array, expected_output):
    x = self.sl.Sequence.from_values(input_array)
    l = self.sl.Squeeze.Config(name='squeeze').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(
        l.get_output_shape_for_sequence(x), expected_output.shape[2:]
    )
    test_receptive_field = np.issubdtype(input_array.dtype, np.inexact)
    y = self.verify_contract(
        l, x, training=False, test_receptive_field=test_receptive_field
    )
    self.assertEmpty(self.get_variables(l))

    # Verify shape
    self.assertEqual(y.values.shape, expected_output.shape)

    # Verify values
    np.testing.assert_allclose(y.values, expected_output)


class ScaleTest(test_utils.SequenceLayerTest):
  """Test behavior of Scale layer."""

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 5, 9),))
  def test_basic(self, shape):
    x = self.random_sequence(*shape)
    l = self.sl.Scale.Config(scale=2.0, name='scale').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify values
    y_expected = x.apply_values(lambda v: v * 2.0)
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    x = self.random_sequence(*shape)
    l = self.sl.Scale.Config(
        scale=np.arange(5, dtype=np.float32), name='scale'
    ).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify values
    y_expected = x.apply_values(lambda v: v * np.arange(5, dtype=np.float32))
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    x = self.random_sequence(2, 3, 5, 1)
    l = self.sl.Scale.Config(scale=np.ones((5, 9))).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))
    y = self.verify_contract(l, x, training=False)
    self.assertEqual(y.values.shape, (2, 3, 5, 9))
    self.assertEmpty(self.get_variables(l))

  def test_too_many_dims(self):
    x = self.random_sequence(2, 3, 5, 1)
    l = self.sl.Scale.Config(scale=np.ones((5, 5, 5))).make()
    l = self.init_layer(l, x, bind_only=True)
    with self.assertRaises(ValueError):
      l.get_output_shape(x.channel_shape)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = self.random_sequence(2, 3, 5, 9)
    l = self.sl.Scale.Config(scale=np.ones((5,))).make()
    l = self.init_layer(l, x, bind_only=True)
    with self.assertRaises(ValueError):
      l.get_output_shape(x.channel_shape)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class AddTest(test_utils.SequenceLayerTest):
  """Test behavior of Add layer."""

  @parameterized.parameters((((2, 13, 5)),), (((2, 13, 5, 9)),))
  def test_add(self, shape):
    x = self.random_sequence(*shape)
    l = self.sl.Add.Config(-2.0, name='add').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify values
    y_expected = x.apply_values(lambda v: v - 2.0).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  @parameterized.parameters(((2, 13, 5),), ((2, 13, 9, 5),))
  def test_ndarray(self, shape):
    x = self.random_sequence(*shape)
    l = self.sl.Add.Config(
        shift=np.arange(5, dtype=np.float32), name='add'
    ).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    y = self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Verify values
    y_expected = x.apply_values(
        lambda v: v + np.arange(5, dtype=np.float32)
    ).mask_invalid()
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    x = self.random_sequence(2, 3, 5, 1)
    l = self.sl.Add.Config(shift=np.ones((5, 9))).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))
    y = self.verify_contract(l, x, training=False)
    self.assertEqual(y.values.shape, (2, 3, 5, 9))
    self.assertEmpty(self.get_variables(l))

  def test_too_many_dims(self):
    x = self.random_sequence(2, 3, 5, 1)
    l = self.sl.Add.Config(shift=np.ones((5, 5, 5))).make()
    l = self.init_layer(l, x, bind_only=True)
    with self.assertRaises(ValueError):
      l.get_output_shape(x.channel_shape)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)

  def test_broadcast_failure(self):
    x = self.random_sequence(2, 3, 5, 9)
    l = self.sl.Add.Config(shift=np.ones((5,))).make()
    l = self.init_layer(l, x, bind_only=True)
    with self.assertRaises(ValueError):
      l.get_output_shape(x.channel_shape)
    with self.assertRaises(ValueError):
      l.layer(x, training=False)


class CastTest(test_utils.SequenceLayerTest):
  """Test behavior of Cast layer."""

  @parameterized.parameters(
      (((2, 3, 5)), np.float16),
      (((2, 3, 5, 9)), np.int32),
  )
  def test_cast(self, shape, target_dtype):
    x = self.random_sequence(*shape, dtype=np.float32)
    l = self.sl.Cast.Config(target_dtype, name='cast').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:])
    test_receptive_field = np.issubdtype(target_dtype, np.inexact)

    pad_value = np.nan if target_dtype == np.float16 else 32768

    y = self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=pad_value,
        test_receptive_field=test_receptive_field,
    )
    self.assertEmpty(self.get_variables(l))

    self.assertEqual(y.values.dtype, target_dtype)


class MaskInvalidTest(test_utils.SequenceLayerTest):
  """Test behavior of MaskInvalid layer."""

  def test_basic(self):
    x = self.random_sequence(2, 15, 5)
    l = self.sl.MaskInvalid.Config(name='mask_invalid').make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    self.assertEqual(l.get_output_shape_for_sequence(x), (5,))
    self.verify_contract(l, x, training=False)
    self.assertEmpty(self.get_variables(l))

    # Now test specific behavior
    # Fill invalid values with NaN
    x_nan = x.mask_invalid(np.nan)

    # Apply layer
    y = l.layer(x_nan, training=False)

    # Verify that invalid values are masked (zeroed)
    self.assertSequencesEqual(x.mask_invalid(), y)


class GatedUnitTest(test_utils.SequenceLayerTest):
  """Test behavior of GatedUnit layers."""

  def test_gated_activation(self):
    shapes = ((2, 13, 6), (2, 13, 5, 10))

    configs = [
        self.sl.GatedUnit.Config(None, None),  # Bilinear
        self.sl.GatedUnit.Config(None, self.nn.swish),  # SwiGLU
        self.sl.GatedUnit.Config(None, self.nn.gelu),  # GeGLU
        self.sl.GatedUnit.Config(lambda x: x, None),  # Bilinear
        self.sl.GatedUnit.Config(self.nn.swish, self.nn.tanh),
        self.sl.GatedTanhUnit.Config(),
        self.sl.GatedLinearUnit.Config(),
    ]

    for shape in shapes:
      for l_config in configs:
        with self.subTest(shape=shape, config=str(l_config)):
          x = self.random_sequence(*shape)
          l = l_config.make()
          l = self.init_layer(l, x)

          self.assertEqual(l.block_size, 1)
          self.assertEqual(l.output_ratio, 1)
          self.assertEqual(
              l.get_output_shape_for_sequence(x),
              shape[2:-1] + (shape[-1] // 2,),
          )
          self.verify_contract(l, x, training=True)


class OneHotTest(test_utils.SequenceLayerTest):
  """Test behavior of OneHot layer."""

  @parameterized.parameters(((1, 2, 3),), ((2, 3, 5, 9),), ((2, 3, 5, 9, 2),))
  def test_one_hot(self, shape):
    depth = 4
    l = self.sl.OneHot.Config(depth, name='one_hot').make()
    x = self.random_sequence(*shape, dtype=self.xp.int32, low=0, high=depth - 1)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), shape[2:] + (depth,))
    self.assertEqual(l.name, 'one_hot')

    l = self.init_layer(l, x)

    y = self.verify_contract(
        l,
        x,
        training=False,
        padding_invariance_pad_value=0,
        test_gradients=False,
        test_receptive_field=False,
    )
    self.assertAllEqual(
        y.values,
        (
            np.eye(depth)[np.array(x.values)].T
            * np.array(x.mask).astype(np.float32).T
        ).T,
    )


class EmbeddingTest(test_utils.SequenceLayerTest):
  """Test behavior of Embedding layer."""

  def test_defaults(self):
    self.assertConfigDefaults(
        self.sl.Embedding.Config,
        {'dimension': 10, 'num_embeddings': 100, 'name': None},
        dimension=10,
        num_embeddings=100,
    )

  def test_embedding(self):
    shapes = [(1, 2, 3), (2, 3, 5, 9)]
    dimension, num_embeddings = 8, 5

    for shape in shapes:
      with self.subTest(shape=shape):
        l = self.sl.Embedding.Config(
            dimension=dimension, num_embeddings=num_embeddings, name='embedding'
        ).make()
        x = self.random_sequence(
            *shape, dtype=self.xp.int32, low=0, high=num_embeddings - 1
        )

        self.assertEqual(l.block_size, 1)
        self.assertEqual(l.output_ratio, 1)
        self.assertEqual(
            l.get_output_shape(x.channel_shape), shape[2:] + (dimension,)
        )

        l = self.init_layer(l, x)

        self.verify_contract(
            l,
            x,
            training=False,
            test_gradients=False,
            test_receptive_field=False,
        )


class LambdaTest(test_utils.SequenceLayerTest):
  """Test behavior of Lambda layer."""

  @parameterized.parameters(True, False)
  def test_array_fn(self, mask_required: bool):
    def fn(v):
      if mask_required:
        # Change the masked status by adding 1.
        v = v + 1.0
      return v.reshape(v.shape + (1,)) > 0.5

    l = self.sl.simple.Lambda.Config(
        fn,
        mask_required=mask_required,
        expected_input_spec=self.sl.types.ChannelSpec((5,), self.xp.float32),
        name='lambda',
    ).make()

    x = self.random_sequence(2, 3, 5)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    # Output spec reflects the changed shape and dtype.
    self.assertEqual(l.get_output_shape(x.channel_shape), (5, 1))
    self.assertEqual(l.get_output_dtype(x.dtype), self.xp.bool_)

    y = self.verify_contract(
        l,
        x,
        training=False,
        # Receptive field test is not supported for bools.
        test_receptive_field=False,
    )

    self.assertSequencesClose(y, x.apply_values(fn).mask_invalid())

  @parameterized.parameters(True, False)
  def test_sequence_fn(self, mask_required: bool):
    def fn(x):
      if mask_required:
        # Change the masked status by adding 1.
        x = x.apply_values(lambda v: v + 1.0)
      return x.apply_values_masked(lambda v: v.reshape(v.shape + (1,)) > 0.5)

    l = self.sl.simple.Lambda.Config(
        fn,
        sequence_input=True,
        expected_input_spec=self.sl.types.ChannelSpec((5,), self.xp.float32),
        name='lambda',
    ).make()

    x = self.random_sequence(2, 3, 5)
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    # Output spec reflects the changed shape and dtype.
    self.assertEqual(l.get_output_shape(x.channel_shape), (5, 1))
    self.assertEqual(l.get_output_dtype(x.dtype), self.xp.bool_)

    y = self.verify_contract(
        l,
        x,
        training=False,
        # Receptive field test is not supported for bools.
        test_receptive_field=False,
    )

    self.assertSequencesClose(y, fn(x).mask_invalid())


class CheckpointNameTest(test_utils.SequenceLayerTest):
  """Test behavior of CheckpointName layer."""

  def test_basic(self):
    x = self.random_sequence(2, 3, 5)
    l = self.sl.simple.CheckpointName.Config(
        checkpoint_name='test', name='checkpoint_name'
    ).make()
    l = self.init_layer(l, x)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.get_output_shape(x.channel_shape), (5,))
    self.verify_contract(l, x, training=False)


# pylint: disable=missing-function-docstring
class Has:
  """A simple `HAS(v)` matcher that tests whether something has `v` in it."""

  def __init__(self, value):
    self._v = value

  @override
  def __eq__(self, o):
    return self._v in o

  @override
  def __ne__(self, o):
    return not self == o

  @override
  def __repr__(self):
    return f'<HAS({self._v!r})>'


class Not:
  """Negates a matcher."""

  def __init__(self, matcher):
    self._matcher = matcher

  @override
  def __eq__(self, o):
    return self._matcher != o

  @override
  def __ne__(self, o):
    return not self == o

  @override
  def __repr__(self):
    return f'<NOT({self._matcher!r})>'


class LoggingTest(test_utils.SequenceLayerTest):
  """Test behavior of Logging layer."""

  @mock.patch.object(logging, 'info', wraps=logging.info)
  def test_logs_tensors(self, mock_logger):
    x = self.sl.types.Sequence.from_values(self.xp.array([[1.414, 2, 3, 4]]))
    training = False

    with self.subTest('prefix'):
      l = self.sl.simple.Logging.Config(prefix='test string').make()
      l = self.init_layer(l, x, bind_only=True)
      l.layer(x, training=training)
      mock_logger.assert_called_with(Has('test string'))
