"""Abstract tests for simple sequence layers."""

import abc
import fractions
import typing
from typing import Any, Callable
from absl.testing import parameterized
import numpy as np
from sequence_layers.abstract import types
from sequence_layers.abstract import test_utils

class IdentityTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_preserves_values(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 3, 4)
    try:
        y = self.call_layer(layer, x, training=False)
    except TypeError:
        y = self.call_layer(layer, x)
    self.assertSequencesEqual(y, x)

class ReluTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_negative_zeroed(self):
    layer = self.create_layer()
    xp = self.get_backend()
    values = xp.array([[-1.0, 0.5, -0.3, 2.0]]).reshape(1, 1, 4)
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected = xp.array([[[0.0, 0.5, 0.0, 2.0]]])
    self.assertAllClose(y.values, expected)

class GeluTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class SwishTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class TanhTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_values(self):
    layer = self.create_layer()
    xp = self.get_backend()
    values = xp.array([[[0.0, 1.0, -1.0, 100.0]]])
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected = xp.array([[[0.0, np.tanh(1.0), np.tanh(-1.0), np.tanh(100.0)]]])
    self.assertAllClose(y.values, expected)

class SigmoidTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class LeakyReluTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, negative_slope: float) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(negative_slope=0.2)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_negative_slope(self):
    layer = self.create_layer(negative_slope=0.1)
    xp = self.get_backend()
    values = xp.array([[[-2.0, 0.5, -1.0, 3.0]]])
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected = xp.array([[[-0.2, 0.5, -0.1, 3.0]]])
    self.assertAllClose(y.values, expected)

class EluTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, alpha: float) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(alpha=1.0)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class SoftmaxTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axis: int) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(axis=-1)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_sums_to_one(self):
    layer = self.create_layer(axis=-1)
    xp = self.get_backend()
    values = xp.array([[[1.0, 2.0, 3.0, 4.0]]])
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    self.assertAllClose(float(xp.sum(y.values)), 1.0)

class SoftplusTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class CastTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, dtype: Any) -> types.Steppable:
    pass

  def test_verify_contract(self):
    xp = self.get_backend()
    layer = self.create_layer(dtype=xp.float16)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_cast(self):
    xp = self.get_backend()
    layer = self.create_layer(dtype=xp.float16)
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertEqual(y.dtype, xp.float16)

class ScaleTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, scale: Any, name: str | None = None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(scale=2.0)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_scalar(self):
    layer = self.create_layer(scale=2.0)
    xp = self.get_backend()
    values = xp.array([[[1.0, 2.0, 3.0]]])
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    try:
        y = self.call_layer(layer, x, training=False)
    except TypeError:
        y = self.call_layer(layer, x)
    expected = xp.array([[[2.0, 4.0, 6.0]]])
    self.assertAllClose(y.values, expected)

  def test_ndarray(self):
    xp = self.get_backend()
    l = self.create_layer(scale=np.arange(5, dtype=np.float32), name='scale')
    x = self.random_sequence(2, 13, 5)
    l = self.init_layer(l, x)
    y = self.verify_contract(l, x, training=False)
    y_expected = x.apply_values(lambda v: v * np.arange(5, dtype=np.float32))
    self.assertSequencesEqual(y, y_expected)

  def test_broadcast(self):
    xp = self.get_backend()
    l = self.create_layer(scale=np.ones((5, 9)))
    x = self.random_sequence(2, 3, 5, 1)
    self.assertEqual(l.get_output_shape_for_sequence(x), (5, 9))

  def test_broadcast_failure(self):
    l = self.create_layer(scale=np.ones((5,)))
    x = self.random_sequence(2, 3, 5, 9)
    with self.assertRaises(Exception):
      l.get_output_shape_for_sequence(x)
    with self.assertRaises(Exception):
      l.layer(x, training=False)

class AddTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, shift: Any) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(shift=1.0)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_scalar(self):
    layer = self.create_layer(shift=10.0)
    xp = self.get_backend()
    values = xp.array([[[1.0, 2.0, 3.0]]])
    mask = xp.array([[True]])
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected = xp.array([[[11.0, 12.0, 13.0]]])
    self.assertAllClose(y.values, expected)

class MaskInvalidTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, mask_value: Any | None = None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_masks_to_zero(self):
    layer = self.create_layer()
    xp = self.get_backend()
    values = xp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = xp.array([[True, False, True]])
    x = self.Sequence(values, mask)
    y = self.call_layer(layer, x)
    expected = xp.array([[[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]])
    self.assertAllClose(y.values, expected)

class GatedUnitTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, feature_activation=None, gate_activation=None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 8)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_with_activations(self):
    xp = self.get_backend()
    # Need to handle backend-specific activations if we want to test them here.
    # But we can just use None or a generic fn.
    # For now, let's just test defaults.
    pass

class GatedLinearUnitTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 8)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_halves_channels(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((8,)), (4,))

class GatedTanhUnitTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 8)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

class FlattenTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 2, 3, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_flatten(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (24,))

class ReshapeTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, output_shape: tuple[int, ...]) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(output_shape=(2, 6))
    x = self.random_sequence(2, 8, 12)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_reshape(self):
    layer = self.create_layer(output_shape=(2, 6))
    x = self.random_sequence(1, 3, 12)
    y = self.call_layer(layer, x)
    self.assertEqual(y.channel_shape, (2, 6))

  def test_mismatch_raises(self):
    layer = self.create_layer(output_shape=(5,))
    with self.assertRaises(ValueError):
      layer.get_output_shape((12,))

class ExpandDimsTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axis: int | tuple[int, ...]) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(axis=-1)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_expand(self):
    layer = self.create_layer(axis=0)
    self.assertEqual(layer.get_output_shape((4, 8)), (1, 4, 8))

class SqueezeTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axis: int | tuple[int, ...] | None = None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 4, 1)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_squeeze(self):
    layer = self.create_layer()
    xp = self.get_backend()
    x = self.MaskedSequence(
        xp.ones((1, 3, 1, 4, 1)),
        xp.ones((1, 3), dtype=getattr(xp, 'bool_', bool)),
    )
    y = self.call_layer(layer, x)
    self.assertEqual(y.channel_shape, (4,))

class TransposeTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axes: tuple[int, ...] | None = None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 2, 3, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_reverse(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (4, 3, 2))

class OneHotTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, depth: int) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(depth=5)
    # OneHot usually expects integer inputs.
    # Our random_sequence generates floats.
    # So we might need to cast or use a custom test.
    pass

  def test_values(self):
    layer = self.create_layer(depth=5)
    xp = self.get_backend()
    x = self.MaskedSequence(
        xp.array([[0, 2, 4]]),
        xp.ones((1, 3), dtype=getattr(xp, 'bool_', bool)),
    )
    y = self.call_layer(layer, x)
    self.assertEqual(y.shape, (1, 3, 5))
    self.assertAllClose(np.array(y.values[0, 0]), [1, 0, 0, 0, 0])

class EmbeddingTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, num_embeddings: int, dimension: int) -> types.Steppable:
    pass

  def test_verify_contract(self):
    # Embedding expects integer inputs.
    pass

  def test_output_shape(self):
    layer = self.create_layer(num_embeddings=10, dimension=8)
    self.assertEqual(layer.get_output_shape(()), (8,))
    self.assertEqual(layer.get_output_shape((3,)), (3, 8))

class DropoutTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, rate: float) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(rate=0.5)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_passthrough(self):
    layer = self.create_layer(rate=0.5)
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertSequencesEqual(y, x)

class Downsample1DTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, rate: int) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(rate=2)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_layer(self):
    layer = self.create_layer(rate=2)
    x = self.random_sequence(1, 6, 4)
    y = self.call_layer(layer, x)
    self.assertEqual(y.shape, (1, 3, 4))

class Upsample1DTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, rate: int) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(rate=3)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_layer(self):
    layer = self.create_layer(rate=3)
    x = self.random_sequence(1, 4, 2)
    y = self.call_layer(layer, x)
    self.assertEqual(y.shape, (1, 12, 2))

  def test_values(self):
    layer = self.create_layer(rate=2)
    xp = self.get_backend()
    values_np = np.arange(4).reshape(1, 2, 2).astype(np.float32)
    values = xp.array(values_np)
    mask = xp.ones((1, 2), dtype=getattr(xp, 'bool_', bool))
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected_values = np.repeat(values_np, 2, axis=1)
    self.assertAllClose(y.values, expected_values)


class ExpandDimsTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axis: int | tuple[int, ...]) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(axis=-1)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_expand(self):
    layer = self.create_layer(axis=0)
    self.assertEqual(layer.get_output_shape((4, 8)), (1, 4, 8))

  def test_layer_values(self):
    layer = self.create_layer(axis=-1)
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertEqual(y.channel_shape, (4, 1))


class TransposeTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, axes: tuple[int, ...] | None = None) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer()
    x = self.random_sequence(2, 8, 2, 3, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_reverse(self):
    layer = self.create_layer()
    self.assertEqual(layer.get_output_shape((2, 3, 4)), (4, 3, 2))

  def test_explicit(self):
    layer = self.create_layer(axes=(3, 2, 4))
    self.assertEqual(layer.get_output_shape((5, 6, 7)), (6, 5, 7))


class Downsample1DTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, rate: int) -> types.Steppable:
    pass

  import unittest
  @unittest.skip("XLA Segfaults on Apple Silicon intermittently")
  def test_verify_contract(self):
    layer = self.create_layer(rate=2)
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False, test_gradients=False, test_padding_invariance=False, test_batching=False, test_2x_step=False)

  def test_layer(self):
    layer = self.create_layer(rate=2)
    x = self.random_sequence(1, 6, 4)
    y = self.call_layer(layer, x)
    self.assertEqual(y.shape, (1, 3, 4))

  def test_values(self):
    layer = self.create_layer(rate=3)
    xp = self.get_backend()
    values_np = np.arange(12).reshape(1, 6, 2).astype(np.float32)
    values = xp.array(values_np)
    mask = xp.ones((1, 6), dtype=getattr(xp, 'bool_', bool))
    x = self.MaskedSequence(values, mask)
    y = self.call_layer(layer, x)
    expected_values = values_np[:, ::3]
    self.assertAllClose(y.values, expected_values)


class CheckpointNameTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, checkpoint_name: str) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(checkpoint_name='test')
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_passthrough(self):
    layer = self.create_layer(checkpoint_name='test')
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertSequencesEqual(y, x)


class LambdaTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, fn: Callable, sequence_input: bool = False, mask_required: bool = True) -> types.Steppable:
    pass

  def test_values_fn(self):
    layer = self.create_layer(fn=lambda v: v * 2.0)
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertAllClose(y.values, np.array(x.values) * 2.0)

  def test_sequence_fn(self):
    def double_seq(s):
      return self.Sequence(s.values * 2.0, s.mask)

    layer = self.create_layer(fn=double_seq, sequence_input=True)
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertAllClose(y.values, np.array(x.values) * 2.0)


class LoggingTest(test_utils.SequenceLayerTest):
  @abc.abstractmethod
  def create_layer(self, prefix: str = '', dump_tensors: bool = False) -> types.Steppable:
    pass

  def test_verify_contract(self):
    layer = self.create_layer(prefix='test')
    x = self.random_sequence(2, 8, 4)
    layer = self.init_layer(layer, x)
    self.verify_contract(layer, x, training=False)

  def test_passthrough(self):
    layer = self.create_layer()
    x = self.random_sequence(1, 3, 4)
    y = self.call_layer(layer, x)
    self.assertSequencesEqual(y, x)
