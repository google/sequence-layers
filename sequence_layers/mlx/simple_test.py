"""Tests for simple MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import typing

from sequence_layers.abstract import simple_test_base
from sequence_layers.mlx import test_utils
from sequence_layers.mlx import simple



class IdentityTest(simple_test_base.IdentityTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Identity()


class ReluTest(simple_test_base.ReluTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Relu()


class GeluTest(simple_test_base.GeluTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Gelu()


class SwishTest(simple_test_base.SwishTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Swish()


class TanhTest(simple_test_base.TanhTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Tanh()


class SigmoidTest(simple_test_base.SigmoidTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Sigmoid()


class LeakyReluTest(simple_test_base.LeakyReluTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, negative_slope=0.01):
    return simple.LeakyRelu(negative_slope=negative_slope)


class EluTest(simple_test_base.EluTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, alpha=1.0):
    return simple.Elu(alpha=alpha)


class SoftmaxTest(simple_test_base.SoftmaxTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, axis=-1):
    return simple.Softmax(axis=axis)


class SoftplusTest(simple_test_base.SoftplusTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Softplus()


class CastTest(simple_test_base.CastTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, dtype):
    return simple.Cast(dtype=dtype)


class ScaleTest(simple_test_base.ScaleTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, scale, name=None):
    return simple.Scale.Config(scale=scale, name=name).make()


class AddTest(simple_test_base.AddTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, shift, name=None):
    return simple.Add.Config(shift=shift, name=name).make()


class MaskInvalidTest(simple_test_base.MaskInvalidTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, mask_value=None):
    return simple.MaskInvalid(mask_value=mask_value)


class GatedUnitTest(simple_test_base.GatedUnitTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, feature_activation=None, gate_activation=None):
    return simple.GatedUnit(
        feature_activation=feature_activation,
        gate_activation=gate_activation,
    )


class GatedLinearUnitTest(simple_test_base.GatedLinearUnitTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.GatedLinearUnit()


class GatedTanhUnitTest(simple_test_base.GatedTanhUnitTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.GatedTanhUnit()


class FlattenTest(simple_test_base.FlattenTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self):
    return simple.Flatten()


class ReshapeTest(simple_test_base.ReshapeTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, output_shape):
    return simple.Reshape(output_shape=output_shape)


class ExpandDimsTest(simple_test_base.ExpandDimsTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, axis):
    return simple.ExpandDims(axis=axis)


class SqueezeTest(simple_test_base.SqueezeTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, axis=None):
    return simple.Squeeze(axis=axis)


class TransposeTest(simple_test_base.TransposeTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, axes=None):
    return simple.Transpose(axes=axes)


class OneHotTest(simple_test_base.OneHotTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, depth):
    return simple.OneHot(depth=depth)


class EmbeddingTest(simple_test_base.EmbeddingTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, num_embeddings, dimension):
    return simple.Embedding(num_embeddings=num_embeddings, dimension=dimension)


class DropoutTest(simple_test_base.DropoutTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, rate=0.0):
    return simple.Dropout(rate=rate)


class Downsample1DTest(simple_test_base.Downsample1DTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, rate):
    return simple.Downsample1D(rate=rate)


class Upsample1DTest(simple_test_base.Upsample1DTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, rate):
    return simple.Upsample1D(rate=rate)


class CheckpointNameTest(simple_test_base.CheckpointNameTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, checkpoint_name=''):
    return simple.CheckpointName(checkpoint_name=checkpoint_name)


class LambdaTest(simple_test_base.LambdaTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, fn, sequence_input=False, mask_required=True):
    return simple.Lambda(
        fn=fn,
        sequence_input=sequence_input,
        mask_required=mask_required,
    )


class LoggingTest(simple_test_base.LoggingTest, test_utils.SequenceLayerTest):
  @typing.override
  def create_layer(self, prefix='', dump_tensors=False):
    return simple.Logging(prefix=prefix, dump_tensors=dump_tensors)




if __name__ == '__main__':
  absltest.main()
