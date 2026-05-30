# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for combinator MLX sequence layers."""

import mlx.core as mx
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import combinators
from sequence_layers.mlx import dense
from sequence_layers.mlx import simple
from sequence_layers.mlx import test_utils
from sequence_layers.specs import combinators_behaviors as spec_behaviors


class CombinatorBehaviorsTest(test_utils.SequenceLayerTest, spec_behaviors.CombinatorBehaviorsTest):
  """Shared behavior tests for combinators in MLX."""


class SerialTest(test_utils.SequenceLayerTest):

  def test_identity_serial(self):
    layer = combinators.Serial([
        simple.Identity.Config().make(),
        simple.Identity.Config().make(),
    ])
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_dense_serial(self):
    layer = combinators.Serial([
        dense.Dense.Config(features=8).make(),
        dense.Dense.Config(features=16).make(),
    ])
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_output_shape(self):
    layer = combinators.Serial([
        dense.Dense.Config(features=8).make(),
        dense.Dense.Config(features=16).make(),
    ])
    self.assertEqual(layer.get_output_shape((4,)), (16,))

  def test_from_config(self):
    import sequence_layers.jax as sl

    config = sl.Serial.Config([
        sl.Identity.Config(),
        sl.Dense.Config(features=8),
    ])
    mlx_layer = combinators.Serial.from_config(config)
    self.assertIsInstance(mlx_layer, combinators.Serial)


class ResidualTest(test_utils.SequenceLayerTest):

  def test_identity_residual(self):
    layer = combinators.Residual([simple.Identity.Config().make()])
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_residual_adds(self):
    layer = combinators.Residual([simple.Identity.Config().make()])
    x = self.random_sequence(1, 3, 4)
    y = layer.layer(x, training=False)
    # y = identity(x) + x = 2 * x
    expected = x.values * 2
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

  def test_from_config(self):
    import sequence_layers.jax as sl

    config = sl.Residual.Config([sl.Identity.Config()])
    mlx_layer = combinators.Residual.from_config(config)
    self.assertIsInstance(mlx_layer, combinators.Residual)


class RepeatTest(test_utils.SequenceLayerTest):

  def test_repeat_identity(self):
    layers = [simple.Identity.Config().make() for _ in range(3)]
    layer = combinators.Repeat(layers)
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_repeat_dense(self):
    layers = [dense.Dense.Config(features=4).make() for _ in range(3)]
    layer = combinators.Repeat(layers)
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_num_repeats(self):
    layers = [simple.Identity.Config().make() for _ in range(5)]
    layer = combinators.Repeat(layers)
    self.assertEqual(layer.num_repeats, 5)

  def test_from_config(self):
    import sequence_layers.jax as sl

    config = sl.Repeat.Config(
        layer=sl.Identity.Config(),
        num_repeats=4,
    )
    mlx_layer = combinators.Repeat.from_config(config)
    self.assertIsInstance(mlx_layer, combinators.Repeat)
    self.assertEqual(mlx_layer.num_repeats, 4)


class ParallelTest(test_utils.SequenceLayerTest):

  def test_stack(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.STACK,
    )
    x = self.random_sequence(1, 4, 3)
    y = layer.layer(x, training=False)
    # STACK: (3,) + (3,) -> (2, 3)
    self.assertEqual(y.channel_shape, (2, 3))

  def test_concat(self):
    layer = combinators.Parallel(
        [
            dense.Dense.Config(features=3).make(),
            dense.Dense.Config(features=5).make(),
        ],
        combination=combinators.CombinationMode.CONCAT,
    )
    x = self.random_sequence(1, 4, 4)
    y = layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (8,))

  def test_add(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.ADD,
    )
    x = self.random_sequence(1, 4, 3)
    y = layer.layer(x, training=False)
    self.assertEqual(y.channel_shape, (3,))
    # ADD of two identities = 2x
    np.testing.assert_allclose(y.values, x.values * 2, atol=1e-6)

  def test_mean(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.MEAN,
    )
    x = self.random_sequence(1, 4, 3)
    y = layer.layer(x, training=False)
    # MEAN of two identities = x
    np.testing.assert_allclose(y.values, x.values, atol=1e-6)

  def test_product(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.PRODUCT,
    )
    x = self.random_sequence(1, 4, 3)
    y = layer.layer(x, training=False)
    # PRODUCT of two identities = x^2
    np.testing.assert_allclose(y.values, x.values * x.values, atol=1e-6)

  def test_step_consistency(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.ADD,
    )
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(layer, x)

  def test_output_shape_stack(self):
    layer = combinators.Parallel(
        [simple.Identity.Config().make(), simple.Identity.Config().make()],
        combination=combinators.CombinationMode.STACK,
    )
    self.assertEqual(layer.get_output_shape((4,)), (2, 4))

  def test_output_shape_concat(self):
    layer = combinators.Parallel(
        [
            dense.Dense.Config(features=3).make(),
            dense.Dense.Config(features=5).make(),
        ],
        combination=combinators.CombinationMode.CONCAT,
    )
    self.assertEqual(layer.get_output_shape((4,)), (8,))

  def test_from_config(self):
    import sequence_layers.jax as sl
    from sequence_layers.jax import utils as jax_utils

    config = sl.Parallel.Config(
        layers=[sl.Identity.Config(), sl.Identity.Config()],
        combination=jax_utils.CombinationMode.ADD,
    )
    mlx_layer = combinators.Parallel.from_config(config)
    self.assertIsInstance(mlx_layer, combinators.Parallel)
    x = self.random_sequence(2, 5, 4)
    self.verify_contract(mlx_layer, x)

  def test_unequal_ratio_raises(self):
    from sequence_layers.mlx import convolution

    with self.assertRaises(ValueError):
      combinators.Parallel([
          simple.Identity.Config().make(),
          convolution.Conv1D(
              in_features=4,
              filters=4,
              kernel_size=3,
              strides=2,
              padding='causal',
          ),
      ])


class TransformerEndToEndTest(test_utils.SequenceLayerTest):
  """End-to-end test with a full Transformer config."""

  def test_decoder_transformer(self):
    import sequence_layers.jax as sl
    from sequence_layers.jax.attention import (
        dot_product_self_attention as dpa,
    )
    import jax

    # Attention outputs [b, t, num_heads, units_per_head].
    # A Dense layer after it projects back to model dim.
    config = sl.Serial.Config([
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            dpa.DotProductSelfAttention.Config(
                num_heads=4,
                units_per_head=8,
                max_past_horizon=64,
            ),
            sl.Flatten.Config(),
            sl.Dense.Config(features=32),
        ]),
        sl.Residual.Config([
            sl.RMSNormalization.Config(),
            sl.Dense.Config(features=64, activation=jax.nn.gelu),
            sl.Dense.Config(features=32),
        ]),
    ])
    model = combinators.Serial.from_config(config)

    # Layer mode.
    x = self.random_sequence(1, 10, 32)
    y = model.layer(x, training=False)
    self.assertEqual(y.shape, (1, 10, 32))

    # Step mode.
    spec = bt.ShapeDType((32,), mx.float32)
    state = model.get_initial_state(1, spec)
    x_step = self.random_sequence(1, 1, 32)
    for _ in range(5):
      y_step, state = model.step(x_step, state, training=False)
    self.assertEqual(y_step.shape, (1, 1, 32))


if __name__ == '__main__':
  absltest.main()
