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
from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
from sequence_layers.jax import test_utils
from sequence_layers.jax.attention import gmm_attention


class GmmAttentionTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (1, 2, True),
      (1, 2, False),
      (3, 5, True),
      (3, 5, False),
  )
  def test_gmm_attention(self, num_heads, units_per_head, monotonic):
    key = jax.random.PRNGKey(1234)
    batch_size, source_time, source_channels = 2, 11, 5
    channels = 3
    num_components = 7
    source_name = 'source'
    l = gmm_attention.GmmAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        num_components=num_components,
        monotonic=monotonic,
        init_offset_bias=1.0,
        init_scale_bias=1.0,
        name='gmm_attention',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}

    x = test_utils.random_sequence(batch_size, 1, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(l.name, 'gmm_attention')

    # Shapes here should be the FlaxEinsumDense equation's second argument.
    chex.assert_trees_all_equal_shapes_and_dtypes(
        flax.core.meta.unbox(l.variables),
        {
            'params': {
                'hidden': {
                    'kernel': jnp.zeros((channels, num_heads, units_per_head)),
                },
                'output': {
                    'kernel': jnp.zeros(
                        (num_heads, units_per_head, num_components, 3)
                    ),
                    'bias': jnp.zeros((num_heads, num_components, 3)),
                },
            },
        },
    )

    for time in [11, 12]:
      x = test_utils.random_sequence(batch_size, time, channels)
      self.assertEqual(
          l.get_output_shape_for_sequence(x, constants=constants),
          (num_heads, source_channels),
      )
      self.verify_contract(
          l,
          x,
          training=False,
          constants=constants,
          grad_atol=1e-5,
          grad_rtol=1e-5,
      )

  def test_emit_outputs(self):
    """Run a separate test since verify_contract() can't return emits."""
    key = jax.random.PRNGKey(1234)
    num_heads, units_per_head, monotonic = 3, 5, True
    batch_size, source_time, source_channels = 2, 11, 2
    source_name = 'source'
    l = gmm_attention.GmmAttention.Config(
        source_name,
        num_heads=num_heads,
        units_per_head=units_per_head,
        num_components=5,
        monotonic=monotonic,
        init_offset_bias=1.0,
        init_scale_bias=1.0,
        name='gmm_attention',
    ).make()
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)

    source = test_utils.random_sequence(
        batch_size, source_time, source_channels
    )
    constants = {source_name: source}
    time, channels = 7, 3
    x = test_utils.random_sequence(batch_size, time, channels)
    l = self.init_and_bind_layer(key, l, x, constants=constants)

    _, emits = l.layer_with_emits(x, training=False, constants=constants)
    self.assertEqual(
        emits.probabilities_by_source[source_name].values.shape,
        (batch_size, time, num_heads, source_time),
    )

    # Only run three timesteps of the sequence.
    x = x[:, :3]
    _, _, emits = l.step_with_emits(
        x,
        l.get_initial_state(
            batch_size=batch_size,
            input_spec=x.channel_spec,
            training=False,
            constants=constants,
        ),
        training=False,
        constants=constants,
    )
    self.assertEqual(
        emits.probabilities_by_source[source_name].values.shape,
        (batch_size, 3, num_heads, source_time),
    )


if __name__ == '__main__':
  test_utils.main()
