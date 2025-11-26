# Copyright 2025 Google LLC
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
from sequence_layers.jax.examples import global_style_token


class StyleTokenTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      ((2, 3, 5),),
      ((2, 3, 5, 9),),
  )
  def test_style_token(self, shape):
    key = jax.random.PRNGKey(1234)
    num_style_tokens, num_heads, units_per_head = 3, 5, 7
    x = test_utils.random_sequence(*shape)
    l = global_style_token.StyleToken.Config(
        num_style_tokens, num_heads, units_per_head, name='style_token'
    ).make()
    l = self.init_and_bind_layer(key, l, x)
    self.assertEqual(l.block_size, 1)
    self.assertEqual(l.output_ratio, 1)
    self.assertEqual(
        l.get_output_shape_for_sequence(x),
        (num_heads, units_per_head),
    )
    self.verify_contract(l, x, training=False)
    # 6 variables:
    # - The style tokens.
    # - Style token keys.
    # - A projection matrix and bias for the query.
    # - A logits projection matrix and bias.
    variables = flax.core.meta.unbox(l.variables)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        variables,
        {
            'params': {
                'style_tokens': jnp.zeros(
                    (num_style_tokens, units_per_head), dtype=jnp.float32
                ),
                'style_token_keys': jnp.zeros(
                    (1, 1, num_heads, num_style_tokens, units_per_head),
                    dtype=jnp.float32,
                ),
                'query_projection': {
                    'kernel': jnp.zeros(
                        x.channel_shape + (num_heads, 1, units_per_head),
                        dtype=jnp.float32,
                    ),
                    'bias': jnp.zeros(
                        (num_heads, 1, units_per_head), dtype=jnp.float32
                    ),
                },
                'to_logits': {
                    'kernel': jnp.zeros((units_per_head, 1), dtype=jnp.float32),
                    'bias': jnp.zeros((1,), dtype=jnp.float32),
                },
            }
        },
    )


if __name__ == '__main__':
  test_utils.main()
