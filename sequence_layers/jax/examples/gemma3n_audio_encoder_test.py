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

import chex
import flax
import jax
import jax.numpy as jnp
from sequence_layers.jax import test_utils
from sequence_layers.jax.examples import gemma3n_audio_encoder


class Gemma3nEncoderTest(test_utils.SequenceLayerTest):

  def test_shapes_dtypes(self):
    key = jax.random.PRNGKey(1234)
    model_dim = 5
    atten_num_heads = 2
    num_layers = 3
    l = gemma3n_audio_encoder.Gemma3nAudioEncoderConfig(
        model_dims=model_dim,
        num_layers=num_layers,
        atten_num_heads=atten_num_heads,
        name='encoder',
    ).make()

    batch, time, channels = 2, 1, 3
    x = test_utils.random_sequence(batch, time, channels)
    l = self.init_and_bind_layer(key, l, x)

    with self.subTest('layer_properties'):
      # Two stride-2 convs (4) times reduction_factor (4) yields block_size=16.
      self.assertEqual(l.block_size, 16)
      self.assertEqual(1 / l.output_ratio, 16)
      self.assertTrue(l.supports_step)
      # The output latency is the number of frames of input it takes to produce
      # the first frame of output. This is determined by the number of
      # subsampling layers.
      self.assertEqual(int(l.output_latency), 1)
      self.assertEqual(l.name, 'encoder')

    with self.subTest('shapes_dtypes'):
      chex.assert_trees_all_equal_shapes_and_dtypes(
          flax.core.meta.unbox(l.variables),
          {
              'params': {
                  'conformer': {
                      'stacked_layers': {
                          'fflayer_end': {
                              'ffn_layer1': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim, model_dim * 4)
                                  )
                              },
                              'ffn_layer2': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim * 4, model_dim)
                                  )
                              },
                              'post_layer_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                              'pre_layer_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                          },
                          'fflayer_start': {
                              'ffn_layer1': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim, model_dim * 4)
                                  )
                              },
                              'ffn_layer2': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim * 4, model_dim)
                                  )
                              },
                              'post_layer_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                              'pre_layer_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                          },
                          'final_ln': {
                              'scale': jnp.zeros((num_layers, model_dim))
                          },
                          'lconv': {
                              'conv_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                              'depthwise_conv1d': {
                                  'kernel': jnp.zeros(
                                      (num_layers, 5, 1, model_dim)
                                  )
                              },
                              'linear_end': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim, model_dim)
                                  )
                              },
                              'linear_start': {
                                  'kernel': jnp.zeros(
                                      (num_layers, model_dim, model_dim * 2)
                                  )
                              },
                              'ln': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                          },
                          'trans_atten': {
                              'post': {
                                  'kernel': jnp.zeros(
                                      (num_layers, 2, 2, model_dim)
                                  )
                              },
                              'post_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                              'pre_norm': {
                                  'scale': jnp.zeros((num_layers, model_dim))
                              },
                              'self_atten': {
                                  'per_dim_scale': jnp.zeros((num_layers, 2)),
                                  'query_key_value_projection': {
                                      'kernel': jnp.zeros(
                                          (num_layers, model_dim, 3, 2, 2)
                                      )
                                  },
                                  'relative_position_embedding': {
                                      'pos_proj': {
                                          'kernel': jnp.zeros(
                                              (num_layers, model_dim, 2, 2)
                                          )
                                      }
                                  },
                              },
                          },
                      },
                  },
                  'feature': {
                      'input_proj': {'kernel': jnp.zeros((1, 32, model_dim))},
                      'norm_0': {'scale': jnp.zeros((128,))},
                      'norm_1': {'scale': jnp.zeros((32,))},
                      'subsampling_0': {'kernel': jnp.zeros((3, 3, 1, 128))},
                      'subsampling_1': {'kernel': jnp.zeros((3, 3, 128, 32))},
                  },
              },
          },
      )

    with self.subTest('verify_contract'):
      time = l.block_size * 2
      x = test_utils.random_sequence(batch, time, channels)
      self.assertEqual(l.get_output_shape(x.channel_shape), (model_dim,))
      self.verify_contract(l, x, training=False, rtol=1e-2, atol=1e-3)


if __name__ == '__main__':
  test_utils.main()
