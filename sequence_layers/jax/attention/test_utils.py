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
"""Test helpers for attention layers."""

import chex
import flax
import jax.numpy as jnp
from sequence_layers.jax import types
from sequence_layers.jax.attention import common
from sequence_layers.jax.attention import shaw_relative_position_embedding
from sequence_layers.jax.attention import t5_relative_position_embedding
from sequence_layers.jax.attention import transformer_xl_relative_position_embedding


def assert_param_dtypes_inits_shapes(
    layer: types.SequenceLayer,
    inputs: types.Sequence,
    input_projection: common.QueryKeyValueProjectionConfig,
    constants: types.Constants | None = None,
    num_sink_embeddings: int = 0,
    use_sink_scalars: bool = False,
) -> None:
  """Asserts that the parameter dtypes, inits, and shapes are correct."""
  config = layer.config
  params = {}

  num_query_heads = config.num_heads
  if hasattr(config, 'num_kv_heads'):
    num_kv_heads = config.num_kv_heads or num_query_heads
  else:
    num_kv_heads = num_query_heads

  # A per-dimension scale factor shared across heads:
  if config.per_dim_scale:
    params['per_dim_scale'] = jnp.zeros(
        (config.units_per_head,), dtype=config.param_dtype
    )

  if num_sink_embeddings > 0:
    params['sink_key_embeddings'] = jnp.zeros(
        (
            num_sink_embeddings,
            num_query_heads,
            config.units_per_head,
        ),
        dtype=config.param_dtype,
    )

    params['sink_value_embeddings'] = jnp.zeros(
        (
            num_sink_embeddings,
            num_kv_heads,
            config.units_per_head,
        ),
        dtype=config.param_dtype,
    )

  if use_sink_scalars:
    params['sink_scalars'] = jnp.zeros(
        (num_query_heads,),
        dtype=config.param_dtype,
    )

  q_in_channels = inputs.shape[2]
  # Cross- versus self-attention:
  if constants is not None:
    if hasattr(config, 'source_names'):
      kv_in_channels = constants[layer._source_names[0]].shape[2]  # pylint: disable=protected-access
    else:
      kv_in_channels = constants[config.source_name].shape[2]
  else:
    kv_in_channels = q_in_channels

  match input_projection:
    case common.CombinedQueryKeyValueProjection():
      # A bias-less dense layer for qkv projection
      params['query_key_value_projection'] = {
          'kernel': jnp.zeros(
              (
                  q_in_channels,
                  2 if input_projection.share_kv_projection else 3,
                  num_query_heads,
                  config.units_per_head,
              ),
              dtype=config.param_dtype,
          ),
      }
    case common.SeparateQueryKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['key_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
    case common.QueryAndKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['key_value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, 2, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
    case common.QueryAndSharedKeyValueProjection():
      params['query_projection'] = {
          'kernel': jnp.zeros(
              (q_in_channels, num_query_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }
      params['shared_key_value_projection'] = {
          'kernel': jnp.zeros(
              (kv_in_channels, num_kv_heads, config.units_per_head),
              dtype=config.param_dtype,
          ),
      }

  # Position embeddings:
  if pos_config := getattr(config, 'relative_position_embedding', None):
    if isinstance(
        pos_config,
        shaw_relative_position_embedding.ShawRelativePositionEmbedding.Config,
    ):
      params['relative_position_embedding'] = {
          'embedding': jnp.zeros(
              (
                  pos_config.max_backward + pos_config.max_forward + 1,
                  num_query_heads,
                  config.units_per_head,
              ),
              dtype=pos_config.param_dtype,
          ),
      }
    if isinstance(
        pos_config,
        t5_relative_position_embedding.T5RelativePositionEmbedding.Config,
    ):
      params['relative_position_embedding'] = {
          'embedding': jnp.zeros(
              (
                  pos_config.num_buckets,
                  num_query_heads,
              ),
              dtype=pos_config.param_dtype,
          ),
      }
    elif isinstance(
        pos_config,
        transformer_xl_relative_position_embedding.TransformerXLRelativePositionEmbedding.Config,
    ):
      params['relative_position_embedding'] = {
          'u': jnp.zeros(
              (num_query_heads, config.units_per_head),
              dtype=pos_config.param_dtype,
          ),
          'v': jnp.zeros(
              (num_query_heads, config.units_per_head),
              dtype=pos_config.param_dtype,
          ),
          'pos_proj': {
              'kernel': jnp.zeros(
                  (
                      pos_config.position_bias_dim,
                      num_query_heads,
                      config.units_per_head,
                  ),
                  dtype=pos_config.param_dtype,
              )
          },
      }

  chex.assert_trees_all_equal_shapes_and_dtypes(
      flax.core.meta.unbox(layer.variables),
      {'params': params},
  )
