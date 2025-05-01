# Copyright 2024 Google LLC
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

"""Transformer using SequenceLayers.

Based on:
"Attention Is All You Need"
https://arxiv.org/abs/1706.03762
"""

from typing import Optional
from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf


def TransformerEncoder(
    num_layers: int,
    dimension: int,
    num_heads: int,
    max_horizon: int,
    max_future_horizon: int,
    ffn_activation=tf.nn.relu,
    attention_probabilities_dropout_rate: float = 0.0,
    ffn_dropout_rate: float = 0.0,
    kernel_initializer=None,
    name: Optional[str] = None,
):
  """Builds a Transformer encoder."""
  assert dimension % num_heads == 0
  units_per_head = dimension // num_heads

  def TransformerLayer(name):
    with tf.name_scope(name):
      return sl.Serial([
          sl.Residual([
              sl.DotProductSelfAttention(
                  num_heads=num_heads,
                  units_per_head=units_per_head,
                  max_horizon=max_horizon,
                  max_future_horizon=max_future_horizon,
                  kernel_initializer=CloneInitializer(kernel_initializer),
                  attention_probabilities_dropout_rate=attention_probabilities_dropout_rate,
              ),
              sl.Flatten(),
          ]),
          sl.LayerNormalization(),
          sl.Residual([
              sl.Dense(
                  dimension * 4,
                  kernel_initializer=CloneInitializer(kernel_initializer),
                  activation=ffn_activation,
              ),
              sl.Dense(
                  dimension,
                  kernel_initializer=CloneInitializer(kernel_initializer),
              ),
              sl.Dropout(ffn_dropout_rate),
          ]),
          sl.LayerNormalization(),
      ])

  with tf.name_scope(name or 'transformer_encoder'):
    return sl.Serial(
        [sl.AddTimingSignal()]
        + [
            TransformerLayer(name=f'layer{layer_i:02d}')
            for layer_i in range(num_layers)
        ]
    )


def TransformerDecoder(
    source_name: str,
    num_layers: int,
    dimension: int,
    num_heads: int,
    max_horizon: int,
    ffn_activation=tf.nn.relu,
    attention_probabilities_dropout_rate: float = 0.0,
    ffn_dropout_rate: float = 0.0,
    kernel_initializer=None,
    name: Optional[str] = None,
):
  """Builds a Transformer decoder."""
  assert dimension % num_heads == 0
  units_per_head = dimension // num_heads

  def TransformerLayer(name):
    with tf.name_scope(name):
      return sl.Serial([
          sl.Residual([
              sl.DotProductSelfAttention(
                  num_heads=num_heads,
                  units_per_head=units_per_head,
                  max_horizon=max_horizon,
                  attention_probabilities_dropout_rate=attention_probabilities_dropout_rate,
                  kernel_initializer=CloneInitializer(kernel_initializer),
              ),
              sl.Flatten(),
          ]),
          sl.LayerNormalization(),
          # TODO(rryan): Timing signal for queries and keys.
          sl.Residual([
              sl.DotProductAttention(
                  source_name,
                  num_heads=num_heads,
                  units_per_head=units_per_head,
                  attention_probabilities_dropout_rate=attention_probabilities_dropout_rate,
                  kernel_initializer=CloneInitializer(kernel_initializer),
              ),
              sl.Flatten(),
          ]),
          sl.LayerNormalization(),
          sl.Residual([
              sl.Dense(
                  dimension * 4,
                  kernel_initializer=CloneInitializer(kernel_initializer),
                  activation=ffn_activation,
              ),
              sl.Dense(
                  dimension,
                  kernel_initializer=CloneInitializer(kernel_initializer),
              ),
              sl.Dropout(ffn_dropout_rate),
          ]),
          sl.LayerNormalization(),
      ])

  with tf.name_scope(name or 'transformer_decoder'):
    return sl.Serial(
        [sl.AddTimingSignal()]
        + [
            TransformerLayer(name=f'layer{layer_i:02d}')
            for layer_i in range(num_layers)
        ]
        + [sl.Dense(dimension, activation=tf.nn.softmax)]
    )


def CloneInitializer(initializer):
  # Keras initializer is going to be stateless, which mean reusing the same
  # initializer will produce same init value when the shapes are the same.
  if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
  # When the input is string/dict or other serialized configs, caller will
  # create a new keras Initializer instance based on that, and we don't need to
  # do anything
  return initializer
