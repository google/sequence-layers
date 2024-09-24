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

"""T5 using SequenceLayers.

Based on:
"Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer"

https://arxiv.org/abs/1910.10683
"""

from typing import Optional
from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf


def FFN(
    dimension: int, hidden_dimension: int, activation: ..., dropout_rate: float
) -> sl.SequenceLayer:
  with tf.name_scope('ffn'):
    return sl.Residual([
        sl.RMSNormalization(epsilon=1e-6),
        sl.Dense(hidden_dimension, use_bias=False, activation=activation),
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
        sl.Dense(dimension, use_bias=False),
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
    ])


def SelfAttention(
    dimension: int,
    num_heads: int,
    units_per_head: int,
    dropout_rate: float,
    relative_position_embedding: sl.RelativePositionEmbedding,
    max_past_horizon: int = -1,
    max_future_horizon: int = -1,
) -> sl.SequenceLayer:
  with tf.name_scope('self_attention'):
    return sl.Residual([
        sl.RMSNormalization(epsilon=1e-6),
        sl.DotProductSelfAttention(
            num_heads=num_heads,
            units_per_head=units_per_head,
            max_horizon=max_past_horizon,
            use_relative_position_embedding=True,
            relative_position_embedding=relative_position_embedding,
            max_future_horizon=max_future_horizon,
            attention_probabilities_dropout_rate=dropout_rate,
            broadcast_dropout_across_queries=True,
            use_bias=False,
        ),
        sl.Flatten(),
        sl.Dense(dimension, use_bias=False),
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
    ])


def CrossAttention(
    source_name: str,
    dimension: int,
    num_heads: int,
    units_per_head: int,
    dropout_rate: float,
) -> sl.SequenceLayer:
  with tf.name_scope('cross_attention'):
    return sl.Residual([
        sl.RMSNormalization(epsilon=1e-6),
        sl.DotProductAttention(
            source_name=source_name,
            num_heads=num_heads,
            units_per_head=units_per_head,
            attention_probabilities_dropout_rate=dropout_rate,
            broadcast_dropout_across_queries=True,
            use_bias=False,
        ),
        sl.Flatten(),
        sl.Dense(dimension, use_bias=False),
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
    ])


def T5Encoder(
    num_layers: int,
    dimension: int,
    num_heads: int,
    ffn_dimension: int,
    ffn_activation: ... = tf.nn.relu,
    dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> sl.SequenceLayer:
  """Builds a T5 encoder."""
  assert dimension % num_heads == 0
  units_per_head = dimension // num_heads

  shared_relative_position_embedding = sl.T5RelativePositionEmbedding(
      num_buckets=32, num_heads=num_heads, bidirectional=True, max_distance=128
  )

  def EncoderBlock(name):
    with tf.name_scope(name):
      return sl.Serial([
          SelfAttention(
              dimension=dimension,
              num_heads=num_heads,
              units_per_head=units_per_head,
              dropout_rate=dropout_rate,
              relative_position_embedding=shared_relative_position_embedding,
              max_past_horizon=-1,
              max_future_horizon=-1,
          ),
          FFN(
              dimension=dimension,
              hidden_dimension=ffn_dimension,
              activation=ffn_activation,
              dropout_rate=dropout_rate,
          ),
      ])

  with tf.name_scope(name or 't5_encoder'):
    return sl.Serial([
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
        sl.Serial(
            [
                EncoderBlock(name=f'layer{layer_i:02d}')
                for layer_i in range(num_layers)
            ]
        ),
        sl.RMSNormalization(epsilon=1e-6),
        sl.Dropout(dropout_rate),
    ])


def T5Decoder(
    source_name: str,
    vocab_size: int,
    num_layers: int,
    dimension: int,
    num_heads: int,
    ffn_dimension: int,
    ffn_activation: ... = tf.nn.relu,
    dropout_rate: float = 0.0,
    max_past_horizon: int = 128,
    name: Optional[str] = None,
) -> sl.SequenceLayer:
  """Builds a T5 decoder."""
  assert dimension % num_heads == 0
  units_per_head = dimension // num_heads

  shared_relative_position_embedding = sl.T5RelativePositionEmbedding(
      num_buckets=32, num_heads=num_heads, bidirectional=False, max_distance=128
  )

  def DecoderBlock(name):
    with tf.name_scope(name):
      return sl.Serial([
          SelfAttention(
              dimension=dimension,
              num_heads=num_heads,
              units_per_head=units_per_head,
              dropout_rate=dropout_rate,
              relative_position_embedding=shared_relative_position_embedding,
              max_past_horizon=max_past_horizon,
              max_future_horizon=0,
          ),
          CrossAttention(
              source_name=source_name,
              dimension=dimension,
              num_heads=num_heads,
              units_per_head=units_per_head,
              dropout_rate=dropout_rate,
          ),
          FFN(
              dimension=dimension,
              hidden_dimension=ffn_dimension,
              activation=ffn_activation,
              dropout_rate=dropout_rate,
          ),
      ])

  with tf.name_scope(name or 'transformer_decoder'):
    return sl.Serial([
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
        sl.Serial(
            [
                DecoderBlock(name=f'layer{layer_i:02d}')
                for layer_i in range(num_layers)
            ]
        ),
        sl.RMSNormalization(epsilon=1e-6),
        sl.Dropout(dropout_rate, noise_shape=[None, 1, None]),
        sl.Dense(vocab_size, name='to_logits'),
    ])
