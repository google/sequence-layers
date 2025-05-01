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

"""Conformer using Sequence Layers.

TODO(rryan): SpecAugment is unimplemented for now.

Based on:
"Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""

from typing import Optional

from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf


def MultiHeadedSelfAttention(
    hidden_size: int, num_heads: int, max_horizon: int, dropout_rate: float
) -> sl.SequenceLayer:
  with tf.name_scope('multi_headed_self_attention_module'):
    return sl.Residual([
        sl.LayerNormalization(),
        # TODO(rryan): Lingvo's Conformer implementation drops out attention
        # probs as well. Add support for this.
        sl.DotProductSelfAttention(
            num_heads=num_heads,
            units_per_head=hidden_size // num_heads,
            max_horizon=max_horizon,
            use_relative_position_embedding=True,
        ),
        # Combine heads with linear projection: [b, t, heads * units_per_head].
        sl.DenseShaped(hidden_size),
        sl.Dropout(dropout_rate),
    ])


def FeedForwardModule(
    hidden_size: int, dropout_rate: float
) -> sl.SequenceLayer:
  with tf.name_scope('feed_forward_module'):
    return sl.Residual([
        sl.LayerNormalization(),
        sl.Dense(4 * hidden_size, activation=tf.nn.swish),
        sl.Dropout(dropout_rate),
        sl.Dense(hidden_size),
        sl.Dropout(dropout_rate),
        sl.Scale(0.5),
    ])


def ConvolutionModule(
    hidden_size: int, dropout_rate: float
) -> sl.SequenceLayer:
  with tf.name_scope('convolution_module'):
    return sl.Residual([
        sl.LayerNormalization(),
        sl.Dense(2 * hidden_size),
        sl.GatedLinearUnit(),
        sl.DepthwiseConv1D(kernel_size=32),
        sl.BatchNormalization(),
        sl.Swish(),
        sl.Dense(hidden_size),
        sl.Dropout(dropout_rate),
    ])


def ConformerBlock(
    hidden_size: int,
    dropout_rate: float,
    max_horizon: int,
    name: Optional[str] = None,
) -> sl.SequenceLayer:
  with tf.name_scope(name or 'conformer_block'):
    # Conformer L uses 8 heads. S and M use 4.
    num_heads = 8
    return sl.Serial([
        FeedForwardModule(hidden_size, dropout_rate),
        MultiHeadedSelfAttention(
            hidden_size, num_heads, max_horizon, dropout_rate
        ),
        ConvolutionModule(hidden_size, dropout_rate),
        FeedForwardModule(hidden_size, dropout_rate),
        sl.LayerNormalization(),
    ])


def ConvolutionSubsampling(hidden_size: int) -> sl.SequenceLayer:
  with tf.name_scope('convolutional_subsampling'):
    return sl.Serial([
        # "Convolutional subsampling". Reduce rate by 4x.
        sl.Conv1D(
            filters=hidden_size,
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
        ),
        sl.Conv1D(
            filters=hidden_size,
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
        ),
    ])


def ConformerEncoder(
    hidden_size: int,
    num_blocks: int,
    max_horizon: int,
    dropout_rate: float = 0.1,
) -> sl.SequenceLayer:
  with tf.name_scope('conformer_encoder'):
    return sl.Serial(
        [
            # TODO(rryan): Implement SpecAugment.
            ConvolutionSubsampling(hidden_size),
            sl.Dense(hidden_size),
            sl.AddTimingSignal(),
            sl.Dropout(dropout_rate),
        ]
        + [
            ConformerBlock(hidden_size, dropout_rate, max_horizon, f'block{i}')
            for i in range(num_blocks)
        ]
    )
