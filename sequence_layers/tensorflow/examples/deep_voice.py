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

"""Deep Voice 3 using SequenceLayers.

Based on:
"Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning"
https://arxiv.org/abs/1710.07654
"""

import math

from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf


def DeepVoice3Decoder(
    source_name: str, reduction_factor: int, target_dimension: int
):
  """Builds the Deep Voice 3 decoder."""

  def ConvBlock():
    return sl.Serial([
        sl.Residual([
            sl.Dropout(0.05),
            sl.Conv1D(filters=512, kernel_size=5, strides=1),
            sl.GatedLinearUnit(),
        ]),
        sl.Scale(math.sqrt(0.5)),
    ])

  def DecoderLayer():
    return sl.Serial([
        ConvBlock(),
        sl.Residual([
            sl.DotProductAttention(
                source_name, num_heads=1, units_per_head=128
            ),
            sl.Flatten(),
        ]),
        sl.Scale(math.sqrt(0.5)),
    ])

  return sl.Serial([
      # Squeeze to [b, t // rf, rf, target_dimension].
      sl.Squeeze(reduction_factor),
      # Flatten to [b, t // rf, rf * target_dimension]
      sl.Flatten(),
      # N_prenet=2 pre-net layers.
      sl.Dropout(0.05),
      sl.Dense(128, activation=tf.nn.relu),
      sl.Dropout(0.05),
      sl.Dense(256, activation=tf.nn.relu),
      # N_decoder=4 decoder layers.
      DecoderLayer(),
      DecoderLayer(),
      DecoderLayer(),
      DecoderLayer(),
      # Predict reduction_factor output frames from the resulting hidden state.
      sl.Dense(target_dimension * reduction_factor),
  ])
