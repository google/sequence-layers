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

"""Tacotron using SequenceLayers.

Based on:
Tacotron: Towards End-to-End Speech Synthesis
https://arxiv.org/abs/1703.10135
"""

from sequence_layers import proto as slp
from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf


def TacotronDecoder(
    source_name: str, reduction_factor: int = 2, target_dimension: int = 128
):
  return sl.Serial([
      # Squeeze to [b, t // rf, rf, target_dimension].
      sl.Squeeze(reduction_factor),
      # Flatten to [b, t // rf, rf * target_dimension]
      sl.Flatten(),
      # Pre-net: Apply an information bottleneck to the previous feature frames.
      sl.Dense(256, activation=tf.nn.relu),
      sl.Dropout(0.5),
      sl.Dense(128, activation=tf.nn.relu),
      sl.Dropout(0.5),
      # Autoregress on the attention output (cell_output, context_vector).
      sl.Autoregressive([
          sl.RNN(tf.keras.layers.LSTMCell(256)),
          # Concatenate cell output with context vector.
          sl.Skip([
              # Attend to the source sequence with
              # query = RNN([previous_context_vector, previous_output]).
              sl.GmmAttention(
                  source_name,
                  num_heads=1,
                  units_per_head=128,
                  num_components=5,
                  monotonic=True,
                  init_offset_bias=1.0,
                  init_scale_bias=1.0,
              ),
              # Flatten the [b, t//2, num_heads, units_per_head] to
              # [b, t//2, num_heads * units_per_head].
              sl.Flatten(),
          ]),
      ]),
      # Transform the [attention RNN output, context vector] with 2 residual RNN
      # layers.
      sl.Residual(sl.RNN(tf.keras.layers.LSTMCell(256))),
      sl.Residual(sl.RNN(tf.keras.layers.LSTMCell(256))),
      # Predict reduction_factor output frames from the resulting hidden state.
      sl.Dense(target_dimension * reduction_factor),
  ])


def TacotronDecoderAsProto(
    source_name: str, reduction_factor: int = 2, target_dimension: int = 128
) -> slp.SequenceLayer:
  """Returns a SequenceLayer protobuf representing a Tacotron decoder."""

  def LSTM(num_units):
    return slp.RNN(cell=[slp.RNN.Cell(lstm=slp.RNN.LSTMCell(units=num_units))])

  return slp.Serial(
      layer=[
          # Squeeze to [b, t // rf, rf, target_dimension].
          slp.Squeeze(factor=reduction_factor),
          # Flatten to [b, t // rf, rf * target_dimension]
          slp.Flatten(),
          # Pre-net: Apply an information bottleneck to the previous feature
          # frames.
          slp.Dense(units=256, activation=slp.Relu()),
          slp.Dropout(rate=0.5),
          slp.Dense(units=128, activation=slp.Relu()),
          slp.Dropout(rate=0.5),
          # Autoregress on the attention output (cell_output, context_vector).
          slp.Autoregressive(
              layer=[
                  LSTM(256),
                  # Concatenate cell output with context vector.
                  slp.Skip(
                      layer=[
                          # Attend to the source sequence with
                          # query = RNN([previous_context_vector,
                          # previous_output]).
                          slp.GmmAttention(
                              source_name=source_name,
                              num_heads=1,
                              units_per_head=128,
                              num_components=5,
                              monotonic=True,
                              init_offset_bias=1.0,
                              init_scale_bias=1.0,
                          ),
                          # Flatten the [b, t//2, num_heads, units_per_head] to
                          # [b, t//2, num_heads * units_per_head].
                          slp.Flatten(),
                      ]
                  ),
              ]
          ),
          # Transform the [attention RNN output, context vector] with 2 residual
          # RNN layers.
          slp.Residual(layer=[LSTM(256)]),
          slp.Residual(layer=[LSTM(256)]),
          # Predict reduction_factor output frames from the resulting hidden
          # state.
          slp.Dense(units=target_dimension * reduction_factor),
      ]
  )
