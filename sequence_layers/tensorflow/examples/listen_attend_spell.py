# Copyright 2023 Google LLC
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
"""Listen Attend and Spell-style seq2seq speech recognizer using SequenceLayers.

Roughly follows the "medium" archictecture in:
K. Irie, R. Prabhavalkar, A. Kannan, A. Bruguier, D. Rybach, and P. Nguyen,
“On the choice of modeling unit for sequence-to-sequence speech recognition”
Interspeech, 2019.
https://arxiv.org/abs/1902.01955

which was itself based on
Y. Zhang, W. Chan, and N. Jaitly
“Very deep convolutional networks for end-to-end speech recognition”
ICASSP, 2017.
https://arxiv.org/abs/1610.03022
"""

from sequence_layers import tensorflow as sl
import tensorflow.compat.v1 as tf


def StreamingEncoder() -> sl.SequenceLayer:
  """Defines of a streaming (unidirectional) speech encoder."""

  def StridedConv2D():
    return sl.Serial([
        sl.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            spatial_padding='valid',
            use_bias=False,
        ),
        sl.BatchNormalization(),
        sl.Relu(),
    ])

  def LSTM():
    return sl.Serial([
        sl.RNN(tf.keras.layers.LSTMCell(512)),
        # Skip projection layer since the RNN is unidirectional.
        sl.BatchNormalization(),
        sl.Relu(),
    ])

  return sl.Serial([
      # Strided convolutions downsample the input sequence.
      StridedConv2D(),
      StridedConv2D(),
      # Flatten the frequency and channels dimensions.
      sl.Flatten(),
      LSTM(),
      LSTM(),
      LSTM(),
      LSTM(),
  ])


def DecoderBody(
    source_name: str, num_output_tokens: int = 128
) -> sl.SequenceLayer:
  """Single decoder step, takes a one-hot of the previous token as input."""
  embedding_dim = 64
  lstm_dim = 512
  return sl.Serial([
      # Hacky embedding lookup, assumes the inputs are one hot.
      sl.Dense(units=embedding_dim, use_bias=False),
      # Use Autoregressive to pass the context vector from the previous step
      # as input to the first decoder RNN layer.
      sl.Autoregressive(
          [
              sl.RNN(tf.keras.layers.LSTMCell(lstm_dim)),
              # Concatenate cell output with the new context vector to feed into
              # the next RNN layer following attention.
              sl.Skip([
                  # Attend to the source sequence with
                  # query = RNN([previous_context_vector, previous_output]).
                  sl.AdditiveAttention(
                      source_name, num_heads=1, units_per_head=128
                  ),
                  # Flatten the num_heads dimension.
                  sl.Flatten(),
              ]),
          ],
          # Only feed the context vector back to the next step.
          feedback_layer=sl.Slice()[lstm_dim:],
      ),
      sl.RNN(tf.keras.layers.LSTMCell(lstm_dim)),
      sl.Dense(num_output_tokens),
      sl.Softmax(),
  ])
