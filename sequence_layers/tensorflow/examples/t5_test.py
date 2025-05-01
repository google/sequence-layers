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

from absl.testing import parameterized
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow.examples import t5
import tensorflow.compat.v2 as tf


class T5Test(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(False, True)
  def test_t5_encoder(self, training):
    batch_size, target_length, target_dimension = 2, 4, 6
    # The target sequence to encode.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    l = t5.T5Encoder(
        num_layers=2,
        dimension=8,
        num_heads=2,
        ffn_dimension=7,
        # So that we can test with training mode enabled.
        dropout_rate=0.0,
    )

    self.verify_contract(l, x, training=training, pad_nan=False)
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, use_flex=True)

  @parameterized.parameters(False, True)
  def test_t5_decoder(self, training):
    batch_size, source_length, source_dim = 2, 3, 5
    target_length, target_dimension = 4, 6
    source_name = 'source'
    # The target sequence to decode.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    l = t5.T5Decoder(
        source_name,
        vocab_size=32,
        num_layers=2,
        dimension=8,
        num_heads=2,
        ffn_dimension=7,
        # So that we can test with training mode enabled.
        dropout_rate=0.0,
        max_past_horizon=128,
    )

    # Check the decoder can be executed step-wise for autoregressive decoding.
    self.assertTrue(l.supports_step)

    # The encoder sequence to attend to.
    source = self.random_sequence(batch_size, source_length, source_dim)
    constants = {source_name: source}

    self.verify_contract(
        l, x, training=training, pad_nan=False, constants=constants
    )
    # Use flex for Einsum.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)


if __name__ == '__main__':
  tf.test.main()
