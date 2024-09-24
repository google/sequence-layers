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

from sequence_layers.examples import listen_attend_spell
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class ListenAttendSpellTest(test_util.SequenceLayerTest):

  def test_encoder(self):
    batch_size, input_length, input_freqs, input_channels = 2, 5, 7, 3
    x = self.random_sequence(
        batch_size, input_length, input_freqs, input_channels
    )

    encoder = listen_attend_spell.StreamingEncoder()

    self.verify_contract(
        encoder,
        x,
        training=False,
    )
    self.verify_tflite_step(encoder, x, rtol=5e-6, atol=5e-6)

  def test_decoder(self):
    batch_size, source_length, source_dim = 2, 3, 5
    target_length, target_dim = 7, 11
    source_name = 'encoded'
    source = self.random_sequence(batch_size, source_length, source_dim)
    target = self.random_sequence(batch_size, target_length, target_dim)
    constants = {source_name: source}

    decoder = listen_attend_spell.DecoderBody(source_name, num_output_tokens=8)

    self.verify_contract(decoder, target, training=False, constants=constants)
    self.verify_tflite_step(
        decoder,
        target,
        constants=constants,
        # Use flex for Einsum.
        use_flex=True,
        rtol=5e-6,
        atol=5e-6,
    )


if __name__ == '__main__':
  tf.test.main()
