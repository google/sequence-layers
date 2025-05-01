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

from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow.examples import conformer
import tensorflow.compat.v2 as tf


class ConformerTest(test_util.SequenceLayerTest):

  def test_encoder(self):
    batch_size, input_length, input_channels = 2, 5, 7
    x = self.random_sequence(batch_size, input_length, input_channels)

    encoder = conformer.ConformerEncoder(
        hidden_size=8, num_blocks=2, max_horizon=4
    )

    self.verify_contract(encoder, x, pad_nan=False, training=False)
    self.verify_tflite_step(encoder, x, use_flex=True, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  tf.test.main()
