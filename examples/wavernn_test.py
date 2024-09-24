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

from sequence_layers.examples import wavernn
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class WaveRNNTest(test_util.SequenceLayerTest):

  def test_wavernn_body(self):
    batch_size, conditioning_length, conditioning_dim = 2, 4, 5
    upsample_ratio = 3
    target_length = conditioning_length * upsample_ratio
    conditioning_name = 'conditioning'

    # The waveform inputs.
    x = self.random_sequence(batch_size, target_length, 1)

    l = wavernn.WaveRNNBody(
        num_units=1,
        num_distribution_parameters=7,
        conditioning_name=conditioning_name,
        upsample_ratio=upsample_ratio,
    )

    # The conditioning sequence.
    conditioning = self.random_sequence(
        batch_size, conditioning_length, conditioning_dim
    )
    constants = {conditioning_name: conditioning}

    # Only test training=False because the model has training noise.
    self.verify_contract(
        l,
        x,
        training=False,
        constants=constants,
        pad_constants=True,
        pad_constants_ratio=upsample_ratio,
    )
    # Flex required for EnsureShape.
    self.verify_tflite_step(l, x, constants=constants, use_flex=True)

  def test_wavernn(self):
    batch_size, conditioning_length, conditioning_dim = 2, 3, 5
    upsample_ratio = 5
    target_length = conditioning_length * upsample_ratio

    # The waveform inputs.
    waveform = self.random_sequence(batch_size, target_length, 1)

    # The conditioning sequence.
    conditioning = self.random_sequence(
        batch_size, conditioning_length, conditioning_dim
    )

    model = wavernn.WaveRNN(num_units=1, upsample_ratio=5)
    log_prob = model.log_prob(waveform, conditioning, training=True)
    self.assertEqual(
        log_prob.values.shape.as_list(), [batch_size, target_length, 1]
    )
    sample = model.sample(conditioning)
    self.assertEqual(
        sample.values.shape.as_list(), [batch_size, target_length, 1]
    )

    self.evaluate(tf1.global_variables_initializer())
    self.evaluate([log_prob, sample])


if __name__ == '__main__':
  tf.test.main()
