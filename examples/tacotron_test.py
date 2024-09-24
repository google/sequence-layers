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
from sequence_layers import proto
from sequence_layers.examples import tacotron
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class TacotronTest(test_util.SequenceLayerTest, parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  def test_tacotron(self, rf):
    batch_size, source_length, source_dim = 2, 3, 5
    target_length, target_dimension = 7, 11
    source_name = 'source'
    # The target sequence to learn.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    l = tacotron.TacotronDecoder(
        source_name, reduction_factor=rf, target_dimension=target_dimension
    )

    # The encoder sequence to attend to.
    source = self.random_sequence(batch_size, source_length, source_dim)
    constants = {source_name: source}

    # Only test training=False because the model has dropout.
    self.verify_contract(l, x, training=False, constants=constants)
    # Use flex for Einsum.
    self.verify_tflite_step(
        l, x, constants=constants, use_flex=True, rtol=5e-6, atol=5e-6
    )

  def test_tacotron_from_proto(self):
    batch_size, source_length, source_dim = 2, 3, 5
    target_length, target_dimension = 7, 11
    source_name = 'source'
    # The target sequence to learn.
    x = self.random_sequence(batch_size, target_length, target_dimension)
    spec = tacotron.TacotronDecoderAsProto(
        source_name, reduction_factor=2, target_dimension=target_dimension
    )

    l = proto.build_sequence_layer(spec)

    # The encoder sequence to attend to.
    source = self.random_sequence(batch_size, source_length, source_dim)
    constants = {source_name: source}

    # Only test training=False because the model has dropout.
    self.verify_contract(l, x, training=False, constants=constants)
    # Use flex for Einsum.
    self.verify_tflite_step(
        l, x, constants=constants, use_flex=True, rtol=5e-6, atol=5e-6
    )


if __name__ == '__main__':
  tf.test.main()
