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
from sequence_layers.tensorflow.examples import mobilenet
import tensorflow.compat.v2 as tf


class MobileNetTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_mobilenet_2d_v1(self):
    batch_size, height, width, channels = 2, 224, 224, 3
    x = self.random_sequence(batch_size, height, width, channels)
    l = mobilenet.MobileNet2D(num_classes=1000)
    # The model requires blocks of [b, 32, 224, 3] frames at a time.
    self.assertEqual(l.block_size, 32)
    # The model downsamples its input in time by a factor of 32x.
    self.assertEqual(1 / l.output_ratio, 32)
    # The output per timestep is a [b, 1, num_classes] tensor of logits for
    # each of the 1234 classes.
    self.assertEqual(l.get_output_shape_for_sequence(x), tf.TensorShape([1000]))
    self.verify_contract(l, x, training=False)

    # Ensure the model can run step-by-step on tf.lite.
    self.verify_tflite_step(l, x)

    num_parameters = sum(var.shape.num_elements() for var in l.variables)
    self.assertEqual(num_parameters, 4253768)


if __name__ == '__main__':
  tf.test.main()
