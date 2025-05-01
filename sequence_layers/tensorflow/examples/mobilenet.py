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

"""MobileNet using SequenceLayers.

Based on:
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications
https://arxiv.org/abs/1704.04861
"""

from sequence_layers import tensorflow as sl


def MobileNet2D(num_classes: int = 1000) -> sl.SequenceLayer:
  """Builds a 2D MobileNet V1 where the time axis is streamed.

  This is useful for streaming audio processing, where the input is a
  time-frequency representation of sound. It is not useful for video processing;
  we will need a MobileNet3D for that so that each timestep of the stream is a
  new image to process.

  Args:
    num_classes: The number of classes to model.

  Returns:
    A SequenceLayer that consumes a [b, t, s, c] tensor and produces a
    [b, t, num_classes] tensor.
  """

  def SeparableConv2D(filters, stride):
    return sl.Serial([
        sl.DepthwiseConv2D(
            kernel_size=3,
            strides=stride,
            spatial_padding='same',
            depth_multiplier=1,
            use_bias=False,
        ),
        # Normalize the channels dimension of the [b, t, s, c] tensor.
        sl.BatchNormalization(),
        sl.Relu(),
        sl.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            spatial_padding='valid',
            use_bias=False,
        ),
        # Normalize the channels dimension of the [b, t, s, c] tensor.
        sl.BatchNormalization(),
        sl.Relu(),
    ])

  return sl.Serial([
      sl.Conv2D(
          filters=32, kernel_size=3, strides=2, spatial_padding='same'
      ),  # -> [b, t//2, s//2, 32]
      SeparableConv2D(filters=64, stride=1),  # -> [b, t//2, s//2, 64]
      SeparableConv2D(filters=128, stride=2),  # -> [b, t//4, s//4, 128]
      SeparableConv2D(filters=128, stride=1),  # -> [b, t//4, s//4, 128]
      SeparableConv2D(filters=256, stride=2),  # -> [b, t//8, s//8, 256]
      SeparableConv2D(filters=256, stride=1),  # -> [b, t//8, s//8, 256]
      SeparableConv2D(filters=512, stride=2),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=512, stride=1),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=512, stride=1),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=512, stride=1),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=512, stride=1),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=512, stride=1),  # -> [b, t//16, s//16, 512]
      SeparableConv2D(filters=1024, stride=2),  # -> [b, t//32, s//32, 1024]
      SeparableConv2D(filters=1024, stride=1),  # -> [b, t//32, s//32, 1024]
      # We don't have a GlobalAveragePooling2D, so assuming the input spatial
      # dimension is 224, we can use stride 7 in the spatial dimension to reduce
      # the spatial dimension to size 1.
      sl.AveragePooling2D(
          pool_size=7, strides=(1, 7)
      ),  # -> [b, t//32, s//224, 1024]
      sl.Flatten(),  # -> [b, t//32, s//224 * 1024]
      sl.Dense(num_classes),  # -> [b, t//32, num_classes]
      sl.Softmax(),  # -> [b, t//32, num_classes]
  ])
