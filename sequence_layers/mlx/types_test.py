# Copyright 2026 Google LLC
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

from absl.testing import absltest
import mlx.core as mx
import numpy as np
from sequence_layers.abstract import types_test_base
from sequence_layers.mlx import types


class SequenceTest(types_test_base.SequenceTest):

  def get_backend(self):
    return mx

  @property
  def Sequence(self):
    return types.Sequence

  @property
  def MaskedSequence(self):
    return types.MaskedSequence

  def assertAllEqual(self, a, b):
    a = np.array(a) if isinstance(a, mx.array) else a
    b = np.array(b) if isinstance(b, mx.array) else b
    np.testing.assert_array_equal(a, b)

  def assertSequencesEqual(self, a, b):
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.mask, b.mask)


class SteppableTest(types_test_base.SteppableTest):

  def create_steppable(self):

    class DefaultSteppable(types.Steppable):

      def layer(self, x, *, constants=None):
        return x

      def step(self, x, state, *, constants=None):
        return x, state

      def get_initial_state(self, batch_size, input_spec, *, constants=None):
        return ()

      def get_output_shape(self, input_shape, *, constants=None):
        return input_shape

      def get_output_dtype(self, input_dtype, *, constants=None):
        return input_dtype

    return DefaultSteppable()


class SequenceLayerConfigTest(types_test_base.SequenceLayerConfigTest):

  def get_config_base_cls(self):
    return types.SequenceLayerConfig


if __name__ == '__main__':
  absltest.main()
