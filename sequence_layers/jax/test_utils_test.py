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
"""Tests for the test utilities."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import sequence_layers.jax as sl
from sequence_layers.jax import test_utils
from sequence_layers.specs import test_utils_behaviors as spec


class ModuleSpecTest(test_utils.SequenceLayerTest, spec.ModuleSpecTest):
  pass


class VerifyContractTest(test_utils.SequenceLayerTest, spec.VerifyContractTest):

  def get_dummy_layer(self, mismatch: bool):
    l = super().get_dummy_layer(mismatch)
    key = jax.random.PRNGKey(1234)
    x = test_utils.random_sequence(2, 5, 10)
    l = self.init_and_bind_layer(key, l, x)
    return l

  def test_verify_contract_with_jax_flags(self):
    """Tests that disabling optional JAX features (gradients, batching) doesn't crash.
    
    Default paths (with these flags as True) are tested in all other tests.
    """
    layer = self.get_dummy_layer(mismatch=False)
    x = sl.Sequence(
        jnp.ones((2, 5, 10)),
        jnp.ones((2, 5), dtype=bool),
    )
    self.verify_contract(
        layer, x, training=False, test_gradients=False, test_batching=False
    )


class StandardDtypeConfigsTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (
          {},
          {
              'p-fp32_i-fp32_c-None',  # default
              'p-bf16_i-bf16_c-bf16',  # praxis
              'p-fp32_i-bf16_c-bf16',
              'p-bf16_i-fp32_c-None',
              'p-fp32_i-fp32_c-bf16',
          },
      ),
      (
          {'param': True, 'compute': True},
          {
              'p-fp32_c-None',  # default
              'p-bf16_c-bf16',  # praxis
              'p-fp32_c-bf16',
              'p-bf16_c-None',
              # 'p-fp32_c-bf16',  # duplicate
          },
      ),
      (
          {'praxis_only': True},
          {
              'p-fp32_i-fp32_c-None',  # default
              'p-bf16_i-bf16_c-bf16',  # praxis
          },
      ),
  )
  def test_standard_dtype_configs_returns_names(self, kwargs, expected):
    names = {
        c['testcase_name']
        for c in test_utils.standard_dtype_configs(named=True, **kwargs)
    }
    self.assertEqual(expected, names)


class NamedProductTest(test_utils.SequenceLayerTest, spec.NamedProductTest):
  pass


class ZipLongestTest(test_utils.SequenceLayerTest, spec.ZipLongestTest):
  pass


class Shear2dTest(test_utils.SequenceLayerTest):

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic_3x3',
          'input_array': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          'expected_output': [
              [0, 0, 1, 2, 3],
              [0, 4, 5, 6, 0],
              [7, 8, 9, 0, 0],
          ],
      },
      {
          'testcase_name': 'rect_more_rows',
          'input_array': [[1, 2], [3, 4], [5, 6]],
          'expected_output': [[0, 0, 1, 2], [0, 3, 4, 0], [5, 6, 0, 0]],
      },
      {
          'testcase_name': 'rect_more_cols',
          'input_array': [[1, 2, 3, 4], [5, 6, 7, 8]],
          'expected_output': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]],
      },
      {
          'testcase_name': 'single_row',
          'input_array': [[1, 2, 3]],
          'expected_output': [[1, 2, 3]],
      },
      {
          'testcase_name': 'single_col',
          'input_array': [[1], [2], [3]],
          'expected_output': [[0, 0, 1], [0, 2, 0], [3, 0, 0]],
      },
      {
          'testcase_name': 'with_zeros',
          'input_array': [[0, 1], [0, 0]],
          'expected_output': [[0, 0, 1], [0, 0, 0]],
      },
  )
  def test_shear_2d(self, input_array, expected_output):
    output = test_utils._shear_2d(jnp.array(input_array))  # pylint: disable=protected-access
    self.assertAllEqual(output, jnp.array(expected_output))


if __name__ == '__main__':
  test_utils.main()
