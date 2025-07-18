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

from unittest import mock
from absl.testing import parameterized
import numpy as np
from sequence_layers.jax import test_utils


class StandardDtypeConfigsTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      (
          dict(),
          {
              'p-fp32_i-fp32_c-None',  # default
              'p-bf16_i-bf16_c-bf16',  # praxis
              'p-fp32_i-bf16_c-bf16',
              'p-bf16_i-fp32_c-None',
              'p-fp32_i-fp32_c-bf16',
          },
      ),
      (
          dict(param=True, compute=True),
          {
              'p-fp32_c-None',  # default
              'p-bf16_c-bf16',  # praxis
              'p-fp32_c-bf16',
              'p-bf16_c-None',
              # 'p-fp32_c-bf16',  # duplicate
          },
      ),
      (
          dict(praxis_only=True),
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


class NamedProductTest(test_utils.SequenceLayerTest):

  @parameterized.parameters(
      dict(
          first=[('a', 'alpha'), ('b', 'beta')],
          second=[('1', 1), ('2', 2), ('3', 3)],
          expected=[
              ('a_1', 'alpha', 1),
              ('a_2', 'alpha', 2),
              ('a_3', 'alpha', 3),
              ('b_1', 'beta', 1),
              ('b_2', 'beta', 2),
              ('b_3', 'beta', 3),
          ],
      ),
      dict(
          first=[{'a': 'alpha', 'testcase_name': 'test'}],
          second=[('1', 1), ('2', 2)],
          expected=[
              ('test_1', 'alpha', 1),
              ('test_2', 'alpha', 2),
          ],
      ),
      dict(
          first=[
              {'letter': 'a', 'testcase_name': 'alpha'},
              {'testcase_name': 'beta', 'letter': 'b'},
          ],
          second=[
              {'testcase_name': 'one', 'number': 1},
              {'number': 2, 'testcase_name': 'two'},
          ],
          expected=[
              {'letter': 'a', 'number': 1, 'testcase_name': 'alpha_one'},
              {'letter': 'a', 'number': 2, 'testcase_name': 'alpha_two'},
              {'letter': 'b', 'number': 1, 'testcase_name': 'beta_one'},
              {'letter': 'b', 'number': 2, 'testcase_name': 'beta_two'},
          ],
      ),
  )
  @mock.patch.object(parameterized, 'named_parameters', autospec=True)
  def test_builds_named_products(self, mock_fn, first, second, expected):
    test_utils.named_product(first, second)
    self.assertSequenceEqual(mock_fn.call_args.args, expected)

  @parameterized.parameters(
      dict(
          first=[{'testcase_name': 'alpha', 'letter': 'a'}, {'letter': 'b'}],
          second=[('1', 1), ('2', 2), ('3', 3)],
          iterator_without_testcase_name=1,
      ),
      dict(
          first=[{'testcase_name': 'alpha', 'letter': 'a'}],
          second=[('1', 1), ()],
          iterator_without_testcase_name=2,
      ),
  )
  def test_raises_on_missing_testcase_names(
      self, first, second, iterator_without_testcase_name
  ):
    with self.assertRaisesRegex(
        ValueError, str(iterator_without_testcase_name)
    ):
      test_utils.named_product(first, second)


class Shear2dTest(test_utils.SequenceLayerTest):

  @parameterized.named_parameters(
      dict(
          testcase_name='basic_3x3',
          input_array=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          expected_output=[
              [0, 0, 1, 2, 3],
              [0, 4, 5, 6, 0],
              [7, 8, 9, 0, 0],
          ],
      ),
      dict(
          testcase_name='rect_more_rows',
          input_array=[[1, 2], [3, 4], [5, 6]],
          expected_output=[[0, 0, 1, 2], [0, 3, 4, 0], [5, 6, 0, 0]],
      ),
      dict(
          testcase_name='rect_more_cols',
          input_array=[[1, 2, 3, 4], [5, 6, 7, 8]],
          expected_output=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 0]],
      ),
      dict(
          testcase_name='single_row',
          input_array=[[1, 2, 3]],
          expected_output=[[1, 2, 3]],
      ),
      dict(
          testcase_name='single_col',
          input_array=[[1], [2], [3]],
          expected_output=[[0, 0, 1], [0, 2, 0], [3, 0, 0]],
      ),
      dict(
          testcase_name='with_zeros',
          input_array=[[0, 1], [0, 0]],
          expected_output=[[0, 0, 1], [0, 0, 0]],
      ),
  )
  def test_shear_2d(self, input_array, expected_output):
    output = test_utils._shear_2d(np.array(input_array))
    self.assertAllEqual(output, np.array(expected_output))


if __name__ == '__main__':
  test_utils.main()
