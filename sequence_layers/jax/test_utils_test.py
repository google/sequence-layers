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
from sequence_layers.abstract import test_utils_test_base
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


class NamedProductTest(
    test_utils_test_base.NamedProductTest, test_utils.SequenceLayerTest
):
  pass


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
