"""Abstract tests for test utilities."""

from unittest import mock
from absl.testing import parameterized

class NamedProductTest(parameterized.TestCase):

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
    from sequence_layers.abstract import test_utils
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
    from sequence_layers.abstract import test_utils
    with self.assertRaisesRegex(
        ValueError, str(iterator_without_testcase_name)
    ):
      test_utils.named_product(first, second)
