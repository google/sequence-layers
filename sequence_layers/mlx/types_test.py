import mlx.core as mx
import numpy as np
from sequence_layers.mlx import types
from absl.testing import parameterized
from absl.testing import absltest

class TypesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('mask_value=None', 0.0),
      ('mask_value=0.0', 0.0),
      ('mask_value=-1.0', -1.0),
  )
  def test_mask_invalid(self, mask_value):
    values = mx.array([
        [1.0, 2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0, 40.0],
    ])
    mask = mx.array([[True, True, False, False], [False, False, False, True]])

    output = types.Sequence(values, mask).mask_invalid(mask_value)
    expected_values = mx.array([
            [1.0, 2.0, mask_value, mask_value],
            [mask_value, mask_value, mask_value, 40.0],
        ])
    self.assertTrue(np.allclose(output.values, expected_values))
    self.assertTrue(np.array_equal(output.mask, mask))

if __name__ == '__main__':
  absltest.main()
