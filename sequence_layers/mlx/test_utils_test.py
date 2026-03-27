"""Tests for the test utilities."""

import typing
from unittest import mock
from absl.testing import parameterized
import numpy as np
from sequence_layers.abstract import test_utils_test_base
from sequence_layers.mlx import test_utils

class NamedProductTest(
    test_utils_test_base.NamedProductTest, test_utils.SequenceLayerTest
):
  pass

if __name__ == '__main__':
  test_utils.main()
