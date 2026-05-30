"""Abstract tests for backend utilities."""

# pylint: disable=abstract-method

from typing import override

import numpy as np

from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import test_utils as test_utils_spec


class ModuleSpecTest(test_utils_spec.ModuleSpecTest):

  @override
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec):
    return {backend_sl.backend: backend_spec.ModuleSpec}


class BackendNNTest(test_utils_spec.SequenceLayerTest):
  """Test behavior of backend.nn operations."""

  def test_relu(self):
    x = self.xp.array(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))
    y = self.nn.relu(x)
    expected = self.xp.array(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    self.assertAllEqual(y, expected)

  def test_sigmoid(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.sigmoid(x)
    expected = self.xp.array(np.array([[0.5]], dtype=np.float32))
    self.assertAllEqual(y, expected)

  def test_tanh(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.tanh(x)
    expected = self.xp.array(np.array([[0.0]], dtype=np.float32))
    self.assertAllEqual(y, expected)

  def test_elu(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.elu(x)
    expected = self.xp.array(np.array([[0.0]], dtype=np.float32))
    self.assertAllEqual(y, expected)

  def test_softplus(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.softplus(x)
    expected = self.xp.array(np.array([[np.log(2.0)]], dtype=np.float32))

    # Wrap in Sequence to satisfy assertSequencesClose in JAX
    y_seq = self.sl.types.Sequence.from_values(y)
    expected_seq = self.sl.types.Sequence.from_values(expected)

    if hasattr(self, 'assertSequencesClose'):
      self.assertSequencesClose(y_seq, expected_seq)
    else:
      self.assertAllEqual(y, expected)

  def test_swish(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.swish(x)
    expected = self.xp.array(np.array([[0.0]], dtype=np.float32))
    self.assertAllEqual(y, expected)

  def test_gelu(self):
    x = self.xp.array(np.array([[0.0]], dtype=np.float32))
    y = self.nn.gelu(x)
    expected = self.xp.array(np.array([[0.0]], dtype=np.float32))
    self.assertAllEqual(y, expected)


class BackendXPTest(test_utils_spec.SequenceLayerTest):
  """Test behavior of backend.xp operations."""

  def test_mean_simple(self):
    x = self.xp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    y = self.xp.mean(x)
    self.assertAllEqual(y, self.xp.array(2.0))

  def test_mean_with_axis(self):
    x = self.xp.array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = self.xp.mean(x, axis=0)
    self.assertAllEqual(y, self.xp.array([2.0, 3.0]))

  def test_mean_with_keepdims(self):
    x = self.xp.array(np.array([[1.0, 2.0]], dtype=np.float32))
    y = self.xp.mean(x, axis=1, keepdims=True)
    self.assertAllEqual(y, self.xp.array([[1.5]]))

  def test_mean_with_where(self):
    x = self.xp.array(np.array([[1.0, 2.0, 10.0]], dtype=np.float32))
    where = self.xp.array(np.array([[True, True, False]], dtype=bool))
    y = self.xp.mean(x, where=where)
    self.assertAllEqual(y, self.xp.array(1.5))

  def test_var_simple(self):
    x = self.xp.array(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    y = self.xp.var(x)
    expected_var = np.var(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    self.assertAllEqual(y, self.xp.array(expected_var))

  def test_var_with_where(self):
    x = self.xp.array(np.array([[1.0, 3.0, 10.0]], dtype=np.float32))
    where = self.xp.array(np.array([[True, True, False]], dtype=bool))
    y = self.xp.var(x, where=where)
    expected_var = np.var(np.array([1.0, 3.0], dtype=np.float32))
    self.assertAllEqual(y, self.xp.array(expected_var))
