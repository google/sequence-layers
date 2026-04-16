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
