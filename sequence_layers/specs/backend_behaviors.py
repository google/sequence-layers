"""Abstract tests for backend utilities."""

# pylint: disable=abstract-method

from typing import override

from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import test_utils as test_utils_spec


class ModuleSpecTest(test_utils_spec.ModuleSpecTest):

  @override
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec):
    return {backend_sl.backend: backend_spec.ModuleSpec}
