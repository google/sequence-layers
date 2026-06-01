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
"""Abstract tests for backend utilities."""

# pylint: disable=abstract-method

from typing import override

from sequence_layers import specs
from sequence_layers.specs import backend as backend_spec
from sequence_layers.specs import test_utils as test_utils_spec


class ModuleSpecTest(test_utils_spec.ModuleSpecTest):

  @override
  def module_spec_pairs(self, backend_sl: specs.ModuleSpec):
    import importlib  # pylint: disable=g-import-not-at-top
    backend = importlib.import_module(backend_sl.__name__ + '.backend')
    return {backend: backend_spec.ModuleSpec}
