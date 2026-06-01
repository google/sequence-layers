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
"""Backend-specific helpers (MLX)."""

from typing import override

import mlx.core as mx
from sequence_layers.specs import backend as spec
from sequence_layers.specs import types as types_spec


class BackendWrapper(spec.xp):
  """Thin wrapper around MLX to match NumPy interface for tests."""

  bool_ = mx.bool_
  int32 = mx.int32
  float32 = mx.float32

  @override
  def array(self, a, dtype=None) -> types_spec.Array:
    return mx.array(a, dtype=dtype)

  @override
  def zeros(self, shape, dtype=None) -> types_spec.Array:
    return mx.zeros(shape, dtype=dtype)

  @override
  def concatenate(self, arrays, axis=0) -> types_spec.Array:
    return mx.concatenate(list(arrays), axis=axis)


xp: spec.xp = BackendWrapper()
