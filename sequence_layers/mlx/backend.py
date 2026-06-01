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
import mlx.nn as nn_mlx
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

  @override
  def broadcast_to(self, array, shape) -> types_spec.Array:
    return mx.broadcast_to(array, shape)

  @override
  def abs(self, x) -> types_spec.Array:
    return mx.abs(x)

  @override
  def exp(self, x) -> types_spec.Array:
    return mx.exp(x)

  @override
  def log(self, x) -> types_spec.Array:
    return mx.log(x)

  @override
  def mean(
      self,
      x,
      axis=None,
      dtype=None,
      keepdims=False,
      where=None,
  ) -> types_spec.Array:
    if where is not None:
      x_masked = mx.where(where, x, 0.0)
      summed = mx.sum(x_masked, axis=axis, keepdims=keepdims)
      counts = mx.sum(where.astype(mx.int32), axis=axis, keepdims=keepdims)
      counts = mx.maximum(counts, 1)
      result = summed / counts
    else:
      result = mx.mean(x, axis=axis, keepdims=keepdims)
    if dtype is not None:
      result = result.astype(dtype)
    return result

  @override
  def var(
      self,
      x,
      axis=None,
      dtype=None,
      keepdims=False,
      where=None,
  ) -> types_spec.Array:
    if where is not None:
      mean_val = self.mean(x, axis=axis, keepdims=True, where=where)
      squared_diff = mx.square(x - mean_val)
      result = self.mean(
          squared_diff, axis=axis, keepdims=keepdims, where=where
      )
    else:
      result = mx.var(x, axis=axis, keepdims=keepdims)
    if dtype is not None:
      result = result.astype(dtype)
    return result


xp: spec.xp = BackendWrapper()


class NNWrapper(spec.nn):
  """Wrapper around MLX activations to match backend protocol."""

  @override
  def relu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.relu(x)

  @override
  def sigmoid(self, x: types_spec.Array) -> types_spec.Array:
    return mx.sigmoid(x)

  @override
  def tanh(self, x: types_spec.Array) -> types_spec.Array:
    return mx.tanh(x)

  @override
  def swish(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.silu(x)

  @override
  def gelu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.gelu(x)

  @override
  def elu(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.elu(x)

  @override
  def softplus(self, x: types_spec.Array) -> types_spec.Array:
    return nn_mlx.softplus(x)

  @override
  def softmax(self, x: types_spec.Array, axis: int = -1) -> types_spec.Array:
    return mx.softmax(x, axis=axis)


nn: spec.nn = NNWrapper()
