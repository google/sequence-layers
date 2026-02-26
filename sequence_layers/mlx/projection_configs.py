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
"""MLX-native attention projection configuration dataclasses.

These are pure-Python dataclasses that mirror the JAX-side projection configs
from sequence_layers.jax.attention.common, but without JAX-specific fields
(sharding, einsum_factory, quantization_provider). They retain initializer
fields as Callable | None so that downstream code can still configure kernel
and bias initialization.
"""

import dataclasses
from typing import Callable


@dataclasses.dataclass(frozen=True)
class QueryKeyValueProjectionConfig:
  """Base class for QKV projection configuration."""
  pass


@dataclasses.dataclass(frozen=True)
class CombinedQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use a single projection matrix for query/key/value projection.

  * Incompatible with Grouped Query Attention (num_query_heads != num_kv_heads).
  * Supports shared key and value projection.
  """

  # Kernel initializer for the combined query/key/value projection.
  # The variable shape is [input_dimension, 3, num_heads, units_per_head].
  # If share_kv_projection is True, the variable shape is [input_dimension, 2,
  # num_heads, units_per_head].
  qkv_kernel_init: Callable | None = None

  # Bias initializer for the combined query/key/value projection.
  # The variable shape is [3, num_heads, units_per_head].
  bias_init: Callable | None = None

  # If true, share the key and value projection matrices.
  share_kv_projection: bool = False


@dataclasses.dataclass(frozen=True)
class SeparateQueryKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate projection matrices for query/key/value projection.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """

  # Kernel initializers for the separate query/key/value projections.
  # The variable shape is [input_dimension, num_heads or num_kv_heads,
  # units_per_head].
  q_kernel_init: Callable | None = None
  k_kernel_init: Callable | None = None
  v_kernel_init: Callable | None = None

  # Bias initializer for the separate query/key/value projections.
  # The variable shape is [num_heads or num_kv_heads, units_per_head].
  bias_init: Callable | None = None


@dataclasses.dataclass(frozen=True)
class QueryAndKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Does not support shared key and value projection. Use
    QueryAndSharedKeyValueProjection.
  """

  # Kernel initializer for the query projection.
  # The variable shape is [input_dimension, num_heads, units_per_head].
  q_kernel_init: Callable | None = None

  # Bias initializer for the query projection.
  # The variable shape is [num_heads, units_per_head].
  q_bias_init: Callable | None = None

  # Kernel initializer for the key/value projection.
  # The variable shape is [input_dimension, 2, num_kv_heads, units_per_head].
  kv_kernel_init: Callable | None = None

  # Bias initializer for the key/value projection.
  # The variable shape is [2, num_kv_heads, units_per_head].
  kv_bias_init: Callable | None = None


@dataclasses.dataclass(frozen=True)
class QueryAndSharedKeyValueProjection(QueryKeyValueProjectionConfig):
  """Use separate query and shared key/value projection matrices.

  * Supports Grouped Query Attention (num_query_heads != num_kv_heads).
  * Requires shared key and value projection.
  """

  # Kernel initializer for the query projection.
  # The variable shape is [input_dimension, num_heads, units_per_head].
  q_kernel_init: Callable | None = None

  # Bias initializer for the query projection.
  # The variable shape is [num_heads, units_per_head].
  q_bias_init: Callable | None = None

  # Kernel initializer for the shared key/value projection.
  # The variable shape is [input_dimension, num_kv_heads, units_per_head].
  kv_kernel_init: Callable | None = None

  # Bias initializer for the shared key/value projection.
  # The variable shape is [num_kv_heads, units_per_head].
  kv_bias_init: Callable | None = None
