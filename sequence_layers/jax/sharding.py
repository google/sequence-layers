# Copyright 2025 Google LLC
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
"""Sharding utilities."""

from typing import Sequence

import jax

UNCONSTRAINED = jax.sharding.PartitionSpec.UNCONSTRAINED
DimSharding = str | Sequence[str] | None | type(UNCONSTRAINED)
Sharding = Sequence[DimSharding] | None

use_mesh = jax.sharding.use_mesh


def shard(x: jax.Array, s: Sharding) -> jax.Array:
  """Annotates x with the provided sharding.

  Args:
    x: The array to annotate.
    s: The sharding to apply.

  Returns:
    The array with sharding constraints applied.
  """
  abstract_mesh = jax.sharding.get_abstract_mesh()

  if s is None or abstract_mesh is None:
    return x

  x = jax.lax.with_sharding_constraint(
      x,
      jax.sharding.NamedSharding(
          abstract_mesh, jax.sharding.PartitionSpec(*s)
      ),
  )
  return x
