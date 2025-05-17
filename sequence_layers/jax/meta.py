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
"""Parameter metadata."""

import enum
import functools
from typing import Any, Callable, Optional, Tuple, Union

import flax

unbox = flax.core.meta.unbox
MESH_AXIS = flax.core.meta.PARTITION_NAME
AXIS_TYPE = 'axis_type'
IS_EMBEDDING = 'is_embedding'
IS_NORMALIZER = 'is_normalizer'


@enum.unique
class AxisType(enum.Enum):
  FANIN = 'FANIN'
  CHANNEL = 'CHANNEL'
  STACKED = 'STACKED'

Partitioned = flax.core.meta.Partitioned


def with_meta(
    fn: Callable[..., Any],
    names: Optional[Tuple[Union[Tuple[str, ...], str, None], ...]] = None,
    **kwargs,
) -> Callable[..., flax.core.meta.Partitioned]:
  """Wraps the function to return a Partitioned value with extra metadata."""

  # Extra arguments are currently ignored.
  del kwargs

  names = names if names is not None else ()
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    value = fn(*args, **kwargs)
    return flax.core.meta.Partitioned(value, names)

  return wrapper
