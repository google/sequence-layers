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
"""Wrappers for making jaxtyping easier to use and understand."""

import jax
import jax.numpy as jnp
from jaxtyping import AbstractDtype, Bool, config as jaxtyping_config, Float, Int, PyTree, Shaped, jaxtyped, TypeCheckError
import numpy as np
import typeguard
from typing import Callable, TypeVar, Union

ArrayT = Union[jax.Array, jax.ShapeDtypeStruct, np.ndarray]
Scalar = Shaped[ArrayT, ''] | Shaped[np.generic, ''] | Shaped[jnp.generic, '']
ScalarInt = Int[ArrayT, ''] | Int[np.generic, ''] | Int[jnp.generic, '']
ScalarFloat = Float[ArrayT, ''] | Float[np.generic, ''] | Float[jnp.generic, '']

AnyArray = Shaped[ArrayT, '...']

AnyPyTree = PyTree[AnyArray]
_F = TypeVar('_F', bound=Callable)

# Configure jaxtyping for cleaner error messages.
jaxtyping_config.update('jaxtyping_remove_typechecker_stack', True)


def typed(function: _F) -> _F:
  # Ensures that type annotations are enforced at runtime.
  return jaxtyped(function, typechecker=typeguard.typechecked)
