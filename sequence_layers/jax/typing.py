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


class _MetaArrayT(type):
  types = ()

  def __instancecheck__(cls, obj):
    return isinstance(obj, cls.types)


class JaxArrayT(metaclass=_MetaArrayT):
  types = (jax.Array, jax.ShapeDtypeStruct)


class ArrayT(metaclass=_MetaArrayT):
  types = (JaxArrayT, np.ndarray)


Scalar = Shaped[ArrayT, ''] | Shaped[np.generic, ''] | Shaped[jnp.generic, '']
ScalarInt = Int[ArrayT, ''] | Int[np.generic, ''] | Int[jnp.generic, '']
ScalarFloat = Float[ArrayT, ''] | Float[np.generic, ''] | Float[jnp.generic, '']

AnyArray = Shaped[ArrayT, '...']

AnyPyTree = PyTree[AnyArray]
_F = TypeVar('_F', bound=Callable)

# Configure jaxtyping for cleaner error messages.
jaxtyping_config.update('jaxtyping_remove_typechecker_stack', True)

# TODO(b/417753029): OSS jaxtyping and typeguard are incompatible.
runtime_type_checking_enabled = False


def typed(function: _F) -> _F:
  """A decorator that enables runtime type checking."""
  if runtime_type_checking_enabled:
    return jaxtyped(function, typechecker=typeguard.typechecked)
  return function
