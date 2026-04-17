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
"""Sequence layers in MLX."""

from . import backend
from . import simple
from . import types
# CRITICAL: Do NOT use wildcard imports (e.g., `from .simple import *`) here.
# Pyrefly (our static analysis tool) has a known limitation with cross-module
# resolution of diamond inheritance chains. When wildcard imports are used to
# re-export classes from `simple.py` (which combine `types` and `spec` bases),
# Pyrefly fails to resolve the concrete method implementations in `mlx/types.py`
# and flags all instances as abstract (`bad-instantiation` false positives).
#
# Explicit imports (e.g., `from .simple import Relu`) DO NOT trigger this issue.
# If you need to expose specific layers at the package level, import them
# explicitly instead of using a star import.
from .simple import Abs
from .simple import Add
from .simple import Cast
from .simple import CheckpointName
from .simple import Downsample1D
from .simple import Dropout
from .simple import Elu
from .simple import Embedding
from .simple import Exp
from .simple import ExpandDims
from .simple import Flatten
from .simple import GatedLinearUnit
from .simple import GatedTanhUnit
from .simple import GatedUnit
from .simple import Gelu
from .simple import Identity
from .simple import Lambda
from .simple import LeakyRelu
from .simple import Log
from .simple import Logging
from .simple import MaskInvalid
from .simple import OneHot
from .simple import Relu
from .simple import Reshape
from .simple import Scale
from .simple import Sigmoid
from .simple import Softmax
from .simple import Softplus
from .simple import Squeeze
from .simple import Swish
from .simple import Tanh
from .simple import Transpose
from .simple import Upsample1D
from .types import MaskedSequence
from .types import Sequence
from .types import SequenceLayer
from .types import SequenceLayerConfig

__all__ = [
    'backend',
    'types',
    'simple',
    'Sequence',
    'MaskedSequence',
    'SequenceLayer',
    'SequenceLayerConfig',
    'Identity',
    'Relu',
    'Gelu',
    'Abs',
    'Exp',
    'Log',
    'Swish',
    'Tanh',
    'Sigmoid',
    'LeakyRelu',
    'Elu',
    'Softmax',
    'Softplus',
    'Cast',
    'Scale',
    'Add',
    'MaskInvalid',
    'GatedUnit',
    'GatedLinearUnit',
    'GatedTanhUnit',
    'Flatten',
    'Reshape',
    'ExpandDims',
    'Squeeze',
    'Transpose',
    'OneHot',
    'Embedding',
    'Dropout',
    'Downsample1D',
    'Upsample1D',
    'CheckpointName',
    'Lambda',
    'Logging',
]
