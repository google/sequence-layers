# Copyright 2023 Google LLC
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
"""Streaming Sequence Layers."""

# TODO(dthkao): Remove the module imports when users are migrated to new deps.
# Alias the module names as well, so that the redirect that exists for legacy
# module imports functions correctly.
from sequence_layers.tensorflow import attention
from sequence_layers.tensorflow import combinators
from sequence_layers.tensorflow import conditioning
from sequence_layers.tensorflow import convolution
from sequence_layers.tensorflow import dense
from sequence_layers.tensorflow import dsp
from sequence_layers.tensorflow import normalization
from sequence_layers.tensorflow import pooling
from sequence_layers.tensorflow import position
from sequence_layers.tensorflow import recurrent
from sequence_layers.tensorflow import simple
from sequence_layers.tensorflow import squeeze
from sequence_layers.tensorflow import time_varying
from sequence_layers.tensorflow import types

# pylint: disable=wildcard-import
# TODO(rryan): Don't wildcard import, or define __all__ explicitly.
from sequence_layers.tensorflow.attention import *
from sequence_layers.tensorflow.combinators import *
from sequence_layers.tensorflow.conditioning import *
from sequence_layers.tensorflow.convolution import *
from sequence_layers.tensorflow.dense import *
from sequence_layers.tensorflow.dsp import *
from sequence_layers.tensorflow.normalization import *
from sequence_layers.tensorflow.pooling import *
from sequence_layers.tensorflow.position import *
from sequence_layers.tensorflow.recurrent import *
from sequence_layers.tensorflow.simple import *
from sequence_layers.tensorflow.squeeze import *
from sequence_layers.tensorflow.time_varying import *
from sequence_layers.tensorflow.types import *
# pylint: enable=wildcard-import
