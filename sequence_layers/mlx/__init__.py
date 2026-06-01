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
from . import attention
from . import backend
from . import dense
from . import dsp
from . import projection_configs
from . import simple
from . import test_utils
from . import types
from . import types as basic_types
from . import utils
from .attention import DotProductAttention
from .attention import DotProductSelfAttention
from .attention import LocalDotProductSelfAttention
from .attention import StreamingDotProductAttention
from .attention import StreamingLocalDotProductAttention
from .combinators import CombinationMode
from .combinators import Parallel
from .combinators import Repeat
from .combinators import Residual
from .combinators import Serial
from .combinators import SerialCombinatorMixin
from .combinators import SerialModules
from .conditioning import Conditioning
from .convolution import Conv1D
from .convolution import Conv1DTranspose
from .convolution import DepthwiseConv1D
from .convolution2d import AveragePooling2D
from .convolution2d import Conv2D
from .convolution2d import Conv2DTranspose
from .convolution2d import ParallelChannels
from .convolution2d import Upsample2D
from .dense import Dense
from .dense import EinsumDense
from .dsp import Delay
from .dsp import FFT
from .dsp import Frame
from .dsp import IFFT
from .dsp import InverseSTFT
from .dsp import IRFFT
from .dsp import LinearToMelSpectrogram
from .dsp import Lookahead
from .dsp import OverlapAdd
from .dsp import RFFT
from .dsp import STFT
from .dsp import Window
from .normalization import BatchNormalization
from .normalization import GroupNormalization
from .normalization import L2Normalize
from .normalization import LayerNormalization
from .normalization import RMSNormalization
from .pooling import AveragePooling1D
from .pooling import MaxPooling1D
from .pooling import MinPooling1D
from .position import AddTimingSignal
from .position import ApplyRotaryPositionalEncoding
from .projection_configs import CombinedQueryKeyValueProjection
from .projection_configs import QueryAndKeyValueProjection
from .projection_configs import QueryAndSharedKeyValueProjection
from .projection_configs import SeparateQueryKeyValueProjection
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
from .test_utils import SequenceLayerTest
from .types import ChannelSpec
from .types import check_layer
from .types import check_step
from .types import Constants
from .types import DType
from .types import Emits
from .types import Emitting
from .types import MaskedSequence
from .types import MaskT
from .types import PreservesShape
from .types import PreservesType
from .types import Sequence
from .types import SequenceLayer
from .types import SequenceLayerConfig
from .types import Shape
from .types import ShapeDType
from .types import ShapeLike
from .types import State
from .types import Stateless
from .types import StatelessPointwise

__all__ = [
    'basic_types',
    'dense',
    'backend',
    'simple',
    'types',
    'test_utils',
    'SequenceLayerTest',
    'Constants',
    'Sequence',
    'MaskedSequence',
    'SequenceLayer',
    'SequenceLayerConfig',
    'check_layer',
    'check_step',
    'Stateless',
    'StatelessPointwise',
    'PreservesShape',
    'PreservesType',
    'MaskT',
    'Shape',
    'ShapeDType',
    'ShapeLike',
    'DType',
    'State',
    'Emits',
    'Emitting',
    'ChannelSpec',
    'Conditioning',
    'Dense',
    'EinsumDense',
    'Conv1D',
    'DepthwiseConv1D',
    'Conv1DTranspose',
    'Conv2D',
    'Conv2DTranspose',
    'AveragePooling2D',
    'Upsample2D',
    'ParallelChannels',
    'MaxPooling1D',
    'MinPooling1D',
    'AveragePooling1D',
    'Serial',
    'SerialModules',
    'SerialCombinatorMixin',
    'Residual',
    'Repeat',
    'Parallel',
    'CombinationMode',
    'DotProductSelfAttention',
    'DotProductAttention',
    'StreamingDotProductAttention',
    'StreamingLocalDotProductAttention',
    'LocalDotProductSelfAttention',
    'CombinedQueryKeyValueProjection',
    'QueryAndKeyValueProjection',
    'QueryAndSharedKeyValueProjection',
    'SeparateQueryKeyValueProjection',
    'AddTimingSignal',
    'ApplyRotaryPositionalEncoding',
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
    'L2Normalize',
    'RMSNormalization',
    'LayerNormalization',
    'BatchNormalization',
    'GroupNormalization',
    'dsp',
    'Delay',
    'FFT',
    'Frame',
    'IFFT',
    'IRFFT',
    'InverseSTFT',
    'LinearToMelSpectrogram',
    'Lookahead',
    'OverlapAdd',
    'RFFT',
    'STFT',
    'Window',
]
