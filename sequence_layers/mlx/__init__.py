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
# Explicit imports (e.g., `from sequence_layers.mlx.simple import Relu`) DO NOT trigger this issue.
# If you need to expose specific layers at the package level, import them
# explicitly instead of using a star import.
from sequence_layers.mlx import attention
from sequence_layers.mlx import dense
from sequence_layers.mlx import dsp
from sequence_layers.mlx import projection_configs
from sequence_layers.mlx import simple
from sequence_layers.mlx import types
from sequence_layers.mlx import types as basic_types
from sequence_layers.mlx import utils
from sequence_layers.mlx.attention import DotProductAttention
from .attention import DotProductSelfAttention
from sequence_layers.mlx.attention import LocalDotProductSelfAttention
from .attention import StreamingDotProductAttention
from sequence_layers.mlx.attention import StreamingLocalDotProductAttention
from .combinators import CombinationMode
from sequence_layers.mlx.combinators import Parallel
from .combinators import Repeat
from sequence_layers.mlx.combinators import Residual
from .combinators import Serial
from sequence_layers.mlx.combinators import SerialCombinatorMixin
from .combinators import SerialModules
from sequence_layers.mlx.conditioning import Conditioning
from .convolution import Conv1D
from sequence_layers.mlx.convolution import Conv1DTranspose
from .convolution import DepthwiseConv1D
from sequence_layers.mlx.convolution2d import AveragePooling2D
from .convolution2d import Conv2D
from sequence_layers.mlx.convolution2d import Conv2DTranspose
from .convolution2d import ParallelChannels
from sequence_layers.mlx.convolution2d import Upsample2D
from .dense import Dense
from sequence_layers.mlx.dense import EinsumDense
from .dsp import Delay
from sequence_layers.mlx.dsp import FFT
from .dsp import Frame
from sequence_layers.mlx.dsp import IFFT
from .dsp import InverseSTFT
from sequence_layers.mlx.dsp import IRFFT
from .dsp import LinearToMelSpectrogram
from sequence_layers.mlx.dsp import Lookahead
from .dsp import OverlapAdd
from sequence_layers.mlx.dsp import RFFT
from .dsp import STFT
from sequence_layers.mlx.dsp import Window
from .normalization import BatchNormalization
from sequence_layers.mlx.normalization import GroupNormalization
from .normalization import L2Normalize
from sequence_layers.mlx.normalization import LayerNormalization
from .normalization import RMSNormalization
from sequence_layers.mlx.pooling import AveragePooling1D
from .pooling import MaxPooling1D
from sequence_layers.mlx.pooling import MinPooling1D
from .position import AddTimingSignal
from sequence_layers.mlx.position import ApplyRotaryPositionalEncoding
from .projection_configs import CombinedQueryKeyValueProjection
from sequence_layers.mlx.projection_configs import QueryAndKeyValueProjection
from .projection_configs import QueryAndSharedKeyValueProjection
from sequence_layers.mlx.projection_configs import SeparateQueryKeyValueProjection
from .simple import Abs
from sequence_layers.mlx.simple import Add
from .simple import Cast
from sequence_layers.mlx.simple import CheckpointName
from .simple import Downsample1D
from sequence_layers.mlx.simple import Dropout
from .simple import Elu
from sequence_layers.mlx.simple import Embedding
from .simple import Exp
from sequence_layers.mlx.simple import ExpandDims
from .simple import Flatten
from sequence_layers.mlx.simple import GatedLinearUnit
from .simple import GatedTanhUnit
from sequence_layers.mlx.simple import GatedUnit
from .simple import Gelu
from sequence_layers.mlx.simple import Identity
from .simple import Lambda
from sequence_layers.mlx.simple import LeakyRelu
from .simple import Log
from sequence_layers.mlx.simple import Logging
from .simple import MaskInvalid
from sequence_layers.mlx.simple import OneHot
from .simple import Relu
from sequence_layers.mlx.simple import Reshape
from .simple import Scale
from sequence_layers.mlx.simple import Sigmoid
from .simple import Softmax
from sequence_layers.mlx.simple import Softplus
from .simple import Squeeze
from sequence_layers.mlx.simple import Swish
from .simple import Tanh
from sequence_layers.mlx.simple import Transpose
from .simple import Upsample1D
from sequence_layers.mlx.types import ChannelSpec
from .types import check_layer
from sequence_layers.mlx.types import check_step
from .types import Constants
from sequence_layers.mlx.types import DType
from .types import Emits
from sequence_layers.mlx.types import Emitting
from .types import MaskedSequence
from sequence_layers.mlx.types import MaskT
from .types import PreservesShape
from sequence_layers.mlx.types import PreservesType
from .types import Sequence
from sequence_layers.mlx.types import SequenceLayer
from .types import SequenceLayerConfig
from sequence_layers.mlx.types import Shape
from .types import ShapeDType
from sequence_layers.mlx.types import ShapeLike
from .types import State
from sequence_layers.mlx.types import Stateless
from .types import StatelessPointwise

__all__ = [
    'basic_types',
    'dense',
    'simple',
    'types',
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
