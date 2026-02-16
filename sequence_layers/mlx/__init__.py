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

# Re-export basic types.
from sequence_layers.mlx.types import Constants
from sequence_layers.mlx.types import DType
from sequence_layers.mlx.types import Emits
from sequence_layers.mlx.types import mask_invalid
from sequence_layers.mlx.types import MASK_DTYPE
from sequence_layers.mlx.types import MaskedSequence
from sequence_layers.mlx.types import PaddingMode
from sequence_layers.mlx.types import ReceptiveField
from sequence_layers.mlx.types import Sequence
from sequence_layers.mlx.types import sequence_mask
from sequence_layers.mlx.types import Shape
from sequence_layers.mlx.types import ShapeDType
from sequence_layers.mlx.types import ShapeLike
from sequence_layers.mlx.types import State

# Re-export basic_types TypeVars used in type annotations.
from sequence_layers.mlx.types import MaskT

# Re-export attention projection configs (from JAX, used to configure attention).
from sequence_layers.jax.attention.common import CombinedQueryKeyValueProjection
from sequence_layers.jax.attention.common import QueryAndKeyValueProjection
from sequence_layers.jax.attention.common import SeparateQueryKeyValueProjection

# Re-export MLX layer hierarchy.
from sequence_layers.mlx.types import ChannelSpec
from sequence_layers.mlx.types import check_layer
from sequence_layers.mlx.types import check_step
from sequence_layers.mlx.types import Emitting
from sequence_layers.mlx.types import PreservesShape
from sequence_layers.mlx.types import PreservesType
from sequence_layers.mlx.types import SequenceLayer
from sequence_layers.mlx.types import Stateless
from sequence_layers.mlx.types import StatelessEmitting
from sequence_layers.mlx.types import StatelessPointwise
from sequence_layers.mlx.types import StatelessPointwiseFunctor
from sequence_layers.mlx.types import Steppable

# Re-export simple layers.
from sequence_layers.mlx.simple import Add
from sequence_layers.mlx.simple import Cast
from sequence_layers.mlx.simple import CheckpointName
from sequence_layers.mlx.simple import Downsample1D
from sequence_layers.mlx.simple import Dropout
from sequence_layers.mlx.simple import Elu
from sequence_layers.mlx.simple import Embedding
from sequence_layers.mlx.simple import ExpandDims
from sequence_layers.mlx.simple import Flatten
from sequence_layers.mlx.simple import GatedLinearUnit
from sequence_layers.mlx.simple import GatedTanhUnit
from sequence_layers.mlx.simple import GatedUnit
from sequence_layers.mlx.simple import Gelu
from sequence_layers.mlx.simple import Identity
from sequence_layers.mlx.simple import Lambda
from sequence_layers.mlx.simple import LeakyRelu
from sequence_layers.mlx.simple import Logging
from sequence_layers.mlx.simple import MaskInvalid
from sequence_layers.mlx.simple import OneHot
from sequence_layers.mlx.simple import Relu
from sequence_layers.mlx.simple import Reshape
from sequence_layers.mlx.simple import Scale
from sequence_layers.mlx.simple import Sigmoid
from sequence_layers.mlx.simple import Softmax
from sequence_layers.mlx.simple import Softplus
from sequence_layers.mlx.simple import Squeeze
from sequence_layers.mlx.simple import Swish
from sequence_layers.mlx.simple import Tanh
from sequence_layers.mlx.simple import Transpose
from sequence_layers.mlx.simple import Upsample1D

# Re-export dense / normalization / position.
from sequence_layers.mlx.dense import Dense
from sequence_layers.mlx.dense import DenseDeferred
from sequence_layers.mlx.dense import EinsumDense
from sequence_layers.mlx.normalization import BatchNormalization
from sequence_layers.mlx.normalization import GroupNormalization
from sequence_layers.mlx.normalization import L2Normalize
from sequence_layers.mlx.normalization import LayerNormalization
from sequence_layers.mlx.normalization import RMSNormalization
from sequence_layers.mlx.position import ApplyRotaryPositionalEncoding

# Re-export attention layers.
from sequence_layers.mlx.attention import DotProductAttention
from sequence_layers.mlx.attention import DotProductSelfAttention
from sequence_layers.mlx.attention import DeferredDotProductAttention
from sequence_layers.mlx.attention import DeferredDotProductSelfAttention
from sequence_layers.mlx.attention import DeferredLocalDotProductSelfAttention
from sequence_layers.mlx.attention import DeferredStreamingDotProductAttention
from sequence_layers.mlx.attention import LocalDotProductSelfAttention
from sequence_layers.mlx.attention import StreamingDotProductAttention

# Re-export pooling layers.
from sequence_layers.mlx.pooling import AveragePooling1D
from sequence_layers.mlx.pooling import MaxPooling1D
from sequence_layers.mlx.pooling import MinPooling1D

# Re-export convolution layers.
from sequence_layers.mlx.convolution import Conv1D
from sequence_layers.mlx.convolution import Conv1DTranspose
from sequence_layers.mlx.convolution import DeferredConv1D
from sequence_layers.mlx.convolution import DeferredConv1DTranspose
from sequence_layers.mlx.convolution import DeferredDepthwiseConv1D
from sequence_layers.mlx.convolution import DepthwiseConv1D

# Re-export DSP layers.
from sequence_layers.mlx.dsp import Delay
from sequence_layers.mlx.dsp import FFT
from sequence_layers.mlx.dsp import Frame
from sequence_layers.mlx.dsp import IFFT
from sequence_layers.mlx.dsp import InverseSTFT
from sequence_layers.mlx.dsp import IRFFT
from sequence_layers.mlx.dsp import LinearToMelSpectrogram
from sequence_layers.mlx.dsp import Lookahead
from sequence_layers.mlx.dsp import OverlapAdd
from sequence_layers.mlx.dsp import RFFT
from sequence_layers.mlx.dsp import STFT
from sequence_layers.mlx.dsp import Window

# Re-export combinators.
from sequence_layers.mlx.combinators import CombinationMode
from sequence_layers.mlx.combinators import Parallel
from sequence_layers.mlx.combinators import Repeat
from sequence_layers.mlx.combinators import Residual
from sequence_layers.mlx.combinators import Serial

# Re-export conditioning.
from sequence_layers.mlx.conditioning import Conditioning

# Re-export export and weight conversion utilities.
from sequence_layers.mlx import export
from sequence_layers.mlx import weight_converter

# ---------------------------------------------------------------------------
# Backend factory registration
# ---------------------------------------------------------------------------
from sequence_layers.jax.types import SequenceLayerConfig as _SLC


def _register_backends():
  """Register MLX factories for all supported Linen Config classes."""
  from sequence_layers.jax import conditioning as jax_cond
  from sequence_layers.jax import simple as jax_simple
  from sequence_layers.jax import dense as jax_dense
  from sequence_layers.jax import normalization as jax_norm
  from sequence_layers.jax import position as jax_pos
  from sequence_layers.jax.attention import (
      dot_product_self_attention as jax_self_attn,
  )
  from sequence_layers.jax.attention import (
      dot_product_attention as jax_cross_attn,
  )
  from sequence_layers.jax.attention import (
      streaming_dot_product_attention as jax_streaming_attn,
  )
  from sequence_layers.jax.attention import (
      streaming_local_dot_product_attention as jax_streaming_local_attn,
  )
  from sequence_layers.jax.attention import (
      local_dot_product_self_attention as jax_local_attn,
  )
  from sequence_layers.jax import convolution as jax_conv
  from sequence_layers.jax import pooling as jax_pool
  from sequence_layers.jax import dsp as jax_dsp
  from sequence_layers.jax import combinators as jax_comb

  from sequence_layers.mlx import conditioning as mlx_cond
  from sequence_layers.mlx import simple as mlx_simple
  from sequence_layers.mlx import dense as mlx_dense
  from sequence_layers.mlx import normalization as mlx_norm
  from sequence_layers.mlx import position as mlx_pos
  from sequence_layers.mlx import attention as mlx_attn
  from sequence_layers.mlx import convolution as mlx_conv
  from sequence_layers.mlx import pooling as mlx_pool
  from sequence_layers.mlx import dsp as mlx_dsp
  from sequence_layers.mlx import combinators as mlx_comb

  reg = _SLC.register_backend_factory

  # Simple layers — activations.
  reg('mlx', jax_simple.Identity.Config, mlx_simple.Identity.from_config)
  reg('mlx', jax_simple.Relu.Config, mlx_simple.Relu.from_config)
  reg('mlx', jax_simple.Gelu.Config, mlx_simple.Gelu.from_config)
  reg('mlx', jax_simple.Swish.Config, mlx_simple.Swish.from_config)
  reg('mlx', jax_simple.Tanh.Config, mlx_simple.Tanh.from_config)
  reg('mlx', jax_simple.Sigmoid.Config, mlx_simple.Sigmoid.from_config)
  reg('mlx', jax_simple.LeakyRelu.Config, mlx_simple.LeakyRelu.from_config)
  reg('mlx', jax_simple.Elu.Config, mlx_simple.Elu.from_config)
  reg('mlx', jax_simple.Softmax.Config, mlx_simple.Softmax.from_config)
  reg('mlx', jax_simple.Softplus.Config, mlx_simple.Softplus.from_config)

  # Simple layers — value manipulation.
  reg('mlx', jax_simple.Cast.Config, mlx_simple.Cast.from_config)
  reg('mlx', jax_simple.Scale.Config, mlx_simple.Scale.from_config)
  reg('mlx', jax_simple.Add.Config, mlx_simple.Add.from_config)

  # Simple layers — masking.
  reg('mlx', jax_simple.MaskInvalid.Config, mlx_simple.MaskInvalid.from_config)

  # Simple layers — gated units.
  reg('mlx', jax_simple.GatedUnit.Config, mlx_simple.GatedUnit.from_config)
  reg(
      'mlx',
      jax_simple.GatedLinearUnit.Config,
      mlx_simple.GatedLinearUnit.from_config,
  )
  reg(
      'mlx',
      jax_simple.GatedTanhUnit.Config,
      mlx_simple.GatedTanhUnit.from_config,
  )

  # Simple layers — shape manipulation.
  reg('mlx', jax_simple.Flatten.Config, mlx_simple.Flatten.from_config)
  reg('mlx', jax_simple.Reshape.Config, mlx_simple.Reshape.from_config)
  reg('mlx', jax_simple.ExpandDims.Config, mlx_simple.ExpandDims.from_config)
  reg('mlx', jax_simple.Squeeze.Config, mlx_simple.Squeeze.from_config)
  reg('mlx', jax_simple.Transpose.Config, mlx_simple.Transpose.from_config)

  # Simple layers — encoding.
  reg('mlx', jax_simple.OneHot.Config, mlx_simple.OneHot.from_config)
  reg('mlx', jax_simple.Embedding.Config, mlx_simple.Embedding.from_config)

  # Simple layers — regularization.
  reg('mlx', jax_simple.Dropout.Config, mlx_simple.Dropout.from_config)

  # Simple layers — sampling.
  reg(
      'mlx', jax_simple.Downsample1D.Config, mlx_simple.Downsample1D.from_config
  )
  reg('mlx', jax_simple.Upsample1D.Config, mlx_simple.Upsample1D.from_config)

  # Simple layers — misc.
  reg(
      'mlx',
      jax_simple.CheckpointName.Config,
      mlx_simple.CheckpointName.from_config,
  )
  reg('mlx', jax_simple.Lambda.Config, mlx_simple.Lambda.from_config)
  reg('mlx', jax_simple.Logging.Config, mlx_simple.Logging.from_config)

  # Dense.
  reg('mlx', jax_dense.Dense.Config, mlx_dense.DenseDeferred.from_config)
  reg('mlx', jax_dense.EinsumDense.Config, mlx_dense.EinsumDense.from_config)

  # Conditioning.
  reg(
      'mlx',
      jax_cond.Conditioning.Config,
      mlx_cond.Conditioning.from_config,
  )

  # Normalization.
  reg(
      'mlx',
      jax_norm.L2Normalize.Config,
      mlx_norm.L2Normalize.from_config,
  )
  reg(
      'mlx',
      jax_norm.RMSNormalization.Config,
      mlx_norm.RMSNormalization.from_config,
  )
  reg(
      'mlx',
      jax_norm.LayerNormalization.Config,
      mlx_norm.LayerNormalization.from_config,
  )
  reg(
      'mlx',
      jax_norm.GroupNormalization.Config,
      mlx_norm.GroupNormalization.from_config,
  )
  reg(
      'mlx',
      jax_norm.BatchNormalization.Config,
      mlx_norm.BatchNormalization.from_config,
  )

  # Position.
  reg(
      'mlx',
      jax_pos.ApplyRotaryPositionalEncoding.Config,
      mlx_pos.ApplyRotaryPositionalEncoding.from_config,
  )

  # Attention.
  reg(
      'mlx',
      jax_self_attn.DotProductSelfAttention.Config,
      mlx_attn.DotProductSelfAttention.from_config,
  )
  reg(
      'mlx',
      jax_cross_attn.DotProductAttention.Config,
      mlx_attn.DotProductAttention.from_config,
  )
  reg(
      'mlx',
      jax_streaming_attn.StreamingDotProductAttention.Config,
      mlx_attn.StreamingDotProductAttention.from_config,
  )
  reg(
      'mlx',
      jax_streaming_local_attn.StreamingLocalDotProductAttention.Config,
      mlx_attn.StreamingDotProductAttention.from_config,
  )
  reg(
      'mlx',
      jax_local_attn.LocalDotProductSelfAttention.Config,
      mlx_attn.LocalDotProductSelfAttention.from_config,
  )

  # Convolution.
  reg('mlx', jax_conv.Conv1D.Config, mlx_conv.Conv1D.from_config)
  reg(
      'mlx',
      jax_conv.DepthwiseConv1D.Config,
      mlx_conv.DepthwiseConv1D.from_config,
  )
  reg(
      'mlx',
      jax_conv.Conv1DTranspose.Config,
      mlx_conv.Conv1DTranspose.from_config,
  )

  # Pooling.
  reg('mlx', jax_pool.MaxPooling1D.Config, mlx_pool.MaxPooling1D.from_config)
  reg('mlx', jax_pool.MinPooling1D.Config, mlx_pool.MinPooling1D.from_config)
  reg(
      'mlx',
      jax_pool.AveragePooling1D.Config,
      mlx_pool.AveragePooling1D.from_config,
  )

  # DSP.
  reg('mlx', jax_dsp.Delay.Config, mlx_dsp.Delay.from_config)
  reg('mlx', jax_dsp.Lookahead.Config, mlx_dsp.Lookahead.from_config)
  reg('mlx', jax_dsp.Window.Config, mlx_dsp.Window.from_config)
  reg('mlx', jax_dsp.Frame.Config, mlx_dsp.Frame.from_config)
  reg('mlx', jax_dsp.OverlapAdd.Config, mlx_dsp.OverlapAdd.from_config)
  reg('mlx', jax_dsp.FFT.Config, mlx_dsp.FFT.from_config)
  reg('mlx', jax_dsp.IFFT.Config, mlx_dsp.IFFT.from_config)
  reg('mlx', jax_dsp.RFFT.Config, mlx_dsp.RFFT.from_config)
  reg('mlx', jax_dsp.IRFFT.Config, mlx_dsp.IRFFT.from_config)
  reg('mlx', jax_dsp.STFT.Config, mlx_dsp.STFT.from_config)
  reg('mlx', jax_dsp.InverseSTFT.Config, mlx_dsp.InverseSTFT.from_config)
  reg(
      'mlx',
      jax_dsp.LinearToMelSpectrogram.Config,
      mlx_dsp.LinearToMelSpectrogram.from_config,
  )

  # Combinators.
  reg('mlx', jax_comb.Serial.Config, mlx_comb.Serial.from_config)
  reg('mlx', jax_comb.Residual.Config, mlx_comb.Residual.from_config)
  reg('mlx', jax_comb.Repeat.Config, mlx_comb.Repeat.from_config)
  reg('mlx', jax_comb.Parallel.Config, mlx_comb.Parallel.from_config)


_register_backends()
