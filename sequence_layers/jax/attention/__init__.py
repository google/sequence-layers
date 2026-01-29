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
"""Attention layers."""

# pylint: disable=wildcard-import
from sequence_layers.jax.attention.blockwise_dot_product_self_attention import *
from sequence_layers.jax.attention.common import *
from sequence_layers.jax.attention.dot_product_attention import *
from sequence_layers.jax.attention.dot_product_self_attention import *
from sequence_layers.jax.attention.dot_product_self_attention_v2 import *
from sequence_layers.jax.attention.gmm_attention import *
from sequence_layers.jax.attention.local_dot_product_self_attention import *
from sequence_layers.jax.attention.multi_source_dot_product_attention import *
from sequence_layers.jax.attention.shaw_relative_position_embedding import *
from sequence_layers.jax.attention.streaming_dot_product_attention import *
from sequence_layers.jax.attention.streaming_local_dot_product_attention import *
from sequence_layers.jax.attention.t5_relative_position_embedding import *
from sequence_layers.jax.attention.transformer_xl_relative_position_embedding import *


# pylint: disable=undefined-all-variable
__all__ = (
    # go/keep-sorted start
    'BlockwiseDotProductSelfAttention',
    'CombinedQueryKeyValueProjection',
    'CrossAttentionEmits',
    'DotProductAttention',
    'DotProductSelfAttention',
    'DotProductSelfAttentionV2',
    'GmmAttention',
    'LocalDotProductSelfAttention',
    'MultiSourceDotProductAttention',
    'QueryAndKeyValueProjection',
    'QueryAndSharedKeyValueProjection',
    'RelativePositionEmbedding',
    'SelfAttentionEmits',
    'SeparateQueryKeyValueProjection',
    'ShawRelativePositionEmbedding',
    'StreamingDotProductAttention',
    'StreamingLocalDotProductAttention',
    'T5RelativePositionEmbedding',
    'TransformerXLRelativePositionEmbedding',
    # go/keep-sorted end
)
