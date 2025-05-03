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
"""Gemma 3n audio encoder."""

import dataclasses
import enum

import jax
import jax.numpy as jnp
import sequence_layers.jax as sl


@dataclasses.dataclass(frozen=True)
class AttentionEinsumFactoryConfig:
  qkv_einsum: sl.EinsumFactoryT | None = None
  position_embedding: sl.EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class ConvEinsumFactoryConfig:
  linear: sl.EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class FFWEinsumFactoryConfig:
  linear: sl.EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class EinsumFactoryConfig:
  """Configuration for einsum factory replacement."""

  # Attention einsum factory config.
  attention: AttentionEinsumFactoryConfig = AttentionEinsumFactoryConfig()
  # Conv einsum factory config.
  conv: ConvEinsumFactoryConfig = ConvEinsumFactoryConfig()
  # MLP einsum factory config.
  mlp: FFWEinsumFactoryConfig = FFWEinsumFactoryConfig()


@enum.unique
class QuantMode(enum.Enum):
  """Quantization modes."""

  # No quantization will happen in the layer.
  NONE = enum.auto()
  # QAT will happen in the layer if an injection factory is supplied.
  TRAIN = enum.auto()
  # Parameters in the layer will be quantized in a namespace inside
  # the injected layer.
  CONVERT = enum.auto()
  # The layer will not create a parameter and instead use the quantized
  # parameters (and scales) created in the injected namespace.
  SERVE = enum.auto()


def prepare_einsum_factory(
    einsum_factory: sl.EinsumFactoryT | None,
) -> sl.EinsumFactoryT | None:
  # Note, we ignore the `w_wrapper` from cl/549464508.
  # It is unclear if it is still necessary.
  if einsum_factory is None:
    return None
  quant_w_init = jnp.zeros
  return lambda: einsum_factory(
      rhs_quant_mode=QuantMode.TRAIN,
      rhs_init=quant_w_init,
  )


@dataclasses.dataclass(frozen=True)
class Gemma3nAudioEncoderConfig(sl.SequenceLayerConfig):
  """Configuration for the Gemma 3n audio encoder.

  A Conformer / USM-based audio encoder implemented with SequenceLayers.
  """

  # input_latency is the SequenceLayer.output_latency of the layers preceding
  # this network. It's used to insert appropriate delays in the strided
  # convolutions so that stepwise execution matches layerwise execution.
  input_latency: int = 0
  # Default to regular dtype promotion rules based on inputs and parameters.
  compute_dtype: sl.DType | None = None
  # By default store all model weights as float32.
  param_dtype: sl.DType = jnp.float32

  # Only available to be changed for testing.
  num_layers: int = 12
  model_dims: int = 1536
  ffn_residual_weight: float = 0.5
  atten_num_heads: int = 8
  atten_left_context: int = 13
  name: str | None = None

  # Optional quantization
  einsum_factories: EinsumFactoryConfig = EinsumFactoryConfig()

  def make(self) -> sl.SequenceLayer:
    """Builds the Gemma 3n audio encoder."""
    conv_stride = 2
    return sl.Serial.Config(
        [
            sl.Serial.Config(
                [
                    sl.ExpandDims.Config(-1),
                    sl.Delay.Config(
                        -self.input_latency % conv_stride,
                        delay_layer_output=False,
                    ),
                    sl.Conv2D.Config(
                        filters=128,
                        kernel_size=3,
                        strides=conv_stride,
                        time_padding='reverse_causal',
                        spatial_padding='same',
                        use_bias=False,
                        compute_dtype=self.compute_dtype,
                        param_dtype=self.param_dtype,
                        name='subsampling_0',
                    ),
                    sl.GroupNormalization.Config(
                        num_groups=1,
                        epsilon=1e-3,
                        cumulative=True,
                        use_bias=False,
                        name='norm_0',
                    ),
                    sl.Relu.Config(),
                    # Introduce a delay in step mode so that the accumulated
                    # latency is divisible by the stride.
                    sl.Delay.Config(1, delay_layer_output=False),
                    sl.Conv2D.Config(
                        filters=32,
                        kernel_size=3,
                        strides=conv_stride,
                        time_padding='reverse_causal',
                        spatial_padding='same',
                        use_bias=False,
                        compute_dtype=self.compute_dtype,
                        param_dtype=self.param_dtype,
                        name='subsampling_1',
                    ),
                    sl.GroupNormalization.Config(
                        num_groups=1,
                        epsilon=1e-3,
                        cumulative=True,
                        use_bias=False,
                        name='norm_1',
                    ),
                    sl.Relu.Config(),
                    sl.DenseShaped.Config(
                        [self.model_dims],
                        use_bias=False,
                        compute_dtype=None,
                        param_dtype=self.param_dtype,
                        name='input_proj',
                    ),
                ],
                name='feature',
            ),
            sl.Repeat.Config(
                sl.Serial.Config(
                    [
                        sl.Residual.Config(
                            [
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='pre_layer_norm',
                                ),
                                sl.Dense.Config(
                                    self.model_dims * 4,
                                    use_bias=False,
                                    activation=jax.nn.swish,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.mlp.linear,
                                    ),
                                    name='ffn_layer1',
                                ),
                                sl.Dense.Config(
                                    self.model_dims,
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.mlp.linear,
                                    ),
                                    name='ffn_layer2',
                                ),
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='post_layer_norm',
                                ),
                                sl.Scale.Config(self.ffn_residual_weight),
                            ],
                            name='fflayer_start',
                        ),
                        # Attention with residual connection.
                        sl.Residual.Config(
                            [
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='pre_norm',
                                ),
                                sl.LocalDotProductSelfAttention.Config(
                                    input_projection=sl.CombinedQueryKeyValueProjection(
                                        einsum_factory=prepare_einsum_factory(
                                            self.einsum_factories.attention.qkv_einsum,
                                        ),
                                    ),
                                    num_heads=self.atten_num_heads,
                                    units_per_head=self.model_dims
                                    // self.atten_num_heads,
                                    use_bias=False,
                                    block_size=12,
                                    max_past_horizon=self.atten_left_context
                                    - 1,
                                    max_future_horizon=0,
                                    attention_logits_soft_cap=50.0,
                                    relative_position_embedding=sl.TransformerXLRelativePositionEmbedding.Config(
                                        num_heads=self.atten_num_heads,
                                        units_per_head=self.model_dims
                                        // self.atten_num_heads,
                                        max_backward=self.atten_left_context
                                        - 1,
                                        max_forward=0,
                                        position_bias_dim=self.model_dims,
                                        use_bias=False,
                                        param_dtype=self.param_dtype,
                                        einsum_factory=prepare_einsum_factory(
                                            self.einsum_factories.attention.position_embedding,
                                        ),
                                    ),
                                    per_dim_scale=True,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    name='self_atten',
                                ),
                                sl.DenseShaped.Config(
                                    [self.model_dims],
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    name='post',
                                ),
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='post_norm',
                                ),
                            ],
                            name='trans_atten',
                        ),
                        sl.Residual.Config(
                            [
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='ln',
                                ),
                                sl.Dense.Config(
                                    2 * self.model_dims,
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.conv.linear,
                                    ),
                                    name='linear_start',
                                ),
                                sl.GatedLinearUnit.Config(name='glu'),
                                sl.DepthwiseConv1D.Config(
                                    kernel_size=5,
                                    padding='causal',
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    name='depthwise_conv1d',
                                ),
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='conv_norm',
                                ),
                                sl.Swish.Config(name='conv_activation'),
                                sl.Dense.Config(
                                    self.model_dims,
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.conv.linear,
                                    ),
                                    name='linear_end',
                                ),
                            ],
                            name='lconv',
                        ),
                        sl.Residual.Config(
                            [
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='pre_layer_norm',
                                ),
                                sl.Dense.Config(
                                    self.model_dims * 4,
                                    use_bias=False,
                                    activation=jax.nn.swish,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.mlp.linear,
                                    ),
                                    name='ffn_layer1',
                                ),
                                sl.Dense.Config(
                                    self.model_dims,
                                    use_bias=False,
                                    compute_dtype=self.compute_dtype,
                                    param_dtype=self.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.einsum_factories.mlp.linear,
                                    ),
                                    name='ffn_layer2',
                                ),
                                sl.RMSNormalization.Config(
                                    param_dtype=self.param_dtype,
                                    name='post_layer_norm',
                                ),
                                sl.Scale.Config(self.ffn_residual_weight),
                            ],
                            name='fflayer_end',
                        ),
                        sl.RMSNormalization.Config(
                            param_dtype=self.param_dtype,
                            name='final_ln',
                        ),
                    ],
                    name='stacked_layers',
                ),
                num_repeats=self.num_layers,
                name='conformer',
            ),
            sl.Delay.Config(2, delay_layer_output=False),
            sl.Downsample1D.Config(rate=4, name='reducer'),
            sl.MaskInvalid.Config(),
        ],
        name=self.name,
    ).make()
