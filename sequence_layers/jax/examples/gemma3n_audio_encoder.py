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
import fractions

import jax
import jax.numpy as jnp
import sequence_layers.jax as sl
from sequence_layers.jax import types


EinsumFactoryT = types.EinsumFactoryT


@dataclasses.dataclass(frozen=True)
class AttentionEinsumFactoryConfig:
  qkv_einsum: EinsumFactoryT | None = None
  position_embedding: EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class ConvEinsumFactoryConfig:
  linear: EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class FFWEinsumFactoryConfig:
  linear: EinsumFactoryT | None = None


@dataclasses.dataclass(frozen=True)
class EinsumFactoryConfig:
  """Configuration for einsum factory replacement.

  Attributes:
    attention: Attention einsum factory config.
    conv: Conv einsum factory config.
    mlp: MLP einsum factory config.
  """

  attention: AttentionEinsumFactoryConfig = AttentionEinsumFactoryConfig()
  conv: ConvEinsumFactoryConfig = ConvEinsumFactoryConfig()
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
    einsum_factory: EinsumFactoryT | None,
) -> EinsumFactoryT | None:
  # Note, we ignore the `w_wrapper` from cl/549464508.
  # It is unclear if it is still necessary.
  if einsum_factory is None:
    return None
  quant_w_init = jnp.zeros
  return lambda: einsum_factory(
      rhs_quant_mode=QuantMode.TRAIN,
      rhs_init=quant_w_init,
  )


class UniformReducer(sl.PreservesType, sl.PreservesShape, sl.Stateless):
  """Select every N'th vector."""

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    reduction_factor: int
    name: str | None = None

    def make(self) -> 'UniformReducer':
      return UniformReducer(self, name=self.name)

  config: Config

  @property
  def block_size(self) -> int:
    return self.config.reduction_factor

  @property
  def output_ratio(self) -> fractions.Fraction:
    """The number of output frames for one input frame."""
    return fractions.Fraction(1, self.config.reduction_factor)

  @property
  def input_latency(self) -> int:
    return self.config.reduction_factor - 1

  @sl.check_layer
  def layer(
      self,
      x: sl.Sequence,
      *,
      training: bool,
      constants: sl.Constants | None = None,
  ) -> sl.Sequence:
    values = x.values[:, :: self.config.reduction_factor]
    mask = x.mask[:, :: self.config.reduction_factor]
    return sl.Sequence(values, mask)


class Gemma3nAudioEncoder(sl.SerialCombinatorMixin, sl.Emitting):
  """A conformer-based audio encoder implemented with SequenceLayers."""

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    """Configuration and builder for the Gemma3nAudioEncoder."""

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
    conv_causal_time_padding_type: types.PaddingModeString = (
        types.PaddingMode.REVERSE_CAUSAL.value
    )
    conv_spatial_padding_type: types.PaddingModeString = (
        types.PaddingMode.SAME.value
    )
    name: str | None = None

    # Optional quantization
    einsum_factories: EinsumFactoryConfig = EinsumFactoryConfig()

    def make(self) -> 'Gemma3nAudioEncoder':
      return Gemma3nAudioEncoder(config=self, name=self.name)

  config: Config

  def setup(self) -> None:
    """Feature processing Sequence layers."""
    filters = (128, 32)
    kernel_size = ((3, 3), (3, 3))
    stride = ((2, 2), (2, 2))

    self.layers = [
        sl.Serial.Config(
            [
                sl.ExpandDims.Config(-1),
                sl.Delay.Config(
                    # Use stride[0][0] for time dim.
                    -self.config.input_latency % stride[0][0],
                    delay_layer_output=False,
                ),
                sl.Conv2D.Config(
                    filters=filters[0],
                    kernel_size=kernel_size[0],
                    strides=stride[0],
                    time_padding=self.config.conv_causal_time_padding_type,
                    spatial_padding=self.config.conv_spatial_padding_type,
                    use_bias=False,
                    compute_dtype=self.config.compute_dtype,
                    param_dtype=self.config.param_dtype,
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
                    filters=filters[1],
                    kernel_size=kernel_size[1],
                    strides=stride[1],
                    time_padding=self.config.conv_causal_time_padding_type,
                    spatial_padding=self.config.conv_spatial_padding_type,
                    use_bias=False,
                    compute_dtype=self.config.compute_dtype,
                    param_dtype=self.config.param_dtype,
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
                    [self.config.model_dims],
                    use_bias=False,
                    compute_dtype=None,
                    param_dtype=self.config.param_dtype,
                    name='input_proj',
                ),
            ],
            name='feature',
        ).make(),
        sl.Repeat.Config(
            sl.Serial.Config(
                [
                    sl.Residual.Config(
                        [
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='pre_layer_norm',
                            ),
                            sl.Dense.Config(
                                self.config.model_dims * 4,
                                use_bias=False,
                                activation=jax.nn.swish,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.mlp.linear,
                                ),
                                name='ffn_layer1',
                            ),
                            sl.Dense.Config(
                                self.config.model_dims,
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.mlp.linear,
                                ),
                                name='ffn_layer2',
                            ),
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='post_layer_norm',
                            ),
                            sl.Scale.Config(self.config.ffn_residual_weight),
                        ],
                        name='fflayer_start',
                    ),
                    # Attention with residual connection.
                    sl.Residual.Config(
                        [
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='pre_norm',
                            ),
                            sl.LocalDotProductSelfAttention.Config(
                                input_projection=sl.CombinedQueryKeyValueProjection(
                                    einsum_factory=prepare_einsum_factory(
                                        self.config.einsum_factories.attention.qkv_einsum,
                                    ),
                                ),
                                num_heads=self.config.atten_num_heads,
                                units_per_head=self.config.model_dims
                                // self.config.atten_num_heads,
                                use_bias=False,
                                block_size=12,
                                max_past_horizon=self.config.atten_left_context
                                - 1,
                                max_future_horizon=0,
                                attention_logits_soft_cap=50.0,
                                relative_position_embedding=sl.TransformerXLRelativePositionEmbedding.Config(
                                    num_heads=self.config.atten_num_heads,
                                    units_per_head=self.config.model_dims
                                    // self.config.atten_num_heads,
                                    max_backward=self.config.atten_left_context
                                    - 1,
                                    max_forward=0,
                                    position_bias_dim=self.config.model_dims,
                                    use_bias=False,
                                    param_dtype=self.config.param_dtype,
                                    einsum_factory=prepare_einsum_factory(
                                        self.config.einsum_factories.attention.position_embedding,
                                    ),
                                ),
                                attention_probabilities_dropout_rate=0.0,
                                per_dim_scale=True,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                name='self_atten',
                            ),
                            sl.DenseShaped.Config(
                                [self.config.model_dims],
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                name='post',
                            ),
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='post_norm',
                            ),
                        ],
                        name='trans_atten',
                    ),
                    sl.Residual.Config(
                        [
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='ln',
                            ),
                            sl.Dense.Config(
                                2 * self.config.model_dims,
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.conv.linear,
                                ),
                                name='linear_start',
                            ),
                            sl.GatedLinearUnit.Config(name='glu'),
                            sl.DepthwiseConv1D.Config(
                                kernel_size=5,
                                strides=1,
                                depth_multiplier=1,
                                padding='causal',
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                name='depthwise_conv1d',
                            ),
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='conv_norm',
                            ),
                            sl.Swish.Config(name='conv_activation'),
                            sl.Dense.Config(
                                self.config.model_dims,
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.conv.linear,
                                ),
                                name='linear_end',
                            ),
                        ],
                        name='lconv',
                    ),
                    sl.Residual.Config(
                        [
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='pre_layer_norm',
                            ),
                            sl.Dense.Config(
                                self.config.model_dims * 4,
                                use_bias=False,
                                activation=jax.nn.swish,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.mlp.linear,
                                ),
                                name='ffn_layer1',
                            ),
                            sl.Dense.Config(
                                self.config.model_dims,
                                use_bias=False,
                                compute_dtype=self.config.compute_dtype,
                                param_dtype=self.config.param_dtype,
                                einsum_factory=prepare_einsum_factory(
                                    self.config.einsum_factories.mlp.linear,
                                ),
                                name='ffn_layer2',
                            ),
                            sl.RMSNormalization.Config(
                                param_dtype=self.config.param_dtype,
                                reductions_in_at_least_fp32=True,
                                name='post_layer_norm',
                            ),
                            sl.Scale.Config(self.config.ffn_residual_weight),
                        ],
                        name='fflayer_end',
                    ),
                    sl.RMSNormalization.Config(
                        param_dtype=self.config.param_dtype,
                        reductions_in_at_least_fp32=True,
                        name='final_ln',
                    ),
                ],
                name='stacked_layers',
            ),
            num_repeats=self.config.num_layers,
            remat=True,
            name='conformer',
        ).make(),
        sl.Delay.Config(2, delay_layer_output=False).make(),
        UniformReducer.Config(
            reduction_factor=4,
            name='reducer',
        ).make(),
        sl.MaskInvalid.Config().make(),
    ]
