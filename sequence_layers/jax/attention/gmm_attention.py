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
"""A multi-headed Gaussian-mixture attention layer."""

import dataclasses
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import meta
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


class GmmAttention(types.PreservesType, types.Emitting):
  """A multi-headed Gaussian-mixture attention layer.

  Uses GMMs to model the probability distribution of where to focus attention
  in the source sequence.
  """

  @dataclasses.dataclass(frozen=True)
  class Config(types.SequenceLayerConfig):
    """Configuration for GmmAttention."""

    # The name of the source to attend to.
    source_name: str
    # The number of attention heads.
    num_heads: int
    # The number of units per head.
    units_per_head: int
    # The number of Gaussian mixture components.
    num_components: int
    # Whether to restrict the mean of each mixture component to
    # monotonically increase over time.
    monotonic: bool
    # Precision config to use for einsums.
    precision: nn.linear.PrecisionLike = None
    # Kernel initializer for the hidden layer.
    hidden_kernel_init: nn.initializers.Initializer | None = (
        nn.linear.default_kernel_init
    )
    # Sharding configuration for the hidden layer.
    hidden_kernel_sharding: types.Sharding | None = None
    # Kernel initializer for the output layer.
    output_kernel_init: nn.initializers.Initializer | None = (
        nn.linear.default_kernel_init
    )
    # Whether to learn a bias in the output layer.
    output_use_bias: bool = True
    # Bias initializer for the output layer.
    output_bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()
    # If use_bias=True, sharding configuration for output layer bias.
    output_bias_sharding: types.Sharding | None = None
    # Sharding configuration for the output layer kernel.
    output_kernel_sharding: types.Sharding | None = None
    # Initial bias for the offset.
    init_offset_bias: float = 0.0
    # Initial bias for the scale.
    init_scale_bias: float = 0.0
    # Maximum offset.
    max_offset: float = -1.0
    # Name of the layer.
    name: str | None = None

    # TODO(b/330395665): compute and param dtype.

    def make(self) -> 'GmmAttention':
      return GmmAttention(self, name=self.name)

  config: Config

  def setup(self) -> None:
    common.validate_attention(
        self.config.source_name,
        self.config.num_heads,
        self.config.units_per_head,
        self.name,
    )
    if self.config.num_components <= 0:
      raise ValueError(
          f'Expected num_components > 0 for {self.name}. Got:'
          f' {self.config.num_components}.'
      )

    # c: channels, n: num_heads, u: units_per_head.
    self._mlp_hidden = utils.FlaxEinsumDense(
        '...c,cnu->...nu',
        output_shape=(self.config.num_heads, self.config.units_per_head),
        precision=self.config.precision,
        kernel_init=utils.shard_initializer(
            self.config.hidden_kernel_init,
            self.config.hidden_kernel_sharding,
            projectable=True,
            axes_types=(meta.AxisType.FANIN, None, None),
        ),
        activation=nn.relu,
        name='hidden',
    )
    # n: num_heads, u: units_per_head, c: num_components, l: num_logits (3).
    self._mlp_output = utils.FlaxEinsumDense(
        equation='...nu,nucl->...ncl',
        # Final axis holds 3 logits: prior_logits, offset_logits, scale_logits.
        output_shape=(self.config.num_heads, self.config.num_components, 3),
        precision=self.config.precision,
        bias_axes='ncl' if self.config.output_use_bias else None,
        kernel_init=utils.shard_initializer(
            self.config.output_kernel_init,
            self.config.output_kernel_sharding,
            projectable=True,
            axes_types=(None, meta.AxisType.FANIN, None, meta.AxisType.STACKED),
        ),
        bias_init=(
            utils.shard_initializer(
                self.config.output_bias_init,
                self.config.output_bias_sharding,
                projectable=False,
                axes_types=(None, None, None),
            )
            if self.config.output_use_bias
            else None
        ),
        bias_sharding=(
            self.config.output_bias_sharding
            if self.config.output_use_bias
            else None
        ),
        activation=None,
        name='output',
    )

  @property
  def receptive_field_per_step(self) -> dict[int, types.ReceptiveField]:
    start = -np.inf if self.config.monotonic else 0
    end = 0
    return {0: (start, end)}

  def get_initial_state(
      self,
      batch_size: int,
      input_spec: types.ShapeDType,
      *,
      training: bool,
      constants: types.Constants | None = None,
  ) -> types.State:
    if self.config.monotonic:
      # Start from zero positions.
      return jnp.zeros(
          [batch_size, 1, self.config.num_heads, self.config.num_components],
          input_spec.dtype,
      )
    else:
      return ()

  @nn.nowrap
  def get_output_shape(
      self,
      input_shape: types.ShapeLike,
      *,
      constants: types.Constants | None = None,
  ) -> types.Shape:
    if len(input_shape) != 1:
      raise ValueError(
          'GmmAttention requires rank 3 input got:'
          f' {(None, None) + tuple(input_shape)}'
      )
    source = common.get_source(self, self.config.source_name, constants)
    # Unlike other multi-headed attention classes in this file,
    # get_output_shape()[1] is the number of source channels (rather than
    # units_per_head) because GmmAttention does not perform a values projection.
    # (We may try adding this).
    return (self.config.num_heads, source.shape[2])

  @types.check_layer_with_emits
  def layer_with_emits(
      self,
      x: types.Sequence,
      training: bool,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.Emits]:
    position = self.get_initial_state(
        x.shape[0], x.channel_spec, training=training, constants=constants
    )

    if self.config.monotonic and x.shape[1] != 1:
      context_vector, _, attention_weights = utils.step_by_step_dynamic(
          l=self,
          x=x,
          training=training,
          initial_state=position,
          blocks_per_step=1,
          constants=constants,
          with_emits=True,
      )
      return context_vector, attention_weights

    source = common.get_source(self, self.config.source_name, constants)
    # Compute in parallel, since either not monotonic, or num_timesteps=1.
    context_vector, _, attention_weights = self.attention(
        source=source,
        query=x,
        prev_position=position,
    )
    return context_vector, attention_weights

  @types.check_step_with_emits
  def step_with_emits(
      self,
      x: types.Sequence,
      state: types.State,
      training: bool,
      *,
      constants: types.Constants | None = None,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    source = common.get_source(self, self.config.source_name, constants)
    if self.config.monotonic and x.shape[1] != 1:
      # step_by_step_static return types already match those of this function:
      # Sequence(context_vector), State(new_position), Emits(attention_weights).
      return utils.step_by_step_static(
          l=self,
          x=x,
          training=training,
          initial_state=state,
          blocks_per_step=1,
          constants=constants,
          with_emits=True,
      )
    # Compute in parallel, since either not monotonic, or num_timesteps=1.
    context_vector, new_position, attention_weights = self.attention(
        source=source,
        query=x,
        prev_position=state,
    )

    return (
        context_vector,
        new_position,
        attention_weights,
    )

  def attention(
      self,
      source: types.Sequence,
      query: types.Sequence,
      prev_position: types.State,
  ) -> tuple[types.Sequence, types.State, types.Emits]:
    """Computes GMM attention from query and previous position.

    Args:
      source: The source sequence to attend to. A [batch, time, dim]
      query: [batch_size, query_time, query_channels]
      prev_position (state): [batch_size, 1, num_heads, num_components].
        Previous position of each mixture component per-head.

    Returns:
      context_vector: [batch_size, query_time, num_heads,
        source_dimension]. The per-head context vector.
      position: [batch_size, 1, num_heads, num_components]. Current
        position of each mixture component per head.
      attention_weights: [batch_size, query_time, num_heads, source_time].
        Per-head attention values representing the probability density at each
        source sequence position.
    """
    if query.values.ndim != 3:
      raise ValueError(f'Expected [b, 1, d] inputs, got: {query.values.shape}.')

    query_time = query.values.shape[1]
    if self.config.monotonic:
      assert (
          query_time == 1
      ), f'Expected [b, 1, d] inputs, got: {query.values.shape}.'
    query = self._mlp_hidden.project_sequence(query)
    assert query.values.ndim == 4, (
        f'After self._mlp_hidden(), {query.values.shape=}, but expected [b, 1,'
        ' num_heads, units_per_head]'
    )

    # Per-head projection from [b, 1, num_heads, units_per_head] to
    # [b, 1, num_heads, num_components, 3] (the 3 logits are unstacked below).
    query = self._mlp_output.project_sequence(query)
    assert query.values.ndim == 5, (
        f'After self._mlp_output(), {query.values.shape=}, but expected [b, 1,'
        ' num_heads, num_components, 3]'
    )

    # Each is [batch_size, query_time, num_heads, num_components].
    prior_logits, offset_logits, scale_logits = utils.unstack(
        query.values, axis=4
    )

    offset_logits += self.config.init_offset_bias
    scale_logits += self.config.init_scale_bias
    assert prior_logits.shape[-1] == self.config.num_components

    # attention_weights: [b, query_time, num_heads, source_time]
    # new_position: [b, q, num_heads, num_components]
    attention_weights, new_position = self._eval_gmm_pdfs(
        source, prior_logits, offset_logits, scale_logits, prev_position
    )

    # Expand source mask for broadcasting to
    # [b, query_time, num_heads, source_time].
    attention_weights = jnp.where(
        source.mask[:, jnp.newaxis, jnp.newaxis, :],
        attention_weights,
        jnp.zeros_like(attention_weights),
    )

    # [b, query_time, num_heads, source_time]
    # [b, source_time, source_dim]
    # -> [b, query_time, num_heads, source_dim]
    context_vector = jnp.einsum(
        'BiNj,BjS->BiNS',
        attention_weights,
        source.values,
        precision=self.config.precision,
    )

    # Make sure we don't update state if not monotonic (we should override the
    # return value of self.attention() to make state be ()).
    state = new_position if self.config.monotonic else ()
    # Convert types for return value.
    assert isinstance(attention_weights, jax.Array)
    attention_weights = common.CrossAttentionEmits(
        {self.config.source_name: types.Sequence(attention_weights, query.mask)}
    )
    # Returns (output, state, emits)
    return (
        types.Sequence(context_vector, query.mask),
        state,
        attention_weights,
    )

  def _eval_gmm_pdfs(
      self,
      source: types.Sequence,
      prior_logits: jax.Array,
      offset_logits: jax.Array,
      scale_logits: jax.Array,
      prev_position: types.State,
      normalize=True,
  ) -> tuple[jax.Array, jax.Array]:
    """Evaluate the location GMMs on all encoder positions.

    * Uses softmax for the mixture weights.
    * Uses softplus for means and scales.
    * Scale represents the standard deviation.

    Args:
      source: The source sequence to attend to.
      prior_logits: [b, query_time, num_heads, num_components].
      offset_logits: [b, query_time, num_heads, num_components].
      scale_logits: [b, query_time, num_heads, num_components].
      prev_position: [b, 1, num_heads, num_components].
      normalize: Whether to normalize the attention probabilities.

    Returns:
      attention_weights: Shaped [b, query_time, num_heads, source_time],
        probability densities for each attention head over the source sequence.
      new_position: [b, query_time, num_heads, num_components].
    """
    priors = utils.run_in_at_least_fp32(jax.nn.softmax)(prior_logits)
    variances = jnp.square(jax.nn.softplus(scale_logits))
    if self.config.max_offset > 0:
      # softplus(x) - softplus(x - M) gives a sigmoid that saturates outside
      # [0, M] and is approximately linear in between.
      position_offset = jax.nn.softplus(offset_logits) - jax.nn.softplus(
          offset_logits - self.config.max_offset
      )
    else:
      position_offset = jax.nn.softplus(offset_logits)

    # priors, position_offset, variances are shaped
    # [b, query_time, num_heads, num_components].
    utils.assert_is_compatible_with(
        priors.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    utils.assert_is_compatible_with(
        position_offset.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    utils.assert_is_compatible_with(
        variances.shape,
        [None, None, self.config.num_heads, self.config.num_components],
    )
    # prev_position is [b, 1, num_heads, num_components].
    if self.config.monotonic:
      utils.assert_is_compatible_with(
          prev_position.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
      # If we're monotonic, then query time is always 1.
      utils.assert_is_compatible_with(
          position_offset.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
      new_position = prev_position + position_offset
      utils.assert_is_compatible_with(
          new_position.shape,
          [None, 1, self.config.num_heads, self.config.num_components],
      )
    else:
      new_position = position_offset

    # Expand all to [b, query_time, num_heads, source_time (1), num_components]
    priors = priors[:, :, :, jnp.newaxis, :]
    means = new_position[:, :, :, jnp.newaxis, :]
    variances = variances[:, :, :, jnp.newaxis, :]

    # [1, 1, 1, source_timesteps, 1].
    source_length = source.values.shape[1]
    encoder_positions = jnp.asarray(
        jnp.arange(source_length), dtype=means.dtype
    )
    encoder_positions = encoder_positions[
        jnp.newaxis, jnp.newaxis, jnp.newaxis, :, jnp.newaxis
    ]

    if normalize:
      priors *= jnp.sqrt(2 * np.pi * variances + 1e-8)
    # Broadcast source time and query time.
    # encoder_positions is [1, 1, 1, source_time, 1]
    # means/priors/variances are
    # [batch, query_time, num_heads, 1, num_components]
    probabilities = priors * jnp.exp(
        -((encoder_positions - means) ** 2) / (2 * variances + 1e-9)
    )
    # probabilities shaped [batch, query_time, num_heads, source_time].
    probabilities = jnp.sum(probabilities, 4)
    return probabilities, new_position
