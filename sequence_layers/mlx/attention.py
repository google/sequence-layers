"""Dot-product attention layers for MLX."""

import dataclasses
import math
from types import MethodType
from typing import Any, cast, override

import mlx.core as mx

from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import projection_configs
from sequence_layers.mlx import types
from sequence_layers.mlx.projection_configs import \
    CombinedQueryKeyValueProjection
from sequence_layers.mlx.projection_configs import QueryAndKeyValueProjection
from sequence_layers.mlx.projection_configs import \
    QueryAndSharedKeyValueProjection
from sequence_layers.mlx.projection_configs import \
    SeparateQueryKeyValueProjection
from sequence_layers.specs import attention as attention_spec

Sequence = types.Sequence
MaskedSequence = types.MaskedSequence


@dataclasses.dataclass(frozen=True)
class SelfAttentionEmits:
  """A structure for emits produced by self attention layers."""

  probabilities: Sequence


@dataclasses.dataclass(frozen=True)
class CrossAttentionEmits:
  """A structure for emits produced by attention layers."""

  probabilities_by_source: dict[str, Sequence]


__all__ = (
    'DotProductSelfAttention',
    'DotProductAttention',
    'StreamingDotProductAttention',
    'LocalDotProductSelfAttention',
    'CombinedQueryKeyValueProjection',
    'SeparateQueryKeyValueProjection',
    'QueryAndKeyValueProjection',
    'QueryAndSharedKeyValueProjection',
    'SelfAttentionEmits',
    'CrossAttentionEmits',
)


def _quantized_matmul_proj(x, q_weight, q_scales, q_biases, group_size, bits):
  """Computes quantized matrix multiplication projection."""
  return mx.quantized_matmul(
      x,
      q_weight,
      scales=q_scales,
      biases=q_biases,
      transpose=True,
      group_size=group_size,
      bits=bits,
  )


def _query_scale_vector(per_dim_scale, query_scale, units_per_head, dtype):
  """Compute the per-dimension query scale vector.

  Returns:
    scale: [units_per_head] array or scalar float.
  """
  if query_scale is None:
    query_scale = 1.0 / math.sqrt(units_per_head)
  if per_dim_scale is not None:
    r_softplus_0 = 1.442695041
    scale = r_softplus_0 * query_scale
    softplus = mx.log1p(mx.exp(per_dim_scale.astype(dtype)))
    return scale * softplus
  return query_scale


def _scale_queries(queries, per_dim_scale, query_scale, units_per_head):
  """Scale queries, optionally with per-dimension learned scale.

  Matches JAX backend's _scale_query in common.py.

  Args:
    queries: [b, num_heads, q_time, units_per_head].
    per_dim_scale: [units_per_head] learned scale or None.
    query_scale: float scale or None (defaults to 1/sqrt(uph)).
    units_per_head: int.

  Returns:
    Scaled queries, same shape.
  """
  scale = _query_scale_vector(
      per_dim_scale, query_scale, units_per_head, queries.dtype
  )
  return queries * scale


def _causal_mask(q_len, kv_len):
  """Build a [1, 1, q_len, kv_len] causal mask (True = attend)."""
  # Each query at position i can attend to keys at positions
  # [kv_len - q_len, ..., kv_len - q_len + i].
  row = mx.arange(q_len)
  col = mx.arange(kv_len)
  # query i (global pos = kv_len - q_len + i) can see key j
  # if j <= kv_len - q_len + i.
  offset = kv_len - q_len
  mask = mx.expand_dims(col, axis=0) <= (mx.expand_dims(row, axis=1) + offset)
  return mask.reshape(1, 1, q_len, kv_len)


class DotProductSelfAttention(
    types.Emitting,
    attention_spec.DotProductSelfAttention[types.Sequence, types.ChannelSpec],
):
  """Multi-headed dot-product self attention for MLX.

  Supports:
  - Grouped Query Attention (num_kv_heads < num_heads)
  - Causal masking via max_past_horizon
  - KV cache for step-by-step inference
  - Optional query/key/value processing networks (e.g. RoPE)

  Kernels are stored in Linen-compatible shapes:
    q_proj: [in_features, num_heads * units_per_head]
    k_proj: [in_features, num_kv_heads * units_per_head]
    v_proj: [in_features, num_kv_heads * units_per_head]
    out_proj: [num_heads * units_per_head, in_features]
  """

  @dataclasses.dataclass(frozen=True)
  class Config(
      types.SequenceLayerConfig,
      attention_spec.DotProductSelfAttention.Config,
  ):
    """MLX-native configuration for DotProductSelfAttention."""

    num_heads: int
    units_per_head: int
    max_past_horizon: int
    max_future_horizon: int = 0
    num_kv_heads: int | None = None
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    input_projection: projection_configs.QueryKeyValueProjectionConfig = (
        dataclasses.field(
            default_factory=projection_configs.CombinedQueryKeyValueProjection
        )
    )
    query_network: Any = None
    key_network: Any = None
    value_network: Any = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    num_sink_embeddings: int = 0
    use_sink_scalars: bool = False
    use_kv_cache_ringbuffer: bool = False
    name: str | None = None

    @override
    def make(self) -> 'DotProductSelfAttention':
      return DotProductSelfAttention(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      in_features: int | None = None,
      num_heads: int | None = None,
      units_per_head: int | None = None,
      max_past_horizon: int | None = None,
      max_future_horizon: int = 0,
      num_kv_heads: int | None = None,
      use_bias: bool = False,
      query_scale: float | None = None,
      per_dim_scale: bool = False,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      attention_logits_soft_cap: float | None = None,
      num_sink_embeddings: int = 0,
      input_projection=None,
  ):
    super().__init__()
    if config is not None:
      self.config = config
    else:
      if (
          num_heads is None
          or units_per_head is None
          or max_past_horizon is None
      ):
        raise ValueError(
            'Must provide either config or num_heads, units_per_head, and'
            ' max_past_horizon'
        )
      num_heads = cast(int, num_heads)
      units_per_head = cast(int, units_per_head)
      max_past_horizon = cast(int, max_past_horizon)
      input_projection_val = (
          input_projection
          or projection_configs.CombinedQueryKeyValueProjection()
      )
      self.config = self.Config(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_past_horizon=max_past_horizon,
          max_future_horizon=max_future_horizon,
          num_kv_heads=num_kv_heads,
          use_bias=use_bias,
          query_scale=query_scale,
          per_dim_scale=per_dim_scale,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
          attention_logits_soft_cap=attention_logits_soft_cap,
          num_sink_embeddings=num_sink_embeddings,
          input_projection=input_projection_val,
      )

    self.compute_dtype = (
        init_mapping.to_mx_dtype(self.config.compute_dtype)
        if self.config.compute_dtype
        else None
    )
    self._param_dtype = (
        init_mapping.to_mx_dtype(self.config.param_dtype) or mx.float32
    )

    self.in_features = None
    self.num_heads = self.config.num_heads
    self.units_per_head = self.config.units_per_head
    self.max_past_horizon = self.config.max_past_horizon
    self.max_future_horizon = self.config.max_future_horizon
    self.num_kv_heads = self.config.num_kv_heads or self.num_heads
    self.use_bias = self.config.use_bias
    self._query_scale = self.config.query_scale
    self._attention_logits_soft_cap = self.config.attention_logits_soft_cap

    self._kernel_init = kernel_init
    self._bias_init = bias_init
    self._per_dim_scale = None

    self.query_network: Any = query_network or self.config.query_network
    self.key_network: Any = key_network or self.config.key_network
    self.value_network: Any = value_network or self.config.value_network

    self.sink_key_embeddings: Any = None
    self.sink_value_embeddings: Any = None

    self.q_proj: Any = None
    self.kv_proj: Any = None
    self.qkv_proj_qw: Any = None
    self.qkv_proj_qs: Any = None
    self.qkv_proj_qb: Any = None
    self._quant_group_size: int | None = None
    self._quant_bits: int | None = None
    self._project_qkv_fn: Any = None

    self._initialized = False

    if in_features is not None:
      self._ensure_initialized(in_features)

  def _ensure_initialized(self, in_features: int):
    """Ensure parameters and submodules are dynamically initialized."""
    if self._initialized:
      return
    self._initialized = True
    self.in_features = in_features

    # pylint: disable=import-outside-toplevel
    from sequence_layers.mlx import utils as mlx_utils

    if hasattr(self.query_network, 'make'):
      self.query_network = mlx_utils.make_layer(self.query_network)
    if hasattr(self.key_network, 'make'):
      self.key_network = mlx_utils.make_layer(self.key_network)
    if hasattr(self.value_network, 'make'):
      self.value_network = mlx_utils.make_layer(self.value_network)

    param_dtype = self._param_dtype
    per_dim_scale = self.config.per_dim_scale
    units_per_head = self.units_per_head
    num_heads = self.num_heads
    num_kv_heads = self.num_kv_heads
    use_bias = self.use_bias
    input_projection = self.config.input_projection
    num_sink_embeddings = self.config.num_sink_embeddings

    self._per_dim_scale = (
        mx.zeros((units_per_head,), dtype=param_dtype)
        if per_dim_scale
        else None
    )

    kernel_init = self._kernel_init
    bias_init = self._bias_init

    if kernel_init is None:
      qkv_init = (
          getattr(input_projection, 'qkv_kernel_init', None)
          or getattr(input_projection, 'q_kernel_init', None)
          or getattr(input_projection, 'kv_kernel_init', None)
      )
      if qkv_init is not None:
        kernel_init = init_mapping.map_initializer(qkv_init)
      else:
        kernel_init = init_mapping.make_variance_scaling_init(
            'fan_in', 'truncated_normal'
        )

    if bias_init is None:
      qkv_bias_init = (
          getattr(input_projection, 'bias_init', None)
          or getattr(input_projection, 'q_bias_init', None)
          or getattr(input_projection, 'kv_bias_init', None)
      )
      if qkv_bias_init is not None:
        bias_init = init_mapping.map_initializer(qkv_bias_init)
      else:
        bias_init = init_mapping.zeros_init

    key = mx.random.key(0)
    q_dim = num_heads * units_per_head
    kv_dim = num_kv_heads * units_per_head

    self.input_projection = input_projection
    if (
        isinstance(
            input_projection, projection_configs.CombinedQueryKeyValueProjection
        )
        and num_kv_heads == num_heads
    ):
      out_dim = q_dim + 2 * kv_dim
      self.qkv_proj = kernel_init(key, (in_features, out_dim), param_dtype)
      if use_bias:
        self.qkv_bias = bias_init(key, (out_dim,), param_dtype)
    else:
      self.q_proj = kernel_init(key, (in_features, q_dim), param_dtype)
      self.kv_proj = mx.concatenate(
          [
              kernel_init(key, (in_features, kv_dim), param_dtype),
              kernel_init(key, (in_features, kv_dim), param_dtype),
          ],
          axis=-1,
      )
      if use_bias:
        self.q_bias = bias_init(key, (q_dim,), param_dtype)
        self.kv_bias = mx.concatenate(
            [
                bias_init(key, (kv_dim,), param_dtype),
                bias_init(key, (kv_dim,), param_dtype),
            ],
            axis=-1,
        )

    self.num_sink_embeddings = num_sink_embeddings
    if num_sink_embeddings > 0:
      self.sink_key_embeddings = mx.zeros(
          (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
      )
      self.sink_value_embeddings = mx.zeros(
          (num_sink_embeddings, num_kv_heads, units_per_head), dtype=param_dtype
      )
    else:
      self.sink_key_embeddings = None
      self.sink_value_embeddings = None

  @property
  @override
  def supports_step(self):
    supports = self.max_past_horizon >= 0 and self.max_future_horizon >= 0
    if self.query_network is not None:
      supports = supports and self.query_network.supports_step
    if self.key_network is not None:
      supports = supports and self.key_network.supports_step
    if self.value_network is not None:
      supports = supports and self.value_network.supports_step
    return supports

  @property
  @override
  def input_latency(self):
    return max(0, self.max_future_horizon)

  def _project_qkv(self, x):
    """Project input to Q, K, V sequences."""
    self._ensure_initialized(x.shape[-1])
    b, t = x.shape[0], x.shape[1]
    dtype = self.compute_dtype or x.dtype

    v = x.values.astype(dtype)

    if hasattr(self, 'qkv_proj'):
      qkv = mx.matmul(v, self.qkv_proj.astype(dtype))
      if self.use_bias:
        qkv = qkv + self.qkv_bias.astype(dtype)

      q, k, val = mx.split(qkv, 3, axis=-1)
    else:
      q = mx.matmul(v, self.q_proj.astype(dtype))
      kv = mx.matmul(v, self.kv_proj.astype(dtype))
      k, val = mx.split(kv, 2, axis=-1)

      if self.use_bias:
        q = q + self.q_bias.astype(dtype)
        kv_bias = self.kv_bias.astype(dtype)
        kb, vb = mx.split(kv_bias, 2, axis=-1)
        k = k + kb
        val = val + vb

    # Reshape to [b, t, heads, units_per_head].
    q = q.reshape(b, t, self.num_heads, self.units_per_head)
    k = k.reshape(b, t, self.num_kv_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_kv_heads, self.units_per_head)

    return (
        Sequence(q, x.mask),
        Sequence(k, x.mask),
        Sequence(val, x.mask),
    )

  def _compute_attention(
      self, queries, keys, values, mask, emit_attention_weights=False
  ):
    """Compute scaled dot-product attention.

    Args:
      queries: [b, q_t, num_heads, units_per_head]
      keys: [b, kv_t, num_kv_heads, units_per_head]
      values: [b, kv_t, num_kv_heads, units_per_head]
      mask: [b, 1, q_t, kv_t] boolean mask (True = attend)
      emit_attention_weights: bool. Whether to emit attention weights.

    Returns:
      (context, weights) tuple.
        context: [b, q_t, num_heads, units_per_head]
        weights: [b, q_t, num_heads, kv_t] or ()
    """
    # Use mx.fast.scaled_dot_product_attention unless soft_cap forces
    # manual logit manipulation or emits are requested.
    has_soft_cap = getattr(self, '_attention_logits_soft_cap', None) is not None

    if not has_soft_cap and not emit_attention_weights:
      # SDPA path — handles both plain and sink cases.
      q = mx.transpose(queries, (0, 2, 1, 3))
      k = mx.transpose(keys, (0, 2, 1, 3))
      v = mx.transpose(values, (0, 2, 1, 3))

      q = _scale_queries(
          q, self._per_dim_scale, self._query_scale, self.units_per_head
      )

      if self.sink_key_embeddings is not None:
        # JAX computes sink logits with *unscaled* queries.  To use SDPA
        # we pre-divide sink keys by the scale so that:
        #   scaled_q @ (sink_k / scale) == unscaled_q @ sink_k
        scale_vec = _query_scale_vector(
            self._per_dim_scale,
            self._query_scale,
            self.units_per_head,
            q.dtype,
        )
        sink_k = self.sink_key_embeddings.astype(q.dtype) / scale_vec
        sink_v = self.sink_value_embeddings.astype(v.dtype)

        # GQA: repeat sink heads to match query heads.
        num_groups = self.num_heads // self.num_kv_heads
        if num_groups > 1:
          sink_v = mx.repeat(sink_v, num_groups, axis=1)

        # Transpose [K, nh, h] → [nh, K, h] and broadcast batch.
        sink_k_b = mx.broadcast_to(
            mx.transpose(sink_k, (1, 0, 2))[None],
            (q.shape[0], self.num_heads, sink_k.shape[0], self.units_per_head),
        )
        sink_v_b = mx.broadcast_to(
            mx.transpose(sink_v, (1, 0, 2))[None],
            (v.shape[0], self.num_heads, sink_v.shape[0], self.units_per_head),
        )

        # Prepend sinks to K/V.
        k = mx.concatenate([sink_k_b, k], axis=2)
        v = mx.concatenate([sink_v_b, v], axis=2)

        # Extend mask — sinks are always valid.
        if mask is not None:
          num_sinks = self.sink_key_embeddings.shape[0]
          sink_mask = mx.ones(
              (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
              dtype=mx.bool_,
          )
          mask = mx.concatenate([sink_mask, mask], axis=-1)

      context = mx.fast.scaled_dot_product_attention(
          q, k, v, scale=1.0, mask=mask
      )
      return mx.transpose(context, (0, 2, 1, 3)), ()

    # Manual path — for attention_logits_soft_cap or when emits are requested.
    num_groups = self.num_heads // self.num_kv_heads
    if num_groups > 1:
      keys = mx.repeat(keys, num_groups, axis=2)
      values = mx.repeat(values, num_groups, axis=2)

    q = mx.transpose(queries, (0, 2, 1, 3))
    k = mx.transpose(keys, (0, 2, 1, 3))
    v = mx.transpose(values, (0, 2, 1, 3))

    # Compute sink logits BEFORE scaling queries, matching JAX behavior.
    sink_logits = None
    if self.sink_key_embeddings is not None:
      sink_k = self.sink_key_embeddings.astype(q.dtype)
      sink_k_t = mx.transpose(sink_k, (1, 2, 0))
      sink_logits = mx.matmul(q, sink_k_t)

    q = _scale_queries(
        q, self._per_dim_scale, self._query_scale, self.units_per_head
    )
    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

    if self.sink_key_embeddings is not None:
      sink_v = self.sink_value_embeddings.astype(v.dtype)
      if num_groups > 1:
        sink_v = mx.repeat(sink_v, num_groups, axis=1)
      sink_v_t = mx.transpose(sink_v, (1, 0, 2))
      sink_v_b = mx.broadcast_to(sink_v_t[None], (v.shape[0],) + sink_v_t.shape)
      v = mx.concatenate([sink_v_b, v], axis=2)
      assert sink_logits is not None
      logits = mx.concatenate([sink_logits, logits], axis=-1)
      if mask is not None:
        num_sinks = self.sink_key_embeddings.shape[0]
        sink_mask = mx.ones(
            (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
            dtype=mx.bool_,
        )
        mask = mx.concatenate([sink_mask, mask], axis=-1)

    if has_soft_cap:
      cap = cast(Any, self._attention_logits_soft_cap)
      logits = mx.tanh(logits / cap) * cap

    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)

    logits_f32 = (
        logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    )
    weights = mx.softmax(logits_f32, axis=-1).astype(v.dtype)
    context = mx.matmul(weights, v)
    context = mx.transpose(context, (0, 2, 1, 3))

    emits = ()
    if emit_attention_weights:
      # Transpose from [b, nh, q, kv] back to [b, q, nh, kv] to match JAX.
      emits = mx.transpose(weights, (0, 2, 1, 3))

    return context, emits

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductSelfAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @override
  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    self._ensure_initialized(input_spec.shape[-1])
    compute_dtype = self.get_output_dtype(input_spec.dtype)
    max_past = max(0, self.max_past_horizon)
    max_future = max(0, self.max_future_horizon)
    kv_buffer_size = max_past + max_future

    kv_shape = (
        batch_size,
        kv_buffer_size,
        self.num_kv_heads,
        self.units_per_head,
    )
    kv_buffer_keys = mx.zeros(kv_shape, dtype=compute_dtype)
    kv_buffer_values = mx.zeros(kv_shape, dtype=compute_dtype)
    kv_buffer_mask = mx.zeros((batch_size, kv_buffer_size), dtype=mx.bool_)
    time_step = mx.zeros((batch_size,), dtype=mx.int32)

    # Q/K/V network states.
    q_net_state = (
        self.query_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.query_network is not None
        else ()
    )
    k_net_state = (
        self.key_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_kv_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.key_network is not None
        else ()
    )
    v_net_state = (
        self.value_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_kv_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.value_network is not None
        else ()
    )

    # Query delay buffer for future horizon.
    if max_future:
      q_delay_values = mx.zeros(
          (
              batch_size,
              max_future,
              self.num_heads,
              self.units_per_head,
          ),
          dtype=compute_dtype,
      )
      q_delay_mask = mx.zeros((batch_size, max_future), dtype=mx.bool_)
    else:
      q_delay_values = ()
      q_delay_mask = ()

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    )

  @override
  def layer_with_emits(self, x, *, training: bool, constants=None):
    proj_fn = self._project_qkv_fn or self._project_qkv
    queries, keys, values = proj_fn(x)

    # Optional Q/K/V processing networks (e.g. RoPE).
    # Use `is not None` because parameterless nn.Modules are falsy.
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(
              queries, training=training, constants=constants
          ).values,
          queries.mask,
      )
    if self.key_network is not None:
      keys = Sequence(
          self.key_network.layer(
              keys, training=training, constants=constants
          ).values,
          keys.mask,
      )
    if self.value_network is not None:
      values = Sequence(
          self.value_network.layer(
              values, training=training, constants=constants
          ).values,
          values.mask,
      )

    # Mask invalid values.
    values = values.mask_invalid()

    t = x.shape[1]

    # Build visibility mask.
    # Start with key validity: [b, 1, 1, t].
    valid_mask = x.mask[:, None, None, :]

    # Optionally add causal / banded mask.
    if self.max_past_horizon >= 0 or self.max_future_horizon >= 0:
      past = t - 1 if self.max_past_horizon == -1 else self.max_past_horizon
      future = (
          t - 1 if self.max_future_horizon == -1 else self.max_future_horizon
      )
      # Banded visibility matrix.
      row = mx.expand_dims(mx.arange(t), axis=1)
      col = mx.expand_dims(mx.arange(t), axis=0)
      banded = (col >= row - past) & (col <= row + future)
      valid_mask = valid_mask & banded.reshape(1, 1, t, t)

    context, probs = self._compute_attention(
        queries.values,
        keys.values,
        values.values,
        valid_mask,
        emit_attention_weights=self.config.emit_attention_weights,
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = SelfAttentionEmits(Sequence(probs, x.mask))
    return Sequence(context, x.mask), emits

  @override
  def step_with_emits(self, x, state: Any, *, training: bool, constants=None):
    proj_fn = self._project_qkv_fn or self._project_qkv
    queries, keys, values = proj_fn(x)

    (
        kv_buf_k,
        kv_buf_v,
        kv_buf_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    ) = state

    # Optional Q/K/V processing networks.
    # Use `is not None` because parameterless nn.Modules are falsy.
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, training=training, constants=constants
      )
    if self.key_network is not None:
      keys, k_net_state = self.key_network.step(
          keys, k_net_state, training=training, constants=constants
      )
    if self.value_network is not None:
      values, v_net_state = self.value_network.step(
          values, v_net_state, training=training, constants=constants
      )

    # Mask invalid values.
    values = values.mask_invalid()

    x_time = x.shape[1]
    kv_buffer_size = kv_buf_k.shape[1]

    if self.max_future_horizon > 0:
      # Concatenate new queries to delay buffer.
      q_delay_values = mx.concatenate([q_delay_values, queries.values], axis=1)
      q_delay_mask = mx.concatenate([q_delay_mask, x.mask], axis=1)

      # Use the oldest x_time queries as the current step's queries.
      queries = Sequence(
          q_delay_values[:, :x_time],
          q_delay_mask[:, :x_time],
      )

      # Preserve the last max_future_horizon queries for the next step.
      q_delay_values = q_delay_values[:, x_time:]
      q_delay_mask = q_delay_mask[:, x_time:]

    if kv_buffer_size > 0:
      t0 = time_step[0]  # MLX scalar, no eval.

      # Concatenate old buffer with new elements for attention computation.
      # This avoids overwriting history needed by current queries.
      combined_k = mx.concatenate([kv_buf_k, keys.values], axis=1)
      combined_v = mx.concatenate([kv_buf_v, values.values], axis=1)
      combined_mask = mx.concatenate([kv_buf_mask, x.mask], axis=1)

      # Build visibility mask: [b, 1, 1, kv_buffer_size + x_time].
      kv_valid = combined_mask[:, None, None, :]

      # Map physical indices in old buffer to temporal indices.
      # The newest time in the old buffer was t0 - 1.
      newest_time_old = t0 - 1
      newest_pos_old = newest_time_old % kv_buffer_size
      phys_old = mx.arange(kv_buffer_size)
      dist_old = (newest_pos_old - phys_old + kv_buffer_size) % kv_buffer_size
      temporal_old = newest_time_old - dist_old

      # Temporal indices for new elements.
      temporal_new = t0 + mx.arange(x_time)

      # Combine temporal indices.
      temporal = mx.concatenate([temporal_old, temporal_new], axis=0)

      # Add causal mask for multi-step queries (respects ring buffer order).
      q_times = t0 - self.max_future_horizon + mx.arange(x_time)
      causal = temporal[None, :] <= (q_times[:, None] + self.max_future_horizon)

      # Add finite horizon mask.
      past = self.max_past_horizon
      finite_horizon = temporal[None, :] >= (q_times[:, None] - past)

      causal_and_finite = causal & finite_horizon
      kv_valid = kv_valid & causal_and_finite.reshape(
          1, 1, x_time, kv_buffer_size + x_time
      )

      context, probs = self._compute_attention(
          queries.values,
          combined_k,
          combined_v,
          kv_valid,
          emit_attention_weights=self.config.emit_attention_weights,
      )

      if self.config.emit_attention_weights:
        assert isinstance(probs, mx.array)
        sort_idx = mx.argsort(temporal)
        probs = probs[..., sort_idx]

      # Ring buffer write AFTER read: insert new K/V at rotating positions.
      positions = (t0 + mx.arange(x_time)) % kv_buffer_size  # [x_time]

      # Scatter K/V into buffer at ring positions.
      idx_4d = mx.broadcast_to(
          positions.reshape(1, x_time, 1, 1), keys.values.shape
      )
      kv_buf_k = mx.put_along_axis(kv_buf_k, idx_4d, keys.values, axis=1)
      kv_buf_v = mx.put_along_axis(kv_buf_v, idx_4d, values.values, axis=1)

      # Scatter mask into buffer.
      idx_2d = mx.broadcast_to(positions.reshape(1, x_time), x.mask.shape)
      kv_buf_mask = mx.put_along_axis(kv_buf_mask, idx_2d, x.mask, axis=1)
    else:
      # Degenerate: no history buffer, attend only to current step.
      kv_valid = x.mask[:, None, None, :]
      if x_time > 1:
        causal = _causal_mask(x_time, x_time)
        kv_valid = kv_valid & causal
      context, probs = self._compute_attention(
          queries.values,
          keys.values,
          values.values,
          kv_valid,
          emit_attention_weights=self.config.emit_attention_weights,
      )

    new_state = (
        kv_buf_k,
        kv_buf_v,
        kv_buf_mask,
        time_step + x_time,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = SelfAttentionEmits(Sequence(probs, queries.mask))
    return Sequence(context, queries.mask), new_state, emits

  def to_quantized(
      self, group_size: int = 64, bits: int = 4, mode: str = 'affine'
  ):
    """Convert attention projection layers to quantized versions."""
    del mode  # Unused in MLX quantize

    # Determine in_features from whichever projection layout was initialized.
    in_features = None
    if getattr(self, 'qkv_proj', None) is not None:
      in_features = self.qkv_proj.shape[0]
    elif getattr(self, 'q_proj', None) is not None:
      in_features = self.q_proj.shape[0]

    if in_features is None or in_features % group_size != 0:
      return self

    self._quant_group_size = group_size
    self._quant_bits = bits

    # Build the combined QKV weight matrix from whichever layout exists.
    if getattr(self, 'qkv_proj', None) is not None:
      w_qkv = self.qkv_proj.T
    else:
      w_q = self.q_proj.T
      # kv_proj is already combined [in, 2*kv_dim].
      w_kv = self.kv_proj.T
      w_qkv = mx.concatenate([w_q, w_kv], axis=0)

    self.qkv_proj_qw, self.qkv_proj_qs, self.qkv_proj_qb = mx.quantize(
        w_qkv, group_size=group_size, bits=bits
    )

    # Clear all original projection weights.
    self.qkv_proj = cast(Any, None)
    self.q_proj = cast(Any, None)
    self.kv_proj = cast(Any, None)

    # Split combined bias into q_bias / kv_bias for the quantized path.
    if self.use_bias and getattr(self, 'qkv_bias', None) is not None:
      d_q = self.num_heads * self.units_per_head
      self.q_bias, self.kv_bias = mx.split(self.qkv_bias, [d_q], axis=-1)
      self.qkv_bias = cast(Any, None)

    def _project_qkv(self, x):
      b, t = x.shape[0], x.shape[1]
      dtype = self.compute_dtype or x.dtype
      v = x.values.astype(dtype)

      qkv = _quantized_matmul_proj(
          v,
          self.qkv_proj_qw,
          self.qkv_proj_qs,
          self.qkv_proj_qb,
          self._quant_group_size,
          self._quant_bits,
      )

      d_q = self.num_heads * self.units_per_head
      d_k = self.num_kv_heads * self.units_per_head
      q, k, val = mx.split(qkv, [d_q, d_q + d_k], axis=-1)

      if self.use_bias:
        q = q + self.q_bias.astype(dtype)
        kv_bias = self.kv_bias.astype(dtype)
        kb, vb = mx.split(kv_bias, 2, axis=-1)
        k = k + kb
        val = val + vb

      q = q.reshape(b, t, self.num_heads, self.units_per_head)
      k = k.reshape(b, t, self.num_kv_heads, self.units_per_head)
      val = val.reshape(b, t, self.num_kv_heads, self.units_per_head)

      return (
          Sequence(q, x.mask),
          Sequence(k, x.mask),
          Sequence(val, x.mask),
      )

    self._project_qkv_fn = MethodType(_project_qkv, self)
    return self

  @classmethod
  def from_config(cls, config: Any) -> 'DotProductSelfAttention':
    """Create from a Linen DotProductSelfAttention.Config."""
    mlx_config = cls.Config(
        num_heads=config.num_heads,
        units_per_head=config.units_per_head,
        max_past_horizon=config.max_past_horizon,
        max_future_horizon=config.max_future_horizon,
        num_kv_heads=config.num_kv_heads,
        attention_probabilities_dropout_rate=config.attention_probabilities_dropout_rate,
        broadcast_dropout_across_queries=config.broadcast_dropout_across_queries,
        use_bias=config.use_bias,
        input_projection=_map_projection_config(config.input_projection),
        query_network=config.query_network,
        key_network=config.key_network,
        value_network=config.value_network,
        attention_logits_soft_cap=config.attention_logits_soft_cap,
        per_dim_scale=config.per_dim_scale,
        query_scale=config.query_scale,
        zero_fully_masked=config.zero_fully_masked,
        compute_dtype=config.compute_dtype,
        param_dtype=config.param_dtype or mx.float32,
        num_sink_embeddings=config.num_sink_embeddings,
        use_sink_scalars=config.use_sink_scalars,
        use_kv_cache_ringbuffer=config.use_kv_cache_ringbuffer,
        name=config.name,
    )
    return cls(mlx_config)


def _map_projection_config(
    config: attention_spec.QueryKeyValueProjectionConfig,
) -> projection_configs.QueryKeyValueProjectionConfig:
  """Maps a spec-level projection config (which may be JAX) to MLX."""
  if isinstance(config, attention_spec.CombinedQueryKeyValueProjection):
    return projection_configs.CombinedQueryKeyValueProjection(
        share_kv_projection=config.share_kv_projection,
        qkv_kernel_init=getattr(config, 'qkv_kernel_init', None),
        bias_init=getattr(config, 'bias_init', None),
    )
  if isinstance(config, attention_spec.SeparateQueryKeyValueProjection):
    return projection_configs.SeparateQueryKeyValueProjection(
        q_kernel_init=getattr(config, 'q_kernel_init', None),
        k_kernel_init=getattr(config, 'k_kernel_init', None),
        v_kernel_init=getattr(config, 'v_kernel_init', None),
        bias_init=getattr(config, 'bias_init', None),
    )
  if isinstance(config, attention_spec.QueryAndKeyValueProjection):
    return projection_configs.QueryAndKeyValueProjection(
        q_kernel_init=getattr(config, 'q_kernel_init', None),
        q_bias_init=getattr(config, 'q_bias_init', None),
        kv_kernel_init=getattr(config, 'kv_kernel_init', None),
        kv_bias_init=getattr(config, 'kv_bias_init', None),
    )
  if isinstance(config, attention_spec.QueryAndSharedKeyValueProjection):
    return projection_configs.QueryAndSharedKeyValueProjection(
        q_kernel_init=getattr(config, 'q_kernel_init', None),
        q_bias_init=getattr(config, 'q_bias_init', None),
        kv_kernel_init=getattr(config, 'kv_kernel_init', None),
        kv_bias_init=getattr(config, 'kv_bias_init', None),
    )
  return cast(Any, config)


class DotProductAttention(
    types.Emitting,
    attention_spec.DotProductAttention[types.Sequence, types.ChannelSpec],
):
  """Multi-headed dot-product cross attention for MLX."""

  @dataclasses.dataclass(frozen=True)
  class Config(
      types.SequenceLayerConfig,
      attention_spec.DotProductAttention.Config,
  ):
    """MLX-native configuration for DotProductAttention."""

    source_name: str
    num_heads: int
    units_per_head: int
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    input_projection: projection_configs.QueryKeyValueProjectionConfig = (
        dataclasses.field(
            default_factory=projection_configs.QueryAndKeyValueProjection
        )
    )
    query_network: Any = None
    key_network: Any = None
    value_network: Any = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    name: str | None = None

    @override
    def make(self) -> 'DotProductAttention':
      return DotProductAttention(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      in_features: int | None = None,
      source_features: int | None = None,
      source_name: str | None = None,
      num_heads: int | None = None,
      units_per_head: int | None = None,
      use_bias: bool = False,
      query_scale: float | None = None,
      per_dim_scale: bool = False,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
  ):
    super().__init__()
    if config is None:
      if source_name is None or num_heads is None or units_per_head is None:
        raise ValueError(
            'Must provide either config or source_name, num_heads, and'
            ' units_per_head'
        )
      source_name = cast(str, source_name)
      num_heads = cast(int, num_heads)
      units_per_head = cast(int, units_per_head)
      # Reconstruct config to store unified properties
      config = DotProductAttention.Config(
          source_name=source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          use_bias=use_bias,
          query_scale=query_scale,
          per_dim_scale=per_dim_scale,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
      )
    self.config = config

    self.compute_dtype = (
        init_mapping.to_mx_dtype(self.config.compute_dtype)
        if self.config.compute_dtype is not None
        else None
    )
    self._param_dtype = (
        init_mapping.to_mx_dtype(self.config.param_dtype) or mx.float32
    )

    self.in_features = None
    self.source_features = None
    self.source_name = self.config.source_name
    self.num_heads = self.config.num_heads
    self.units_per_head = self.config.units_per_head
    self.use_bias = self.config.use_bias
    self._query_scale = self.config.query_scale

    self._kernel_init = kernel_init
    self._bias_init = bias_init
    self._per_dim_scale = None

    self.query_network: Any = query_network or self.config.query_network
    self.key_network: Any = key_network or self.config.key_network
    self.value_network: Any = value_network or self.config.value_network

    self._initialized = False

    if in_features is not None and source_features is not None:
      self._ensure_initialized(in_features, source_features)

  def _ensure_initialized(self, in_features: int, source_features: int):
    """Ensure parameters and submodules are dynamically initialized."""
    if self._initialized:
      return
    self._initialized = True
    self.in_features = in_features
    self.source_features = source_features

    # pylint: disable=import-outside-toplevel
    from sequence_layers.mlx import utils as mlx_utils

    if hasattr(self.query_network, 'make'):
      self.query_network = mlx_utils.make_layer(self.query_network)
    if hasattr(self.key_network, 'make'):
      self.key_network = mlx_utils.make_layer(self.key_network)
    if hasattr(self.value_network, 'make'):
      self.value_network = mlx_utils.make_layer(self.value_network)

    param_dtype = self._param_dtype
    per_dim_scale = self.config.per_dim_scale
    units_per_head = self.units_per_head
    num_heads = self.num_heads
    use_bias = self.use_bias
    input_projection = self.config.input_projection

    self._per_dim_scale = (
        mx.zeros((units_per_head,), dtype=param_dtype)
        if per_dim_scale
        else None
    )

    kernel_init = self._kernel_init
    bias_init = self._bias_init

    if kernel_init is None:
      qkv_init = (
          getattr(input_projection, 'qkv_kernel_init', None)
          or getattr(input_projection, 'q_kernel_init', None)
          or getattr(input_projection, 'kv_kernel_init', None)
      )
      if qkv_init is not None:
        kernel_init = init_mapping.map_initializer(qkv_init)
      else:
        kernel_init = init_mapping.make_variance_scaling_init(
            'fan_in', 'truncated_normal'
        )

    if bias_init is None:
      qkv_bias_init = (
          getattr(input_projection, 'bias_init', None)
          or getattr(input_projection, 'q_bias_init', None)
          or getattr(input_projection, 'kv_bias_init', None)
      )
      if qkv_bias_init is not None:
        bias_init = init_mapping.map_initializer(qkv_bias_init)
      else:
        bias_init = init_mapping.zeros_init

    key = mx.random.key(0)
    qkv_dim = num_heads * units_per_head

    self.q_proj = kernel_init(key, (in_features, qkv_dim), param_dtype)
    self.kv_proj = mx.concatenate(
        [
            kernel_init(key, (source_features, qkv_dim), param_dtype),
            kernel_init(key, (source_features, qkv_dim), param_dtype),
        ],
        axis=-1,
    )
    if use_bias:
      self.q_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.kv_bias = mx.concatenate(
          [
              bias_init(key, (qkv_dim,), param_dtype),
              bias_init(key, (qkv_dim,), param_dtype),
          ],
          axis=-1,
      )

  @property
  @override
  def supports_step(self):
    if self.query_network is not None:
      return self.query_network.supports_step
    return True

  @property
  @override
  def input_latency(self):
    return 0

  def _project_q(self, x):
    """Project input query sequence."""
    b, t = x.shape[0], x.shape[1]
    dtype = self.compute_dtype or x.dtype
    v = x.values.astype(dtype)
    q = mx.matmul(v, self.q_proj.astype(dtype))
    if self.use_bias:
      q = q + self.q_bias.astype(dtype)
    q = q.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(q, x.mask)

  def _project_kv(self, source):
    """Project external source sequence to key/value matrices."""
    b, t = source.shape[0], source.shape[1]
    dtype = self.compute_dtype or source.dtype
    v = source.values.astype(dtype)
    kv = mx.matmul(v, self.kv_proj.astype(dtype))
    k, val = mx.split(kv, 2, axis=-1)
    if self.use_bias:
      kv_bias = self.kv_bias.astype(dtype)
      kb, vb = mx.split(kv_bias, 2, axis=-1)
      k = k + kb
      val = val + vb
    k = k.reshape(b, t, self.num_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(k, source.mask), Sequence(val, source.mask)

  def _get_source(self, constants):
    """Helper to resolve the external source sequence from constants."""
    if constants is None or self.source_name not in constants:
      raise ValueError(f'Source "{self.source_name}" not found in constants.')
    return constants[self.source_name]

  def _compute_attention(
      self, queries, keys, values, mask, emit_attention_weights=False
  ):
    """Compute scaled dot-product attention (no causal mask)."""
    q = mx.transpose(queries, (0, 2, 1, 3))
    k = mx.transpose(keys, (0, 2, 1, 3))
    v = mx.transpose(values, (0, 2, 1, 3))

    q = _scale_queries(
        q, self._per_dim_scale, self._query_scale, self.units_per_head
    )

    if not emit_attention_weights:
      context = mx.fast.scaled_dot_product_attention(
          q, k, v, scale=1.0, mask=mask
      )
      return mx.transpose(context, (0, 2, 1, 3)), ()

    # Manual path for emits
    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)

    logits_f32 = (
        logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    )
    weights = mx.softmax(logits_f32, axis=-1).astype(v.dtype)
    context = mx.matmul(weights, v)
    context = mx.transpose(context, (0, 2, 1, 3))

    emits = mx.transpose(weights, (0, 2, 1, 3))
    return context, emits

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Returns the output shape of the layer's features."""
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    """Returns the computation output dtype."""
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @override
  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    """Computes and returns the initial cache states for cross attention."""
    source = self._get_source(constants)
    self._ensure_initialized(input_spec.shape[-1], source.shape[-1])

    keys, values = self._project_kv(source)

    if self.key_network is not None:
      keys = self.key_network.layer(
          keys, training=training, constants=constants
      )
    if self.value_network is not None:
      values = self.value_network.layer(
          values, training=training, constants=constants
      )

    keys = keys.mask_invalid()
    values = values.mask_invalid()

    q_net_state = (
        self.query_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_heads, self.units_per_head),
                self.get_output_dtype(input_spec.dtype),
            ),
            training=training,
            constants=constants,
        )
        if self.query_network is not None
        else ()
    )

    time_step = mx.zeros((batch_size,), dtype=mx.int32)
    return (
        keys.values,
        values.values,
        keys.mask,
        q_net_state,
        time_step,
    )

  @override
  def layer_with_emits(self, x, *, training: bool, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])

    keys, values = self._project_kv(source)

    if self.key_network is not None:
      keys = self.key_network.layer(
          keys, training=training, constants=constants
      )
    if self.value_network is not None:
      values = self.value_network.layer(
          values, training=training, constants=constants
      )

    queries = self._project_q(x)
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(
              queries, training=training, constants=constants
          ).values,
          queries.mask,
      )

    values = values.mask_invalid()
    valid_mask = source.mask[:, None, None, :]
    context, probs = self._compute_attention(
        queries.values,
        keys.values,
        values.values,
        valid_mask,
        emit_attention_weights=self.config.emit_attention_weights,
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = CrossAttentionEmits({self.source_name: Sequence(probs, x.mask)})
    return Sequence(context, x.mask), emits

  @override
  def step_with_emits(self, x, state: Any, *, training: bool, constants=None):
    keys_v, values_v, kv_mask, q_net_state, time_step = state
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])

    queries = self._project_q(x)
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, training=training, constants=constants
      )

    valid_mask = kv_mask[:, None, None, :]
    context, probs = self._compute_attention(
        queries.values,
        keys_v,
        values_v,
        valid_mask,
        emit_attention_weights=self.config.emit_attention_weights,
    )

    new_state = (
        keys_v,
        values_v,
        kv_mask,
        q_net_state,
        time_step + x.shape[1],
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = CrossAttentionEmits({self.source_name: Sequence(probs, x.mask)})
    return Sequence(context, x.mask), new_state, emits

  @classmethod
  def from_config(cls, config: Any) -> 'DotProductAttention':
    """Create from a Linen DotProductAttention.Config."""
    mlx_config = cls.Config(
        source_name=config.source_name,
        num_heads=config.num_heads,
        units_per_head=config.units_per_head,
        attention_probabilities_dropout_rate=config.attention_probabilities_dropout_rate,
        broadcast_dropout_across_queries=config.broadcast_dropout_across_queries,
        use_bias=config.use_bias,
        input_projection=_map_projection_config(config.input_projection),
        query_network=config.query_network,
        key_network=config.key_network,
        value_network=config.value_network,
        attention_logits_soft_cap=config.attention_logits_soft_cap,
        per_dim_scale=config.per_dim_scale,
        query_scale=config.query_scale,
        zero_fully_masked=config.zero_fully_masked,
        compute_dtype=config.compute_dtype,
        param_dtype=config.param_dtype or mx.float32,
        name=config.name,
    )
    return cls(mlx_config)


def _banded_mask(q_len, kv_len, num_lower, num_upper):
  """Build a [1, 1, q_len, kv_len] banded visibility mask.

  Position (i, j) is True iff j >= i - num_lower and j <= i + num_upper.
  """
  row = mx.expand_dims(mx.arange(q_len), axis=1)
  col = mx.expand_dims(mx.arange(kv_len), axis=0)
  mask = (col >= row - num_lower) & (col <= row + num_upper)
  return mask.reshape(1, 1, q_len, kv_len)


def _step_visibility_mask(
    max_past_horizon, max_future_horizon, query_time, key_time
):
  """Compute step-wise banded visibility mask.

  For a single query (query_time=1), returns None since no causal mask
  is needed — the KV buffer already contains only visible positions.

  For multi-step queries, returns a banded matrix with num_lower=0 and
  num_upper=max_past_horizon + max_future_horizon.
  """
  if query_time == 1:
    return None
  return _banded_mask(
      query_time,
      key_time,
      num_lower=0,
      num_upper=max_past_horizon + max_future_horizon,
  )


class StreamingDotProductAttention(
    types.Emitting,
    attention_spec.StreamingDotProductAttention[
        types.Sequence, types.ChannelSpec
    ],
):
  """Multi-headed streaming cross-attention for MLX.

  Also covers StreamingLocalDotProductAttention from the JAX backend.

  Queries come from the input; keys and values come from a source
  sequence provided in constants at the same streaming rate as input.

  Unlike DotProductAttention (which pre-projects the full source in
  get_initial_state), this class projects source chunks per-step and
  maintains a rolling KV buffer, enabling streaming cross-attention.

  Covers both StreamingDotProductAttention and
  StreamingLocalDotProductAttention from the JAX backend (which differ
  only in layer-mode efficiency, not in step-mode behavior or output).

  Kernels stored in Linen-compatible shapes:
    q_proj: [in_features, num_heads * units_per_head]
    k_proj: [source_features, num_heads * units_per_head]
    v_proj: [source_features, num_heads * units_per_head]
  """

  @dataclasses.dataclass(frozen=True)
  class Config(
      types.SequenceLayerConfig,
      attention_spec.StreamingDotProductAttention.Config,
  ):
    """MLX-native configuration for StreamingDotProductAttention.

    This Config also serves as the MLX-native equivalent of the JAX
    StreamingLocalDotProductAttention.Config.
    """

    source_name: str
    num_heads: int
    units_per_head: int
    block_size: int = 1
    max_past_horizon: int = 1
    max_future_horizon: int = 0
    attention_probabilities_dropout_rate: float = 0.0
    broadcast_dropout_across_queries: bool = False
    use_bias: bool = False
    use_query_delay_buffer: bool = True
    input_projection: projection_configs.QueryKeyValueProjectionConfig = (
        dataclasses.field(
            default_factory=projection_configs.QueryAndKeyValueProjection
        )
    )
    query_network: Any = None
    key_network: Any = None
    value_network: Any = None
    attention_logits_soft_cap: float | None = None
    per_dim_scale: bool = False
    query_scale: float | None = None
    zero_fully_masked: bool = False
    compute_dtype: types.DType | None = None
    param_dtype: types.DType = mx.float32
    num_sink_embeddings: int = 0
    use_sink_scalars: bool = False
    use_kv_cache_ringbuffer: bool = False
    name: str | None = None

    @override
    def make(self) -> 'StreamingDotProductAttention':
      return StreamingDotProductAttention(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      in_features: int | None = None,
      source_features: int | None = None,
      source_name: str | None = None,
      num_heads: int | None = None,
      units_per_head: int | None = None,
      max_past_horizon: int | None = None,
      max_future_horizon: int = 0,
      use_bias: bool = False,
      use_query_delay_buffer: bool = True,
      query_scale: float | None = None,
      per_dim_scale: bool = False,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      num_sink_embeddings: int = 0,
      input_projection=None,
  ):
    super().__init__()
    if config is None:
      if (
          source_name is None
          or num_heads is None
          or units_per_head is None
          or max_past_horizon is None
      ):
        raise ValueError(
            'Must provide either config or source_name, num_heads, '
            'units_per_head, and max_past_horizon'
        )
      source_name = cast(str, source_name)
      num_heads = cast(int, num_heads)
      units_per_head = cast(int, units_per_head)
      max_past_horizon = cast(int, max_past_horizon)
      input_projection_val = (
          input_projection or projection_configs.QueryAndKeyValueProjection()
      )
      config = StreamingDotProductAttention.Config(
          source_name=source_name,
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_past_horizon=max_past_horizon,
          max_future_horizon=max_future_horizon,
          use_bias=use_bias,
          use_query_delay_buffer=use_query_delay_buffer,
          query_scale=query_scale,
          per_dim_scale=per_dim_scale,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
          num_sink_embeddings=num_sink_embeddings,
          input_projection=input_projection_val,
      )
    self.config = config

    if self.config.max_past_horizon < 1:
      raise ValueError(
          f'max_past_horizon must be >= 1, got {self.config.max_past_horizon}.'
      )
    if self.config.max_future_horizon < 0:
      raise ValueError(
          'max_future_horizon must be >= 0, got'
          f' {self.config.max_future_horizon}.'
      )

    self.compute_dtype = (
        init_mapping.to_mx_dtype(self.config.compute_dtype)
        if self.config.compute_dtype is not None
        else None
    )
    self._param_dtype = (
        init_mapping.to_mx_dtype(self.config.param_dtype) or mx.float32
    )

    self.in_features = None
    self.source_features = None
    self.source_name = self.config.source_name
    self.num_heads = self.config.num_heads
    self.units_per_head = self.config.units_per_head
    self.max_past_horizon = self.config.max_past_horizon
    self.max_future_horizon = self.config.max_future_horizon
    self.use_bias = self.config.use_bias
    self.use_query_delay_buffer = self.config.use_query_delay_buffer
    self._query_scale = self.config.query_scale

    self._kernel_init = kernel_init
    self._bias_init = bias_init
    self._per_dim_scale = None

    self.query_network: Any = query_network or self.config.query_network
    self.key_network: Any = key_network or self.config.key_network
    self.value_network: Any = value_network or self.config.value_network

    self.num_sink_embeddings = self.config.num_sink_embeddings
    self.sink_key_embeddings: Any = None
    self.sink_value_embeddings: Any = None

    self.q_proj: Any = None
    self.kv_proj: Any = None
    self.q_proj_qw: Any = None
    self.q_proj_qs: Any = None
    self.q_proj_qb: Any = None
    self.kv_proj_qw: Any = None
    self.kv_proj_qs: Any = None
    self.kv_proj_qb: Any = None
    self._quant_group_size: int | None = None
    self._quant_bits: int | None = None
    self._project_q_fn: Any = None
    self._project_kv_fn: Any = None

    self._initialized = False

    if in_features is not None and source_features is not None:
      self._ensure_initialized(in_features, source_features)

  def _ensure_initialized(self, in_features: int, source_features: int):
    """Ensure parameters and submodules are dynamically initialized."""
    if self._initialized:
      return
    self._initialized = True
    self.in_features = in_features
    self.source_features = source_features

    # pylint: disable=import-outside-toplevel
    from sequence_layers.mlx import utils as mlx_utils

    if hasattr(self.query_network, 'make'):
      self.query_network = mlx_utils.make_layer(self.query_network)
    if hasattr(self.key_network, 'make'):
      self.key_network = mlx_utils.make_layer(self.key_network)
    if hasattr(self.value_network, 'make'):
      self.value_network = mlx_utils.make_layer(self.value_network)

    param_dtype = self._param_dtype
    per_dim_scale = self.config.per_dim_scale
    units_per_head = self.units_per_head
    num_heads = self.num_heads
    use_bias = self.use_bias
    input_projection = self.config.input_projection
    num_sink_embeddings = self.num_sink_embeddings

    self._per_dim_scale = (
        mx.zeros((units_per_head,), dtype=param_dtype)
        if per_dim_scale
        else None
    )

    kernel_init = self._kernel_init
    bias_init = self._bias_init

    if kernel_init is None:
      qkv_init = (
          getattr(input_projection, 'qkv_kernel_init', None)
          or getattr(input_projection, 'q_kernel_init', None)
          or getattr(input_projection, 'kv_kernel_init', None)
      )
      if qkv_init is not None:
        kernel_init = init_mapping.map_initializer(qkv_init)
      else:
        kernel_init = init_mapping.make_variance_scaling_init(
            'fan_in', 'truncated_normal'
        )

    if bias_init is None:
      qkv_bias_init = (
          getattr(input_projection, 'bias_init', None)
          or getattr(input_projection, 'q_bias_init', None)
          or getattr(input_projection, 'kv_bias_init', None)
      )
      if qkv_bias_init is not None:
        bias_init = init_mapping.map_initializer(qkv_bias_init)
      else:
        bias_init = init_mapping.zeros_init

    key = mx.random.key(0)
    qkv_dim = num_heads * units_per_head

    self.q_proj = kernel_init(key, (in_features, qkv_dim), param_dtype)
    self.kv_proj = mx.concatenate(
        [
            kernel_init(key, (source_features, qkv_dim), param_dtype),
            kernel_init(key, (source_features, qkv_dim), param_dtype),
        ],
        axis=-1,
    )
    if use_bias:
      self.q_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.kv_bias = mx.concatenate(
          [
              bias_init(key, (qkv_dim,), param_dtype),
              bias_init(key, (qkv_dim,), param_dtype),
          ],
          axis=-1,
      )

    if num_sink_embeddings > 0:
      self.sink_key_embeddings = mx.zeros(
          (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
      )
      self.sink_value_embeddings = mx.zeros(
          (num_sink_embeddings, num_heads, units_per_head), dtype=param_dtype
      )

  @property
  @override
  def supports_step(self):
    supports = True
    if self.query_network is not None:
      supports = supports and self.query_network.supports_step
    if self.key_network is not None:
      supports = supports and self.key_network.supports_step
    if self.value_network is not None:
      supports = supports and self.value_network.supports_step
    return supports

  @property
  @override
  def input_latency(self):
    if self.max_future_horizon > 0 and self.use_query_delay_buffer:
      return self.max_future_horizon
    return 0

  def _project_q(self, x):
    """Project input to query sequence."""
    b, t = x.shape[0], x.shape[1]
    dtype = self.compute_dtype or x.dtype
    v = x.values.astype(dtype)
    q = mx.matmul(v, self.q_proj.astype(dtype))
    if self.use_bias:
      q = q + self.q_bias.astype(dtype)
    q = q.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(q, x.mask)

  def _project_kv(self, source):
    """Project source to key/value sequences."""
    b, t = source.shape[0], source.shape[1]
    dtype = self.compute_dtype or source.dtype
    v = source.values.astype(dtype)
    kv = mx.matmul(v, self.kv_proj.astype(dtype))
    k, val = mx.split(kv, 2, axis=-1)
    if self.use_bias:
      kv_bias = self.kv_bias.astype(dtype)
      kb, vb = mx.split(kv_bias, 2, axis=-1)
      k = k + kb
      val = val + vb
    k = k.reshape(b, t, self.num_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(k, source.mask), Sequence(val, source.mask)

  def _get_source(self, constants):
    """Helper to resolve the external source sequence from constants."""
    if constants is None or self.source_name not in constants:
      raise ValueError(f'Source "{self.source_name}" not found in constants.')
    return constants[self.source_name]

  def _compute_attention(
      self, queries, keys, values, mask, emit_attention_weights=False
  ):
    """Compute scaled dot-product attention."""
    q = mx.transpose(queries, (0, 2, 1, 3))
    k = mx.transpose(keys, (0, 2, 1, 3))
    v = mx.transpose(values, (0, 2, 1, 3))

    q = _scale_queries(
        q, self._per_dim_scale, self._query_scale, self.units_per_head
    )

    if not emit_attention_weights:
      if self.sink_key_embeddings is not None:
        # JAX computes sink logits with *unscaled* queries.  Pre-divide
        # sink keys by the scale so that SDPA produces equivalent logits:
        #   scaled_q @ (sink_k / scale) == unscaled_q @ sink_k
        scale_vec = _query_scale_vector(
            self._per_dim_scale,
            self._query_scale,
            self.units_per_head,
            q.dtype,
        )
        sink_k = self.sink_key_embeddings.astype(q.dtype) / scale_vec
        sink_v = self.sink_value_embeddings.astype(v.dtype)

        sink_k_b = mx.broadcast_to(
            mx.transpose(sink_k, (1, 0, 2))[None],
            (q.shape[0], self.num_heads, sink_k.shape[0], self.units_per_head),
        )
        sink_v_b = mx.broadcast_to(
            mx.transpose(sink_v, (1, 0, 2))[None],
            (v.shape[0], self.num_heads, sink_v.shape[0], self.units_per_head),
        )

        k = mx.concatenate([sink_k_b, k], axis=2)
        v = mx.concatenate([sink_v_b, v], axis=2)

        if mask is not None:
          num_sinks = self.sink_key_embeddings.shape[0]
          sink_mask = mx.ones(
              (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
              dtype=mx.bool_,
          )
          mask = mx.concatenate([sink_mask, mask], axis=-1)

      context = mx.fast.scaled_dot_product_attention(
          q, k, v, scale=1.0, mask=mask
      )
      return mx.transpose(context, (0, 2, 1, 3)), ()

    # Manual path for emits
    sink_logits = None
    if self.sink_key_embeddings is not None:
      sink_k = self.sink_key_embeddings.astype(q.dtype)
      sink_k_t = mx.transpose(sink_k, (1, 2, 0))
      sink_logits = mx.matmul(q, sink_k_t)

    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

    if self.sink_key_embeddings is not None:
      sink_v = self.sink_value_embeddings.astype(v.dtype)
      sink_v_t = mx.transpose(sink_v, (1, 0, 2))
      sink_v_b = mx.broadcast_to(sink_v_t[None], (v.shape[0],) + sink_v_t.shape)
      v = mx.concatenate([sink_v_b, v], axis=2)
      assert sink_logits is not None
      logits = mx.concatenate([sink_logits, logits], axis=-1)
      if mask is not None:
        num_sinks = self.sink_key_embeddings.shape[0]
        sink_mask = mx.ones(
            (mask.shape[0], mask.shape[1], mask.shape[2], num_sinks),
            dtype=mx.bool_,
        )
        mask = mx.concatenate([sink_mask, mask], axis=-1)

    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)

    logits_f32 = (
        logits.astype(mx.float32) if logits.dtype != mx.float32 else logits
    )
    weights = mx.softmax(logits_f32, axis=-1).astype(v.dtype)
    context = mx.matmul(weights, v)
    context = mx.transpose(context, (0, 2, 1, 3))

    emits = mx.transpose(weights, (0, 2, 1, 3))
    return context, emits

  @override
  def get_output_shape(self, input_shape, *, constants=None):
    """Returns the feature output shape."""
    if len(input_shape) != 1:
      raise ValueError(
          'StreamingDotProductAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  @override
  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  @override
  def get_initial_state(
      self, batch_size, input_spec, *, training: bool, constants=None
  ):
    source = self._get_source(constants)
    self._ensure_initialized(input_spec.shape[-1], source.shape[-1])

    compute_dtype = self.get_output_dtype(input_spec.dtype)
    max_past = max(0, self.max_past_horizon)
    max_future = max(0, self.max_future_horizon)
    kv_buffer_size = max_past + max_future

    kv_shape = (
        batch_size,
        kv_buffer_size,
        self.num_heads,
        self.units_per_head,
    )
    kv_buffer_keys = mx.zeros(kv_shape, dtype=compute_dtype)
    kv_buffer_values = mx.zeros(kv_shape, dtype=compute_dtype)
    kv_buffer_mask = mx.zeros((batch_size, kv_buffer_size), dtype=mx.bool_)
    time_step = mx.zeros((batch_size,), dtype=mx.int32)

    # Q/K/V network states.
    q_net_state = (
        self.query_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.query_network is not None
        else ()
    )
    k_net_state = (
        self.key_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.key_network is not None
        else ()
    )
    v_net_state = (
        self.value_network.get_initial_state(
            batch_size,
            types.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            training=training,
            constants=constants,
        )
        if self.value_network is not None
        else ()
    )

    # Query delay buffer for future horizon.
    if max_future and self.use_query_delay_buffer:
      q_delay_values = mx.zeros(
          (
              batch_size,
              max_future,
              self.num_heads,
              self.units_per_head,
          ),
          dtype=compute_dtype,
      )
      q_delay_mask = mx.zeros((batch_size, max_future), dtype=mx.bool_)
    else:
      q_delay_values = ()
      q_delay_mask = ()

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    )

  @override
  def layer_with_emits(self, x, *, training: bool, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])

    proj_q = self._project_q_fn or self._project_q
    queries = proj_q(x)
    proj_kv = self._project_kv_fn or self._project_kv
    keys, values = proj_kv(source)
    queries_time = queries.shape[1]
    keys_time = keys.shape[1]

    # Optional Q/K/V processing networks.
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(
              queries, training=training, constants=constants
          ).values,
          queries.mask,
      )
    if self.key_network is not None:
      keys = Sequence(
          self.key_network.layer(
              keys, training=training, constants=constants
          ).values,
          keys.mask,
      )
    if self.value_network is not None:
      values = Sequence(
          self.value_network.layer(
              values, training=training, constants=constants
          ).values,
          values.mask,
      )

    # Mask invalid values.
    values = values.mask_invalid()

    # Build visibility mask: banded + source validity.
    valid_mask = source.mask[:, None, None, :]
    banded = _banded_mask(
        queries_time,
        keys_time,
        num_lower=self.max_past_horizon,
        num_upper=self.max_future_horizon,
    )
    valid_mask = valid_mask & banded

    context, probs = self._compute_attention(
        queries.values,
        keys.values,
        values.values,
        valid_mask,
        emit_attention_weights=self.config.emit_attention_weights,
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = CrossAttentionEmits({self.source_name: Sequence(probs, x.mask)})
    return Sequence(context, x.mask), emits

  @override
  def step_with_emits(self, x, state: Any, *, training: bool, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])

    if x.shape[1] != source.shape[1]:
      raise ValueError(
          f'Expected x.shape[1]={x.shape[1]} to match'
          f' source.shape[1]={source.shape[1]}'
      )

    (
        kv_buf_k,
        kv_buf_v,
        kv_buf_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    ) = state

    kv_buffer_size = kv_buf_k.shape[1]
    x_time = x.shape[1]

    proj_q = self._project_q_fn or self._project_q
    queries = proj_q(x)
    proj_kv = self._project_kv_fn or self._project_kv
    keys, values = proj_kv(source)

    # Optional Q/K/V processing networks.
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, training=training, constants=constants
      )
    if self.key_network is not None:
      keys, k_net_state = self.key_network.step(
          keys, k_net_state, training=training, constants=constants
      )
    if self.value_network is not None:
      values, v_net_state = self.value_network.step(
          values, v_net_state, training=training, constants=constants
      )

    # Mask invalid values.
    values = values.mask_invalid()

    # Concatenate new K/V to buffer.
    new_k = mx.concatenate([kv_buf_k, keys.values], axis=1)
    new_v = mx.concatenate([kv_buf_v, values.values], axis=1)
    new_mask = mx.concatenate([kv_buf_mask, source.mask], axis=1)

    # Handle query delay buffer.
    has_delay_buffer = not isinstance(q_delay_values, tuple)
    if has_delay_buffer:
      # Insert new queries into delay buffer.
      all_q_values = mx.concatenate([q_delay_values, queries.values], axis=1)
      all_q_mask = mx.concatenate([q_delay_mask, queries.mask], axis=1)
      # Pop oldest x_time queries as current.
      queries = Sequence(all_q_values[:, :x_time], all_q_mask[:, :x_time])
      # Preserve remaining for next step.
      q_delay_values = all_q_values[:, -self.max_future_horizon :]
      q_delay_mask = all_q_mask[:, -self.max_future_horizon :]

    # Build visibility mask.
    kv_time = new_k.shape[1]
    valid_mask = new_mask[:, None, None, :]

    vis_mask = _step_visibility_mask(
        self.max_past_horizon,
        self.max_future_horizon,
        x_time,
        kv_time,
    )
    if vis_mask is not None:
      valid_mask = valid_mask & vis_mask

    context, probs = self._compute_attention(
        queries.values,
        new_k,
        new_v,
        valid_mask,
        emit_attention_weights=self.config.emit_attention_weights,
    )

    # Trim KV buffer to keep only last kv_buffer_size entries.
    new_k = new_k[:, -kv_buffer_size:]
    new_v = new_v[:, -kv_buffer_size:]
    new_mask = new_mask[:, -kv_buffer_size:]

    new_state = (
        new_k,
        new_v,
        new_mask,
        time_step + x_time,
        q_net_state,
        k_net_state,
        v_net_state,
        q_delay_values,
        q_delay_mask,
    )
    emits = ()
    if self.config.emit_attention_weights:
      emits = CrossAttentionEmits(
          {self.source_name: Sequence(probs, queries.mask)}
      )
    return Sequence(context, queries.mask), new_state, emits

  def to_quantized(
      self, group_size: int = 64, bits: int = 4, mode: str = 'affine'
  ):
    """Convert projection layers to quantized equivalents."""
    del mode  # Unused in MLX quantize
    if (
        getattr(self, 'q_proj', None) is None
        or self.q_proj.shape[0] % group_size != 0
    ):
      return self

    self._quant_group_size = group_size
    self._quant_bits = bits

    w_q = self.q_proj.T
    self.q_proj_qw, self.q_proj_qs, self.q_proj_qb = mx.quantize(
        w_q, group_size=group_size, bits=bits
    )

    # kv_proj is already combined [source, 2*qkv_dim].
    w_kv = self.kv_proj.T
    self.kv_proj_qw, self.kv_proj_qs, self.kv_proj_qb = mx.quantize(
        w_kv, group_size=group_size, bits=bits
    )

    self.q_proj = cast(Any, None)
    self.kv_proj = cast(Any, None)

    def _project_q(self, x):
      b, t = x.shape[0], x.shape[1]
      dtype = self.compute_dtype or x.dtype
      v = x.values.astype(dtype)
      q = _quantized_matmul_proj(
          v,
          self.q_proj_qw,
          self.q_proj_qs,
          self.q_proj_qb,
          self._quant_group_size,
          self._quant_bits,
      )
      if self.use_bias:
        q = q + self.q_bias.astype(dtype)
      q = q.reshape(b, t, self.num_heads, self.units_per_head)
      return Sequence(q, x.mask)

    def _project_kv(self, source):
      b, t = source.shape[0], source.shape[1]
      dtype = self.compute_dtype or source.dtype
      v = source.values.astype(dtype)
      kv = _quantized_matmul_proj(
          v,
          self.kv_proj_qw,
          self.kv_proj_qs,
          self.kv_proj_qb,
          self._quant_group_size,
          self._quant_bits,
      )
      k, val = mx.split(kv, 2, axis=-1)
      if self.use_bias:
        kv_bias = self.kv_bias.astype(dtype)
        kb, vb = mx.split(kv_bias, 2, axis=-1)
        k = k + kb
        val = val + vb
      k = k.reshape(b, t, self.num_heads, self.units_per_head)
      val = val.reshape(b, t, self.num_heads, self.units_per_head)
      return Sequence(k, source.mask), Sequence(val, source.mask)

    self._project_q_fn = MethodType(_project_q, self)
    self._project_kv_fn = MethodType(_project_kv, self)

    return self

  @classmethod
  def from_config(cls, config: Any) -> 'StreamingDotProductAttention':
    """Create from a Linen StreamingDotProductAttention.Config."""
    mlx_config = cls.Config(
        source_name=config.source_name,
        num_heads=config.num_heads,
        units_per_head=config.units_per_head,
        block_size=getattr(config, 'block_size', 1),
        max_past_horizon=config.max_past_horizon,
        max_future_horizon=config.max_future_horizon,
        attention_probabilities_dropout_rate=config.attention_probabilities_dropout_rate,
        broadcast_dropout_across_queries=config.broadcast_dropout_across_queries,
        use_bias=config.use_bias,
        use_query_delay_buffer=getattr(config, 'use_query_delay_buffer', True),
        input_projection=_map_projection_config(config.input_projection),
        query_network=config.query_network,
        key_network=config.key_network,
        value_network=config.value_network,
        attention_logits_soft_cap=config.attention_logits_soft_cap,
        per_dim_scale=config.per_dim_scale,
        query_scale=config.query_scale,
        zero_fully_masked=config.zero_fully_masked,
        compute_dtype=config.compute_dtype,
        param_dtype=config.param_dtype or mx.float32,
        num_sink_embeddings=getattr(config, 'num_sink_embeddings', 0),
        use_sink_scalars=getattr(config, 'use_sink_scalars', False),
        use_kv_cache_ringbuffer=getattr(
            config, 'use_kv_cache_ringbuffer', False
        ),
        name=config.name,
    )
    return cls(mlx_config)


class LocalDotProductSelfAttention(
    DotProductSelfAttention,
    attention_spec.LocalDotProductSelfAttention[
        types.Sequence, types.ChannelSpec
    ],
):
  """Local dot-product self attention with configurable block_size."""

  @dataclasses.dataclass(frozen=True)
  class Config(
      DotProductSelfAttention.Config,
      attention_spec.LocalDotProductSelfAttention.Config,
  ):
    """MLX-native configuration for LocalDotProductSelfAttention."""

    block_size: int = 1

    @override
    def make(self) -> 'LocalDotProductSelfAttention':
      return LocalDotProductSelfAttention(self)

  def __init__(
      self,
      config: Config | None = None,
      *,
      in_features: int | None = None,
      num_heads: int | None = None,
      units_per_head: int | None = None,
      max_past_horizon: int | None = None,
      max_future_horizon: int = 0,
      num_kv_heads: int | None = None,
      use_bias: bool = False,
      query_scale: float | None = None,
      per_dim_scale: bool = False,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      attention_logits_soft_cap: float | None = None,
      num_sink_embeddings: int = 0,
      input_projection=None,
      block_size: int | None = None,
      block_size_config: int | None = None,
  ):
    if block_size is None:
      block_size = block_size_config if block_size_config is not None else 1

    if config is None:
      if (
          num_heads is None
          or units_per_head is None
          or max_past_horizon is None
      ):
        raise ValueError(
            'Must provide either config or num_heads, units_per_head, and'
            ' max_past_horizon'
        )
      num_heads = cast(int, num_heads)
      units_per_head = cast(int, units_per_head)
      max_past_horizon = cast(int, max_past_horizon)
      input_projection_val = (
          input_projection
          or projection_configs.CombinedQueryKeyValueProjection()
      )
      config = LocalDotProductSelfAttention.Config(
          num_heads=num_heads,
          units_per_head=units_per_head,
          max_past_horizon=max_past_horizon,
          max_future_horizon=max_future_horizon,
          num_kv_heads=num_kv_heads,
          use_bias=use_bias,
          query_scale=query_scale,
          per_dim_scale=per_dim_scale,
          compute_dtype=compute_dtype,
          param_dtype=param_dtype,
          query_network=query_network,
          key_network=key_network,
          value_network=value_network,
          attention_logits_soft_cap=attention_logits_soft_cap,
          num_sink_embeddings=num_sink_embeddings,
          input_projection=input_projection_val,
          block_size=block_size,
      )
    super().__init__(
        config,
        in_features=in_features,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )
    self._block_size_config = config.block_size

  @property
  @override
  def block_size(self):
    return self._block_size_config

  @classmethod
  @override
  def from_config(cls, config: Any) -> 'LocalDotProductSelfAttention':
    # pylint: disable=unexpected-keyword-arg
    mlx_config = LocalDotProductSelfAttention.Config(
        num_heads=config.num_heads,
        units_per_head=config.units_per_head,
        max_past_horizon=config.max_past_horizon,
        max_future_horizon=config.max_future_horizon,
        num_kv_heads=getattr(config, 'num_kv_heads', None),
        attention_probabilities_dropout_rate=config.attention_probabilities_dropout_rate,
        broadcast_dropout_across_queries=config.broadcast_dropout_across_queries,
        use_bias=config.use_bias,
        input_projection=_map_projection_config(config.input_projection),
        query_network=config.query_network,
        key_network=config.key_network,
        value_network=config.value_network,
        attention_logits_soft_cap=config.attention_logits_soft_cap,
        per_dim_scale=config.per_dim_scale,
        query_scale=config.query_scale,
        zero_fully_masked=config.zero_fully_masked,
        compute_dtype=config.compute_dtype,
        param_dtype=config.param_dtype or mx.float32,
        num_sink_embeddings=config.num_sink_embeddings,
        use_sink_scalars=config.use_sink_scalars,
        use_kv_cache_ringbuffer=config.use_kv_cache_ringbuffer,
        block_size=config.block_size,
        name=config.name,
    )
    return cls(mlx_config)


StreamingLocalDotProductAttention = StreamingDotProductAttention
