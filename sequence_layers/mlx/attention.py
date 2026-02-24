"""Dot-product attention layers for MLX."""

import math

import mlx.core as mx
import numpy as np

from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import init_mapping
from sequence_layers.mlx import types

Sequence = bt.Sequence
MaskedSequence = bt.MaskedSequence


def _causal_mask(q_len, kv_len):
  """Build a [1, 1, q_len, kv_len] causal mask (True = attend)."""
  # Each query at position i can attend to keys at positions
  # [kv_len - q_len, ..., kv_len - q_len + i].
  row = mx.arange(q_len)
  col = mx.arange(kv_len)
  # query i (global pos = kv_len - q_len + i) can see key j
  # if j <= kv_len - q_len + i.
  offset = kv_len - q_len
  mask = col[None, :] <= (row[:, None] + offset)
  return mask.reshape(1, 1, q_len, kv_len)


class DotProductSelfAttention(types.Emitting):
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

  def __init__(
      self,
      *,
      in_features: int,
      num_heads: int,
      units_per_head: int,
      max_past_horizon: int,
      max_future_horizon: int = 0,
      num_kv_heads: int | None = None,
      use_bias: bool = False,
      query_scale: float | None = None,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
      attention_logits_soft_cap: float | None = None,
  ):
    super().__init__()
    if num_kv_heads is None:
      num_kv_heads = num_heads
    if num_heads % num_kv_heads != 0:
      raise ValueError(f'{num_heads=} must be divisible by {num_kv_heads=}.')
    if max_past_horizon < -1:
      raise ValueError(
          f'max_past_horizon must be >= -1, got {max_past_horizon}.'
      )
    if max_future_horizon < -1:
      raise ValueError(
          f'max_future_horizon must be >= -1, got {max_future_horizon}.'
      )

    self.in_features = in_features
    self.num_heads = num_heads
    self.units_per_head = units_per_head
    self.max_past_horizon = max_past_horizon
    self.max_future_horizon = max_future_horizon
    self.num_kv_heads = num_kv_heads
    self.use_bias = use_bias
    self._query_scale = query_scale
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype
    self._attention_logits_soft_cap = attention_logits_soft_cap

    if kernel_init is None:
      kernel_init = init_mapping._make_variance_scaling_init(
          'fan_in', 'truncated_normal'
      )
    if bias_init is None:
      bias_init = init_mapping._zeros_init

    key = mx.random.key(0)
    q_dim = num_heads * units_per_head
    kv_dim = num_kv_heads * units_per_head

    # Projections stored as [in, out] to match Linen convention.
    self.q_proj = kernel_init(key, (in_features, q_dim), param_dtype)
    self.k_proj = kernel_init(key, (in_features, kv_dim), param_dtype)
    self.v_proj = kernel_init(key, (in_features, kv_dim), param_dtype)
    if use_bias:
      self.q_bias = bias_init(key, (q_dim,), param_dtype)
      self.k_bias = bias_init(key, (kv_dim,), param_dtype)
      self.v_bias = bias_init(key, (kv_dim,), param_dtype)

    self.query_network = query_network
    self.key_network = key_network
    self.value_network = value_network

  @property
  def supports_step(self):
    return self.max_past_horizon >= 0 and self.max_future_horizon >= 0

  @property
  def input_latency(self):
    return max(0, self.max_future_horizon)

  def _project_qkv(self, x):
    """Project input to Q, K, V sequences."""
    b, t = x.shape[0], x.shape[1]
    dtype = self.compute_dtype or x.dtype

    v = x.values.astype(dtype)
    q = mx.matmul(v, self.q_proj.astype(dtype))
    k = mx.matmul(v, self.k_proj.astype(dtype))
    val = mx.matmul(v, self.v_proj.astype(dtype))

    if self.use_bias:
      q = q + self.q_bias.astype(dtype)
      k = k + self.k_bias.astype(dtype)
      val = val + self.v_bias.astype(dtype)

    # Reshape to [b, t, heads, units_per_head].
    q = q.reshape(b, t, self.num_heads, self.units_per_head)
    k = k.reshape(b, t, self.num_kv_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_kv_heads, self.units_per_head)

    return (
        Sequence(q, x.mask),
        Sequence(k, x.mask),
        Sequence(val, x.mask),
    )

  def _compute_attention(self, queries, keys, values, mask):
    """Compute scaled dot-product attention.

    Args:
      queries: [b, q_t, num_heads, units_per_head]
      keys: [b, kv_t, num_kv_heads, units_per_head]
      values: [b, kv_t, num_kv_heads, units_per_head]
      mask: [b, 1, q_t, kv_t] boolean mask (True = attend)

    Returns:
      context: [b, q_t, num_heads, units_per_head]
    """
    scale = self._query_scale or (1.0 / math.sqrt(self.units_per_head))

    # GQA: repeat K/V heads to match query heads.
    num_groups = self.num_heads // self.num_kv_heads
    if num_groups > 1:
      b, kv_t, nk, h = keys.shape
      keys = mx.repeat(keys, num_groups, axis=2)
      values = mx.repeat(values, num_groups, axis=2)

    # Transpose to [b, heads, t, h] for batched matmul.
    q = mx.transpose(queries, (0, 2, 1, 3))  # [b, nh, qt, h]
    k = mx.transpose(keys, (0, 2, 1, 3))  # [b, nh, kvt, h]
    v = mx.transpose(values, (0, 2, 1, 3))  # [b, nh, kvt, h]

    # Scaled dot-product attention.
    q = q * scale
    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

    # Optional soft cap on logits (e.g., Gemma 2 uses cap=50.0).
    if self._attention_logits_soft_cap is not None:
      cap = self._attention_logits_soft_cap
      logits = mx.tanh(logits / cap) * cap

    # Apply mask: set masked positions to large negative.
    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)

    weights = mx.softmax(logits, axis=-1)
    context = mx.matmul(weights, v)  # [b, nh, qt, h]

    # Transpose back to [b, qt, nh, h].
    context = mx.transpose(context, (0, 2, 1, 3))
    return context

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductSelfAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
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
            bt.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            constants=constants,
        )
        if self.query_network is not None
        else ()
    )
    k_net_state = (
        self.key_network.get_initial_state(
            batch_size,
            bt.ShapeDType(
                (self.num_kv_heads, self.units_per_head),
                compute_dtype,
            ),
            constants=constants,
        )
        if self.key_network is not None
        else ()
    )
    v_net_state = (
        self.value_network.get_initial_state(
            batch_size,
            bt.ShapeDType(
                (self.num_kv_heads, self.units_per_head),
                compute_dtype,
            ),
            constants=constants,
        )
        if self.value_network is not None
        else ()
    )

    return (
        kv_buffer_keys,
        kv_buffer_values,
        kv_buffer_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
    )

  def layer_with_emits(self, x, *, constants=None):
    queries, keys, values = self._project_qkv(x)

    # Optional Q/K/V processing networks (e.g. RoPE).
    # Use `is not None` because parameterless nn.Modules are falsy.
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(queries, constants=constants).values,
          queries.mask,
      )
    if self.key_network is not None:
      keys = Sequence(
          self.key_network.layer(keys, constants=constants).values,
          keys.mask,
      )
    if self.value_network is not None:
      values = Sequence(
          self.value_network.layer(values, constants=constants).values,
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
      row = mx.arange(t)[:, None]
      col = mx.arange(t)[None, :]
      banded = (col >= row - past) & (col <= row + future)
      valid_mask = valid_mask & banded.reshape(1, 1, t, t)

    context = self._compute_attention(
        queries.values, keys.values, values.values, valid_mask
    )
    return Sequence(context, x.mask), ()

  def step_with_emits(self, x, state, *, constants=None):
    queries, keys, values = self._project_qkv(x)

    (
        kv_buf_k,
        kv_buf_v,
        kv_buf_mask,
        time_step,
        q_net_state,
        k_net_state,
        v_net_state,
    ) = state

    # Optional Q/K/V processing networks.
    # Use `is not None` because parameterless nn.Modules are falsy.
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, constants=constants
      )
    if self.key_network is not None:
      keys, k_net_state = self.key_network.step(
          keys, k_net_state, constants=constants
      )
    if self.value_network is not None:
      values, v_net_state = self.value_network.step(
          values, v_net_state, constants=constants
      )

    # Mask invalid values.
    values = values.mask_invalid()

    x_time = x.shape[1]
    kv_buffer_size = kv_buf_k.shape[1]

    if kv_buffer_size > 0:
      # Ring buffer write: insert new K/V at rotating positions.
      # Uses put_along_axis to scatter into pre-allocated buffers,
      # compatible with mx.compile / mx.export_function (no Python
      # int conversion needed).
      t0 = time_step[0]  # MLX scalar, no eval.
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

      # Build visibility mask: [b, 1, 1, kv_buffer_size].
      kv_valid = kv_buf_mask[:, None, None, :]

      # Add causal mask for multi-step queries (respects ring buffer order).
      if x_time > 1:
        newest_time = t0 + x_time - 1
        newest_pos = newest_time % kv_buffer_size
        phys = mx.arange(kv_buffer_size)
        dist = (newest_pos - phys + kv_buffer_size) % kv_buffer_size
        temporal = newest_time - dist
        q_times = t0 + mx.arange(x_time)
        causal = temporal[None, :] <= q_times[:, None]
        kv_valid = kv_valid & causal.reshape(1, 1, x_time, kv_buffer_size)

      context = self._compute_attention(
          queries.values, kv_buf_k, kv_buf_v, kv_valid
      )
    else:
      # Degenerate: no history buffer, attend only to current step.
      kv_valid = x.mask[:, None, None, :]
      if x_time > 1:
        causal = _causal_mask(x_time, x_time)
        kv_valid = kv_valid & causal
      context = self._compute_attention(
          queries.values, keys.values, values.values, kv_valid
      )

    new_state = (
        kv_buf_k,
        kv_buf_v,
        kv_buf_mask,
        time_step + x_time,
        q_net_state,
        k_net_state,
        v_net_state,
    )
    return Sequence(context, x.mask), new_state, ()

  @classmethod
  def from_config(cls, config):
    """Create from a Linen DotProductSelfAttention.Config.

    Since in_features is not in the config (it's inferred), we
    return a _DeferredDotProductSelfAttention that creates
    projections on first use.
    """
    return DeferredDotProductSelfAttention(config)


class DeferredDotProductSelfAttention(types.Emitting):
  """Wrapper that defers projection creation until first input.

  Linen DotProductSelfAttention.Config doesn't specify in_features;
  it is inferred from the first input.
  """

  def __init__(self, config):
    super().__init__()
    self._config = config
    self._inner = None

  def _ensure_initialized(self, in_features, backend='mlx'):
    if self._inner is not None:
      return

    # Build optional Q/K/V networks.
    query_network = None
    key_network = None
    value_network = None
    if self._config.query_network:
      query_network = self._config.query_network.make(backend=backend)
    if self._config.key_network:
      key_network = self._config.key_network.make(backend=backend)
    if self._config.value_network:
      value_network = self._config.value_network.make(backend=backend)

    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(self._config.param_dtype)
    self._inner = DotProductSelfAttention(
        in_features=in_features,
        num_heads=self._config.num_heads,
        units_per_head=self._config.units_per_head,
        max_past_horizon=self._config.max_past_horizon,
        max_future_horizon=self._config.max_future_horizon,
        num_kv_heads=self._config.num_kv_heads,
        use_bias=self._config.use_bias,
        query_scale=getattr(self._config, 'query_scale', None),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        kernel_init=init_mapping.map_initializer(
            getattr(self._config, 'input_projection', None)
            and getattr(
                self._config.input_projection,
                'qkv_kernel_init',
                None,
            )
        ),
        query_network=query_network,
        key_network=key_network,
        value_network=value_network,
    )

  @property
  def supports_step(self):
    mph = self._config.max_past_horizon
    mfh = self._config.max_future_horizon
    return mph >= 0 and mfh >= 0

  @property
  def input_latency(self):
    return max(0, self._config.max_future_horizon)

  def get_output_shape(self, input_shape, *, constants=None):
    return (
        self._config.num_heads,
        self._config.units_per_head,
    )

  def get_output_dtype(self, input_dtype, *, constants=None):
    if getattr(self._config, 'compute_dtype', None):
      return init_mapping._to_mx_dtype(self._config.compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_initialized(input_spec.shape[-1])
    return self._inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer_with_emits(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self._inner.layer_with_emits(x, constants=constants)

  def step_with_emits(self, x, state, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self._inner.step_with_emits(x, state, constants=constants)


class DotProductAttention(types.Emitting):
  """Multi-headed dot-product cross attention for MLX.

  Queries come from the input sequence; keys and values come from a
  source sequence looked up in the ``constants`` dictionary.

  In ``layer()`` mode the K/V projections and optional networks are
  applied to the source on-the-fly.  In ``step()`` mode they are
  pre-computed during ``get_initial_state()`` so that each step only
  needs to project and attend queries.

  Kernels are stored in Linen-compatible shapes:
    q_proj:   [in_features, num_heads * units_per_head]
    k_proj:   [source_features, num_heads * units_per_head]
    v_proj:   [source_features, num_heads * units_per_head]
    out_proj: [num_heads * units_per_head, in_features]
  """

  def __init__(
      self,
      *,
      in_features: int,
      source_features: int,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      use_bias: bool = False,
      query_scale: float | None = None,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
  ):
    super().__init__()
    self.in_features = in_features
    self.source_features = source_features
    self.source_name = source_name
    self.num_heads = num_heads
    self.units_per_head = units_per_head
    self.use_bias = use_bias
    self._query_scale = query_scale
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    if kernel_init is None:
      kernel_init = init_mapping._make_variance_scaling_init(
          'fan_in', 'truncated_normal'
      )
    if bias_init is None:
      bias_init = init_mapping._zeros_init

    key = mx.random.key(0)
    qkv_dim = num_heads * units_per_head

    self.q_proj = kernel_init(key, (in_features, qkv_dim), param_dtype)
    self.k_proj = kernel_init(key, (source_features, qkv_dim), param_dtype)
    self.v_proj = kernel_init(key, (source_features, qkv_dim), param_dtype)
    if use_bias:
      self.q_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.k_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.v_bias = bias_init(key, (qkv_dim,), param_dtype)

    self.query_network = query_network
    self.key_network = key_network
    self.value_network = value_network

  @property
  def supports_step(self):
    if self.query_network is not None:
      return self.query_network.supports_step
    return True

  @property
  def input_latency(self):
    return 0

  def _project_q(self, x):
    b, t = x.shape[0], x.shape[1]
    dtype = self.compute_dtype or x.dtype
    v = x.values.astype(dtype)
    q = mx.matmul(v, self.q_proj.astype(dtype))
    if self.use_bias:
      q = q + self.q_bias.astype(dtype)
    q = q.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(q, x.mask)

  def _project_kv(self, source):
    b, t = source.shape[0], source.shape[1]
    dtype = self.compute_dtype or source.dtype
    v = source.values.astype(dtype)
    k = mx.matmul(v, self.k_proj.astype(dtype))
    val = mx.matmul(v, self.v_proj.astype(dtype))
    if self.use_bias:
      k = k + self.k_bias.astype(dtype)
      val = val + self.v_bias.astype(dtype)
    k = k.reshape(b, t, self.num_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(k, source.mask), Sequence(val, source.mask)

  def _get_source(self, constants):
    if constants is None or self.source_name not in constants:
      raise ValueError(f'Source "{self.source_name}" not found in constants.')
    return constants[self.source_name]

  def _compute_attention(self, queries, keys, values, mask):
    """Compute scaled dot-product attention (no causal mask)."""
    scale = self._query_scale or (1.0 / math.sqrt(self.units_per_head))

    q = mx.transpose(queries, (0, 2, 1, 3))
    k = mx.transpose(keys, (0, 2, 1, 3))
    v = mx.transpose(values, (0, 2, 1, 3))

    q = q * scale
    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))

    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)

    weights = mx.softmax(logits, axis=-1)
    context = mx.matmul(weights, v)
    context = mx.transpose(context, (0, 2, 1, 3))
    return context

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'DotProductAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    # Pre-project source keys and values.
    source = self._get_source(constants)
    keys, values = self._project_kv(source)

    if self.key_network is not None:
      keys = self.key_network.layer(keys, constants=constants)
    if self.value_network is not None:
      values = self.value_network.layer(values, constants=constants)

    keys = keys.mask_invalid()
    values = values.mask_invalid()

    q_net_state = (
        self.query_network.get_initial_state(
            batch_size,
            bt.ShapeDType(
                (self.num_heads, self.units_per_head),
                self.get_output_dtype(input_spec.dtype),
            ),
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

  def layer_with_emits(self, x, *, constants=None):
    source = self._get_source(constants)
    keys, values = self._project_kv(source)

    if self.key_network is not None:
      keys = self.key_network.layer(keys, constants=constants)
    if self.value_network is not None:
      values = self.value_network.layer(values, constants=constants)

    queries = self._project_q(x)
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(queries, constants=constants).values,
          queries.mask,
      )

    values = values.mask_invalid()
    valid_mask = source.mask[:, None, None, :]
    context = self._compute_attention(
        queries.values, keys.values, values.values, valid_mask
    )
    return Sequence(context, x.mask), ()

  def step_with_emits(self, x, state, *, constants=None):
    keys_v, values_v, kv_mask, q_net_state, time_step = state

    queries = self._project_q(x)
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, constants=constants
      )

    valid_mask = kv_mask[:, None, None, :]
    context = self._compute_attention(
        queries.values, keys_v, values_v, valid_mask
    )

    new_state = (
        keys_v,
        values_v,
        kv_mask,
        q_net_state,
        time_step + x.shape[1],
    )
    return Sequence(context, x.mask), new_state, ()

  @classmethod
  def from_config(cls, config):
    return DeferredDotProductAttention(config)


class DeferredDotProductAttention(types.Emitting):
  """Deferred DotProductAttention that creates projections on first use."""

  def __init__(self, config):
    super().__init__()
    self._config = config
    self._inner = None

  def _ensure_initialized(self, in_features, source_features, backend='mlx'):
    if self._inner is not None:
      return

    query_network = None
    key_network = None
    value_network = None
    if self._config.query_network:
      query_network = self._config.query_network.make(backend=backend)
    if self._config.key_network:
      key_network = self._config.key_network.make(backend=backend)
    if self._config.value_network:
      value_network = self._config.value_network.make(backend=backend)

    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(self._config.param_dtype)

    self._inner = DotProductAttention(
        in_features=in_features,
        source_features=source_features,
        source_name=self._config.source_name,
        num_heads=self._config.num_heads,
        units_per_head=self._config.units_per_head,
        use_bias=self._config.use_bias,
        query_scale=getattr(self._config, 'query_scale', None),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        kernel_init=init_mapping.map_initializer(
            getattr(self._config, 'input_projection', None)
            and getattr(
                self._config.input_projection,
                'qkv_kernel_init',
                None,
            )
        ),
        query_network=query_network,
        key_network=key_network,
        value_network=value_network,
    )

  def _get_source(self, constants):
    if constants is None:
      raise ValueError('Constants required for cross-attention.')
    if self._config.source_name not in constants:
      raise ValueError(f'Source "{self._config.source_name}" not found.')
    return constants[self._config.source_name]

  @property
  def supports_step(self):
    if self._config.query_network is not None:
      # Can't easily check without building; assume True.
      return True
    return True

  @property
  def input_latency(self):
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return (
        self._config.num_heads,
        self._config.units_per_head,
    )

  def get_output_dtype(self, input_dtype, *, constants=None):
    if getattr(self._config, 'compute_dtype', None):
      return init_mapping._to_mx_dtype(self._config.compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(input_spec.shape[-1], source.shape[-1])
    return self._inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer_with_emits(self, x, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])
    return self._inner.layer_with_emits(x, constants=constants)

  def step_with_emits(self, x, state, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])
    return self._inner.step_with_emits(x, state, constants=constants)


def _banded_mask(q_len, kv_len, num_lower, num_upper):
  """Build a [1, 1, q_len, kv_len] banded visibility mask.

  Position (i, j) is True iff j >= i - num_lower and j <= i + num_upper.
  """
  row = mx.arange(q_len)[:, None]
  col = mx.arange(kv_len)[None, :]
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


class StreamingDotProductAttention(types.Emitting):
  """Multi-headed streaming cross-attention for MLX.

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

  def __init__(
      self,
      *,
      in_features: int,
      source_features: int,
      source_name: str,
      num_heads: int,
      units_per_head: int,
      max_past_horizon: int,
      max_future_horizon: int = 0,
      use_bias: bool = False,
      use_query_delay_buffer: bool = True,
      query_scale: float | None = None,
      compute_dtype=None,
      param_dtype=mx.float32,
      kernel_init=None,
      bias_init=None,
      query_network: types.SequenceLayer | None = None,
      key_network: types.SequenceLayer | None = None,
      value_network: types.SequenceLayer | None = None,
  ):
    super().__init__()
    if max_past_horizon < 1:
      raise ValueError(
          f'max_past_horizon must be >= 1, got {max_past_horizon}.'
      )
    if max_future_horizon < 0:
      raise ValueError(
          f'max_future_horizon must be >= 0, got {max_future_horizon}.'
      )

    self.in_features = in_features
    self.source_features = source_features
    self.source_name = source_name
    self.num_heads = num_heads
    self.units_per_head = units_per_head
    self.max_past_horizon = max_past_horizon
    self.max_future_horizon = max_future_horizon
    self.use_bias = use_bias
    self.use_query_delay_buffer = use_query_delay_buffer
    self._query_scale = query_scale
    self.compute_dtype = compute_dtype
    self._param_dtype = param_dtype

    if kernel_init is None:
      kernel_init = init_mapping._make_variance_scaling_init(
          'fan_in', 'truncated_normal'
      )
    if bias_init is None:
      bias_init = init_mapping._zeros_init

    key = mx.random.key(0)
    qkv_dim = num_heads * units_per_head

    # Q projection from input.
    self.q_proj = kernel_init(key, (in_features, qkv_dim), param_dtype)
    # K/V projections from source.
    self.k_proj = kernel_init(key, (source_features, qkv_dim), param_dtype)
    self.v_proj = kernel_init(key, (source_features, qkv_dim), param_dtype)
    if use_bias:
      self.q_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.k_bias = bias_init(key, (qkv_dim,), param_dtype)
      self.v_bias = bias_init(key, (qkv_dim,), param_dtype)

    self.query_network = query_network
    self.key_network = key_network
    self.value_network = value_network

  @property
  def supports_step(self):
    return True

  @property
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
    k = mx.matmul(v, self.k_proj.astype(dtype))
    val = mx.matmul(v, self.v_proj.astype(dtype))
    if self.use_bias:
      k = k + self.k_bias.astype(dtype)
      val = val + self.v_bias.astype(dtype)
    k = k.reshape(b, t, self.num_heads, self.units_per_head)
    val = val.reshape(b, t, self.num_heads, self.units_per_head)
    return Sequence(k, source.mask), Sequence(val, source.mask)

  def _get_source(self, constants):
    if constants is None or self.source_name not in constants:
      raise ValueError(f'Source "{self.source_name}" not found in constants.')
    return constants[self.source_name]

  def _compute_attention(self, queries, keys, values, mask):
    """Compute scaled dot-product attention."""
    scale = self._query_scale or (1.0 / math.sqrt(self.units_per_head))
    q = mx.transpose(queries, (0, 2, 1, 3))
    k = mx.transpose(keys, (0, 2, 1, 3))
    v = mx.transpose(values, (0, 2, 1, 3))
    q = q * scale
    logits = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    if mask is not None:
      large_neg = mx.array(-1e9, dtype=logits.dtype)
      logits = mx.where(mask, logits, large_neg)
    weights = mx.softmax(logits, axis=-1)
    context = mx.matmul(weights, v)
    context = mx.transpose(context, (0, 2, 1, 3))
    return context

  def get_output_shape(self, input_shape, *, constants=None):
    if len(input_shape) != 1:
      raise ValueError(
          'StreamingDotProductAttention requires rank 3 input,'
          f' got channel_shape={input_shape}.'
      )
    return (self.num_heads, self.units_per_head)

  def get_output_dtype(self, input_dtype, *, constants=None):
    if self.compute_dtype is not None:
      return self.compute_dtype
    return self._param_dtype

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
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
            bt.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            constants=constants,
        )
        if self.query_network is not None
        else ()
    )
    k_net_state = (
        self.key_network.get_initial_state(
            batch_size,
            bt.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
            constants=constants,
        )
        if self.key_network is not None
        else ()
    )
    v_net_state = (
        self.value_network.get_initial_state(
            batch_size,
            bt.ShapeDType(
                (self.num_heads, self.units_per_head),
                compute_dtype,
            ),
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

  def layer_with_emits(self, x, *, constants=None):
    source = self._get_source(constants)

    queries = self._project_q(x)
    keys, values = self._project_kv(source)
    queries_time = queries.shape[1]
    keys_time = keys.shape[1]

    # Optional Q/K/V processing networks.
    if self.query_network is not None:
      queries = Sequence(
          self.query_network.layer(queries, constants=constants).values,
          queries.mask,
      )
    if self.key_network is not None:
      keys = Sequence(
          self.key_network.layer(keys, constants=constants).values,
          keys.mask,
      )
    if self.value_network is not None:
      values = Sequence(
          self.value_network.layer(values, constants=constants).values,
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

    context = self._compute_attention(
        queries.values, keys.values, values.values, valid_mask
    )
    return Sequence(context, x.mask), ()

  def step_with_emits(self, x, state, *, constants=None):
    source = self._get_source(constants)

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

    queries = self._project_q(x)
    keys, values = self._project_kv(source)

    # Optional Q/K/V processing networks.
    if self.query_network is not None:
      queries, q_net_state = self.query_network.step(
          queries, q_net_state, constants=constants
      )
    if self.key_network is not None:
      keys, k_net_state = self.key_network.step(
          keys, k_net_state, constants=constants
      )
    if self.value_network is not None:
      values, v_net_state = self.value_network.step(
          values, v_net_state, constants=constants
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

    context = self._compute_attention(queries.values, new_k, new_v, valid_mask)

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
    return Sequence(context, queries.mask), new_state, ()

  @classmethod
  def from_config(cls, config):
    return DeferredStreamingDotProductAttention(config)


class DeferredStreamingDotProductAttention(types.Emitting):
  """Deferred StreamingDotProductAttention.

  Creates the inner attention on first use when in_features and
  source_features are known.
  """

  def __init__(self, config):
    super().__init__()
    self._config = config
    self._inner = None

  def _ensure_initialized(self, in_features, source_features, backend='mlx'):
    if self._inner is not None:
      return

    query_network = None
    key_network = None
    value_network = None
    if self._config.query_network:
      query_network = self._config.query_network.make(backend=backend)
    if self._config.key_network:
      key_network = self._config.key_network.make(backend=backend)
    if self._config.value_network:
      value_network = self._config.value_network.make(backend=backend)

    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(self._config.param_dtype)

    self._inner = StreamingDotProductAttention(
        in_features=in_features,
        source_features=source_features,
        source_name=self._config.source_name,
        num_heads=self._config.num_heads,
        units_per_head=self._config.units_per_head,
        max_past_horizon=self._config.max_past_horizon,
        max_future_horizon=self._config.max_future_horizon,
        use_bias=self._config.use_bias,
        use_query_delay_buffer=getattr(
            self._config, 'use_query_delay_buffer', True
        ),
        query_scale=getattr(self._config, 'query_scale', None),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        kernel_init=init_mapping.map_initializer(
            getattr(self._config, 'input_projection', None)
            and getattr(
                self._config.input_projection,
                'q_kernel_init',
                None,
            )
        ),
        query_network=query_network,
        key_network=key_network,
        value_network=value_network,
    )

  def _get_source(self, constants):
    if constants is None:
      raise ValueError('Constants required for streaming attention.')
    if self._config.source_name not in constants:
      raise ValueError(f'Source "{self._config.source_name}" not found.')
    return constants[self._config.source_name]

  @property
  def supports_step(self):
    return True

  @property
  def input_latency(self):
    mfh = self._config.max_future_horizon
    uqdb = getattr(self._config, 'use_query_delay_buffer', True)
    if mfh > 0 and uqdb:
      return mfh
    return 0

  def get_output_shape(self, input_shape, *, constants=None):
    return (
        self._config.num_heads,
        self._config.units_per_head,
    )

  def get_output_dtype(self, input_dtype, *, constants=None):
    if getattr(self._config, 'compute_dtype', None):
      return init_mapping._to_mx_dtype(self._config.compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(input_spec.shape[-1], source.shape[-1])
    return self._inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer_with_emits(self, x, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])
    return self._inner.layer_with_emits(x, constants=constants)

  def step_with_emits(self, x, state, *, constants=None):
    source = self._get_source(constants)
    self._ensure_initialized(x.shape[-1], source.shape[-1])
    return self._inner.step_with_emits(x, state, constants=constants)


class LocalDotProductSelfAttention(DotProductSelfAttention):
  """Local dot-product self attention with configurable block_size.

  Extends DotProductSelfAttention with a configurable block_size for
  step-mode processing. The sliding window behavior is already handled
  by the base class's banded visibility mask via max_past_horizon and
  max_future_horizon.
  """

  def __init__(self, *, block_size_config: int = 1, **kwargs):
    super().__init__(**kwargs)
    self._block_size_config = block_size_config

  @property
  def block_size(self):
    return self._block_size_config

  @classmethod
  def from_config(cls, config):
    return DeferredLocalDotProductSelfAttention(config)


class DeferredLocalDotProductSelfAttention(types.Emitting):
  """Deferred LocalDotProductSelfAttention.

  Creates the inner attention on first use when in_features is known.
  """

  def __init__(self, config):
    super().__init__()
    self._config = config
    self._inner = None

  def _ensure_initialized(self, in_features, backend='mlx'):
    if self._inner is not None:
      return

    query_network = None
    key_network = None
    value_network = None
    if self._config.query_network:
      query_network = self._config.query_network.make(backend=backend)
    if self._config.key_network:
      key_network = self._config.key_network.make(backend=backend)
    if self._config.value_network:
      value_network = self._config.value_network.make(backend=backend)

    compute_dtype = getattr(self._config, 'compute_dtype', None)
    if compute_dtype is not None:
      compute_dtype = init_mapping._to_mx_dtype(compute_dtype)
    param_dtype = init_mapping._to_mx_dtype(self._config.param_dtype)

    self._inner = LocalDotProductSelfAttention(
        in_features=in_features,
        num_heads=self._config.num_heads,
        units_per_head=self._config.units_per_head,
        max_past_horizon=self._config.max_past_horizon,
        max_future_horizon=self._config.max_future_horizon,
        use_bias=self._config.use_bias,
        block_size_config=self._config.block_size,
        query_scale=getattr(self._config, 'query_scale', None),
        compute_dtype=compute_dtype,
        param_dtype=param_dtype,
        attention_logits_soft_cap=getattr(
            self._config, 'attention_logits_soft_cap', None
        ),
        kernel_init=init_mapping.map_initializer(
            getattr(self._config, 'input_projection', None)
            and getattr(
                self._config.input_projection,
                'qkv_kernel_init',
                None,
            )
        ),
        query_network=query_network,
        key_network=key_network,
        value_network=value_network,
    )

  @property
  def supports_step(self):
    mph = self._config.max_past_horizon
    mfh = self._config.max_future_horizon
    return mph >= 0 and mfh >= 0

  @property
  def block_size(self):
    return self._config.block_size

  @property
  def input_latency(self):
    return max(0, self._config.max_future_horizon)

  def get_output_shape(self, input_shape, *, constants=None):
    return (
        self._config.num_heads,
        self._config.units_per_head,
    )

  def get_output_dtype(self, input_dtype, *, constants=None):
    if getattr(self._config, 'compute_dtype', None):
      return init_mapping._to_mx_dtype(self._config.compute_dtype)
    return init_mapping._to_mx_dtype(self._config.param_dtype)

  def get_initial_state(self, batch_size, input_spec, *, constants=None):
    self._ensure_initialized(input_spec.shape[-1])
    return self._inner.get_initial_state(
        batch_size, input_spec, constants=constants
    )

  def layer_with_emits(self, x, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self._inner.layer_with_emits(x, constants=constants)

  def step_with_emits(self, x, state, *, constants=None):
    self._ensure_initialized(x.shape[-1])
    return self._inner.step_with_emits(x, state, constants=constants)
