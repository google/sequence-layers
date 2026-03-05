"""Convert Linen-trained params to MLX model weights.

Handles the structural differences between Linen (JAX/Flax) and MLX:
  - Linen Dense kernel [in, out] → MLX nn.Linear weight [out, in]
  - Linen combined QKV kernel [in, 3, heads, uph] → separate q/k/v
  - Linen Repeat stacked params [N, ...] → per-copy params [...]
  - Linen Partitioned wrappers → unwrapped arrays
"""

import mlx.core as mx
import numpy as np


def _unbox_params(params):
  """Unwrap Flax Partitioned wrappers and convert to numpy.

  Args:
    params: A Linen param dict (possibly with Partitioned values).

  Returns:
    A nested dict of numpy arrays.
  """
  import jax
  from flax import linen as nn

  params = nn.unbox(params)
  return jax.tree_util.tree_map(lambda x: np.array(x), params)


def _set_weight(module, attr_name, value):
  """Set a weight on an MLX module.

  Handles both direct array attributes and nn.Module child params.

  Args:
    module: An MLX nn.Module.
    attr_name: Dot-separated attribute path (e.g. '_linear.weight').
    value: An mx.array value.
  """
  parts = attr_name.split('.')
  obj = module
  for part in parts[:-1]:
    obj = getattr(obj, part)
  setattr(obj, parts[-1], value)


def load_linen_params(
    mlx_model,
    linen_params,
    config,
    *,
    input_spec=None,
    batch_stats=None,
    constants=None,
):
  """Load Linen-trained params into an MLX model.

  Uses the config tree to guide the conversion, handling structural
  differences between Linen and MLX parameter layouts.

  Args:
    mlx_model: An MLX SequenceLayer (already initialized via
        config.make(backend='mlx')).
    linen_params: A Linen param dict from model.init(...)['params'].
    config: The SequenceLayerConfig used to create both models.
    input_spec: Optional ShapeDType for the input. Defaults to scalar int32
        (for token models). For float models (e.g. convolution), pass
        ShapeDType((channels,), mx.float32).
    batch_stats: Optional batch_stats dict from model.init(...)['batch_stats'].
        Required for BatchNormalization layers.
    constants: Optional constants dict for layers that need a source sequence
        during deferred initialization (e.g. cross-attention).
  """
  from sequence_layers.mlx import export
  from sequence_layers.mlx import basic_types as bt

  if input_spec is None:
    input_spec = bt.ShapeDType((), mx.int32)

  # Materialize deferred layers with a dummy forward pass.
  # Slice constants to time=1 to match the dummy input.
  init_constants = None
  if constants is not None:
    init_constants = {}
    for k, v in constants.items():
      if hasattr(v, 'values') and hasattr(v, 'mask'):
        # Slice Sequence to time=1.
        init_constants[k] = bt.Sequence(v.values[:1, :1], v.mask[:1, :1])
      else:
        init_constants[k] = v
  export._materialize_deferred(
      mlx_model,
      batch_size=1,
      input_spec=input_spec,
      constants=init_constants,
  )

  # Unbox and convert to numpy.
  params = _unbox_params(linen_params)
  bs = _unbox_params(batch_stats) if batch_stats is not None else None

  # Walk the config tree and load params.
  _load_config(mlx_model, params, config, batch_stats=bs)
  mx.eval(mlx_model.parameters())


def _load_config(mlx_module, linen_params, config, batch_stats=None):
  """Recursively load params guided by config type."""
  from sequence_layers.jax import combinators as jax_comb
  from sequence_layers.jax import conditioning as jax_cond
  from sequence_layers.jax import convolution as jax_conv
  from sequence_layers.jax import dense as jax_dense
  from sequence_layers.jax import normalization as jax_norm
  from sequence_layers.jax import simple as jax_simple
  from sequence_layers.jax.attention import (
      dot_product_attention as jax_cross_attn,
  )
  from sequence_layers.jax.attention import (
      dot_product_self_attention as jax_self_attn,
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

  if isinstance(config, jax_comb.Serial.Config):
    _load_serial(mlx_module, linen_params, config, batch_stats)
  elif isinstance(config, jax_comb.Parallel.Config):
    _load_parallel(mlx_module, linen_params, config, batch_stats)
  elif isinstance(config, jax_comb.Repeat.Config):
    _load_repeat(mlx_module, linen_params, config, batch_stats)
  elif isinstance(config, jax_comb.Residual.Config):
    _load_residual(mlx_module, linen_params, config, batch_stats)
  elif isinstance(
      config,
      (
          jax_cross_attn.DotProductAttention.Config,
          jax_streaming_attn.StreamingDotProductAttention.Config,
          jax_streaming_local_attn.StreamingLocalDotProductAttention.Config,
      ),
  ):
    _load_streaming_attention(mlx_module, linen_params, config)
  elif isinstance(config, jax_local_attn.LocalDotProductSelfAttention.Config):
    _load_attention(mlx_module, linen_params, config)
  elif isinstance(config, jax_self_attn.DotProductSelfAttention.Config):
    _load_attention(mlx_module, linen_params, config)
  elif isinstance(config, jax_dense.Dense.Config):
    _load_dense(mlx_module, linen_params, config)
  elif isinstance(config, jax_conv.Conv1D.Config):
    _load_conv1d(mlx_module, linen_params, config)
  elif isinstance(config, jax_conv.DepthwiseConv1D.Config):
    _load_depthwise_conv1d(mlx_module, linen_params, config)
  elif isinstance(config, jax_conv.Conv1DTranspose.Config):
    _load_conv1d_transpose(mlx_module, linen_params, config)
  elif isinstance(config, jax_norm.RMSNormalization.Config):
    _load_rms_norm(mlx_module, linen_params, config)
  elif isinstance(config, jax_norm.LayerNormalization.Config):
    _load_layer_norm(mlx_module, linen_params, config)
  elif isinstance(config, jax_norm.BatchNormalization.Config):
    _load_batch_norm(mlx_module, linen_params, config, batch_stats)
  elif isinstance(config, jax_norm.GroupNormalization.Config):
    _load_group_norm(mlx_module, linen_params, config)
  elif isinstance(config, jax_dense.EinsumDense.Config):
    _load_einsum_dense(mlx_module, linen_params, config)
  elif isinstance(config, jax_cond.Conditioning.Config):
    _load_conditioning(mlx_module, linen_params, config)
  elif isinstance(config, jax_simple.Embedding.Config):
    _load_embedding(mlx_module, linen_params, config)
  # Stateless layers (Flatten, Identity, RoPE, pooling, etc.) have no params.


def _load_serial(mlx_serial, linen_params, config, batch_stats=None):
  """Load Serial: walk layers_{i} in Linen, model.layers[i] in MLX."""
  for i, layer_config in enumerate(config.layers):
    key = f'layers_{i}'
    child_params = linen_params.get(key, {})
    child_bs = batch_stats.get(key, {}) if batch_stats else None
    _load_config(
        mlx_serial.layers[i],
        child_params,
        layer_config,
        batch_stats=child_bs,
    )


def _load_parallel(mlx_parallel, linen_params, config, batch_stats=None):
  """Load Parallel: walk layers_{i}, same as Serial."""
  for i, layer_config in enumerate(config.layers):
    key = f'layers_{i}'
    child_params = linen_params.get(key, {})
    child_bs = batch_stats.get(key, {}) if batch_stats else None
    _load_config(
        mlx_parallel.layers[i],
        child_params,
        layer_config,
        batch_stats=child_bs,
    )


def _load_repeat(mlx_repeat, linen_params, config, batch_stats=None):
  """Load Repeat: slice stacked Linen params for each MLX copy."""
  child_params = linen_params.get('child_layer', {})
  child_bs = batch_stats.get('child_layer', {}) if batch_stats else None

  # Linen Repeat stacks all child params with leading [num_repeats].
  # Slice axis 0 for each copy.
  for i in range(config.num_repeats):
    sliced = _slice_params(child_params, i)
    sliced_bs = _slice_params(child_bs, i) if child_bs else None
    _load_config(
        mlx_repeat.layers[i],
        sliced,
        config.layer,
        batch_stats=sliced_bs,
    )


def _slice_params(params, index):
  """Slice the leading axis of all arrays in a param dict."""
  result = {}
  for key, value in params.items():
    if isinstance(value, dict):
      result[key] = _slice_params(value, index)
    elif isinstance(value, np.ndarray):
      result[key] = value[index]
    else:
      result[key] = value
  return result


def _load_residual(mlx_residual, linen_params, config, batch_stats=None):
  """Load Residual: body is layers_{i}, shortcut is shortcut_layer."""
  # Body is a Serial inside the Residual.
  body = mlx_residual.body
  for i, layer_config in enumerate(config.layers):
    key = f'layers_{i}'
    child_params = linen_params.get(key, {})
    child_bs = batch_stats.get(key, {}) if batch_stats else None
    _load_config(
        body.layers[i],
        child_params,
        layer_config,
        batch_stats=child_bs,
    )

  # Shortcut (usually Identity — no params).
  if config.shortcut_layers:
    shortcut_params = linen_params.get('shortcut_layer', {})
    shortcut_bs = batch_stats.get('shortcut_layer', {}) if batch_stats else None
    for i, sc_config in enumerate(config.shortcut_layers):
      sc_key = f'layers_{i}'
      sc_bs = shortcut_bs.get(sc_key, {}) if shortcut_bs else None
      _load_config(
          mlx_residual.shortcut,
          shortcut_params.get(sc_key, {}),
          sc_config,
          batch_stats=sc_bs,
      )


def _load_dense(mlx_dense, linen_params, config):
  """Load Dense: transpose kernel [in, out] → [out, in]."""
  # Handle DenseDeferred wrapper.
  inner = mlx_dense
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner

  kernel = linen_params.get('kernel')
  if kernel is not None:
    # Linen: [in, out], MLX nn.Linear: [out, in]
    weight = mx.array(kernel.T)
    inner._linear.weight = weight

  bias = linen_params.get('bias')
  if bias is not None:
    inner._linear.bias = mx.array(bias)


def _load_einsum_dense(mlx_einsum, linen_params, config):
  """Load EinsumDense: kernel shape matches directly (einsum notation)."""
  kernel = linen_params.get('kernel')
  if kernel is not None:
    mlx_einsum.kernel = mx.array(kernel)
    mlx_einsum._initialized = True
  bias = linen_params.get('bias')
  if bias is not None:
    mlx_einsum.bias = mx.array(bias)


def _load_attention(mlx_attn, linen_params, config):
  """Load DotProductSelfAttention.

  Handles:
    - CombinedQueryKeyValueProjection:
        query_key_value_projection/kernel [in, 3, heads, uph]
    - SeparateQueryKeyValueProjection:
        query_projection/kernel [in, heads, uph]
        key_projection/kernel [in, kv_heads, uph]
        value_projection/kernel [in, kv_heads, uph]
  """
  from sequence_layers.jax.attention import common as attn_common
  from sequence_layers.mlx import projection_configs as mlx_proj

  # Handle Deferred wrapper.
  inner = mlx_attn
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner

  input_projection = config.input_projection

  if isinstance(input_projection, (attn_common.CombinedQueryKeyValueProjection, mlx_proj.CombinedQueryKeyValueProjection)):
    # Combined QKV: kernel [in, 3, heads, uph]
    qkv_params = linen_params.get('query_key_value_projection', {})
    combined_kernel = qkv_params.get('kernel')
    if combined_kernel is not None:
      in_features = combined_kernel.shape[0]
      if hasattr(inner, 'qkv_proj'):
        inner.qkv_proj = mx.array(combined_kernel.reshape(in_features, -1))
      else:
        # Separate Q + combined KV layout.
        q, k, v = np.split(combined_kernel, 3, axis=1)
        inner.q_proj = mx.array(q.reshape(in_features, -1))
        k_flat = k.reshape(in_features, -1)
        v_flat = v.reshape(in_features, -1)
        inner.kv_proj = mx.array(
            np.concatenate([k_flat, v_flat], axis=-1)
        )

    combined_bias = qkv_params.get('bias')
    if combined_bias is not None:
      if hasattr(inner, 'qkv_bias'):
        inner.qkv_bias = mx.array(combined_bias.reshape(-1))
      else:
        qb, kb, vb = np.split(combined_bias, 3, axis=0)
        inner.q_bias = mx.array(qb.reshape(-1))
        inner.kv_bias = mx.array(
            np.concatenate([kb.reshape(-1), vb.reshape(-1)], axis=-1)
        )

  elif isinstance(
      input_projection, (attn_common.SeparateQueryKeyValueProjection, mlx_proj.SeparateQueryKeyValueProjection)
  ):
    # Separate Q/K/V projections (used for GQA where num_kv_heads < num_heads).
    q_params = linen_params.get('query_projection', {})
    q_kernel = q_params.get('kernel')
    if q_kernel is not None:
      in_features = q_kernel.shape[0]
      inner.q_proj = mx.array(q_kernel.reshape(in_features, -1))
    q_bias = q_params.get('bias')
    if q_bias is not None:
      inner.q_bias = mx.array(q_bias.reshape(-1))

    k_params = linen_params.get('key_projection', {})
    k_kernel = k_params.get('kernel')
    v_params = linen_params.get('value_projection', {})
    v_kernel = v_params.get('kernel')
    if k_kernel is not None and v_kernel is not None:
      in_features = k_kernel.shape[0]
      k_flat = k_kernel.reshape(in_features, -1)
      v_flat = v_kernel.reshape(in_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    k_bias = k_params.get('bias')
    v_bias = v_params.get('bias')
    if k_bias is not None and v_bias is not None:
      inner.kv_bias = mx.array(
          np.concatenate([k_bias.reshape(-1), v_bias.reshape(-1)], axis=-1)
      )

  # per_dim_scale: learned [units_per_head] query scale.
  per_dim_scale = linen_params.get('per_dim_scale')
  if per_dim_scale is not None:
    inner._per_dim_scale = mx.array(per_dim_scale)

  # Q/K/V processing networks have no trainable params
  # (RoPE is stateless with no learned weights).


def _load_streaming_attention(mlx_attn, linen_params, config):
  """Load StreamingDotProductAttention.

  Handles different projection layouts:
    - QueryAndKeyValueProjection (default):
        query_projection/kernel [in, heads, uph]
        key_value_projection/kernel [source, 2, heads, uph]
    - SeparateQueryKeyValueProjection:
        query_projection/kernel [in, heads, uph]
        key_projection/kernel [source, heads, uph]
        value_projection/kernel [source, heads, uph]
    - QueryAndSharedKeyValueProjection:
        query_projection/kernel [in, heads, uph]
        shared_key_value_projection/kernel [source, heads, uph]
  """
  from sequence_layers.jax.attention import common as attn_common
  from sequence_layers.mlx import projection_configs as mlx_proj

  # Handle Deferred wrapper.
  inner = mlx_attn
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner

  input_projection = config.input_projection

  # Load query projection.
  q_params = linen_params.get('query_projection', {})
  q_kernel = q_params.get('kernel')
  if q_kernel is not None:
    # Shape: [in_features, num_heads, units_per_head] → [in, heads*uph]
    in_features = q_kernel.shape[0]
    inner.q_proj = mx.array(q_kernel.reshape(in_features, -1))
  q_bias = q_params.get('bias')
  if q_bias is not None:
    inner.q_bias = mx.array(q_bias.reshape(-1))

  if isinstance(input_projection, (attn_common.QueryAndKeyValueProjection, mlx_proj.QueryAndKeyValueProjection)):
    # Combined KV: kernel [source, 2, heads, uph] → combined kv_proj.
    kv_params = linen_params.get('key_value_projection', {})
    kv_kernel = kv_params.get('kernel')
    if kv_kernel is not None:
      source_features = kv_kernel.shape[0]
      # Split along axis 1 (the '2' axis for K/V), flatten, recombine.
      k, v = np.split(kv_kernel, 2, axis=1)
      k_flat = k.reshape(source_features, -1)
      v_flat = v.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    kv_bias = kv_params.get('bias')
    if kv_bias is not None:
      kb, vb = np.split(kv_bias, 2, axis=0)
      inner.kv_bias = mx.array(
          np.concatenate([kb.reshape(-1), vb.reshape(-1)], axis=-1)
      )

  elif isinstance(
      input_projection, (attn_common.SeparateQueryKeyValueProjection, mlx_proj.SeparateQueryKeyValueProjection)
  ):
    # Separate K and V projections → combined kv_proj.
    k_params = linen_params.get('key_projection', {})
    k_kernel = k_params.get('kernel')
    v_params = linen_params.get('value_projection', {})
    v_kernel = v_params.get('kernel')
    if k_kernel is not None and v_kernel is not None:
      source_features = k_kernel.shape[0]
      k_flat = k_kernel.reshape(source_features, -1)
      v_flat = v_kernel.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    k_bias = k_params.get('bias')
    v_bias = v_params.get('bias')
    if k_bias is not None and v_bias is not None:
      inner.kv_bias = mx.array(
          np.concatenate([k_bias.reshape(-1), v_bias.reshape(-1)], axis=-1)
      )

  elif isinstance(
      input_projection, (attn_common.QueryAndSharedKeyValueProjection, mlx_proj.QueryAndSharedKeyValueProjection)
  ):
    # Shared K/V projection: same weights for both K and V → combined kv_proj.
    shared_params = linen_params.get('shared_key_value_projection', {})
    shared_kernel = shared_params.get('kernel')
    if shared_kernel is not None:
      source_features = shared_kernel.shape[0]
      proj = shared_kernel.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([proj, proj], axis=-1))
    shared_bias = shared_params.get('bias')
    if shared_bias is not None:
      b = shared_bias.reshape(-1)
      inner.kv_bias = mx.array(np.concatenate([b, b], axis=-1))

  # per_dim_scale: learned [units_per_head] query scale.
  per_dim_scale = linen_params.get('per_dim_scale')
  if per_dim_scale is not None:
    inner._per_dim_scale = mx.array(per_dim_scale)


def _load_rms_norm(mlx_norm, linen_params, config):
  """Load RMSNormalization: scale [dim] → same."""
  scale = linen_params.get('scale')
  if scale is not None:
    scale_mx = mx.array(scale)
    if mlx_norm._use_builtin and mlx_norm._rms_norm is not None:
      mlx_norm._rms_norm.weight = scale_mx
    elif hasattr(mlx_norm, '_scale'):
      mlx_norm._scale = scale_mx


def _load_layer_norm(mlx_norm, linen_params, config):
  """Load LayerNormalization: scale and bias."""
  scale = linen_params.get('scale')
  bias = linen_params.get('bias')

  if mlx_norm._use_builtin and mlx_norm._layer_norm is not None:
    if scale is not None:
      mlx_norm._layer_norm.weight = mx.array(scale)
    if bias is not None:
      mlx_norm._layer_norm.bias = mx.array(bias)
  else:
    if scale is not None and mlx_norm._manual_scale is not None:
      mlx_norm._manual_scale = mx.array(scale)
    if bias is not None and mlx_norm._manual_bias is not None:
      mlx_norm._manual_bias = mx.array(bias)


def _load_embedding(mlx_emb, linen_params, config):
  """Load Embedding: table [vocab, dim] → same."""
  embedding = linen_params.get('embedding')
  if embedding is not None:
    mlx_emb._embedding.weight = mx.array(embedding)


def _load_batch_norm(mlx_bn, linen_params, config, batch_stats=None):
  """Load BatchNormalization: scale/bias from params, mean/var from batch_stats."""
  scale = linen_params.get('scale')
  bias = linen_params.get('bias')

  if scale is not None and mlx_bn.use_scale:
    mlx_bn._scale = mx.array(scale)
  if bias is not None and mlx_bn.use_bias:
    mlx_bn._bias = mx.array(bias)

  if batch_stats is not None:
    mean = batch_stats.get('mean')
    var = batch_stats.get('var')
    if mean is not None:
      mlx_bn._running_mean = mx.array(mean)
    if var is not None:
      mlx_bn._running_var = mx.array(var)


def _load_group_norm(mlx_gn, linen_params, config):
  """Load GroupNormalization: scale and bias."""
  scale = linen_params.get('scale')
  if scale is not None and mlx_gn.use_scale:
    mlx_gn._scale = mx.array(scale)
  bias = linen_params.get('bias')
  if bias is not None and mlx_gn.use_bias:
    mlx_gn._bias = mx.array(bias)


def _load_conv1d(mlx_conv, linen_params, config):
  """Load Conv1D: kernel [k, in, out] → [out, k, in]."""
  inner = mlx_conv
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner

  kernel = linen_params.get('kernel')
  if kernel is not None:
    inner._conv.weight = mx.array(kernel.transpose(2, 0, 1))

  bias = linen_params.get('bias')
  if bias is not None:
    inner._conv.bias = mx.array(bias)


def _load_depthwise_conv1d(mlx_conv, linen_params, config):
  """Load DepthwiseConv1D: same kernel layout as Conv1D."""
  _load_conv1d(mlx_conv, linen_params, config)


def _load_conv1d_transpose(mlx_conv, linen_params, config):
  """Load Conv1DTranspose: kernel [k, in, out] → [out, k, in].

  The kernel is flipped along the spatial axis because Linen uses
  conv_general_dilated with lhs_dilation (correlation), while MLX uses
  conv_transpose1d which reverses the kernel direction.
  """
  inner = mlx_conv
  if hasattr(inner, 'inner') and inner.inner is not None:
    inner = inner.inner

  kernel = linen_params.get('kernel')
  if kernel is not None:
    # Flip spatial axis, then transpose to MLX layout.
    inner.kernel = mx.array(kernel[::-1].transpose(2, 0, 1))

  bias = linen_params.get('bias')
  if bias is not None:
    inner.bias = mx.array(bias)


def _load_conditioning(mlx_cond, linen_params, config):
  """Load Conditioning: projection Dense kernel/bias from 'dense' subdict.

  Linen Conditioning creates a DenseShaped under the name 'dense' for
  LINEAR and LINEAR_AFFINE projections. The kernel shape matches directly
  (input_kernel_shape + output_kernel_shape) since we use the same einsum
  equation.
  """
  from sequence_layers.jax import conditioning as jax_cond

  projection = config.projection
  if projection == jax_cond.BaseConditioning.Projection.IDENTITY:
    return  # No params for identity projection.

  dense_params = linen_params.get('dense', {})
  kernel = dense_params.get('kernel')
  if kernel is not None:
    mlx_cond.kernel = mx.array(kernel)
    mlx_cond._proj_initialized = True
  bias = dense_params.get('bias')
  if bias is not None:
    mlx_cond.bias = mx.array(bias)
