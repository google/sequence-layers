"""Convert Linen-trained params to MLX model weights.

Handles the structural differences between Linen (JAX/Flax) and MLX:
  - Linen Dense kernel [in, out] → MLX nn.Linear weight [out, in]
  - Linen combined QKV kernel [in, 3, heads, uph] → separate q/k/v
  - Linen Repeat stacked params [N, ...] → per-copy params [...]
  - Linen Partitioned wrappers → unwrapped arrays
"""

# pylint: disable=protected-access,redefined-outer-name,unused-argument,unnecessary-lambda,unbalanced-tuple-unpacking

import importlib

from flax import linen as flax_nn
import jax
import mlx.core as mx
import mlx.nn as mlx_nn
import numpy as np

from sequence_layers.jax import attention as jax_attn
from sequence_layers.jax import combinators as jax_comb
from sequence_layers.jax import conditioning as jax_cond
from sequence_layers.jax import convolution as jax_conv
from sequence_layers.jax import dense as jax_dense
from sequence_layers.jax import normalization as jax_norm
from sequence_layers.jax import pooling as jax_pooling
from sequence_layers.jax import simple as jax_simple
from sequence_layers.jax.attention import common as attn_common
from sequence_layers.mlx import attention as mlx_attn
from sequence_layers.mlx import combinators as mlx_comb
from sequence_layers.mlx import convolution2d as mlx_conv
from sequence_layers.mlx import dense as mlx_dense
from sequence_layers.mlx import dsp as mlx_dsp
from sequence_layers.mlx import export as mlx_export
from sequence_layers.mlx import normalization as mlx_norm
from sequence_layers.mlx import projection_configs as mlx_proj
from sequence_layers.mlx import simple as mlx_simple
from sequence_layers.mlx import types as bt


def _get_inner(layer):
  """Unwrap Deferred wrapper to get the actual layer."""
  inner = layer
  if hasattr(inner, "inner") and inner.inner is not None:
    inner = inner.inner
  return inner


def _unbox_params(params):
  """Unwrap Flax Partitioned wrappers and convert to numpy.

  Args:
    params: A Linen param dict (possibly with Partitioned values).

  Returns:
    A nested dict of numpy arrays.
  """
  params = flax_nn.unbox(params)
  return jax.tree_util.tree_map(lambda x: np.array(x), params)


def _set_weight(module, attr_name, value):
  """Set a weight on an MLX module.

  Handles both direct array attributes and nn.Module child params.

  Args:
    module: An MLX nn.Module.
    attr_name: Dot-separated attribute path (e.g. '_linear.weight').
    value: An mx.array value.
  """
  parts = attr_name.split(".")
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
    skip_materialization=False,
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
    skip_materialization: If True, skip materializing deferred layers.
  """
  if not skip_materialization:
    if input_spec is None:
      input_spec = bt.ShapeDType((), mx.int32)

    # Materialize deferred layers with a dummy forward pass.
    # Slice constants to time=1 to match the dummy input.
    init_constants = None
    if constants is not None:
      init_constants = {}
      for k, v in constants.items():
        if hasattr(v, "values") and hasattr(v, "mask"):
          # Slice Sequence to time=1.
          init_constants[k] = bt.Sequence(v.values[:1, :1], v.mask[:1, :1])
        else:
          init_constants[k] = v
    mlx_export._materialize_deferred(
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


def collect_all_params(module):
  """Collect all parameters, including those in unregistered lists."""
  params = {}

  def _recurse(obj, path):
    if isinstance(obj, mx.array):
      params[path] = obj
    elif isinstance(obj, mlx_nn.Module):
      _recurse(obj.parameters(), path)
    elif isinstance(obj, dict):
      for k, v in obj.items():
        _recurse(v, path + (k,))
    elif isinstance(obj, list):
      for i, v in enumerate(obj):
        _recurse(v, path + (str(i),))

  _recurse(module, ())
  return params


def check_unupdated_params(module, params_before):
  """Check if any parameters in the module were NOT updated (same id)."""
  params_after = collect_all_params(module)
  unupdated = []
  for k in params_before:
    if k in params_after and id(params_before[k]) == id(params_after[k]):
      unupdated.append(k)
  return unupdated


_MLX_CONVERTERS = {}


def register_converter(config_cls, mapping_or_fn):
  """Registers a converter for a given Config class."""
  _MLX_CONVERTERS[config_cls] = mapping_or_fn


def _load_config(mlx_module, linen_params, config, batch_stats=None):
  """Recursively load params guided by config type.

  Contract: the config tree must be identical across JAX and MLX — i.e.,
  the same config produced both models, with only the backend import
  swapped (``sequence_layers.jax`` → ``sequence_layers.mlx``). Under this
  contract, both name-based and positional matching in ``_load_serial`` /
  ``_load_residual`` are correct.

  If your model has structural divergence between JAX and MLX (e.g., a layer
  that exists on one side but not the other), register a custom converter
  via ``register_converter`` that remaps JAX param keys before delegating
  to ``load_linen_params``.
  """
  inner = mlx_module
  inner = _get_inner(inner)

  config_cls = type(config)
  original_config_cls = config_cls
  if ".mlx." in config_cls.__module__:
    jax_module_name = config_cls.__module__.replace(".mlx.", ".jax.")
    try:
      jax_module = importlib.import_module(jax_module_name)
      obj = jax_module
      for part in config_cls.__qualname__.split("."):
        obj = getattr(obj, part)
      config_cls = obj
    except (ImportError, AttributeError):
      pass  # Fall back to the actual class if JAX counterpart not found

  mapping_or_fn = None
  if hasattr(config, "name") and config.name:
    mapping_or_fn = _MLX_CONVERTERS.get(config.name)

  if mapping_or_fn is None:
    mapping_or_fn = _MLX_CONVERTERS.get(config_cls)

  if mapping_or_fn is None and original_config_cls != config_cls:
    mapping_or_fn = _MLX_CONVERTERS.get(original_config_cls)

  if mapping_or_fn is None:
    raise ValueError(f"No converter registered for config type: {config_cls}")

  if callable(mapping_or_fn):
    # Custom function fallback
    mapping_or_fn(inner, linen_params, config, batch_stats=batch_stats)
  else:
    # Declarative mapping (dict)
    for jax_name, (mlx_path, transpose) in mapping_or_fn.items():
      if jax_name in linen_params:
        val = mx.array(linen_params[jax_name])
        if transpose:
          val = val.T
        _set_weight(inner, mlx_path, val)
  # Stateless layers (Flatten, Identity, RoPE, pooling, etc.) have no params.


def _load_serial(mlx_serial, linen_params, config, batch_stats=None):
  """Load Serial: try name first, fallback to layers_{i}."""
  for i, layer_config in enumerate(config.layers):
    name = mlx_serial._layer_names[i]

    # Try semantic name first, fallback to positional key.
    if name in linen_params:
      key = name
    else:
      key = f"layers_{i}"

    child_params = linen_params.get(key, {})
    child_bs = batch_stats.get(key, {}) if batch_stats else None

    _load_config(
        mlx_serial.layers[i],
        child_params,
        layer_config,
        batch_stats=child_bs,
    )


def _load_parallel(mlx_parallel, linen_params, config, batch_stats=None):
  """Load Parallel: walk layer names or fallback to layers_{i}."""
  for i, layer_config in enumerate(config.layers):
    name = getattr(layer_config, "name", None)
    if name and name in linen_params:
      key = name
    else:
      key = f"layers_{i}"
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
  child_params = linen_params.get("child_layer", {})
  child_bs = batch_stats.get("child_layer", {}) if batch_stats else None

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
  """Load Residual: body is Serial, shortcut is shortcut_layer."""
  # Body is a Serial inside the Residual.
  body = mlx_residual.body
  for i, layer_config in enumerate(config.layers):
    name = body._layer_names[i]

    # Try semantic name first, fallback to positional key.
    if name in linen_params:
      key = name
    else:
      key = f"layers_{i}"

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
    shortcut_params = linen_params.get("shortcut_layer", {})
    shortcut_bs = batch_stats.get("shortcut_layer", {}) if batch_stats else None
    if len(config.shortcut_layers) == 1:
      shortcut_layers_mlx = [mlx_residual.shortcut]
    else:
      shortcut_layers_mlx = mlx_residual.shortcut.layers

    for i, sc_config in enumerate(config.shortcut_layers):
      if len(config.shortcut_layers) == 1:
        name = mlx_residual.shortcut.name
      else:
        name = mlx_residual.shortcut._layer_names[i]

      if name in shortcut_params:
        sc_key = name
      else:
        sc_key = f"layers_{i}"

      sc_bs = shortcut_bs.get(sc_key, {}) if shortcut_bs else None
      _load_config(
          shortcut_layers_mlx[i],
          shortcut_params.get(sc_key, {}),
          sc_config,
          batch_stats=sc_bs,
      )


def _load_dense(mlx_dense, linen_params, config, batch_stats=None):
  """Load Dense: transpose kernel [in, out] → [out, in]."""
  # Handle DenseDeferred wrapper.
  inner = mlx_dense
  inner = _get_inner(inner)

  kernel = linen_params.get("kernel")
  if kernel is not None:
    # Linen: [in, out], MLX nn.Linear: [out, in]
    weight = mx.array(kernel.T)
    inner._linear.weight = weight

  bias = linen_params.get("bias")
  if bias is not None:
    inner._linear.bias = mx.array(bias)


def _load_einsum_dense(mlx_einsum, linen_params, config, batch_stats=None):
  """Load EinsumDense: kernel shape matches directly (einsum notation)."""
  kernel = linen_params.get("kernel")
  if kernel is not None:
    mlx_einsum.kernel = mx.array(kernel)
    mlx_einsum._initialized = True
  bias = linen_params.get("bias")
  if bias is not None:
    mlx_einsum.bias = mx.array(bias)


def _load_attention(mlx_attn, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load DotProductSelfAttention.

  Handles:
    - CombinedQueryKeyValueProjection:
        query_key_value_projection/kernel [in, 3, heads, uph]
    - SeparateQueryKeyValueProjection:
        query_projection/kernel [in, heads, uph]
        key_projection/kernel [in, kv_heads, uph]
        value_projection/kernel [in, kv_heads, uph]
  """

  # Handle Deferred wrapper.
  inner = mlx_attn
  inner = _get_inner(inner)

  input_projection = config.input_projection

  if isinstance(
      input_projection,
      (
          attn_common.CombinedQueryKeyValueProjection,
          mlx_proj.CombinedQueryKeyValueProjection,
      ),
  ):
    # Combined QKV: kernel [in, 3, heads, uph]
    qkv_params = linen_params.get("query_key_value_projection", {})
    combined_kernel = qkv_params.get("kernel")
    if combined_kernel is not None:
      in_features = combined_kernel.shape[0]
      if hasattr(inner, "qkv_proj"):
        inner.qkv_proj = mx.array(combined_kernel.reshape(in_features, -1))
      else:
        # Separate Q + combined KV layout.
        q, k, v = np.split(combined_kernel, 3, axis=1)
        inner.q_proj = mx.array(q.reshape(in_features, -1))
        k_flat = k.reshape(in_features, -1)
        v_flat = v.reshape(in_features, -1)
        inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))

    combined_bias = qkv_params.get("bias")
    if combined_bias is not None:
      if hasattr(inner, "qkv_bias"):
        inner.qkv_bias = mx.array(combined_bias.reshape(-1))
      else:
        qb, kb, vb = np.split(combined_bias, 3, axis=0)
        inner.q_bias = mx.array(qb.reshape(-1))
        inner.kv_bias = mx.array(
            np.concatenate([kb.reshape(-1), vb.reshape(-1)], axis=-1)
        )

  elif isinstance(
      input_projection,
      (
          attn_common.SeparateQueryKeyValueProjection,
          mlx_proj.SeparateQueryKeyValueProjection,
      ),
  ):
    # Separate Q/K/V projections (used for GQA where num_kv_heads < num_heads).
    q_params = linen_params.get("query_projection", {})
    q_kernel = q_params.get("kernel")
    if q_kernel is not None:
      in_features = q_kernel.shape[0]
      inner.q_proj = mx.array(q_kernel.reshape(in_features, -1))
    q_bias = q_params.get("bias")
    if q_bias is not None:
      inner.q_bias = mx.array(q_bias.reshape(-1))

    k_params = linen_params.get("key_projection", {})
    k_kernel = k_params.get("kernel")
    v_params = linen_params.get("value_projection", {})
    v_kernel = v_params.get("kernel")
    if k_kernel is not None and v_kernel is not None:
      in_features = k_kernel.shape[0]
      k_flat = k_kernel.reshape(in_features, -1)
      v_flat = v_kernel.reshape(in_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    k_bias = k_params.get("bias")
    v_bias = v_params.get("bias")
    if k_bias is not None and v_bias is not None:
      inner.kv_bias = mx.array(
          np.concatenate([k_bias.reshape(-1), v_bias.reshape(-1)], axis=-1)
      )

  # per_dim_scale: learned [units_per_head] query scale.
  per_dim_scale = linen_params.get("per_dim_scale")
  if per_dim_scale is not None:
    inner._per_dim_scale = mx.array(per_dim_scale)

  # Attention sink embeddings.
  sink_key = linen_params.get("sink_key_embeddings")
  if sink_key is not None:
    inner.sink_key_embeddings = mx.array(sink_key)
  sink_value = linen_params.get("sink_value_embeddings")
  if sink_value is not None:
    inner.sink_value_embeddings = mx.array(sink_value)

  # Q/K/V processing networks have no trainable params
  # (RoPE is stateless with no learned weights).


def _load_streaming_attention(mlx_attn, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
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

  # Handle Deferred wrapper.
  inner = mlx_attn
  inner = _get_inner(inner)

  input_projection = config.input_projection

  # Load query projection.
  q_params = linen_params.get("query_projection", {})
  q_kernel = q_params.get("kernel")
  if q_kernel is not None:
    # Shape: [in_features, num_heads, units_per_head] → [in, heads*uph]
    in_features = q_kernel.shape[0]
    inner.q_proj = mx.array(q_kernel.reshape(in_features, -1))
  q_bias = q_params.get("bias")
  if q_bias is not None:
    inner.q_bias = mx.array(q_bias.reshape(-1))

  if isinstance(
      input_projection,
      (
          attn_common.QueryAndKeyValueProjection,
          mlx_proj.QueryAndKeyValueProjection,
      ),
  ):
    # Combined KV: kernel [source, 2, heads, uph] → combined kv_proj.
    kv_params = linen_params.get("key_value_projection", {})
    kv_kernel = kv_params.get("kernel")
    if kv_kernel is not None:
      source_features = kv_kernel.shape[0]
      # Split along axis 1 (the '2' axis for K/V), flatten, recombine.
      k, v = np.split(kv_kernel, 2, axis=1)
      k_flat = k.reshape(source_features, -1)
      v_flat = v.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    kv_bias = kv_params.get("bias")
    if kv_bias is not None:
      kb, vb = np.split(kv_bias, 2, axis=0)
      inner.kv_bias = mx.array(
          np.concatenate([kb.reshape(-1), vb.reshape(-1)], axis=-1)
      )

  elif isinstance(
      input_projection,
      (
          attn_common.SeparateQueryKeyValueProjection,
          mlx_proj.SeparateQueryKeyValueProjection,
      ),
  ):
    # Separate K and V projections → combined kv_proj.
    k_params = linen_params.get("key_projection", {})
    k_kernel = k_params.get("kernel")
    v_params = linen_params.get("value_projection", {})
    v_kernel = v_params.get("kernel")
    if k_kernel is not None and v_kernel is not None:
      source_features = k_kernel.shape[0]
      k_flat = k_kernel.reshape(source_features, -1)
      v_flat = v_kernel.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([k_flat, v_flat], axis=-1))
    k_bias = k_params.get("bias")
    v_bias = v_params.get("bias")
    if k_bias is not None and v_bias is not None:
      inner.kv_bias = mx.array(
          np.concatenate([k_bias.reshape(-1), v_bias.reshape(-1)], axis=-1)
      )

  elif isinstance(
      input_projection,
      (
          attn_common.QueryAndSharedKeyValueProjection,
          mlx_proj.QueryAndSharedKeyValueProjection,
      ),
  ):
    # Shared K/V projection: same weights for both K and V → combined kv_proj.
    shared_params = linen_params.get("shared_key_value_projection", {})
    shared_kernel = shared_params.get("kernel")
    if shared_kernel is not None:
      source_features = shared_kernel.shape[0]
      proj = shared_kernel.reshape(source_features, -1)
      inner.kv_proj = mx.array(np.concatenate([proj, proj], axis=-1))
    shared_bias = shared_params.get("bias")
    if shared_bias is not None:
      b = shared_bias.reshape(-1)
      inner.kv_bias = mx.array(np.concatenate([b, b], axis=-1))

  # per_dim_scale: learned [units_per_head] query scale.
  per_dim_scale = linen_params.get("per_dim_scale")
  if per_dim_scale is not None:
    inner._per_dim_scale = mx.array(per_dim_scale)

  # Attention sink embeddings.
  sink_key = linen_params.get("sink_key_embeddings")
  if sink_key is not None:
    inner.sink_key_embeddings = mx.array(sink_key)
  sink_value = linen_params.get("sink_value_embeddings")
  if sink_value is not None:
    inner.sink_value_embeddings = mx.array(sink_value)


def _load_rms_norm(mlx_norm, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load RMSNormalization: scale [dim] → same."""
  scale = linen_params.get("scale")
  if scale is not None:
    scale_mx = mx.array(scale)
    if mlx_norm._use_builtin and mlx_norm._rms_norm is not None:
      mlx_norm._rms_norm.weight = scale_mx
    elif hasattr(mlx_norm, "_scale"):
      mlx_norm._scale = scale_mx


def _load_layer_norm(mlx_norm, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load LayerNormalization: scale and bias."""
  scale = linen_params.get("scale")
  bias = linen_params.get("bias")

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


def _load_embedding(mlx_emb, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Embedding: table [vocab, dim] → same."""
  embedding = linen_params.get("embedding")
  if embedding is not None:
    mlx_emb._embedding.weight = mx.array(embedding)


def _load_batch_norm(mlx_bn, linen_params, config, batch_stats=None):
  """Load BatchNormalization: scale/bias from params, mean/var from batch_stats."""
  scale = linen_params.get("scale")
  bias = linen_params.get("bias")

  if scale is not None and mlx_bn.use_scale:
    mlx_bn._scale = mx.array(scale)
  if bias is not None and mlx_bn.use_bias:
    mlx_bn._bias = mx.array(bias)

  if batch_stats is not None:
    mean = batch_stats.get("mean")
    var = batch_stats.get("var")
    if mean is not None:
      mlx_bn._running_mean = mx.array(mean)
    if var is not None:
      mlx_bn._running_var = mx.array(var)


def _load_group_norm(mlx_gn, linen_params, config, batch_stats=None):
  """Load GroupNormalization: scale and bias."""
  scale = linen_params.get("scale")
  if scale is not None and mlx_gn.use_scale:
    mlx_gn._scale = mx.array(scale)
  bias = linen_params.get("bias")
  if bias is not None and mlx_gn.use_bias:
    mlx_gn._bias = mx.array(bias)


def _load_conv1d(mlx_conv, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Conv1D: kernel [k, in, out] → [out, k, in]."""
  inner = mlx_conv
  inner = _get_inner(inner)

  kernel = linen_params.get("kernel")
  if kernel is not None:
    inner._conv.weight = mx.array(kernel.transpose(2, 0, 1))

  bias = linen_params.get("bias")
  if bias is not None:
    inner._conv.bias = mx.array(bias)


def _load_depthwise_conv1d(mlx_conv, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load DepthwiseConv1D: same kernel layout as Conv1D."""
  _load_conv1d(mlx_conv, linen_params, config)


def _load_conv1d_transpose(mlx_conv, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Conv1DTranspose: kernel [k, in, out] → [out, k, in].

  The kernel is flipped along the spatial axis because Linen uses
  conv_general_dilated with lhs_dilation (correlation), while MLX uses
  conv_transpose1d which reverses the kernel direction.
  """
  inner = mlx_conv
  inner = _get_inner(inner)

  kernel = linen_params.get("kernel")
  if kernel is not None:
    # Flip spatial axis, then transpose to MLX layout.
    inner.kernel = mx.array(kernel[::-1].transpose(2, 0, 1))

  bias = linen_params.get("bias")
  if bias is not None:
    inner.bias = mx.array(bias)


def _load_conv2d(mlx_conv, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Conv2D: kernel [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin]."""
  inner = mlx_conv
  inner = _get_inner(inner)

  kernel = linen_params.get("kernel")
  if kernel is not None:
    # Reorder: [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin]
    inner.kernel = mx.array(kernel.transpose(3, 0, 1, 2))

  bias = linen_params.get("bias")
  if bias is not None:
    inner.bias = mx.array(bias)


def _load_conv2d_transpose(mlx_conv, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Conv2DTranspose: kernel [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin].

  JAX's conv_transpose flips the kernel (true mathematical transposition),
  but MLX's conv_transpose2d does NOT. We must flip the kernel along both
  spatial dimensions (kH, kW) to compensate.
  """
  inner = mlx_conv
  inner = _get_inner(inner)

  kernel = linen_params.get("kernel")
  if kernel is not None:
    # Flip along kH and kW to match JAX's implicit kernel flip.
    kernel = kernel[::-1, ::-1, :, :]
    # Reorder: [kH, kW, Cin, Cout] -> [Cout, kH, kW, Cin]
    inner.kernel = mx.array(kernel.transpose(3, 0, 1, 2))

  bias = linen_params.get("bias")
  if bias is not None:
    inner.bias = mx.array(bias)


def _load_conditioning(mlx_cond, linen_params, config, batch_stats=None):
  # pylint: disable=unused-argument
  """Load Conditioning: projection Dense kernel/bias from 'dense' subdict.

  Linen Conditioning creates a DenseShaped under the name 'dense' for
  LINEAR and LINEAR_AFFINE projections. The kernel shape matches directly
  (input_kernel_shape + output_kernel_shape) since we use the same einsum
  equation.
  """

  projection = config.projection
  if projection == jax_cond.BaseConditioning.Projection.IDENTITY:
    return  # No params for identity projection.

  dense_params = linen_params.get("dense", {})
  kernel = dense_params.get("kernel")
  if kernel is not None:
    mlx_cond.kernel = mx.array(kernel)
    mlx_cond._proj_initialized = True
  bias = dense_params.get("bias")
  if bias is not None:
    mlx_cond.bias = mx.array(bias)


# === Registrations ===


register_converter(jax_comb.Serial.Config, _load_serial)
register_converter(mlx_comb.Serial.Config, _load_serial)
register_converter(jax_comb.Parallel.Config, _load_parallel)
register_converter(mlx_comb.Parallel.Config, _load_parallel)
register_converter(jax_comb.Repeat.Config, _load_repeat)
register_converter(jax_comb.Residual.Config, _load_residual)
register_converter(mlx_comb.Residual.Config, _load_residual)

register_converter(
    jax_dense.Dense.Config,
    {
        "kernel": ("_linear.weight", True),
        "bias": ("_linear.bias", False),
    },
)
register_converter(
    mlx_dense.Dense.Config,
    {
        "kernel": ("_linear.weight", True),
        "bias": ("_linear.bias", False),
    },
)

register_converter(jax_dense.EinsumDense.Config, _load_einsum_dense)
register_converter(jax_norm.RMSNormalization.Config, _load_rms_norm)
register_converter(mlx_norm.RMSNormalization.Config, _load_rms_norm)
register_converter(jax_norm.LayerNormalization.Config, _load_layer_norm)
register_converter(mlx_norm.LayerNormalization.Config, _load_layer_norm)
register_converter(jax_norm.BatchNormalization.Config, _load_batch_norm)
register_converter(jax_norm.GroupNormalization.Config, _load_group_norm)

register_converter(jax_conv.Conv1D.Config, _load_conv1d)
register_converter(jax_conv.DepthwiseConv1D.Config, _load_depthwise_conv1d)
register_converter(jax_conv.Conv1DTranspose.Config, _load_conv1d_transpose)
register_converter(jax_conv.Conv2D.Config, _load_conv2d)
register_converter(mlx_conv.Conv2D.Config, _load_conv2d)
register_converter(jax_conv.Conv2DTranspose.Config, _load_conv2d_transpose)
register_converter(mlx_conv.Conv2DTranspose.Config, _load_conv2d_transpose)

register_converter(
    jax_attn.DotProductAttention.Config, _load_streaming_attention
)
register_converter(
    jax_attn.StreamingDotProductAttention.Config,
    _load_streaming_attention,
)
register_converter(
    jax_attn.StreamingLocalDotProductAttention.Config,
    _load_streaming_attention,
)
register_converter(
    jax_attn.LocalDotProductSelfAttention.Config, _load_attention
)
register_converter(jax_attn.DotProductSelfAttention.Config, _load_attention)

register_converter(jax_cond.Conditioning.Config, _load_conditioning)
register_converter(
    jax_simple.Embedding.Config,
    {
        "embedding": ("_embedding.weight", False),
    },
)
register_converter(
    mlx_simple.Embedding.Config,
    {
        "embedding": ("_embedding.weight", False),
    },
)


register_converter(
    mlx_attn.LocalDotProductSelfAttention.Config, _load_attention
)
register_converter(
    mlx_attn.StreamingDotProductAttention.Config,
    _load_streaming_attention,
)
register_converter(mlx_dense.EinsumDense.Config, _load_einsum_dense)
register_converter(mlx_norm.GroupNormalization.Config, _load_group_norm)


# Stateless layers (no params to load).
for config_cls in [
    jax_simple.Identity.Config,
    jax_simple.Logging.Config,
    jax_simple.Relu.Config,
    jax_simple.Gelu.Config,
    jax_simple.Abs.Config,
    jax_simple.Exp.Config,
    jax_simple.Log.Config,
    jax_simple.Swish.Config,
    jax_simple.Tanh.Config,
    jax_simple.Sigmoid.Config,
    jax_simple.LeakyRelu.Config,
    jax_simple.Elu.Config,
    jax_simple.Softmax.Config,
    jax_simple.Softplus.Config,
    jax_simple.Cast.Config,
    jax_simple.Flatten.Config,
    jax_simple.Reshape.Config,
    jax_simple.ExpandDims.Config,
    jax_simple.Squeeze.Config,
    jax_simple.Transpose.Config,
    jax_simple.OneHot.Config,
    jax_simple.Lambda.Config,
    jax_simple.Upsample2D.Config,
    jax_pooling.MaxPooling1D.Config,
    jax_pooling.MinPooling1D.Config,
    jax_pooling.AveragePooling1D.Config,
    jax_pooling.AveragePooling2D.Config,
    # MLX-native stateless layers.
    mlx_simple.Identity.Config,
    mlx_simple.Logging.Config,
    mlx_simple.Scale.Config,
    mlx_simple.Add.Config,
    mlx_simple.Relu.Config,
    mlx_simple.Gelu.Config,
    mlx_simple.Swish.Config,
    mlx_simple.Elu.Config,
    mlx_simple.Flatten.Config,
    mlx_simple.Reshape.Config,
    mlx_simple.ExpandDims.Config,
    mlx_simple.GatedUnit.Config,
    mlx_simple.GatedLinearUnit.Config,
    mlx_simple.GatedTanhUnit.Config,
    mlx_simple.Lambda.Config,
    mlx_conv.Upsample2D.Config,
    mlx_conv.AveragePooling2D.Config,
]:
  register_converter(config_cls, {})

# MLX-native layers with special handling.


for config_cls in [
    mlx_simple.CheckpointName.Config,
    mlx_simple.Dropout.Config,
    mlx_dsp.Delay.Config,
]:
  register_converter(config_cls, {})
