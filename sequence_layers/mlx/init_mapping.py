"""Mapping JAX/Flax initializers and activations to MLX equivalents."""

import functools
import math

import jax
import jax.numpy as jnp
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from flax.linen import initializers as flax_init


def _variance_scaling(key, shape, dtype, mode, distribution, fan_in, fan_out):
  """Variance scaling initializer core logic."""
  dtype = _to_mx_dtype(dtype)
  if mode == 'fan_in':
    denominator = max(fan_in, 1)
  elif mode == 'fan_out':
    denominator = max(fan_out, 1)
  elif mode == 'fan_avg':
    denominator = max((fan_in + fan_out) / 2.0, 1)
  else:
    raise ValueError(f'Unknown mode: {mode}')

  variance = 1.0 / denominator
  if distribution == 'truncated_normal':
    stddev = math.sqrt(variance) / 0.87962566103423978
    return (
        mx.random.truncated_normal(-2.0, 2.0, shape=shape, key=key).astype(
            dtype
        )
        * stddev
    )
  elif distribution == 'normal':
    return mx.random.normal(shape=shape, key=key).astype(dtype) * math.sqrt(
        variance
    )
  elif distribution == 'uniform':
    limit = math.sqrt(3.0 * variance)
    return mx.random.uniform(-limit, limit, shape=shape, key=key).astype(dtype)
  else:
    raise ValueError(f'Unknown distribution: {distribution}')


def _compute_fans(shape):
  """Compute fan_in and fan_out for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    # Conv kernels: last two dims are (fan_in, fan_out), rest are spatial.
    receptive_field_size = 1
    for s in shape[:-2]:
      receptive_field_size *= s
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


def _make_variance_scaling_init(mode, distribution):
  """Create an MLX variance scaling initializer."""

  def init_fn(key, shape, dtype=mx.float32):
    fan_in, fan_out = _compute_fans(shape)
    return _variance_scaling(
        key, shape, dtype, mode, distribution, fan_in, fan_out
    )

  return init_fn


def _to_mx_dtype(dtype):
  """Convert any dtype (JAX, numpy, MLX) to an MLX dtype."""
  if isinstance(dtype, mx.Dtype):
    return dtype
  name = getattr(dtype, '__name__', '') or str(dtype)
  mapping = {
      'float32': mx.float32,
      'float16': mx.float16,
      'bfloat16': mx.bfloat16,
      'float64': mx.float32,  # MLX lacks float64.
      'int32': mx.int32,
      'int64': mx.int32,  # MLX lacks int64.
      'int16': mx.int16,
      'int8': mx.int8,
      'uint8': mx.uint8,
      'uint32': mx.uint32,
      'bool': mx.bool_,
      'bool_': mx.bool_,
      'complex64': mx.complex64,
  }
  for key, val in mapping.items():
    if key in name:
      return val
  return mx.float32


def _zeros_init(key, shape, dtype=mx.float32):
  del key
  return mx.zeros(shape, dtype=_to_mx_dtype(dtype))


def _ones_init(key, shape, dtype=mx.float32):
  del key
  return mx.ones(shape, dtype=_to_mx_dtype(dtype))


def _normal_init(stddev=0.01):
  def init_fn(key, shape, dtype=mx.float32):
    dtype = _to_mx_dtype(dtype)
    return mx.random.normal(shape=shape, key=key).astype(dtype) * stddev

  return init_fn


def map_initializer(jax_init):
  """Convert a JAX/Flax initializer to an MLX-compatible initializer.

  Args:
    jax_init: A JAX/Flax initializer function.

  Returns:
    An MLX initializer function with signature (key, shape, dtype).
  """
  if jax_init is None:
    return None

  # Check for common Flax initializer instances by calling with
  # a probe to determine behavior.
  try:
    # Test with a small shape to determine the initializer type.
    test_key = jax.random.PRNGKey(0)
    test_shape = (4, 4)
    test_out = jax_init(test_key, test_shape, jnp.float32)
    test_np = np.array(test_out)

    # Check if it's zeros.
    if np.allclose(test_np, 0.0):
      return _zeros_init

    # Check if it's ones.
    if np.allclose(test_np, 1.0):
      return _ones_init
  except Exception:
    pass

  # Try to identify by function name or attributes.
  name = getattr(jax_init, '__name__', '')
  qualname = getattr(jax_init, '__qualname__', '')
  func = getattr(jax_init, 'func', None)
  func_qualname = getattr(func, '__qualname__', '') if func else ''

  # Variance scaling variants.
  if 'lecun_normal' in name or 'lecun_normal' in qualname:
    return _make_variance_scaling_init('fan_in', 'truncated_normal')
  if 'lecun_uniform' in name or 'lecun_uniform' in qualname:
    return _make_variance_scaling_init('fan_in', 'uniform')
  if 'glorot_normal' in name or 'glorot_normal' in qualname:
    return _make_variance_scaling_init('fan_avg', 'truncated_normal')
  if 'glorot_uniform' in name or 'glorot_uniform' in qualname:
    return _make_variance_scaling_init('fan_avg', 'uniform')
  if 'he_normal' in name or 'he_normal' in qualname:
    return _make_variance_scaling_init('fan_in', 'normal')
  if 'he_uniform' in name or 'he_uniform' in qualname:
    return _make_variance_scaling_init('fan_in', 'uniform')
  if 'xavier_normal' in name or 'xavier_normal' in qualname:
    return _make_variance_scaling_init('fan_avg', 'normal')
  if 'xavier_uniform' in name or 'xavier_uniform' in qualname:
    return _make_variance_scaling_init('fan_avg', 'uniform')

  # Check for variance_scaling in qualname/func.
  if 'variance_scaling' in qualname or 'variance_scaling' in func_qualname:
    return _make_variance_scaling_init('fan_in', 'truncated_normal')

  if 'zeros' in name or 'zeros' in qualname:
    return _zeros_init
  if 'ones' in name or 'ones' in qualname:
    return _ones_init

  # Default fallback: lecun_normal equivalent.
  return _make_variance_scaling_init('fan_in', 'truncated_normal')


# ---------------------------------------------------------------------------
# Activation mapping
# ---------------------------------------------------------------------------

_ACTIVATION_MAP = {}


def _build_activation_map():
  """Build the JAX -> MLX activation mapping lazily."""
  if _ACTIVATION_MAP:
    return
  _ACTIVATION_MAP.update({
      jax.nn.relu: nn.relu,
      jax.nn.gelu: nn.gelu,
      jax.nn.silu: nn.silu,
      jax.nn.swish: nn.silu,  # swish == silu
      jax.nn.sigmoid: mx.sigmoid,
      jax.nn.tanh: mx.tanh,
      jax.nn.softmax: mx.softmax,
      jax.nn.elu: nn.elu,
      jax.nn.leaky_relu: nn.leaky_relu,
      jax.nn.log_softmax: mx.log,  # Approximate.
  })
  # Also add jnp versions.
  for k, v in list(_ACTIVATION_MAP.items()):
    name = getattr(k, '__name__', '')
    jnp_fn = getattr(jnp, name, None)
    if jnp_fn is not None and jnp_fn not in _ACTIVATION_MAP:
      _ACTIVATION_MAP[jnp_fn] = v


def map_activation(jax_activation):
  """Convert a JAX activation function to its MLX equivalent.

  Args:
    jax_activation: A JAX activation function (e.g. jax.nn.relu).

  Returns:
    The corresponding MLX activation, or the original function
    if no mapping is found.
  """
  if jax_activation is None:
    return None
  _build_activation_map()
  return _ACTIVATION_MAP.get(jax_activation, jax_activation)
