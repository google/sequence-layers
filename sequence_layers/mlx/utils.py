"""Utility functions for MLX sequence layers."""

import dataclasses
import inspect
from typing import Any

from mlx import nn
import mlx.core as mx
import numpy as np

from sequence_layers.mlx import init_mapping
from sequence_layers.specs import combinators as spec_combinators
from sequence_layers.specs import types as specs_types

CombinationMode = spec_combinators.CombinationMode


def get_output_latency(config, accumulated_output_latency=0):
  """Returns the output latency of the provided SequenceLayerConfig.

  In MLX, we can simply instantiate the layer and compute the latency
  directly without needing JAX's eval_shape.

  Args:
    config: A SequenceLayerConfig to compute output latency for.
    accumulated_output_latency: The accumulated output latency of preceding
      layers. Defaults to 0.

  Returns:
    The output latency of the layer.
  """
  layer = config.make()
  return _get_accumulated_output_latency(layer, accumulated_output_latency)


def _get_accumulated_output_latency(layer, output_latency):
  """Computes accumulated output latency for a layer.

  Mirrors SequenceLayer.get_accumulated_output_latency from JAX types.
  """
  # Check for Serial-like combinators that chain layers.
  if hasattr(layer, 'layers') and isinstance(layer.layers, (list, tuple)):
    for sub in layer.layers:
      output_latency = _get_accumulated_output_latency(sub, output_latency)
    return output_latency

  # Check for internal body (Residual stores layers in _body).
  if hasattr(layer, '_body'):
    return _get_accumulated_output_latency(layer.body, output_latency)

  # Check for deferred layers that wrap another layer.
  if hasattr(layer, '_layer') and layer.inner is not None:
    return _get_accumulated_output_latency(layer.inner, output_latency)
  if hasattr(layer, '_child'):
    return _get_accumulated_output_latency(layer.child, output_latency)

  # Single layer: compute latency.
  return layer.get_accumulated_output_latency(output_latency)


def get_required_stepwise_delay(output_ratio, input_latency):
  """Returns the delay required so input_latency is divisible by 1/output_ratio.

  When combining upsampling and downsampling layers with latency,
  layer/step equivalence requires inserting delays. This function returns the
  correct amount of step-wise delay to insert.

  Args:
    output_ratio: The output ratio of the layer (a fractions.Fraction).
    input_latency: The accumulated input latency of layers preceding the layer.

  Returns:
    The amount of delay required to ensure input latency is divisible by
    output_ratio.
  """
  if 1 not in output_ratio.as_integer_ratio():
    raise NotImplementedError(
        'get_required_stepwise_delay expects integer upsampling or'
        f' downsampling, got {output_ratio=}'
    )
  return int(-input_latency % (1 / output_ratio))


def _to_mx_dtype(dtype: Any) -> Any:
  """Converts various dtype representations to MLX DType."""
  if dtype is None:
    return None
  return init_mapping._to_mx_dtype(dtype)


def _map_activation(act: Any) -> Any:
  """Maps an activation function or its name to the corresponding MLX activation."""
  if act is None:
    return None
  if not callable(act):
    return act

  name = getattr(act, '__name__', None)
  if name is None:
    return act

  activations = {
      'relu': nn.relu,
      'gelu': nn.gelu,
      'silu': nn.silu,
      'swish': nn.silu,
      'sigmoid': mx.sigmoid,
      'tanh': mx.tanh,
      'elu': nn.elu,
      'softmax': mx.softmax,
      'softplus': nn.softplus,
  }
  return activations.get(name, act)


# pylint: disable=too-many-nested-blocks
def make_layer(config, backend='mlx') -> Any:
  """Instantiates an MLX layer from a JAX or Spec config."""

  # 1. Try calling config.make() if it supports backend argument.
  if (
      hasattr(config, 'make')
      and type(config).make != specs_types.SequenceLayerConfig.make
  ):
    sig = inspect.signature(config.make)
    if 'backend' in sig.parameters:
      layer = config.make(backend=backend)
      if layer is not None:
        return layer
    # If it's an MLX-specific config, it might have no-arg make() returning MLX layer.
    config_module = config.__class__.__module__
    if 'mlx' in config_module:
      layer = config.make()
      if layer is not None:
        return layer

  # 2. Fallback to from_config resolution.
  config_class = config.__class__
  parts = config_class.__qualname__.split('.')
  if len(parts) > 1 and parts[-1] == 'Config':
    class_name = parts[-2]
  else:
    class_name = config_class.__name__
    if class_name.endswith('Config'):
      class_name = class_name[:-6]

  import sequence_layers.mlx as mlx_module  # pylint: disable=import-outside-toplevel

  if not hasattr(mlx_module, class_name):
    raise AttributeError(
        f"Concrete MLX class '{class_name}' not found in sequence_layers.mlx."
        ' Make sure it is imported and exposed in'
        ' sequence_layers/mlx/__init__.py.'
    )
  mlx_class = getattr(mlx_module, class_name)

  if hasattr(mlx_class, 'from_config'):
    sig = inspect.signature(mlx_class.from_config)
    if 'backend' in sig.parameters:
      return mlx_class.from_config(config, backend=backend)
    return mlx_class.from_config(config)

  # 3. Dynamic conversion fallback for leaf layers without from_config.
  if hasattr(mlx_class, 'Config') and dataclasses.is_dataclass(
      mlx_class.Config
  ):
    mlx_config_class = mlx_class.Config
    mlx_fields = {f.name: f for f in dataclasses.fields(mlx_config_class)}

    kwargs = {}
    for f in dataclasses.fields(config):
      if f.name in mlx_fields:
        val = getattr(config, f.name)

        # Map activations and dtypes
        if f.name == 'activation':
          val = _map_activation(val)
        elif 'dtype' in f.name:
          val = _to_mx_dtype(val)

        # Recursively convert nested configs
        if isinstance(val, (list, tuple)):
          new_val = []
          for item in val:
            if hasattr(item, '__class__') and dataclasses.is_dataclass(item):
              try:
                new_val.append(make_layer(item, backend=backend))
              except Exception:  # pylint: disable=broad-exception-caught
                new_val.append(item)
            else:
              new_val.append(item)
          val = type(val)(new_val)
        elif hasattr(val, '__class__') and dataclasses.is_dataclass(val):
          try:
            val = make_layer(val, backend=backend)
          except Exception:  # pylint: disable=broad-exception-caught
            pass

        kwargs[f.name] = val

    try:
      mlx_config = mlx_config_class(**kwargs)
      if (
          hasattr(mlx_config, 'make')
          and type(mlx_config).make != specs_types.SequenceLayerConfig.make
      ):
        return mlx_config.make()
      return mlx_class(mlx_config)
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise AttributeError(
          f"Concrete MLX class '{class_name}' does not implement from_config "
          f'and dynamic instantiation failed: {e}'
      ) from e

  raise AttributeError(
      f"Concrete MLX class '{class_name}' does not implement from_config "
      'and has no Config dataclass for dynamic instantiation.'
  )


# pylint: enable=too-many-nested-blocks


def call_layer_with_emits(
    layer, x, *, training=False, constants=None, **kwargs
):
  """Calls layer_with_emits, forwarding training and constants."""
  return layer.layer_with_emits(x, training=training, constants=constants, **kwargs)


def call_step_with_emits(
    layer, x, state, *, training=False, constants=None, **kwargs
):
  """Calls step_with_emits, forwarding training and constants."""
  return layer.step_with_emits(x, state, training=training, constants=constants, **kwargs)


def call_get_initial_state(
    layer, batch_size, input_spec, *, training=False, constants=None, **kwargs
):
  """Calls get_initial_state, forwarding training and constants."""
  return layer.get_initial_state(batch_size, input_spec, training=training, constants=constants, **kwargs)


def _patch_spec_configs():
  # pylint: disable=import-outside-toplevel,missing-function-docstring,reimported
  def _make(self):
    return make_layer(self)

  # Patch the base class
  specs_types.SequenceLayerConfig.make = _make

  # Patch all spec modules dynamically
  modules: list[Any] = []
  try:
    from sequence_layers.specs import combinators as spec_comb

    modules.append(spec_comb)
  except ImportError:
    pass
  try:
    from sequence_layers.specs import convolution as spec_conv

    modules.append(spec_conv)
  except ImportError:
    pass
  try:
    from sequence_layers.specs import dense as spec_dense

    modules.append(spec_dense)
  except ImportError:
    pass
  try:
    from sequence_layers.specs import normalization as spec_norm

    modules.append(spec_norm)
  except ImportError:
    pass
  try:
    from sequence_layers.specs import pooling as spec_pool

    modules.append(spec_pool)
  except ImportError:
    pass
  try:
    from sequence_layers.specs import simple as spec_simple

    modules.append(spec_simple)
  except ImportError:
    pass

  for mod in modules:
    for name in dir(mod):
      attr = getattr(mod, name)
      if isinstance(attr, type) and hasattr(attr, 'Config'):
        config_cls = getattr(attr, 'Config')
        if isinstance(config_cls, type) and issubclass(
            config_cls, specs_types.SequenceLayerConfig
        ):
          config_cls.make = _make


_patch_spec_configs()
