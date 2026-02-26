"""Utility functions for MLX sequence layers."""

import fractions

from sequence_layers.mlx.combinators import CombinationMode


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
  output_ratio = layer.output_ratio
  return int(output_latency * output_ratio) + layer.output_latency


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
