"""Export MLX SequenceLayer step() to .mlxfn for streaming inference."""

import mlx.core as mx

from sequence_layers.mlx import basic_types as bt

Sequence = bt.Sequence


# ---------------------------------------------------------------------------
# State flattening / unflattening
# ---------------------------------------------------------------------------


def _flatten_state(state):
  """Flatten a nested pytree state into a list of mx.array.

  Handles tuples, lists, and mx.array leaves. Empty tuples contribute
  zero arrays.

  Args:
    state: Nested tuple/list of mx.array.

  Returns:
    (flat_arrays, structure) where structure encodes the nesting.
  """
  flat = []

  def _record(node):
    if isinstance(node, mx.array):
      flat.append(node)
      return 'array'
    elif isinstance(node, tuple):
      children = [_record(child) for child in node]
      return ('tuple', children)
    elif isinstance(node, list):
      children = [_record(child) for child in node]
      return ('list', children)
    else:
      raise TypeError(f'Unsupported state node type: {type(node)}')

  structure = _record(state)
  return flat, structure


def _unflatten_state(flat, structure):
  """Reconstruct a nested state from a flat array list and structure.

  Args:
    flat: List of mx.array.
    structure: Structure descriptor from _flatten_state.

  Returns:
    Nested tuple/list matching the original structure.
  """
  idx = [0]

  def _rebuild(struct):
    if struct == 'array':
      result = flat[idx[0]]
      idx[0] += 1
      return result
    elif isinstance(struct, tuple) and struct[0] == 'tuple':
      return tuple(_rebuild(s) for s in struct[1])
    elif isinstance(struct, tuple) and struct[0] == 'list':
      return [_rebuild(s) for s in struct[1]]
    else:
      raise ValueError(f'Unknown structure node: {struct}')

  result = _rebuild(structure)
  if idx[0] != len(flat):
    raise ValueError(f'Not all arrays consumed: used {idx[0]} of {len(flat)}')
  return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _materialize_deferred(model, batch_size, input_spec, *, constants=None):
  """Run a dummy forward pass to materialize all deferred layers."""
  x_values = mx.zeros(
      (batch_size, 1) + input_spec.shape, dtype=input_spec.dtype
  )
  x_mask = mx.ones((batch_size, 1), dtype=mx.bool_)
  x = Sequence(x_values, x_mask)
  state = model.get_initial_state(batch_size, input_spec, constants=constants)
  model.step(x, state, constants=constants)
  mx.eval(model.parameters())


def get_initial_state_flat(model, batch_size, input_spec, *, constants=None):
  """Get flattened initial state arrays and structure for a model.

  Args:
    model: An MLX SequenceLayer.
    batch_size: Batch size.
    input_spec: A ShapeDType describing the input channels.
    constants: Optional constants dict.

  Returns:
    (flat_arrays, structure) where flat_arrays is a list of mx.array
    and structure can be used with _unflatten_state.
  """
  state = model.get_initial_state(batch_size, input_spec, constants=constants)
  flat, structure = _flatten_state(state)
  mx.eval(*flat) if flat else None
  return flat, structure


def export_step(
    model,
    path,
    batch_size,
    input_spec,
    *,
    constants=None,
    time_steps=1,
):
  """Export model.step() to a .mlxfn file.

  The exported function signature is:
    (x_values, x_mask, *state_flat) -> (y_values, y_mask, *new_state_flat)

  Model weights are captured in the closure and embedded in the .mlxfn
  file. State arrays (e.g. KV cache) are explicit I/O.

  The exported function uses fixed shapes (batch_size, time_steps).
  For streaming generation, time_steps=1 is typical.

  Args:
    model: An MLX SequenceLayer with supports_step.
    path: Output file path (should end in .mlxfn).
    batch_size: Batch size for the exported function.
    input_spec: A ShapeDType describing the input channel shape and dtype.
    constants: Optional constants dict for cross-attention.
    time_steps: Number of time steps per call (default 1).
  """
  if not model.supports_step:
    raise ValueError(f'{model.__class__.__name__} does not support step().')

  # Materialize all deferred layers.
  _materialize_deferred(model, batch_size, input_spec, constants=constants)

  # Get initial state and flatten.
  flat_state, structure = get_initial_state_flat(
      model, batch_size, input_spec, constants=constants
  )

  # Make sure all model params are evaluated.
  mx.eval(model.parameters())

  def step_fn(x_values, x_mask, *state_flat):
    state = _unflatten_state(list(state_flat), structure)
    x = Sequence(x_values, x_mask)
    y, new_state = model.step(x, state, constants=constants)
    new_flat, _ = _flatten_state(new_state)
    return (y.values, y.mask, *new_flat)

  # Create example inputs for tracing.
  x_values = mx.zeros(
      (batch_size, time_steps) + input_spec.shape,
      dtype=input_spec.dtype,
  )
  x_mask = mx.ones((batch_size, time_steps), dtype=mx.bool_)
  mx.eval(x_values, x_mask)

  mx.export_function(
      path,
      step_fn,
      x_values,
      x_mask,
      *flat_state,
  )


def run_exported(imported_fn, x_values, x_mask, state_flat):
  """Call an imported .mlxfn step function.

  Args:
    imported_fn: A function from mx.import_function().
    x_values: Input values array [batch, time, ...channels].
    x_mask: Input mask array [batch, time].
    state_flat: List of flat state arrays.

  Returns:
    (y_values, y_mask, new_state_flat) where new_state_flat is a list.
  """
  results = imported_fn(x_values, x_mask, *state_flat)
  y_values = results[0]
  y_mask = results[1]
  new_state_flat = list(results[2:])
  return y_values, y_mask, new_state_flat
