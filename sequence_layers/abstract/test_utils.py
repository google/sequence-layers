"""Abstract test utilities."""

import abc
import fractions
import typing
from typing import Any, Callable, Sequence as TypingSequence
import dataclasses
from absl.testing import parameterized
import numpy as np
from sequence_layers.abstract import types
import itertools
from typing import Any, Callable, Sequence as TypingSequence, Iterable, Mapping, TypeVar

_T = TypeVar('_T')
_TestFnT = Callable[..., None]

def zip_longest(
    targets: Iterable[Iterable[Any]],
    sources: Iterable[_T],
) -> list[_T]:
  """Applies zip_longest, specialized to @parameterized's argument format.

  Args:
    targets: Iterable of parameterized test arguments.
    sources: Iterable of parameterized test arguments. If `targets` is a mapping
      `sources` must be a mapping as well.

  Returns:
    A list of the zipped arguments, of the type of `targets` and with each
    zipped argument internally sorted (target, source). If either input sequence
    was longer, the last element of the shorter input sequence is repeated.
  """

  results = []
  prev_source, prev_target = None, None
  for source, target in itertools.zip_longest(sources, targets):
    # If either runs out ahead-of-time, we repeat the final non-None element.
    # (This is safest as we cannot inspect the function's defaults.)
    if source is None:
      source = prev_source
    elif target is None:
      target = prev_target

    if isinstance(target, Mapping):
      assert isinstance(source, Mapping)
      results.append({**target, **source})
    elif isinstance(sources, Iterable):
      # target is a non-mapping iterable, like tuple or list.
      if isinstance(source, Mapping):
        # To match the target, we replace the source with its unlabeled values.
        source = source.values()
      results.append((*target, *source))
      prev_source, prev_target = source, target
    else:
      raise NotImplementedError(
          f'Targets of type {type(target)=} are unsupported.'
      )

  return results


def named_product(
    first: Iterable[TypingSequence[Any] | Mapping[str, Any]],
    second: Iterable[TypingSequence[Any] | Mapping[str, Any]],
) -> Callable[[_TestFnT], _TestFnT]:
  """Builds named parameters from the product of iterators of named parameters.

  As in parameterized.named_parameters, if an iterator's items are sequences,
  the first element is interpreted as the name. If an iterator's items are
  mappings, the `testcase_name` key is used.

  Args:
    first: Iterable of named parameters, whose names will be the first part of
      the named product's test names.
    second: Iterable of named parameters, whose names will be the second part of
      the named product's test names.

  Returns:
    A decorator that calls the test function with the cartesian product of the
    given iterators, whose items are named parameters with names of the form
    `{first_item_name}_{second_item_name}`. If both iterators' items are
    mappings, the product's items are mappings; otherwise they are ordered
    tuples.

    For example, if `first` is
    `[{**foo, 'testcase_name': 'foo'}, {**bar, 'testcase_name': 'bar'}]` and
    `second` is `[['baz', *baz], ['qux', *qux]]`, the items will be
    `('foo_baz', *foo.values(), *baz), ('foo_qux', *foo.values(), *qux), ...`

  Raises:
    ValueError: A testcase_name is missing; either an iterator item is empty, or
      one is a mapping without a `testcase_name` key.
  """

  results = []

  for p1, p2 in itertools.product(first, second):

    for source, parameters in enumerate([p1, p2]):
      if isinstance(parameters, Mapping):
        if 'testcase_name' not in parameters:
          raise ValueError(
              f'Mapping {parameters} from iterable #{source+1} does not have'
              ' key `testcase_name`.'
          )
      elif not parameters:
        raise ValueError(
            f'An sequence from iterable #{source+1} is empty; the first entry'
            ' is expected to be a testcase name.'
        )

    # When both are mappings, we merge by key:
    if isinstance(p1, Mapping) and isinstance(p2, Mapping):
      testcase_name = f'{p1["testcase_name"]}_{p2["testcase_name"]}'
      p1 = {k: v for k, v in p1.items() if k != 'testcase_name'}
      p2 = {k: v for k, v in p2.items() if k != 'testcase_name'}
      results.append({**p1, **p2, 'testcase_name': testcase_name})

    # Else, we return an ordered tuple based on each parameter set's order:
    else:

      if isinstance(p1, Mapping):
        p1_name = p1['testcase_name']
        p1 = tuple(v for k, v in p1.items() if k != 'testcase_name')
      else:
        p1_name = p1[0]
        p1 = p1[1:]

      if isinstance(p2, Mapping):
        p2_name = p2['testcase_name']
        p2 = tuple(v for k, v in p2.items() if k != 'testcase_name')
      else:
        p2_name = p2[0]
        p2 = p2[1:]

      testcase_name = f'{p1_name}_{p2_name}'
      results.append((testcase_name, *p1, *p2))

  return parameterized.named_parameters(*results)

class SequenceLayerTest(parameterized.TestCase):
  """Base abstract test class providing common sequence testing assertions."""

  @abc.abstractmethod
  def get_backend(self) -> Any:
    """Returns the backend module (jax.numpy or mlx.core)."""

  @property
  @abc.abstractmethod
  def Sequence(self) -> type[types.Sequence]:
    """Returns the Sequence class for the backend."""

  @property
  @abc.abstractmethod
  def MaskedSequence(self) -> Any:
     """Returns the MaskedSequence class for the backend."""

  @abc.abstractmethod
  def random_sequence(
      self,
      *dims: int,
      dtype=None,
      random_mask: bool = False,
      random_lengths: bool | None = None,
      low: int | None = 0,
      high: int | None = 10,
      low_length: int = 0,
      high_length: int | None = None,
  ) -> Any:
    """Generates a random Sequence with dims dimension."""

  def init_layer(self, layer, x, **kwargs):
      """Initialize and bind variables if required by the backend."""
      return layer

  def call_layer(self, layer, x, training: bool = False, **kwargs):
      return layer.layer(x, training=training, **kwargs)

  def call_get_initial_state(self, layer, batch, spec, **kwargs):
      return layer.get_initial_state(batch, spec, **kwargs)

  def call_step(self, layer, x, state, training: bool = False, **kwargs):
      return layer.step(x, state, training=training, **kwargs)

  def _step_by_step(
      self,
      layer,
      x: types.Sequence,
      *,
      block_size: int = 1,
      constants=None,
      stream_constants=None,
  ) -> tuple[types.Sequence, Any]:
      batch = x.shape[0]
      time = x.shape[1]
      spec = x.channel_spec

      init_constants = dict(constants) if constants else {}
      if stream_constants:
          init_constants.update(stream_constants)

      state = self.call_get_initial_state(
          layer, batch, spec, constants=init_constants or None
      )

      outputs_values = []
      outputs_masks = []

      xp = self.get_backend()

      for t in range(0, time, block_size):
          x_block = self.Sequence(
              x.values[:, t : t + block_size],
              x.mask[:, t : t + block_size],
          )

          step_constants = dict(constants) if constants else {}
          if stream_constants:
              for name, seq in stream_constants.items():
                  step_constants[name] = self.Sequence(
                      seq.values[:, t : t + block_size],
                      seq.mask[:, t : t + block_size],
                  )

          y_block, state = self.call_step(
              layer,
              x_block,
              state,
              constants=step_constants or None,
          )
          outputs_values.append(y_block.values)
          outputs_masks.append(y_block.mask)

      y_values = xp.concatenate(outputs_values, axis=1)
      y_mask = xp.concatenate(outputs_masks, axis=1)
      
      return self.Sequence(y_values, y_mask), state

  def verify_config_metadata(self, config: types.SequenceLayerConfig):
      """Verifies that the concrete config satisfies 'up-front' design principles."""
      cls = type(config)
      
      # For concrete subclasses, collect all annotations and their defaults.
      # We just look at the direct abstract parent (if it exists) to enforce its contract!
      base = cls.__mro__[1] if len(cls.__mro__) > 1 else None
      
      # Determine if the direct parent is actually an abstract interface from the .abstract module
      is_abstract_parent = base and getattr(base, '__module__', '').startswith('sequence_layers.abstract')
      
      if not is_abstract_parent:
          # If the parent is not abstract, we skip the up-front enforcement, 
          # as the base could be another intermediate concrete class.
          # We only enforce against the pure abstract definition.
          return
      
      if getattr(cls, '__module__', '').startswith('sequence_layers.abstract'):
          raise AssertionError(f"Layer returns an abstract config {cls.__name__} from {cls.__module__} instead of defining its own concrete Config subclass!")

      # The parent is an abstract Config. Collect its generic specification:
      required_fields = {}
      if hasattr(base, '__annotations__'):
        for k in base.__annotations__:
          if hasattr(base, k):
            required_fields[k] = getattr(base, k)
          else:
            required_fields[k] = ...  # Use Ellipsis to denote no default

      if not required_fields:
          return

      # Check if the concrete class explicitly redefines every single annotation.
      cls_annotations = getattr(cls, '__annotations__', {})
      missing_fields = set(required_fields.keys()) - set(cls_annotations.keys())
      if missing_fields:
        raise AssertionError(
            f"Concrete config class {cls.__name__} violates the up-front"
            f" design principle! It must explicitly redeclare these"
            f" inherited fields from {base.__name__}: {sorted(list(missing_fields))}"
        )

      # Enforce strictly matching defaults to ensure purity.
      divergent_defaults = {}
      for k, base_default in required_fields.items():
        if base_default is not ...:
          cls_default = getattr(cls, k) if hasattr(cls, k) else ...
          
          match = (cls_default == base_default)
          if isinstance(base_default, types.SemanticDType):
              match = base_default.matches(cls_default)
          
          if not match:
            divergent_defaults[k] = (base_default, cls_default)
      
      if divergent_defaults:
        msg = ", ".join([f"'{k}': expected {expect!r}, got {got!r}" 
                         for k, (expect, got) in divergent_defaults.items()])
        raise AssertionError(
            f"Concrete config class {cls.__name__} violates the up-front"
            f" design principle! Its defaults diverge from the abstract spec:\n{msg}"
        )

  def verify_contract(
      self,
      layer,
      x: types.Sequence,
      *,
      training: bool = False,
      constants=None,
      atol: float = 1e-5,
      rtol: float = 1e-5,
      test_step: bool = True,
      **kwargs,
  ):
      self.verify_config_metadata(layer.config)
      
      xp = self.get_backend()
      
      if hasattr(x, 'channel_shape'):
          input_shape = x.channel_shape
      elif hasattr(x, 'shape'):
           input_shape = x.shape[2:]
      else:
           input_shape = x
      dtype = x.dtype if hasattr(x, 'dtype') else xp.float32

      y_layer = self.call_layer(layer,  x, constants=constants)

      expected_shape = layer.get_output_shape(input_shape, constants=constants)
      self.assertEqual(y_layer.channel_shape, expected_shape)

      expected_dtype = layer.get_output_dtype(dtype, constants=constants)
      self.assertEqual(y_layer.dtype, expected_dtype)

      if not test_step or not layer.supports_step:
          return y_layer

      block_size = layer.block_size
      y_step, _ = self._step_by_step(layer, x, block_size=block_size, constants=constants)

      self.assertEqual(y_step.shape, y_layer.shape)
      self.assertSequencesClose(y_layer, y_step, atol=atol, rtol=rtol)
      
      return y_layer

  def assertSequencesClose(self, x: Any, y: Any, **kwargs):
      x_np = np.array(x.values) if hasattr(x, 'values') else np.array(x)
      y_np = np.array(y.values) if hasattr(y, 'values') else np.array(y)
      np.testing.assert_allclose(x_np, y_np, **kwargs)
      if hasattr(x, 'mask') and hasattr(y, 'mask'):
          mask_x = np.array(x.mask)
          mask_y = np.array(y.mask)
          np.testing.assert_array_equal(mask_x, mask_y)

  def assertSequencesEqual(self, x: Any, y: Any):
      x_np = np.array(x.values) if hasattr(x, 'values') else np.array(x)
      y_np = np.array(y.values) if hasattr(y, 'values') else np.array(y)
      np.testing.assert_array_equal(x_np, y_np)
      if hasattr(x, 'mask') and hasattr(y, 'mask'):
          mask_x = np.array(x.mask)
          mask_y = np.array(y.mask)
          np.testing.assert_array_equal(mask_x, mask_y)

  def assertAllClose(self, x: Any, y: Any, **kwargs):
      x_np = np.array(x)
      y_np = np.array(y)
      np.testing.assert_allclose(x_np, y_np, **kwargs)

  def assertAllEqual(self, x: Any, y: Any):
      x_np = np.array(x)
      y_np = np.array(y)
      np.testing.assert_array_equal(x_np, y_np)

  def assertSequencesNotClose(self, x: Any, y: Any, **kwargs):
      x_np = np.array(x.values) if hasattr(x, 'values') else np.array(x)
      y_np = np.array(y.values) if hasattr(y, 'values') else np.array(y)
      try:
          np.testing.assert_allclose(x_np, y_np, **kwargs)
      except AssertionError:
          return
      if hasattr(x, 'mask') and hasattr(y, 'mask'):
          mask_x = np.array(x.mask)
          mask_y = np.array(y.mask)
          try:
              np.testing.assert_array_equal(mask_x, mask_y)
          except AssertionError:
              return
      raise AssertionError("Sequences are close")

  def assertSequencesNotEqual(self, x: Any, y: Any):
      x_np = np.array(x.values) if hasattr(x, 'values') else np.array(x)
      y_np = np.array(y.values) if hasattr(y, 'values') else np.array(y)
      try:
          np.testing.assert_array_equal(x_np, y_np)
      except AssertionError:
          return
      if hasattr(x, 'mask') and hasattr(y, 'mask'):
          mask_x = np.array(x.mask)
          mask_y = np.array(y.mask)
          try:
              np.testing.assert_array_equal(mask_x, mask_y)
          except AssertionError:
              return
      raise AssertionError("Sequences are equal")

  def assertNotAllEqual(self, x: Any, y: Any):
      x_np = np.array(x)
      y_np = np.array(y)
      try:
          np.testing.assert_array_equal(x_np, y_np)
      except AssertionError:
          return
      raise AssertionError("All equal")

  def assertNotAllClose(self, x: Any, y: Any, **kwargs):
      x_np = np.array(x)
      y_np = np.array(y)
      try:
          np.testing.assert_allclose(x_np, y_np, **kwargs)
      except AssertionError:
          return
      raise AssertionError("All close")
