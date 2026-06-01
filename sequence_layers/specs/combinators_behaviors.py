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
"""Shared behavior tests for combinators."""

# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation

import dataclasses
import fractions
from typing import Any, override

import numpy as np

from sequence_layers.specs import combinators as spec_combinators
from sequence_layers.specs import test_utils as test_utils_spec
from sequence_layers.specs import types as types_spec


# pylint: disable=abstract-method
# pyrefly: disable=bad-instantiation
class CombinatorBehaviorsTest(test_utils_spec.SequenceLayerTest):
  """Base test class for shared combinator tests."""

  def create_dummy_layer_config(
      self,
      val: float = 1.0,
      state_val: float = 0.0,
      block_size: int = 1,
      output_ratio: int = 1,
      input_latency: int = 0,
      out_features: int | None = None,
  ) -> Any:
    """Helper to create a dummy layer config tied to the active backend."""
    # pylint: disable=missing-class-docstring,missing-function-docstring,unused-argument
    backend_sl = self.sl
    xp = self.xp

    if "jax" in backend_sl.__name__:  # pyrefly: ignore[missing-attribute]

      @dataclasses.dataclass
      class DummyAddLayer(backend_sl.types.Emitting):
        val: float = 1.0
        state_val: float = 0.0
        _block_size: int = 1
        _output_ratio: fractions.Fraction = fractions.Fraction(1)
        _input_latency: int = 0
        out_features: int | None = None

        @property
        @override
        def supports_step(self) -> bool:
          return True

        @property
        @override
        def block_size(self) -> int:
          return self._block_size

        @property
        @override
        def output_ratio(self) -> fractions.Fraction:
          return self._output_ratio

        @property
        @override
        def input_latency(self) -> int:
          return self._input_latency

        @property
        @override
        def receptive_field_per_step(  # pyrefly: ignore[bad-override]
            self,
        ) -> dict[int, Any]:
          return {0: (0, 0)}

        @override
        def get_output_shape(
            self, input_shape, *, constants=None
        ) -> tuple[int, ...]:
          if self.out_features is not None:
            return (self.out_features,)
          return tuple(input_shape)

        @override
        def get_output_dtype(self, input_dtype, *, constants=None):
          return input_dtype

        @override
        def get_initial_state(
            self,
            batch_size,
            input_spec,
            *,
            training=False,
            constants=None,
            **kwargs,
        ):
          return xp.broadcast_to(
              xp.array(self.state_val, dtype=xp.float32), (batch_size,)
          )

        @override
        def layer_with_emits(
            self, x, *, training=False, constants=None, **kwargs
        ):
          y_values = x.values + self.val
          if self.out_features is not None:
            in_ch = x.values.shape[-1]
            if self.out_features > in_ch:
              pad_shape = list(y_values.shape)
              pad_shape[-1] = self.out_features - in_ch
              zeros = xp.zeros(tuple(pad_shape), dtype=y_values.dtype)
              y_values = xp.concatenate([y_values, zeros], axis=-1)
            else:
              y_values = y_values[..., : self.out_features]
          y_values = y_values * x.mask[..., None]
          emit_val = y_values * 0 + self.val
          return type(x)(  # pyrefly: ignore[bad-instantiation]
              y_values, x.mask
          ), {"emit_val": emit_val}

        @override
        def step_with_emits(
            self, x, state, *, training=False, constants=None, **kwargs
        ):
          y_values = x.values + self.val
          if self.out_features is not None:
            in_ch = x.values.shape[-1]
            if self.out_features > in_ch:
              pad_shape = list(y_values.shape)
              pad_shape[-1] = self.out_features - in_ch
              zeros = xp.zeros(tuple(pad_shape), dtype=y_values.dtype)
              y_values = xp.concatenate([y_values, zeros], axis=-1)
            else:
              y_values = y_values[..., : self.out_features]
          y_values = y_values * x.mask[..., None]
          emit_val = y_values * 0 + self.val
          return (
              type(x)(y_values, x.mask),  # pyrefly: ignore[bad-instantiation]
              state + 1.0,
              {"emit_val": emit_val},
          )

    else:

      class DummyAddLayer(backend_sl.types.Emitting):

        def __init__(
            self,
            val,
            state_val,
            _block_size,
            _output_ratio,
            _input_latency,
            out_features,
        ):
          super().__init__()
          self.val = val
          self.state_val = state_val
          self._block_size = _block_size
          self._output_ratio = _output_ratio
          self._input_latency = _input_latency
          self.out_features = out_features

        @property
        @override
        def supports_step(self) -> bool:
          return True

        @property
        @override
        def block_size(self) -> int:
          return self._block_size

        @property
        @override
        def output_ratio(self) -> fractions.Fraction:
          return self._output_ratio

        @property
        @override
        def input_latency(self) -> int:
          return self._input_latency

        @property
        @override
        def receptive_field_per_step(  # pyrefly: ignore[bad-override]
            self,
        ) -> dict[int, Any]:
          return {0: (0, 0)}

        @override
        def get_output_shape(
            self, input_shape, *, constants=None
        ) -> tuple[int, ...]:
          if self.out_features is not None:
            return (self.out_features,)
          return tuple(input_shape)

        @override
        def get_output_dtype(self, input_dtype, *, constants=None):
          return input_dtype

        @override
        def get_initial_state(
            self,
            batch_size,
            input_spec,
            *,
            training=False,
            constants=None,
            **kwargs,
        ):
          return xp.broadcast_to(
              xp.array(self.state_val, dtype=xp.float32), (batch_size,)
          )

        @override
        def layer_with_emits(
            self, x, *, training=False, constants=None, **kwargs
        ):
          y_values = x.values + self.val
          if self.out_features is not None:
            in_ch = x.values.shape[-1]
            if self.out_features > in_ch:
              pad_shape = list(y_values.shape)
              pad_shape[-1] = self.out_features - in_ch
              zeros = xp.zeros(tuple(pad_shape), dtype=y_values.dtype)
              y_values = xp.concatenate([y_values, zeros], axis=-1)
            else:
              y_values = y_values[..., : self.out_features]
          y_values = y_values * x.mask[..., None]
          emit_val = y_values * 0 + self.val
          return type(x)(  # pyrefly: ignore[bad-instantiation]
              y_values, x.mask
          ), {"emit_val": emit_val}

        @override
        def step_with_emits(
            self, x, state, *, training=False, constants=None, **kwargs
        ):
          y_values = x.values + self.val
          if self.out_features is not None:
            in_ch = x.values.shape[-1]
            if self.out_features > in_ch:
              pad_shape = list(y_values.shape)
              pad_shape[-1] = self.out_features - in_ch
              zeros = xp.zeros(tuple(pad_shape), dtype=y_values.dtype)
              y_values = xp.concatenate([y_values, zeros], axis=-1)
            else:
              y_values = y_values[..., : self.out_features]
          y_values = y_values * x.mask[..., None]
          emit_val = y_values * 0 + self.val
          return (
              type(x)(y_values, x.mask),  # pyrefly: ignore[bad-instantiation]
              state + 1.0,
              {"emit_val": emit_val},
          )

    @dataclasses.dataclass(frozen=True)
    class DummyConfig(types_spec.SequenceLayerConfig):
      val: float
      state_val: float
      block_size: int
      output_ratio: int
      input_latency: int
      out_features: int | None

      @override
      def make(self, backend="jax"):
        return DummyAddLayer(  # pyrefly: ignore[bad-instantiation]
            val=self.val,
            state_val=self.state_val,
            _block_size=self.block_size,
            _output_ratio=fractions.Fraction(self.output_ratio),
            _input_latency=self.input_latency,
            out_features=self.out_features,
        )

    return DummyConfig(
        val=val,
        state_val=state_val,
        block_size=block_size,
        output_ratio=output_ratio,
        input_latency=input_latency,
        out_features=out_features,
    )

  def test_serial_basic(self):
    config = self.sl.combinators.Serial.Config([
        self.create_dummy_layer_config(val=1.0),
        self.create_dummy_layer_config(val=2.0),
    ])
    layer = self.make_layer(config)
    x = self.random_sequence(2, 5, 3)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    # Serial adds: x + 1.0 + 2.0 = x + 3.0
    expected = (x.values + 3.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_serial_empty(self):
    config = self.sl.combinators.Serial.Config([])
    layer = self.make_layer(config)
    x = self.random_sequence(2, 5, 3)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)
    np.testing.assert_allclose(y.values, x.values, atol=1e-6)
    self.verify_contract(layer, x)

  def test_serial_state_tracking(self):
    config = self.sl.combinators.Serial.Config([
        self.create_dummy_layer_config(val=1.0, state_val=10.0),
        self.create_dummy_layer_config(val=2.0, state_val=20.0),
    ])
    layer = self.make_layer(config)
    x = self.random_sequence(1, 1, 2)
    layer = self.init_layer(layer, x)
    state = layer.get_initial_state(1, x.channel_spec, training=False)
    self.assertEqual(state, (10.0, 20.0))

    y, next_state = layer.step(x, state, training=False)
    # States should increment
    self.assertEqual(next_state, (11.0, 21.0))
    expected = (x.values + 3.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

  def test_residual_basic(self):
    # y = body(x) + shortcut(x)
    # body = add 2.0, shortcut = identity
    config = self.sl.combinators.Residual.Config(
        [self.create_dummy_layer_config(val=2.0)]
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 4, 3)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    # expected = (x + 2.0) + x = 2x + 2.0
    expected = (x.values * 2.0 + 2.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_residual_with_custom_shortcut(self):
    # body = add 2.0, shortcut = add 5.0
    config = self.sl.combinators.Residual.Config(
        [self.create_dummy_layer_config(val=2.0)],
        shortcut_layers=[self.create_dummy_layer_config(val=5.0)],
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 4, 3)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    # expected = (x + 2.0) + (x + 5.0) = 2x + 7.0
    expected = (x.values * 2.0 + 7.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_repeat_basic(self):
    # Repeats a layer N times
    config = self.sl.combinators.Repeat.Config(
        layer=self.create_dummy_layer_config(val=1.5),
        num_repeats=4,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 5, 3)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    # expected = x + 4 * 1.5 = x + 6.0
    expected = (x.values + 6.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_parallel_stack(self):
    config = self.sl.combinators.Parallel.Config(
        [
            self.create_dummy_layer_config(val=1.0),
            self.create_dummy_layer_config(val=2.0),
        ],
        combination=spec_combinators.CombinationMode.STACK,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    self.assertEqual(y.channel_shape, (2, 4))
    self.verify_contract(layer, x)

  def test_parallel_concat(self):
    config = self.sl.combinators.Parallel.Config(
        [
            self.create_dummy_layer_config(val=1.0, out_features=3),
            self.create_dummy_layer_config(val=2.0, out_features=5),
        ],
        combination=spec_combinators.CombinationMode.CONCAT,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    self.assertEqual(y.channel_shape, (8,))
    self.verify_contract(layer, x)

  def test_parallel_add(self):
    config = self.sl.combinators.Parallel.Config(
        [
            self.create_dummy_layer_config(val=1.0),
            self.create_dummy_layer_config(val=2.0),
        ],
        combination=spec_combinators.CombinationMode.ADD,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    expected = (x.values * 2.0 + 3.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_parallel_mean(self):
    config = self.sl.combinators.Parallel.Config(
        [
            self.create_dummy_layer_config(val=1.0),
            self.create_dummy_layer_config(val=3.0),
        ],
        combination=spec_combinators.CombinationMode.MEAN,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    expected = (x.values + 2.0) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)

  def test_parallel_product(self):
    config = self.sl.combinators.Parallel.Config(
        [
            self.create_dummy_layer_config(val=2.0),
            self.create_dummy_layer_config(val=3.0),
        ],
        combination=spec_combinators.CombinationMode.PRODUCT,
    )
    layer = self.make_layer(config)
    x = self.random_sequence(2, 3, 4)
    layer = self.init_layer(layer, x)
    y = layer.layer(x, training=False)

    expected = ((x.values + 2.0) * (x.values + 3.0)) * x.mask[..., None]
    np.testing.assert_allclose(y.values, expected, atol=1e-6)

    self.verify_contract(layer, x)
