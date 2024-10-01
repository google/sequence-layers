# Copyright 2023 Google LLC
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
"""Test utilities for SequenceLayers."""

import contextlib
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import sequence_layers.tensorflow as sl
from sequence_layers.tensorflow import utils
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


# List of Keras mixed precision policies SequenceLayers supports.
# https://www.tensorflow.org/guide/mixed_precision
SUPPORTED_PRECISION_POLICIES = ('float32', 'mixed_bfloat16', 'bfloat16')


def _grad_ys(sequence: sl.Sequence) -> tf.Tensor:
  if sequence.values.dtype.is_complex:
    return tf.ones_like(sequence.values) + 1j * tf.ones_like(sequence.values)
  else:
    return tf.ones_like(sequence.values)


def _tf_gradients(y: sl.Sequence, x: sl.Sequence) -> sl.Sequence:
  return sl.Sequence(
      tf.gradients(y.values, x.values, grad_ys=_grad_ys(y))[0], x.mask
  ).mask_invalid()


def _tape_gradient(
    tape: tf.GradientTape, y: sl.Sequence, x: sl.Sequence
) -> sl.Sequence:
  def grad_fn(values: tf.Tensor) -> tf.Tensor:
    grad = tape.gradient(y.values, values, _grad_ys(y))
    if grad is None:
      raise ValueError('Gradient of y with respect to x does not exist.')
    return grad

  return x.apply_values(grad_fn).mask_invalid()


class SequenceLayerTest(tf.test.TestCase):
  """Helper functions for SequenceLayer tests."""

  def assertAllFinite(self, a):
    """Assert that all entries in a `Tensor` are finite.

    Args:
      a: A `Tensor` whose entries are checked for finiteness.
    """
    is_finite = np.isfinite(self._GetNdArray(a))
    all_true = np.ones_like(is_finite, dtype=bool)
    self.assertAllEqual(all_true, is_finite)

  def assertAllNan(self, a):
    """Assert that every entry in a `Tensor` is NaN.

    Args:
      a: A `Tensor` whose entries must be verified as NaN.
    """
    is_nan = np.isnan(self._GetNdArray(a))
    all_true = np.ones_like(is_nan, dtype=bool)
    self.assertAllEqual(all_true, is_nan)

  def verify_contract(
      self,
      l: sl.SequenceLayer,
      x: sl.Sequence,
      *,
      training: bool,
      constants: sl.Constants = None,
      test_causality: bool = True,
      pad_nan: bool = True,
      pad_constants: bool = False,
      pad_constants_ratio: int = 1,
      test_gradients: bool = True,
      rtol=1e-6,
      atol=1e-6,
      grad_rtol=None,
      grad_atol=None,
  ) -> Tuple[sl.Sequence, sl.Sequence]:
    """Verifies the SequenceLayer contract outlined in go/tf-sequence-layers.

    In graph mode, x is modified so that batch and time dimension are unknown.

    Args:
      l: The SequenceLayer to test.
      x: A Sequence to use as input.
      training: Whether to run the layer in training mode.
      constants: A dictionary of constant name to tf.Tensor or sl.Sequence.
        Values or sequences that are "constant" with respect to the
        SequenceLayer, but may affect its processing. For example, for an
        Attention layer this may contain the key/value sequence to attend to.
      test_causality: Whether to test causality.
      pad_nan: Whether to test causality by padding x with NaNs and verifying
        that NaNs do not leak into valid regions of the sequence.
      pad_constants: Whether to pad sl.Sequence instances in constants along
        with padding x with NaNs, when verifying causality.
      pad_constants_ratio: The ratio between the time dimension of x and
        sequences in constants.
      test_gradients: Whether to compute and compare the gradients for l applied
        to x in step-wise and layer-wise mode.
      rtol: The relative tolerance to test layer/step equivalence to.
      atol: The absolute tolerance to test layer/step equivalence to.
      grad_rtol: Optional relative tolerance to test layer/step gradient
        equivalence to. Uses rtol if not specified.
      grad_atol: Optional absolute tolerance to test layer/step gradient
        equivalence to. Uses atol if not specified.

    Returns:
      x: The input Sequence, converted to numpy arrays.
      y_layer: The result of running x layer-by-layer, converted to numpy
        arrays. If the function returns without throwing an exception then
        layer-by-layer processing is equivalent to step-by-step processing
        (up to rtol/atol).
    """
    if not tf.executing_eagerly():
      raise ValueError('verify_contract is only supported in eager mode.')

    if grad_rtol is None:
      grad_rtol = rtol
    if grad_atol is None:
      grad_atol = atol
    # Forget batch and time dimension inside tf.function.
    unknown_batch_and_time_spec = sl.Sequence(
        tf.TensorSpec(
            tf.TensorShape([None, None]).concatenate(x.channel_shape),
            x.values.dtype,
        ),
        tf.TensorSpec(tf.TensorShape([None, None]), x.mask.dtype),
    )

    @tf.function(input_signature=[unknown_batch_and_time_spec])
    def layer_fn(x):
      y = l.layer(x, training=training, initial_state=s0, constants=constants)
      if not y.channel_shape.is_fully_defined():
        raise ValueError(
            f'Sequence enters {l}.layer with shape {x.values.shape} '
            f'and loses channel shape information: {y.values.shape}'
        )
      if test_gradients:
        return y, _tf_gradients(y, x)
      else:
        return y

    @tf.function(input_signature=[unknown_batch_and_time_spec])
    def step_fn(x):
      y, _, _ = utils.step_by_step_dynamic(
          l, x, training, s0, constants=constants
      )
      if not y.channel_shape.is_fully_defined():
        raise ValueError(
            f'Sequence enters {l}.step with shape {x.values.shape} '
            f'and loses channel shape information: {y.values.shape}'
        )
      if test_gradients:
        return y, _tf_gradients(y, x)
      else:
        return y

    s0 = l.get_initial_state(x, constants)
    output_spec = l.get_output_spec_for_sequence(x, constants)

    # Compute outputs layer-by-layer, step-by-step, and step-by-step with a
    # block size of 2x its minimum.
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      y_layer = l.layer(x, training, initial_state=s0, constants=constants)
      if l.supports_step:
        y_step, _, _ = utils.step_by_step_dynamic(
            l, x, training, s0, constants=constants
        )
        y_step_2x, _, _ = utils.step_by_step_dynamic(
            l, x, training, s0, blocks_per_step=2, constants=constants
        )
    if test_gradients:
      y_layer_fn, y_layer_fn_grad = layer_fn(x)
      if l.supports_step:
        y_step_fn, y_step_fn_grad = step_fn(x)
    else:
      y_layer_fn, y_layer_fn_grad = layer_fn(x), None
      if l.supports_step:
        y_step_fn, y_step_fn_grad = step_fn(x), None

    # The actual layer output spec matches get_output_spec.
    self.assertEqual(y_layer.channel_spec, output_spec)
    self.assertEqual(y_layer_fn.channel_spec, output_spec)
    if l.supports_step:
      self.assertEqual(y_step.channel_spec, output_spec)
      self.assertEqual(y_step_2x.channel_spec, output_spec)
      self.assertEqual(y_step_fn.channel_spec, output_spec)

    x_np = x.apply(lambda v, m: (v.numpy(), m.numpy()))
    # In Eager mode we use the above GradientTape to compute gradients.
    if test_gradients:
      y_layer_grad = _tape_gradient(tape, y_layer, x)
      if l.supports_step:
        y_step_grad = _tape_gradient(tape, y_step, x)
        y_step_2x_grad = _tape_gradient(tape, y_step_2x, x)

    # 1. Layer-wise / step-wise equivalent (values, masks, gradients).
    if l.supports_step:
      self.assertSequencesClose(y_layer, y_step, rtol=rtol, atol=atol)
      self.assertSequencesClose(y_layer, y_step_2x, rtol=rtol, atol=atol)
      self.assertSequencesClose(y_layer, y_step_fn, rtol=rtol, atol=atol)
    self.assertSequencesClose(y_layer, y_layer_fn, rtol=rtol, atol=atol)

    if test_gradients:
      if l.supports_step:
        self.assertSequencesClose(
            y_layer_grad, y_step_grad, rtol=grad_rtol, atol=grad_atol
        )
        self.assertSequencesClose(
            y_layer_grad, y_step_2x_grad, rtol=grad_rtol, atol=grad_atol
        )
        self.assertSequencesClose(
            y_layer_grad, y_step_fn_grad, rtol=grad_rtol, atol=grad_atol
        )
      self.assertSequencesClose(
          y_layer_grad, y_layer_fn_grad, rtol=grad_rtol, atol=grad_atol
      )

    # 6. Masking.
    self.verify_masked(y_layer)
    self.verify_masked(y_layer_fn)
    if l.supports_step:
      self.verify_masked(y_step)
      self.verify_masked(y_step_2x)
      self.verify_masked(y_step_fn)

    # 2. No implicit end padding.
    # TODO(rryan): Change mask to test this?

    # 3. Causality.
    # 5. Tensor shape invariance.
    # Check causality and tensor shape invariance by replacing all padding with
    # NaN and padding with a block of NaNs.
    # TODO(rryan): Make this test stronger by padding with valid NaNs at the
    # end of the sequence to show that they don't "leak" backwards.

    # Causality is only required if stepping is supported.
    if test_causality and l.supports_step:
      pad_value = np.nan if pad_nan else tf.cast(1e9, dtype=x.values.dtype)
      # Replace all masked values with NaN and pad with one block of invalid
      # NaNs.
      x_nan = x.apply_values(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              x.expanded_mask() > 0.0, v, tf.ones_like(v) * pad_value
          )
      )

      def _pad(x, pad_back):
        return x.pad_time(0, pad_back, valid=False, pad_value=pad_value)

      x_nan = _pad(x_nan, l.block_size * pad_constants_ratio)
      if constants is not None and pad_constants:
        constants_nan = {
            k: _pad(v, l.block_size) if isinstance(v, sl.Sequence) else v
            for k, v in constants.items()
        }
      else:
        constants_nan = constants

      y_layer_nan = l.layer(
          x_nan, training, initial_state=s0, constants=constants_nan
      )
      # Replace NaNs in the padding with zeros.
      y_layer_nan = y_layer_nan.apply_values(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              y_layer_nan.expanded_mask() > 0.0, v, tf.zeros_like(v)
          )
      )
      self.assertSequencesClose(y_layer_nan, y_layer, rtol=rtol, atol=atol)

      y_step_nan, _, _ = utils.step_by_step_dynamic(
          l, x_nan, training, s0, constants=constants_nan
      )
      # Replace NaNs in the padding with zeros.
      y_step_nan = y_step_nan.apply_values(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              y_step_nan.expanded_mask() > 0.0, v, tf.zeros_like(v)
          )
      )
      self.assertSequencesClose(y_step_nan, y_step, rtol=rtol, atol=atol)

    # 4. Time alignment.
    # TODO(rryan): How? Layer specific. Maybe pad with a block size and verify
    # a right shift?

    # Return the input and layer-by-layer result so that tests can check for
    # correctness.
    return x_np, y_layer

  def verify_masked(self, x: sl.Sequence):
    zeros = tf.zeros_like(x.values)
    invalid_mask = tf.broadcast_to(1.0 - x.expanded_mask(), tf.shape(x.values))
    unmasked = tf.where(invalid_mask > 0.0, x.values, zeros)
    self.assertAllEqual(unmasked, zeros)

  def verify_tflite_step(
      self,
      l: sl.SequenceLayer,
      x: sl.Sequence,
      constants: sl.Constants = None,
      allow_custom_ops: bool = False,
      use_flex: bool = False,
      rtol: float = 1e-7,
      atol: float = 1e-7,
  ):
    """Builds a tf.lite model for a step and verifies identical outputs."""
    if not l.supports_step:
      return

    if constants is None:
      constants = {}

    def fn(
        x: sl.Sequence, state: sl.State, constants: sl.Constants
    ) -> Tuple[sl.Sequence, sl.State]:
      return l.step(x, state, training=False, constants=constants)

    x_block_shape = tf.TensorShape(
        [x.values.shape.dims[0].value, l.block_size]
    ).concatenate(x.channel_shape)
    x_block = self.random_sequence(
        *x_block_shape.as_list(), dtype=x.values.dtype
    )
    s0 = l.get_initial_state(x_block, constants)
    y_step_tf, s1_tf = l.step(x_block, s0, training=False, constants=constants)

    tflite_model = tflite_convert(
        fn,
        input_templates=[x_block, s0, constants],
        allow_custom_ops=allow_custom_ops,
        use_flex=use_flex,
    )

    tflite_flat_outputs = evaluate_tflite_model(
        tflite_model, tf.nest.flatten([x_block, s0, constants])
    )
    y_step_tflite, s1_tflite = tf.nest.pack_sequence_as(
        [y_step_tf, s1_tf], tflite_flat_outputs
    )

    self.assertAllClose(
        y_step_tflite.values, y_step_tf.values, rtol=rtol, atol=atol
    )
    self.assertAllEqual(y_step_tflite.mask, y_step_tf.mask)
    tf.nest.map_structure(
        lambda a, b: self.assertAllClose(a, b, rtol=rtol, atol=atol),
        s1_tflite,
        s1_tf,
    )

  def random_sequence(self, *args, **kwargs) -> sl.Sequence:
    return random_sequence(*args, **kwargs)

  def assertEmitsCompatible(self, emit_specs: sl.EmitSpecs, emits: sl.Emits):
    tf.nest.assert_same_structure(emit_specs, emits)

    def _check_spec(spec: tf.TensorSpec, emit: tf.Tensor):
      actual_spec = tf.TensorSpec(emit.shape[2:], emit.dtype)
      if not spec.is_compatible_with(actual_spec):
        raise ValueError(
            f"{emit}'s spec ({actual_spec}) does not match expected ({spec})."
        )

    tf.nest.map_structure(_check_spec, emit_specs, emits)

  def assertSequencesClose(
      self,
      a: sl.Sequence,
      b: sl.Sequence,
      atol: float = 1e-6,
      rtol: float = 1e-6,
      msg: Optional[str] = None,
  ):
    """After padding, checks sequence values are close and masks are equal."""
    a_time = a.values.shape[1]
    b_time = b.values.shape[1]
    max_time = max(a_time, b_time)
    a = a.pad_time(0, max_time - a_time, valid=False)
    b = b.pad_time(0, max_time - b_time, valid=False)
    a, b = self.evaluate([a, b])
    self.assertAllClose(a.values, b.values, atol=atol, rtol=rtol, msg=msg)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesNotClose(
      self,
      a: sl.Sequence,
      b: sl.Sequence,
      atol: float = 1e-6,
      rtol: float = 1e-6,
      msg: Optional[str] = None,
  ):
    """After padding, check sequence values aren't close and masks are equal."""
    a_time = a.values.shape[1]
    b_time = b.values.shape[1]
    max_time = max(a_time, b_time)
    a = a.pad_time(0, max_time - a_time, valid=False)
    b = b.pad_time(0, max_time - b_time, valid=False)
    a, b = self.evaluate([a, b])
    self.assertNotAllClose(a.values, b.values, atol=atol, rtol=rtol, msg=msg)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesEqual(
      self, a: sl.Sequence, b: sl.Sequence, msg: Optional[str] = None
  ):
    """After padding, checks sequence values are equal and masks are equal."""
    a_time = a.values.shape[1]
    b_time = b.values.shape[1]
    max_time = max(a_time, b_time)
    a = a.pad_time(0, max_time - a_time, valid=False)
    b = b.pad_time(0, max_time - b_time, valid=False)
    a, b = self.evaluate([a, b])
    self.assertAllEqual(a.values, b.values, msg=msg)
    self.assertAllEqual(a.mask, b.mask)

  def assertSequencesNotEqual(
      self, a: sl.Sequence, b: sl.Sequence, msg: Optional[str] = None
  ):
    """After padding, checks sequence values aren't equal, masks are equal."""
    a_time = a.values.shape[1]
    b_time = b.values.shape[1]
    max_time = max(a_time, b_time)
    a = a.pad_time(0, max_time - a_time, valid=False)
    b = b.pad_time(0, max_time - b_time, valid=False)
    a, b = self.evaluate([a, b])
    self.assertNotAllEqual(a.values, b.values, msg=msg)
    self.assertAllEqual(a.mask, b.mask)


def random_sequence(*dims, **kwargs) -> sl.Sequence:
  """Generates a random Sequence with dims dimension.

  Each batch item has a randomly generated length in [0, dims[1]) which is used
  to compute the mask.

  Args:
    *dims: The dimensions of the sequence.
    **kwargs: An optional "dtype" keyword argument.

  Returns:
    A Sequence with the specified dimensions and type. Invalid values timesteps
    are masked.
  """
  # TODO(rryan): Once we switch to Python 3 add this as a kwarg.
  dtype = kwargs.pop('dtype', tf.float32)
  # If random_mask is disabled, default to random_lengths.
  random_mask = kwargs.pop('random_mask', False)
  random_lengths = kwargs.pop('random_lengths', not random_mask)
  if random_mask and random_lengths:
    raise ValueError('Must not specify random_mask and random_lengths.')
  if len(dims) < 2:
    raise ValueError(
        'random_sequence expects at least 2 dimensions, got: %s' % (dims,)
    )
  if dtype.is_complex:
    np_values = np.random.normal(size=dims) + 1j * np.random.normal(size=dims)
  elif dtype.is_integer:
    low = kwargs.pop('low', 0)
    high = kwargs.pop('high', 10)
    np_values = np.random.randint(low, high, size=dims)
  else:
    np_values = np.random.normal(size=dims)

  batch_size, time = dims[0], dims[1]
  values = tf.convert_to_tensor(np_values.astype(dtype.as_numpy_dtype))
  if random_mask:
    mask = tf.cast(
        np.random.uniform(size=(batch_size, time)) > 0.5, dtype=sl.MASK_DTYPE
    )
  else:
    if time > 0:
      if random_lengths:
        low_length = kwargs.pop('low_length', 0)
        high_length = kwargs.pop('high_length', time)
        lengths = tf.convert_to_tensor(
            np.random.randint(low_length, high_length, size=[batch_size]),
            tf.int32,
        )
      else:
        lengths = tf.fill(
            [batch_size], tf.convert_to_tensor(time, dtype=tf.int32)
        )
    else:
      lengths = tf.fill([batch_size], tf.convert_to_tensor(0, dtype=tf.int32))
    mask = tf.sequence_mask(lengths, time, dtype=sl.MASK_DTYPE)
  return sl.Sequence(values, mask).mask_invalid()


def tflite_convert(
    fn: Callable[..., Any],
    input_templates: List[Any],
    allow_custom_ops: bool = False,
    use_flex: bool = False,
) -> bytes:
  """Converts the provided fn to a tf.lite model.

  Args:
    fn: A callable that expects a list of inputs like input_templates that
      returns a tensor or structure of tensors.
    input_templates: A list of Tensors, ndarrays or TensorSpecs describing the
      inputs that fn expects. The actual values of the Tensors or ndarrays are
      unused.
    allow_custom_ops: Whether to allow custom ops.
    use_flex: Enable tf.lite flex ops. Requires linking
      //third_party/tensorflow/lite/delegates/flex:delegate

  Returns:
    The serialized tf.lite model.
  """
  fn = tf.function(fn)
  concrete_func = fn.get_concrete_function(*input_templates)
  converter = tf.lite.TFLiteConverter([concrete_func])
  converter.allow_custom_ops = allow_custom_ops
  if use_flex:
    converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)
  return converter.convert()


def evaluate_tflite_model(
    tflite_model: bytes, input_ndarrays: List[np.ndarray]
) -> List[np.ndarray]:
  """Evaluates the provided tf.lite model with the given input ndarrays.

  Args:
    tflite_model: bytes. The serialized tf.lite model.
    input_ndarrays: A list of NumPy arrays to feed as input to the model.

  Returns:
    A list of ndarrays produced by the model.

  Raises:
    ValueError: If the number of input arrays does not match the number of
      inputs the model expects.
  """
  the_interpreter = tf.lite.Interpreter(model_content=tflite_model)
  the_interpreter.allocate_tensors()

  input_details = the_interpreter.get_input_details()
  output_details = the_interpreter.get_output_details()

  if len(input_details) != len(input_ndarrays):
    raise ValueError(
        'Wrong number of inputs: provided=%s, '
        'input_details=%s output_details=%s'
        % (input_ndarrays, input_details, output_details)
    )
  for input_tensor, data in zip(input_details, input_ndarrays):
    the_interpreter.set_tensor(input_tensor['index'], data)
  the_interpreter.invoke()
  return [
      the_interpreter.get_tensor(details['index']) for details in output_details
  ]


def conv1d_mask(
    mask: tf.Tensor,
    kernel_size: int,
    stride: int,
    dilation_rate: int,
    padding: str,
) -> tf.Tensor:
  """Output timesteps are valid iff they only touch valid input timesteps."""
  if padding == sl.PaddingMode.SAME.value:
    return tf.nn.conv1d(
        mask[:, :, tf.newaxis],
        tf.ones([1, 1, 1]),
        stride=stride,
        dilations=dilation_rate,
        padding='VALID',
    )[:, :, 0]
  else:
    if padding == sl.PaddingMode.CAUSAL.value:
      # We assume causal padding has been applied outside of this method.
      explicit_padding = (0, 0)
    else:
      explicit_padding = convolution_explicit_padding(
          padding, kernel_size, dilation_rate
      )

    # Use conv2d since conv1d doesn't support explicit padding.
    mask_golden = tf.nn.conv2d(
        mask[:, :, tf.newaxis, tf.newaxis],
        tf.ones([kernel_size, 1, 1, 1]),
        strides=[stride, 1],
        dilations=[dilation_rate, 1],
        padding=[(0, 0), explicit_padding, (0, 0), (0, 0)],
    )[:, :, 0, 0]
    # Only timesteps that add up to kernel_size are valid.
    return tf.cast(mask_golden > kernel_size - 1e-3, mask_golden.dtype)


def conv1d_transpose_mask(
    mask: tf.Tensor, kernel_size: int, stride: int, padding: str
) -> tf.Tensor:
  """Output timesteps are valid iff they only touch valid input timesteps."""
  padding = sl.validate_padding(padding)
  if kernel_size <= stride or padding == 'same':
    return tf.repeat(mask, stride, axis=1)
  batch_size, time = tf.unstack(tf.shape(mask))
  # Based on tf-keras (tf_keras/utils/conv_utils.py).
  if padding == 'same':
    output_time = time * stride
  else:
    output_time = time * stride + max(kernel_size - stride, 0)
  # Upsample the "invalid" mask so we can tell which timesteps are affected by
  # invalid timesteps.
  mask_golden = tf.nn.conv2d_transpose(
      1.0 - mask[:, :, tf.newaxis, tf.newaxis],
      tf.ones([kernel_size, 1, 1, 1]),
      output_shape=[batch_size, output_time, 1, 1],
      strides=stride,
      padding='SAME' if padding == 'same' else 'VALID',
  )[:, :, 0, 0]
  # Any timesteps that have invalid timesteps "mixed" into them are invalid.
  return tf.where(
      mask_golden > 0.0, tf.zeros_like(mask_golden), tf.ones_like(mask_golden)
  )


class AssertConstantsLayer(sl.StatelessPointwise):
  """Identity layer that raises ValueError if 'test' is not in constants."""

  def get_initial_state(
      self, x: sl.Sequence, constants: sl.Constants = None
  ) -> sl.State:
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return super(AssertConstantsLayer, self).get_initial_state(x, constants)

  def get_output_shape(
      self, x: sl.Sequence, constants: sl.Constants = None
  ) -> tf.TensorShape:
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return super(AssertConstantsLayer, self).get_output_shape(x, constants)

  def layer(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State = None,
      constants: sl.Constants = None,
  ) -> sl.Sequence:
    del training
    del initial_state
    if constants is None or 'test' not in constants:
      raise ValueError('constant not present')
    return x


class NonSteppableLayer(sl.StatelessPointwise):
  """A test layer that does not support stepping."""

  @property
  def supports_step(self):
    return False

  def layer(
      self,
      x: sl.Sequence,
      training: bool,
      initial_state: sl.State = None,
      constants: sl.Constants = None,
  ) -> sl.Sequence:
    del training
    del initial_state
    del constants
    return x


def rtol_atol_for_dtype(dtype: tf.DType) -> tuple[float, float]:
  rtol_atol_for_size = {
      2: (1e-2, 1e-2),
      4: (1e-6, 1e-6),
  }
  assert dtype.is_floating
  return rtol_atol_for_size[dtype.size]


@contextlib.contextmanager
def keras_precision_policy_scope(
    policy: str | tf.keras.mixed_precision.Policy,
):
  old_policy = tf.keras.mixed_precision.global_policy()
  try:
    tf.keras.mixed_precision.set_global_policy(policy)
    yield
  finally:
    tf.keras.mixed_precision.set_global_policy(old_policy)


def _flatten(structure):
  """Wrapper around nest.flatten for TPU in/outfeeds.

  Args:
    structure: An arbitrarily nested structure or a scalar object.

  Returns:
    A Python list, the flattened version of the input.
  """
  flat_sequence = tf1.nest.flatten(structure)
  # TODO(rryan): Handle tf1.SparseTensor.
  return [
      t
      for t in flat_sequence
      if t is not None and isinstance(t, tf1.Tensor) and t.dtype != tf1.string
  ]


def _pack_sequence_as(template, flat_structure):
  """Wrapper around nest.pack_sequence_as for TPU in/out-feeds.

  Args:
    template: Nested structure, whose structure is given by nested lists,
      tuples, and dicts.
    flat_structure: A flat sequence of tensors to pack into the same structure
      as template.

  Returns:
    A structure with the same structure as template, but whose flattened form
    equals flat_structure: i.e.
      structure = _pack_sequence_as(template, flat_structure)
      tf1.nest.assert_same_structure(structure, template)
      tf1.nest.flatten(structure) == flat_structure
  """
  flat_template = tf1.nest.flatten(template)
  fields = []
  flat_structure = list(flat_structure)
  for index, t in enumerate(flat_template):
    if t is None or not isinstance(t, tf1.Tensor) or t.dtype == tf1.string:
      # For non-Tensors, use the value from the template.
      # TODO(rryan): Handle tf1.SparseTensor.
      fields.append(t if not isinstance(t, tf1.Tensor) else None)
    else:
      entry = flat_structure.pop(0)
      if entry.dtype != t.dtype:
        raise ValueError(
            (
                'flat_utterance dtype (%s) does not match the %d-th element'
                ' of the template (%s).'
            )
            % (entry.dtype, index, t.dtype)
        )
      if t.shape.ndims == 0:
        # Squeeze the entry because the template expects a scalar.
        entry = tf1.squeeze(entry)
      fields.append(entry)
  return tf1.nest.pack_sequence_as(template, fields)


def structured_tpu_rewrite(
    fn,
    inputs: Optional[List[Any]] = None,
    rewrite_fn=tf1.tpu.rewrite,
    **rewrite_kwargs,
):
  """Call rewrite_fn on a function with non-Tensor or tf1.string inputs/outputs.

  tpu.rewrite can only handle functions where every input/output element can be
  converted to a tensor (and tf1.string is unsupported). This wrapper makes it
  convenient to call on functions where some inputs or return values may
  optionally be non-Tensors (protos, None, etc.) or tf1.string.

  Args:
    fn: A callable taking nested inputs and returning nested outputs.
    inputs: Optional list of inputs to fn.
    rewrite_fn: A replicating or sharding rewrite fn, such as tf1.tpu.rewrite or
      tf1.tpu.batch_parallel, or tpu_utils.tpu_partitioned_call_rewrite_fn.
    **rewrite_kwargs: kwargs to pass through to rewrite_fn.

  Returns:
    The outputs of the rewritten fn.
  """
  # List for capturing the true output structure from the user-provided fn.
  output_template = []

  def compute(*flat_inputs):
    device_inputs = []
    if inputs is not None:
      device_inputs = _pack_sequence_as(inputs, flat_inputs)
    ret = fn(*device_inputs)
    output_template.append(ret)
    return _flatten(ret)

  flat_output_shapes = []

  def batch_fn(*flat_inputs):
    flat_outputs = rewrite_fn(compute, list(flat_inputs), **rewrite_kwargs)
    flat_output_shapes.extend(t.shape for t in flat_outputs)
    return flat_outputs

  flat_inputs = _flatten(inputs) if inputs is not None else []
  flat_outputs = batch_fn(*flat_inputs)
  for t, shape in zip(flat_outputs, flat_output_shapes):
    t.set_shape(shape)
  return _pack_sequence_as(output_template[0], flat_outputs)


def convolution_explicit_padding(
    padding: str, kernel_size: int, dilation_rate: int
) -> tuple[int, int]:
  """Returns the amount of padding to add for the desired padding mode."""
  effective_kernel_size = (kernel_size - 1) * dilation_rate + 1
  match padding:
    case sl.PaddingMode.CAUSAL.value:
      return (effective_kernel_size - 1, 0)
    case sl.PaddingMode.REVERSE_CAUSAL.value:
      return (0, effective_kernel_size - 1)
    case sl.PaddingMode.SAME.value:
      pad_amount = effective_kernel_size - 1
      pad_left = pad_amount // 2
      pad_right = pad_amount - pad_left
      return pad_left, pad_right
    case sl.PaddingMode.VALID.value:
      return 0, 0
    case _:
      raise ValueError(f'Unsupported padding: {padding}')
