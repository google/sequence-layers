# SequenceLayers

This directory contains the JAX implementation of
https://github.com/google/sequence-layers, an open-source library for building
streamable sequence models.

SequenceLayers are layer neural network layers for sequence processing that take
a single sequence as an input and produce a single sequence as an output.

The key feature the SequenceLayer API provides, is that all layers offer a
layer-wise (standard) processing interface, and additionally a streaming
step-by-step processing interface (if supported by the layer).

This feature is achieved by requiring layers to explicitly support an RNN-style
interface where layer state over time is explicitly represented by a PyTree of
arrays.

With composition primitives like `Serial` and `Residual`, it's easy to build
stacks of `SequenceLayer`s that can in turn be processed layer-wise or
step-wise.

## Overview Talk

Watch
[the recording](https://drive.google.com/file/d/1Cnrs-uk61L-3POLk8Ej2WG_dtHoHUmq4/preview)
or browse the slides below.

<iframe src="https://docs.google.com/presentation/d/1l_qpLjtrIt9bZNICQxn2QJRUzrJGzmrlig8hqRjbYDg/embed#slide=id.p" width="100%" height="500" allow="fullscreen"></iframe>

## Tour

### The `Sequence` type:

A `Sequence` is a PyTree dataclass that has two fields: `values` and `mask`.

```python
x = sl.Sequence(
  values=jnp.zeros((2, 3, 5)),
  mask=jnp.ones((2, 3)),
)
```

Values are batched sequences with shape at least `[batch, time, ...]`, where
`...` is the **channel shape** of the sequence. The channel shape can be `[]` or
any shape.

Masks are bool tensors of shape `[batch, time]`. If `mask[b, t]` is `True`, it
indicates the entire **channel shape** frame at `values[b, t]` is valid data. If
`False`, then `values[b, t]` is invalid, and should **never be used to compute
any valid timepoint.**

Since `Sequence` is a PyTree dataclass, it can be passed through JAX jit
boundaries like any other PyTree.

#### `Sequence.mask_invalid()``

The `mask_invalid()` method returns a **new `Sequence`**, where the `values` are
masked with `mask`, setting invalid timesteps to zero (or the specified pad
value). `mask_invalid()` returns a `MaskedSequence`, which is nothing more than
a subclass of `Sequence` with `mask_invalid()` replaced with a no-op.

```python
x = sl.Sequence(
  values=jnp.zeros((2, 3, 5)),
  mask=jnp.ones((2, 3), dtype=jnp.bool_),
)

x = x.mask_invalid()  # Values are masked.

x = x.mask_invalid()  # No-op.
```

This enables defensive masking without a performance penalty if the sequence is
already masked.

#### `Sequence.apply_values`

For convenience, simple transformations of the values can be applied with
`Sequence.apply_values`.

```python
x = sl.MaskedSequence(
  values=jnp.zeros((2, 3, 5)),
  mask=jnp.ones((2, 3), dtype=jnp.bool_),
)

y = x.apply_values(lambda v: v + 1.0)
assert isinstance(y, sl.Sequence)
```

By default, `apply_values` assumes the transformation leaves the sequence
unmasked. If you know the transformation preserves masking, use
`apply_values_masked`.

```python
x = sl.MaskedSequence(
  values=jnp.zeros((2, 3, 5)),
  mask=jnp.ones((2, 3), dtype=jnp.bool_),
)

# Relu maps 0.0 to 0.0, so masking is preseved.
y = x.apply_values_masked(jax.nn.relu)
assert isinstance(y, sl.MaskedSequence)
```

#### Type-checking `Sequence`s: `SequenceT[jt.Float, 'B T *C']`

To support jaxtyping-style type checking of sequence values and masks,
`sl.SequenceT[ArrayType, ShapeSpec]` can be used in typing declarations to check
`Sequence` values types and shapes. `ShapeSpec` describes the full shape of the
sequence.

```python
from google3.learning.deepmind.jax.typing import typing as jt

@jt.typed
def slice_head(
  x: sl.SequenceT[jt.Float, 'B T H C'],
  i: int
) -> sl.SequenceT[jt.Float, 'B T C']:
  # Return value 'C' is checked against input dimension 'C'.
  return x.apply_values(lambda v: v[:, :, i, :])

y = slice_head(
  sl.Sequence(jnp.ones((2, 3, 5, 7)), jnp.ones((2, 3), dtype=jnp.bool_)), 0)
assert y.shape == (2, 3, 7)

# Raises due to wrong shape input:
try:
  slice_head(
    sl.Sequence(jnp.ones((2, 3, 5)), jnp.ones((2, 3), dtype=jnp.bool_)), 0)
except jt.TypeCheckError:
  pass
```

### The `SequenceLayer` type:

A `SequenceLayer` is a Flax `nn.Module` subclass that implements a set of APIs
to enable layer-by-layer processing and step-by-step processing of a sequence,
where the layer-wise and step-wise processing are **required** to produce
identical results.

### Example of layer-wise and step-wise processing.

In this example, we create a causal, strided Conv1D layer and execute it
layer-wise and step-wise, verifying the output is identical in both cases.

```python
k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)

# A randomly generated input sequence.
x = sl.Sequence(
  values=jax.random.normal(k1, (2, 13, 5)),
  mask=jnp.ones((2, 13), dtype=jnp.bool_),
)
assert x.channel_shape == (5,)
assert x.dtype == jnp.float32

# A simple causal 1D strided convolution.
l = sl.Conv1D.Config(7, kernel_size=3, strides=2, padding='causal_valid').make()

# For imperative demonstration, bind variables.
l = l.bind(l.init(k1, x, training=False))

# This layer supports step-wise processing (causal convolution is streamable).
assert l.supports_step

# The output ratio tells us the layer produces one output for 2 inputs.
assert l.output_ratio == 1/2

# Inputs to the `step` method must be multiples of the block size. Since 2
# inputs are required to produce one output at stride 2, the block size is 2.
assert l.block_size == 2

# Output shape is (7,)
assert l.get_output_shape(x.channel_shape) == (7,)
assert l.get_output_dtype(x.dtype) == jnp.float32

# Layer-wise execution:
y = l.layer(x, training=False)

# Verify the output is as expected.
assert y.shape == (2, 7, 7)
assert y.channel_shape == (7,)
assert y.dtype == jnp.float32

# Step-wise execution:

# Get a PyTree of state over time for the layer.
state = l.get_initial_state(
  batch_size=x.shape[0], input_spec=x.channel_spec, training=False)

# Pad x so we can slice it into blocks in the loop.
x = x.pad_time(0, 1, valid=False)
num_blocks = x.shape[1] // l.block_size

y_blocks = []
for i in range(num_blocks):
  x_i = x[:, i * l.block_size : (i + 1) * l.block_size]
  y_i, state = l.step(x_i, state, training=False)
  y_blocks.append(y_i)

# Concatenate resulting outputs on the time dimension.
y_blocks = sl.Sequence.concatenate_sequences(y_blocks)

# Verify layer-wise and step-wise processing are equivalent:
np.testing.assert_array_equal(y.values, y_blocks.values)
```

### API

The above example made use of these `SequenceLayer` APIs:

#### `block_size: int`

A property denoting the multiple of input timesteps required for the layer's
step-wise processing. `step()` requires inputs whose time dimension is divisible
by `block_size`.

#### `output_ratio: fractions.Fraction`

A `fractions.Fraction`. The ratio of output timesteps to input timesteps. If `>
1`, the layer "upsamples" its input. If `< 1`, the layer "downsamples" its
input.

#### `supports_step: bool`

A property indicating whether the layer supports the `step()` method. For
example, a causal convolution supports `step()`, while a bidirectional RNN does
not.

#### `layer(x: Sequence, *, training: bool, constants: Constants = None) -> Sequence`:

Process `Sequence` `x` (`[b, t, ...]`) layer-wise, producing a `Sequence` `[b, t
* layer.output_ratio, ...output_shape]`.

#### `step(x: Sequence, state: State, *, training: bool, constants: Constants = None) -> tuple[Sequence, State]`:

Process `Sequence` `x` (`[b, t, ...]`) step-wise where `x` contains a multiple
of `block_size` timesteps (`t % layer.block_size == 0`).

`t` is typically the smallest block of timesteps that can be processed in one
step to produce one output timestep from a chain of `SequenceLayer`s. Produces a
`Sequence` `[b, t * layer.output_ratio, output_shape]` output and a structure of
state tensors matching `get_initial_state`.

#### `get_initial_state(batch_size: int, input_spec: DTypeShape, constants: Constants) -> State`:

Get the initial state for a given batch size and shape/dtype input. An arbitrary
PyTree of state tensors or sequences.

#### `get_output_shape(input_shape: Shape, constants: Constants) -> tf.TensorShape`:

Get the output shape for the layer. The input and output shape only refer to the
channels dimensions of the sequence (i.e. do not include the batch or time
dimension).

#### `get_output_dtype(input_dtype: DType) -> DType`:

Get the output dtype for a given input dtype.

### Contract

`SequenceLayer`s must obey the following contract:

*   **Layer-wise and step-wise equivalence**: If `SequenceLayer.supports_step`,
    `SequenceLayer.layer` and `SequenceLayer.step` must produce identical
    results when fed identical data and starting state (slicing the data up into
    blocks of multiples of `SequenceLayer.block_size` timesteps.

    *   Stateful stochastic layers (e.g. `Dropout`) should obey this property if
        RNG state were made deterministic.

*   **Padding invariance**: The overall input `Sequence` `values` (`[b, t,
    ...]`) batch size `b` or length `t` must have no impact on the resulting
    computation for an individual sequence in the batch. For example, adding
    arbitrary amounts of padding to the end of the input `values` must not
    change the valid portions of the returned sequence.

    *   **Corollary:** Padding values must not affect the calculation of
        non-padding values.

## The `SequenceLayerConfig` type:

A `SequenceLayerConfig` is a `MakeableConfig[SequenceLayer]` alias, i.e. a
protocol that provides a `def make(self) -> SequenceLayer` method that
constructs a `SequenceLayer`. Conventionally, a `SequenceLayerConfig` is a
frozen dataclass containing configuration for the layer returned by the `make()`
method. By convention, the config class is nested within the layer it serves as
configuration for.

```python

class MyLayer(sl.SequenceLayer):

  @dataclasses.dataclass(frozen=True)
  class Config(sl.SequenceLayerConfig):
    num_units: int
    name: str

    def make(self) -> 'MyLayer':
      return MyLayer(config=self, name=self.name)

  config: Config

l = MyLayer.Config(num_units=5).make()

```

## Combinator `SequenceLayer`s

### `Serial`

The `Serial` `SequenceLayer` is a combinator that allows executing a sequence of
layers in serial.

```python
key = jax.random.PRNGKey(42)

x = sl.Sequence(
  values=jax.random.normal(key, (2, 13, 5)),
  mask=jnp.ones((2, 13), dtype=jnp.bool_),
)

l = sl.Serial.Config([
  sl.Conv1D.Config(8, kernel_size=3, strides=2, padding='causal_valid'),
  sl.GroupNormalization.Config(num_groups=4, cumulative=True),
  sl.Relu.Config(),

  sl.Conv1D.Config(8, kernel_size=3, strides=2, padding='causal_valid'),
  sl.GroupNormalization.Config(num_groups=4, cumulative=True),
  sl.Relu.Config(),

  sl.Conv1D.Config(8, kernel_size=3, strides=2, padding='causal_valid'),
  sl.GroupNormalization.Config(num_groups=4, cumulative=True),
  sl.Relu.Config()
]).make()


# For imperative demonstration, bind variables to l.
l = l.bind(l.init(k1, x, training=False))

assert l.supports_step
assert l.output_ratio == 1/8
assert l.block_size == 8
assert l.get_output_shape(x.channel_shape) == (8,)
assert l.get_output_dtype(x.dtype) == jnp.float32
```

Since `Serial` knows the `output_ratio` and `block_size`, and output
shapes/types and steppability of its children, it can compute the overall output
ratio, block size, and output shape/dtype, and steppability of the serial
composition.

### `Residual`

Like `Serial`, `Residual` is a combinator that allows executing a sequence of
layers in serial, with the added ability to sum the output with the input to
achieve the common residual pattern found in many networks succinctly and with
no risk of bugs due to misshaped residuals or time misalignment due to
down/upsampling in the residual body.

```python
key = jax.random.PRNGKey(42)

x = sl.Sequence(
  values=jax.random.normal(key, (2, 13, 8)),
  mask=jnp.ones((2, 13), dtype=jnp.bool_),
)

l = sl.Residual.Config([
  sl.Conv1D.Config(8, kernel_size=3, strides=1, padding='causal_valid'),
  sl.GroupNormalization.Config(num_groups=4, cumulative=True),
  sl.Relu.Config(),
]).make()

# For imperative demonstration, bind variables to l.
l = l.bind(l.init(k1, x, training=False))

assert l.supports_step
assert l.output_ratio == 1
assert l.block_size == 1
assert l.get_output_shape(x.channel_shape) == (8,)
assert l.get_output_dtype(x.dtype) == jnp.float32
```
