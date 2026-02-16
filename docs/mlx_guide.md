# MLX Backend Guide

This guide covers using Sequence Layers with the MLX backend for inference on Apple Silicon.

## Installation

```bash
pip install sequence-layers[mlx]
```

## Workflow

The MLX backend lets you define architectures with the same Linen configs used
for JAX training, then run inference on Apple Silicon GPUs via MLX.

### 1. Define the architecture

Use Linen configs exactly as you would for JAX:

```python
import jax
import sequence_layers.jax as sl
from sequence_layers.jax.attention import dot_product_self_attention as dpa

config = sl.Serial.Config([
    sl.Residual.Config([
        sl.RMSNormalization.Config(),
        dpa.DotProductSelfAttention.Config(
            num_heads=4, units_per_head=32,
            max_past_horizon=512, max_future_horizon=0,
        ),
        sl.Flatten.Config(),
        sl.Dense.Config(features=128),
    ]),
    sl.Residual.Config([
        sl.RMSNormalization.Config(),
        sl.Dense.Config(features=256, activation=jax.nn.gelu),
        sl.Dense.Config(features=128),
    ]),
])
```

### 2. Train in JAX (or load existing weights)

```python
linen_model = config.make()
variables = linen_model.init(
    jax.random.PRNGKey(0), dummy_input, training=False,
)
params = variables['params']
# ... train ...
```

### 3. Create the MLX model

```python
import sequence_layers.mlx  # Registers MLX backend factories.

mlx_model = config.make(backend='mlx')
```

### 4. Load weights

```python
import mlx.core as mx
from sequence_layers.mlx import weight_converter
from sequence_layers.mlx import ShapeDType

weight_converter.load_linen_params(
    mlx_model, params, config,
    input_spec=ShapeDType((128,), mx.float32),
)
```

For models with `BatchNormalization`, pass `batch_stats` too:

```python
weight_converter.load_linen_params(
    mlx_model, params, config,
    input_spec=ShapeDType((128,), mx.float32),
    batch_stats=variables['batch_stats'],
)
```

For models with cross-attention (e.g. `DotProductAttention` or
`StreamingDotProductAttention`), pass `constants` so that deferred
layers can determine source dimensions:

```python
from sequence_layers.mlx import Sequence

source = Sequence(mx.zeros((1, 1, 64)), mx.ones((1, 1), dtype=mx.bool_))
weight_converter.load_linen_params(
    mlx_model, params, config,
    input_spec=ShapeDType((128,), mx.float32),
    constants={'encoder': source},
)
```

### 5. Run inference

**Full-sequence (layer mode):**

```python
from sequence_layers.mlx import Sequence

values = mx.random.normal(shape=(1, 100, 128))
mask = mx.ones((1, 100), dtype=mx.bool_)
x = Sequence(values, mask)
y = mlx_model.layer(x)
```

**Streaming (step mode):**

```python
spec = ShapeDType((128,), mx.float32)
state = mlx_model.get_initial_state(batch_size=1, input_spec=spec)

for frame in audio_frames:
    x = Sequence(frame, mx.ones((1, 1), dtype=mx.bool_))
    y, state = mlx_model.step(x, state)
    # Process y...
```

**Streaming with cross-attention constants:**

For models that use `DotProductAttention` (static cross-attention), pass
the full source as constants. Keys and values are pre-projected once in
`get_initial_state`:

```python
source = Sequence(encoder_output, encoder_mask)
constants = {'encoder': source}

state = mlx_model.get_initial_state(
    batch_size=1, input_spec=spec, constants=constants,
)
for frame in audio_frames:
    x = Sequence(frame, mx.ones((1, 1), dtype=mx.bool_))
    y, state = mlx_model.step(x, state, constants=constants)
```

For models that use `StreamingDotProductAttention`, source chunks arrive
at the same rate as input. Each step receives the corresponding source
slice:

```python
source_chunks = [...]  # Same number of chunks as input frames.
state = mlx_model.get_initial_state(
    batch_size=1, input_spec=spec,
    constants={'encoder': source_chunks[0]},
)
for frame, src in zip(audio_frames, source_chunks):
    x = Sequence(frame, mx.ones((1, 1), dtype=mx.bool_))
    y, state = mlx_model.step(x, state, constants={'encoder': src})
```

### 6. Export for deployment

```python
from sequence_layers.mlx import export

export.export_step(mlx_model, 'model.mlxfn', batch_size=1, input_spec=spec)
```

## Supported Layers

The MLX backend supports the following JAX configs via `config.make(backend='mlx')`.
Layers not listed here (e.g. Conv2D/3D, Pooling2D/3D, LSTM, RGLRU,
DotProductSelfAttentionV2, Bidirectional, etc.) are JAX-only.

| Category       | Layers |
|---------------|--------|
| Simple         | Identity, Relu, Gelu, Swish, Tanh, Sigmoid, LeakyRelu, Elu, Softmax, Softplus, Cast, Scale, Add, MaskInvalid, GatedUnit, GatedLinearUnit, GatedTanhUnit, Flatten, Reshape, ExpandDims, Squeeze, Transpose, OneHot, Embedding, Dropout, Downsample1D, Upsample1D, CheckpointName, Lambda, Logging |
| Dense          | Dense, EinsumDense |
| Normalization  | RMSNormalization, LayerNormalization, GroupNormalization, BatchNormalization, L2Normalize |
| Position       | ApplyRotaryPositionalEncoding |
| Attention      | DotProductSelfAttention, LocalDotProductSelfAttention, DotProductAttention, StreamingDotProductAttention, StreamingLocalDotProductAttention |
| Conditioning   | Conditioning |
| Convolution    | Conv1D, DepthwiseConv1D, Conv1DTranspose |
| Pooling        | MaxPooling1D, MinPooling1D, AveragePooling1D |
| DSP            | Delay, Lookahead, Window, Frame, OverlapAdd, FFT, IFFT, RFFT, IRFFT, STFT, InverseSTFT, LinearToMelSpectrogram |
| Combinators    | Serial, Residual, Repeat, Parallel |

## Key Differences from JAX

- **Inference only** -- no training, no gradient computation.
- **Deferred initialization** -- Dense, Conv, and Attention layers create weights
  on the first forward pass (Linen configs don't specify `in_features`).
- **No scan/vmap** -- `Repeat` uses N independent copies instead of stacked
  params.
- **Kernel layouts** -- weights are automatically transposed by
  `load_linen_params` (e.g., Dense `[in, out]` to `[out, in]`).
- **BatchNormalization** -- inference-only; uses running mean/variance. Training
  mode raises an error.

## Attention Variants

### Self-Attention (`DotProductSelfAttention`)

Queries, keys, and values all come from the input sequence. Supports causal
masking, grouped query attention (GQA), and optional Q/K processing networks
(e.g. RoPE). In step mode, uses a rolling KV cache.

### Local Self-Attention (`LocalDotProductSelfAttention`)

Extends `DotProductSelfAttention` with a configurable `block_size` for step-mode
processing. The sliding window behavior uses banded visibility masks via
`max_past_horizon` and `max_future_horizon`. Also supports
`attention_logits_soft_cap` for logit capping (e.g. Gemma 2 uses 50.0).

### Cross-Attention (`DotProductAttention`)

Queries come from the input; keys/values come from a source sequence in
`constants`. In step mode, keys and values are pre-projected once during
`get_initial_state`, so each step only projects queries.

### Streaming Cross-Attention (`StreamingDotProductAttention`)

Like cross-attention, but the source arrives in streaming chunks at the same
rate as the input. Keys and values are projected per-step and stored in a
rolling KV buffer. Layer mode uses a banded visibility mask.

This class handles both `StreamingDotProductAttention.Config` and
`StreamingLocalDotProductAttention.Config` from the JAX backend (they differ
only in layer-mode efficiency optimizations).

## Weight Conversion Details

`load_linen_params` handles all structural differences between Linen and MLX:

| Layer | Linen shape | MLX shape | Transform |
|-------|------------|-----------|-----------|
| Dense | `[in, out]` | `[out, in]` | Transpose |
| Conv1D | `[k, in, out]` | `[out, k, in]` | `transpose(2,0,1)` |
| Conv1DTranspose | `[k, in, out]` | `[out, k, in]` | Flip spatial + `transpose(2,0,1)` |
| DepthwiseConv1D | `[k, in, 1]` | `[1, k, in]` | Same as Conv1D |
| Self-Attention (Combined QKV) | `[in, 3, heads, uph]` | 3x `[in, heads*uph]` | Split axis 1, reshape |
| Self-Attention (Separate Q/K/V, GQA) | Q: `[in, heads, uph]`, K: `[in, kv_heads, uph]`, V: same | Q: `[in, heads*uph]`, K/V: `[in, kv_heads*uph]` | Reshape each |
| Cross-Attention Q+KV | Q: `[in, heads, uph]`, KV: `[src, 2, heads, uph]` | Q: `[in, heads*uph]`, K/V: `[src, heads*uph]` | Reshape, split KV axis 1 |
| Repeat | `[N, ...]` | N copies of `[...]` | Slice axis 0 |
| Embedding | `[vocab, dim]` | `[vocab, dim]` | No change |
| RMS/LayerNorm | `[dim]` | `[dim]` | No change |
| GroupNorm | scale: `[dim]`, bias: `[dim]` | Same | No change |
| BatchNorm | scale/bias from `params`, mean/var from `batch_stats` | Same | No change |
| EinsumDense | `kernel` (einsum-shaped) | Same | No change |
| Conditioning (LINEAR) | `dense/kernel`, `dense/bias` | Same | No change (same einsum equation) |
