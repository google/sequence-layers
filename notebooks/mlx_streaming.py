# %% [markdown]
# # MLX Streaming Inference Demo
#
# This notebook demonstrates the full MLX streaming inference pipeline:
#
# 1. Define a decoder transformer using SequenceLayers configs
# 2. Initialize weights in JAX (Linen)
# 3. Convert weights to MLX
# 4. Stream tokens natively in MLX
# 5. Export to `.mlxfn` for deployment
# 6. Stream tokens from the exported function
#
# No checkpoint is needed — we use random init weights throughout.
#
# **Requires:** `pip install sequence-layers[mlx]`

# %% [markdown]
# ## 1. Setup

# %%
import os
import tempfile

import flax.core.meta
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

import sequence_layers.jax as sl
from sequence_layers.mlx import basic_types as bt
from sequence_layers.mlx import export
from sequence_layers.mlx import weight_converter

Sequence = bt.Sequence
ShapeDType = bt.ShapeDType

# Hyperparameters (small model, fast to run).
VOCAB_SIZE = 256
DIM = 64
NUM_HEADS = 4
UNITS_PER_HEAD = DIM // NUM_HEADS  # 16
NUM_LAYERS = 2
FFN_DIM = DIM * 4  # 256
MAX_PAST = 128
BATCH_SIZE = 1
NUM_TOKENS = 16

print('Setup complete.')

# %% [markdown]
# ## 2. Define Architecture
#
# A small decoder-only transformer:
# ```
# Embedding → Repeat(N, [
#   Residual([RMSNorm, SelfAttention(RoPE), Flatten]),
#   Residual([RMSNorm, Dense(FFN, gelu), Dense(dim)]),
# ]) → RMSNorm → Dense(vocab_size)
# ```

# %%
config = sl.Serial.Config([
    sl.Embedding.Config(
        num_embeddings=VOCAB_SIZE,
        dimension=DIM,
    ),
    sl.Repeat.Config(
        num_repeats=NUM_LAYERS,
        layer=sl.Serial.Config([
            sl.Residual.Config([
                sl.RMSNormalization.Config(),
                sl.DotProductSelfAttention.Config(
                    num_heads=NUM_HEADS,
                    units_per_head=UNITS_PER_HEAD,
                    max_past_horizon=MAX_PAST,
                    max_future_horizon=0,
                    query_network=sl.ApplyRotaryPositionalEncoding.Config(
                        max_wavelength=10_000.0,
                    ),
                    key_network=sl.ApplyRotaryPositionalEncoding.Config(
                        max_wavelength=10_000.0,
                    ),
                ),
                sl.Flatten.Config(),
            ]),
            sl.Residual.Config([
                sl.RMSNormalization.Config(),
                sl.Dense.Config(features=FFN_DIM, activation=jax.nn.gelu),
                sl.Dense.Config(features=DIM),
            ]),
        ]),
    ),
    sl.RMSNormalization.Config(),
    sl.Dense.Config(features=VOCAB_SIZE),
])

print(f'Architecture: vocab={VOCAB_SIZE}, dim={DIM}, heads={NUM_HEADS}, '
      f'uph={UNITS_PER_HEAD}, layers={NUM_LAYERS}, ffn={FFN_DIM}, '
      f'max_past={MAX_PAST}')

# %% [markdown]
# ## 3. Initialize in JAX
#
# Build the Linen model, init with a dummy input, count parameters.

# %%
jax_model = config.make()
key = jax.random.key(0)

# Dummy input for init: a single token.
x_init = sl.Sequence(
    jnp.zeros((BATCH_SIZE, 1), dtype=jnp.int32),
    jnp.ones((BATCH_SIZE, 1), dtype=jnp.bool_),
)

model_vars = jax_model.init(key, x_init, training=False)
jax_params = flax.core.meta.unbox(model_vars)['params']

param_count = sum(
    x.size for x in jax.tree_util.tree_leaves(jax_params)
)
print(f'JAX model initialized: {param_count:,} parameters')

# %% [markdown]
# ## 4. Convert Weights to MLX
#
# Build an MLX model from the same config, load Linen weights, verify
# that `layer()` outputs match between JAX and MLX.

# %%
mlx_model = config.make(backend='mlx')
weight_converter.load_linen_params(mlx_model, jax_params, config)
print('Weights loaded into MLX model.')

# Verify layer() outputs match on a short sequence.
tokens = np.array([[0, 42, 7, 13, 99, 200, 1, 128]], dtype=np.int32)
mask = np.ones(tokens.shape, dtype=bool)

# JAX forward.
x_jax = sl.Sequence(jnp.array(tokens), jnp.array(mask))
jax_bound = jax_model.bind(model_vars)
y_jax = jax_bound.layer(x_jax, training=False)

# MLX forward.
x_mlx = Sequence(mx.array(tokens), mx.array(mask))
y_mlx = mlx_model.layer(x_mlx)
mx.eval(y_mlx.values)

np.testing.assert_allclose(
    np.array(y_mlx.values),
    np.array(y_jax.values),
    atol=1e-3,
    rtol=1e-3,
)
print(f'JAX and MLX layer() outputs match (atol=1e-3).')
print(f'  Output shape: {y_mlx.values.shape}')

# %% [markdown]
# ## 5. Native MLX Streaming
#
# Generate tokens one at a time using `model.step()` with greedy
# (argmax) decoding. This uses the KV cache internally.

# %%
input_spec = ShapeDType((), mx.int32)
state = mlx_model.get_initial_state(BATCH_SIZE, input_spec)

token = 0  # Start-of-sequence token.
generated = [token]

for _ in range(NUM_TOKENS - 1):
  x = Sequence(
      mx.array([[token]], dtype=mx.int32),
      mx.ones((1, 1), dtype=mx.bool_),
  )
  y, state = mlx_model.step(x, state)
  mx.eval(y.values)
  logits = y.values[0, 0]  # [vocab_size]
  token = int(mx.argmax(logits))
  generated.append(token)

print(f'Generated {len(generated)} tokens (native step):')
print(generated)

# %% [markdown]
# ## 6. Export to .mlxfn
#
# Export the step function to a `.mlxfn` file. Model weights are
# captured in the closure; state arrays (KV cache) are explicit I/O.

# %%
export_path = os.path.join(tempfile.gettempdir(), 'decoder_demo.mlxfn')
export.export_step(
    mlx_model, export_path, batch_size=BATCH_SIZE, input_spec=input_spec
)
size_kb = os.path.getsize(export_path) / 1024
print(f'Exported to: {export_path}')
print(f'File size: {size_kb:.1f} KB')

# %% [markdown]
# ## 7. Streaming from Exported Function
#
# Load the `.mlxfn` back and run the same generation loop. Outputs
# must match the native step exactly (bit-for-bit).

# %%
imported = mx.import_function(export_path)
flat_state, structure = export.get_initial_state_flat(
    mlx_model, BATCH_SIZE, input_spec
)

token = 0
exported_generated = [token]

for _ in range(NUM_TOKENS - 1):
  x_values = mx.array([[token]], dtype=mx.int32)
  x_mask = mx.ones((1, 1), dtype=mx.bool_)
  y_vals, y_mask, flat_state = export.run_exported(
      imported, x_values, x_mask, flat_state
  )
  mx.eval(y_vals)
  logits = y_vals[0, 0]
  token = int(mx.argmax(logits))
  exported_generated.append(token)

print(f'Generated {len(exported_generated)} tokens (exported step):')
print(exported_generated)

assert generated == exported_generated, (
    f'Mismatch!\n  native:   {generated}\n  exported: {exported_generated}'
)
print('Native and exported outputs match exactly.')

# %% [markdown]
# ## 8. Cleanup

# %%
os.remove(export_path)
print(f'Removed {export_path}')
print('Done!')
