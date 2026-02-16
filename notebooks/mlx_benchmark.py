# %% [markdown]
# # JAX vs MLX Step Latency Benchmark
#
# Measures token-by-token autoregressive step latency for decoder
# transformers across three backends:
#
# 1. **JAX (jitted)** — `jax.jit` compiled step with `block_until_ready()`
# 2. **MLX (native)** — direct `model.step()` with `mx.eval()`
# 3. **MLX (exported)** — `.mlxfn` exported step with `mx.eval()`
#
# Runs multiple model sizes to show the crossover point where GPU
# throughput overtakes CPU.
#
# **Requires:** `pip install sequence-layers[mlx]`

# %% [markdown]
# ## 1. Setup

# %%
import os
import tempfile
import time

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

BATCH_SIZE = 1
MAX_PAST = 128
WARMUP = 10
NUM_STEPS = 50

CONFIGS = [
    {'label': 'Small', 'dim': 64, 'heads': 4, 'layers': 2},
    {'label': 'Medium', 'dim': 256, 'heads': 8, 'layers': 4},
    {'label': 'Large', 'dim': 512, 'heads': 8, 'layers': 8},
]
VOCAB_SIZE = 256

print(f'Benchmark config: warmup={WARMUP}, steps={NUM_STEPS}')
print(f'Model sizes: {[c["label"] for c in CONFIGS]}')

# %% [markdown]
# ## 2. Helpers


# %%
def make_config(dim, heads, layers):
  return sl.Serial.Config([
      sl.Embedding.Config(
          num_embeddings=VOCAB_SIZE,
          dimension=dim,
      ),
      sl.Repeat.Config(
          num_repeats=layers,
          layer=sl.Serial.Config([
              sl.Residual.Config([
                  sl.RMSNormalization.Config(),
                  sl.DotProductSelfAttention.Config(
                      num_heads=heads,
                      units_per_head=dim // heads,
                      max_past_horizon=MAX_PAST,
                      max_future_horizon=0,
                      query_network=(
                          sl.ApplyRotaryPositionalEncoding.Config(
                              max_wavelength=10_000.0,
                          )
                      ),
                      key_network=(
                          sl.ApplyRotaryPositionalEncoding.Config(
                              max_wavelength=10_000.0,
                          )
                      ),
                  ),
                  sl.Flatten.Config(),
              ]),
              sl.Residual.Config([
                  sl.RMSNormalization.Config(),
                  sl.Dense.Config(
                      features=dim * 4, activation=jax.nn.gelu
                  ),
                  sl.Dense.Config(features=dim),
              ]),
          ]),
      ),
      sl.RMSNormalization.Config(),
      sl.Dense.Config(features=VOCAB_SIZE),
  ])


def bench_jax(config, model_vars, jax_model):
  """Benchmark JAX jitted step."""
  jax_bound = jax_model.bind(model_vars)
  jax_input_spec = jax.ShapeDtypeStruct((), jnp.int32)
  jax_state = jax_bound.get_initial_state(
      BATCH_SIZE, jax_input_spec, training=False
  )

  @jax.jit
  def jax_step(x_values, x_mask, state):
    x = sl.Sequence(x_values, x_mask)
    y, new_state = jax_bound.step(x, state, training=False)
    return y.values, y.mask, new_state

  x_val = jnp.zeros((BATCH_SIZE, 1), dtype=jnp.int32)
  x_msk = jnp.ones((BATCH_SIZE, 1), dtype=jnp.bool_)

  # Warmup.
  state = jax_state
  for _ in range(WARMUP):
    y_vals, y_mask, state = jax_step(x_val, x_msk, state)
    jax.block_until_ready((y_vals, y_mask, state))

  # Timed.
  state = jax_state
  times = []
  for _ in range(NUM_STEPS):
    t0 = time.perf_counter()
    y_vals, y_mask, state = jax_step(x_val, x_msk, state)
    jax.block_until_ready((y_vals, y_mask, state))
    times.append(time.perf_counter() - t0)
  return times


def bench_mlx_native(mlx_model):
  """Benchmark MLX native step."""
  mlx_input_spec = ShapeDType((), mx.int32)
  export._materialize_deferred(mlx_model, BATCH_SIZE, mlx_input_spec)

  x = Sequence(
      mx.zeros((BATCH_SIZE, 1), dtype=mx.int32),
      mx.ones((BATCH_SIZE, 1), dtype=mx.bool_),
  )

  # Warmup.
  state = mlx_model.get_initial_state(BATCH_SIZE, mlx_input_spec)
  for _ in range(WARMUP):
    y, state = mlx_model.step(x, state)
    mx.eval(y.values)

  # Timed.
  state = mlx_model.get_initial_state(BATCH_SIZE, mlx_input_spec)
  times = []
  for _ in range(NUM_STEPS):
    t0 = time.perf_counter()
    y, state = mlx_model.step(x, state)
    mx.eval(y.values)
    times.append(time.perf_counter() - t0)
  return times


def bench_mlx_exported(mlx_model):
  """Benchmark MLX exported step."""
  mlx_input_spec = ShapeDType((), mx.int32)
  path = os.path.join(tempfile.gettempdir(), 'benchmark_decoder.mlxfn')
  export.export_step(
      mlx_model, path, batch_size=BATCH_SIZE, input_spec=mlx_input_spec
  )
  imported = mx.import_function(path)

  x_val = mx.zeros((BATCH_SIZE, 1), dtype=mx.int32)
  x_msk = mx.ones((BATCH_SIZE, 1), dtype=mx.bool_)

  # Warmup.
  flat_state, _ = export.get_initial_state_flat(
      mlx_model, BATCH_SIZE, mlx_input_spec
  )
  for _ in range(WARMUP):
    y_vals, y_mask, flat_state = export.run_exported(
        imported, x_val, x_msk, flat_state
    )
    mx.eval(y_vals)

  # Timed.
  flat_state, _ = export.get_initial_state_flat(
      mlx_model, BATCH_SIZE, mlx_input_spec
  )
  times = []
  for _ in range(NUM_STEPS):
    t0 = time.perf_counter()
    y_vals, y_mask, flat_state = export.run_exported(
        imported, x_val, x_msk, flat_state
    )
    mx.eval(y_vals)
    times.append(time.perf_counter() - t0)

  os.remove(path)
  return times


# %% [markdown]
# ## 3. Run Benchmarks

# %%
all_results = []

for cfg in CONFIGS:
  label = cfg['label']
  dim, heads, layers = cfg['dim'], cfg['heads'], cfg['layers']
  print(f'\n{"=" * 60}')
  print(f'{label}: dim={dim}, heads={heads}, layers={layers}')
  print('=' * 60)

  config = make_config(dim, heads, layers)

  # JAX init.
  jax_model = config.make()
  x_init = sl.Sequence(
      jnp.zeros((BATCH_SIZE, 1), dtype=jnp.int32),
      jnp.ones((BATCH_SIZE, 1), dtype=jnp.bool_),
  )
  model_vars = jax_model.init(jax.random.key(0), x_init, training=False)
  jax_params = flax.core.meta.unbox(model_vars)['params']
  param_count = sum(
      x.size for x in jax.tree_util.tree_leaves(jax_params)
  )
  print(f'Parameters: {param_count:,}')

  # MLX init.
  mlx_model = config.make(backend='mlx')
  weight_converter.load_linen_params(mlx_model, jax_params, config)

  # Benchmark all three.
  print('  JAX jitted...', end='', flush=True)
  jax_times = bench_jax(config, model_vars, jax_model)
  jax_mean = np.mean(jax_times) * 1000
  jax_std = np.std(jax_times) * 1000
  print(f' {jax_mean:.3f} ms')

  print('  MLX native...', end='', flush=True)
  mlx_times = bench_mlx_native(mlx_model)
  mlx_mean = np.mean(mlx_times) * 1000
  mlx_std = np.std(mlx_times) * 1000
  print(f' {mlx_mean:.3f} ms')

  print('  MLX exported...', end='', flush=True)
  exp_times = bench_mlx_exported(mlx_model)
  exp_mean = np.mean(exp_times) * 1000
  exp_std = np.std(exp_times) * 1000
  print(f' {exp_mean:.3f} ms')

  all_results.append({
      'label': label,
      'params': param_count,
      'jax': (jax_mean, jax_std),
      'mlx_native': (mlx_mean, mlx_std),
      'mlx_exported': (exp_mean, exp_std),
  })

# %% [markdown]
# ## 4. Results Summary

# %%
print('\n')
print(f'{"Model":<10} {"Params":>10}  '
      f'{"JAX (ms)":>12} {"MLX nat (ms)":>14} {"MLX exp (ms)":>14}')
print('-' * 66)
for r in all_results:
  jm, js = r['jax']
  nm, ns = r['mlx_native']
  em, es = r['mlx_exported']
  print(
      f'{r["label"]:<10} {r["params"]:>10,}  '
      f'{jm:>6.2f}+/-{js:<4.2f} '
      f'{nm:>7.2f}+/-{ns:<4.2f} '
      f'{em:>7.2f}+/-{es:<4.2f}'
  )

print()
print('Tokens/sec:')
print(f'{"Model":<10} {"JAX":>10} {"MLX native":>12} {"MLX exported":>14}')
print('-' * 50)
for r in all_results:
  jt = 1000.0 / r['jax'][0]
  nt = 1000.0 / r['mlx_native'][0]
  et = 1000.0 / r['mlx_exported'][0]
  print(f'{r["label"]:<10} {jt:>10.0f} {nt:>12.0f} {et:>14.0f}')

print('\nDone!')
