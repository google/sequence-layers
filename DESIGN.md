# SequenceLayers Design & Philosophy

This document summarizes the core design principles, primitives, and contracts
of the `SequenceLayers` library, as detailed in `tech-report.pdf`. It is
designed as a highly readable reference for both human developers and AI coding
agents.

--------------------------------------------------------------------------------

## 1. Core Philosophy

SequenceLayers is a design pattern and library for sequence modeling. It is
built around three core features:

1.  **Streamable:** Gives you streaming "for free". Every streamable layer
    implements an explicit state and a `step` method to evolve that state,
    allowing easy transition from offline training to online streaming.
2.  **Correct by Default:** Eliminates entire classes of bugs related to
    masking, padding, causality, and lookahead by enforcing a strict
    mathematical contract and using unified `Sequence` containers.
3.  **Composable:** Uses a declarative, compositional API (combinators like
    `Serial`, `Residual`) that allows complex models to be defined like block
    diagrams, automatically handling state plumbing and aggregate properties
    (latency, receptive fields).

--------------------------------------------------------------------------------

## 2. Core Primitives

### `Sequence` and `MaskedSequence`

Instead of raw tensors, SequenceLayers APIs consume and produce `Sequence`
objects.

*   **Structure:** A PyTree dataclass pairing `values` (shape `[batch, time,
    ...channels]`) with a boolean `mask` (shape `[batch, time]`).
*   **Masking:** A `Sequence` is "masked" if all invalid positions (`mask[b,
    t] == False`) have their corresponding `values[b, t, ...]` zeroed out.
*   **`MaskedSequence`:** A subclass of `Sequence` that statically guarantees
    that invalid positions are already zeroed out. Calling `.mask_invalid()` on
    it is a no-op.

--------------------------------------------------------------------------------

## 3. The `SequenceLayer` API

A `SequenceLayer` is a functional component that supports two primary execution
modes:

### A. Layer-wise (Offline / Training)

Used for parallel processing (e.g., teacher-forced training).

```python
y = layer.layer(x, training=training)
```

*   **Input `x`:** `Sequence` of shape `[b, t_in, ...]`
*   **Output `y`:** `Sequence` of shape `[b, t_out, ...]` where `t_out = t_in *
    output_ratio`.

### B. Step-wise (Online / Streaming / Inference)

Used for autoregressive generation or streaming inference.

```python
state = layer.get_initial_state(batch_size, input_spec, training=training)
# In a loop:
y_step, state = layer.step(x_step, state, training=training)
```

*   **Input `x_step`:** `Sequence` of shape `[b, block_size, ...]` (usually
    `block_size = 1`)
*   **State:** An explicit PyTree of arrays representing the layer's temporal
    state (e.g., KV cache, convolution buffer). No state is stored internally in
    the layer object.

### Constants

Layers may accept a `Constants` dictionary for time-synchronized conditioning
signals (e.g., speaker embeddings, language IDs). Constants are propagated
through combinators alongside `Sequence` and `State`.

### Emits

Since `layer()` and `step()` return a single `Sequence`, layers that need
auxiliary debugging output use the **Emits** API:

*   `layer_with_emits(x, constants) -> (Sequence, Emits)`: Layer-wise with
    auxiliary outputs.
*   `step_with_emits(x, state, constants) -> (Sequence, State, Emits)`:
    Step-wise with auxiliary outputs.

The `Emitting` subclass of `SequenceLayer` implements `layer`/`step` in terms of
the `_with_emits` variants. The `Emit` layer simply emits its input for tapping
into intermediate sequences.

### Receptive Field

The `receptive_field` property computes the (start, end) range of input
timesteps affecting each output timestep. Key details:

*   For layers with `output_ratio != 1`, the receptive field is relative to
    `t_i = t_o // output_ratio`.
*   `receptive_field_per_step` tracks step-specific receptive fields (used by
    combinators like `Serial` for precise composition).
*   Special cases: infinite receptive fields (e.g., LSTM → `(-inf, 0)`), `None`
    for timesteps with no receptive field (e.g., transposed convolution holes).

--------------------------------------------------------------------------------

## 4. The SequenceLayer Contract (CRITICAL)

For a layer to be correct, it **MUST** satisfy the following properties,
verified via the `verify_contract` test utility:

1.  **Layer-Step Equivalence:** Running a sequence through `layer()` must
    produce mathematically identical results (values and mask) to feeding it
    chunk-by-chunk through `step()` and concatenating the outputs, once latency
    is accounted for. Stateful stochastic layers (e.g., `Dropout`) should obey
    this when the starting RNG state is equivalent.
2.  **Padding Invariance:** Appending padding (invalid timesteps) to the end of
    an input sequence must not affect the output values of the non-padding
    (valid) timesteps. *Note: This is currently only required for end padding.
    Start or interior padding may affect behavior.*
3.  **Batching Invariance:** The position of an example in a batch, or the
    lengths of other examples in the batch, must not affect its computed output.
4.  **Masked Inputs/Outputs:** Layers must NOT assume input `values` are masked.
    If a layer's computation requires masked inputs (e.g., it mixes information
    across timesteps), it must call `mask_invalid()` on the input before use.

## `verify_contract` checks: layer-step output equivalence, gradient equivalence (parameters and inputs), consistency with metadata (`get_output_spec`, `output_ratio`, `block_size`, latencies), receptive field matching (via gradient-based calculation), batching invariance (inserting invalid batch items), and padding invariance (replacing invalid timesteps with NaNs or large integers).

## 5. Latency and Lookahead in Streaming

When a layer requires future context (lookahead) or introduces delay, it manages
this via two properties:

*   **`input_latency`:** The number of future timesteps required to produce the
    current output. To get all valid outputs, the caller must "flush" the layer
    at the end of the sequence by feeding it `input_latency` invalid (padded)
    timesteps.
*   **`output_latency`:** The delay introduced by the layer. The first
    `output_latency` timesteps returned by `step()` will be invalid (`mask =
    False`) and must be discarded by the caller before expecting valid outputs.

*For a causal layer, both latencies are `0`.*

--------------------------------------------------------------------------------

## 6. Combinators

Layers are composed into larger architectures using backend-agnostic
combinators:

*   **`Serial`:** Executes a list of layers sequentially. Automatically handles
    the nesting and plumbing of sub-layer states into a single aggregate state.
*   **`Parallel`:** Executes multiple layers on the same input in parallel,
    combining their outputs.
*   **`Residual`:** Implements `F(x) + x`, managing state for `F`.
*   **`Repeat`:** Repeats a layer `N` times, using control flow primitives like
    `scan` to minimize compilation time.
*   **`Blockwise`:** Dynamically adjusts the execution block size of any layer,
    automatically implementing `layer` in terms of `step` to reduce peak memory.

--------------------------------------------------------------------------------
