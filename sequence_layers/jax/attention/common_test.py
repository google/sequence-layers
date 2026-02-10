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
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from sequence_layers.jax import test_utils
from sequence_layers.jax import types
from sequence_layers.jax import utils
from sequence_layers.jax.attention import common


def _dot_product_attention_reference(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    query_ids: jax.Array,
    key_ids: jax.Array,
    key_mask: jax.Array,
    query_pos: jax.Array,
    key_pos: jax.Array,
    max_past_horizon: int | None,
    max_future_horizon: int | None,
    query_scale: jax.Array | None,
    per_dim_scale: jax.Array | None,
    attention_logits_soft_cap: float | None = None,
    compute_dtype: jnp.dtype | None = None,
) -> jax.Array:
  assert query_ids.shape == query.shape[:2], (query_ids.shape, query.shape)
  assert query_pos.shape == query.shape[:2]
  assert key_ids.shape == key.shape[:2]
  assert key_mask.shape == key.shape[:2]
  assert key_pos.shape == key.shape[:2]

  if compute_dtype is not None:
    query = query.astype(compute_dtype)
    key = key.astype(compute_dtype)
    value = value.astype(compute_dtype)

  query = common._scale_query(query, per_dim_scale, query_scale)

  num_query_heads = query.shape[-2]
  num_key_heads = key.shape[-2]

  if num_query_heads % num_key_heads != 0:
    raise ValueError(
        f'num_query_heads {num_query_heads} must be divisible by num_key_heads'
        f' {num_key_heads}'
    )

  query_heads_per_kv_head = num_query_heads // num_key_heads
  query = utils.split_dimension(
      query, axis=2, shape=(query_heads_per_kv_head, num_key_heads)
  )

  logits = jnp.einsum('biqnh,bjnh->bqnij', query, key)
  logits = logits.astype(jnp.float32)

  if attention_logits_soft_cap is not None:
    logits = attention_logits_soft_cap * jax.nn.tanh(
        logits / attention_logits_soft_cap
    )

  mask = key_mask[:, jnp.newaxis, :]
  mask &= query_ids[:, :, jnp.newaxis] == key_ids[:, jnp.newaxis, :]

  distance = query_pos[:, :, jnp.newaxis] - key_pos[:, jnp.newaxis, :]

  # Positive distance: query is ahead of key by distance timesteps.
  if max_past_horizon is not None:
    mask &= distance <= max_past_horizon

  # Negative distance: query is behind key by distance timesteps.
  if max_future_horizon is not None:
    mask &= distance >= -max_future_horizon

  logits = jnp.where(mask[:, jnp.newaxis, jnp.newaxis, :, :], logits, -1e9)
  probs = jax.nn.softmax(logits, axis=-1)
  assert probs.dtype == jnp.float32

  value = jnp.where(key_mask[:, :, jnp.newaxis, jnp.newaxis], value, 0)
  if compute_dtype is not None:
    assert value.dtype == compute_dtype

  context = jnp.einsum('bqnij,bjnh->biqnh', probs, value)

  context = context.reshape(
      (*context.shape[:2], num_query_heads, *context.shape[4:])
  )

  # Handle the case where no keys are valid for a given query.
  mask = jnp.any(mask, axis=-1, keepdims=True)[:, :, :, jnp.newaxis]
  context = jnp.where(mask, context, 0)

  context = context.astype(compute_dtype)

  return context


class DotProductAttentionHelperTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      query_scale=(None, 2.0), use_per_dim_scale=(False, True)
  )
  def test_query_scale(
      self, query_scale: float | None, use_per_dim_scale: bool
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 3, 5, 7, 11
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    valid_mask = jnp.ones((batch, num_heads, query_time, kv_time), jnp.bool_)

    if query_scale is None:
      query_scale = 1 / jnp.sqrt(units_per_head)

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
      queries_scaled = (
          queries * query_scale * 1.442695041 * jax.nn.softplus(per_dim_scale)
      )
    else:
      per_dim_scale = None
      queries_scaled = queries * query_scale

    _, probabilities = common.dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=per_dim_scale,
        per_dim_key_scale=None,
        query_scale=query_scale,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_positions=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    expected_probabilities = jax.nn.softmax(
        jnp.einsum('bqnh,bknh->bqnk', queries_scaled, keys)
    )
    self.assertAllClose(probabilities, expected_probabilities)

  @parameterized.product(
      (
          {'num_query_heads': 1, 'num_kv_heads': 1},
          {'num_query_heads': 3, 'num_kv_heads': 1},
          {'num_query_heads': 4, 'num_kv_heads': 2},
      ),
      broadcast_valid_mask=(True, False),
  )
  def test_grouped_query_attention(
      self,
      num_query_heads: int,
      num_kv_heads: int,
      broadcast_valid_mask: bool,
  ):
    batch, query_time, kv_time, units_per_head = 1, 3, 5, 11
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(42), 3)
    queries = jax.random.normal(
        k1, (batch, query_time, num_query_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_kv_heads, units_per_head))
    values = jax.random.normal(
        k3, (batch, kv_time, num_kv_heads, units_per_head)
    )
    valid_mask = jnp.ones(
        (
            batch,
            1 if broadcast_valid_mask else num_query_heads,
            query_time,
            kv_time,
        ),
        jnp.bool_,
    )

    _, probabilities = common.dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        per_dim_key_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_positions=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    query_scale = 1 / jnp.sqrt(units_per_head)

    num_query_heads_per_kv_head = num_query_heads // num_kv_heads
    queries = utils.split_dimension(
        queries, axis=2, shape=(num_kv_heads, num_query_heads_per_kv_head)
    )

    expected_probabilities = jax.nn.softmax(
        jnp.einsum('bikqh,bjkh->bikqj', queries * query_scale, keys)
    ).reshape((batch, query_time, num_query_heads, kv_time))
    self.assertAllClose(probabilities, expected_probabilities)

  def test_zero_fully_masked(self):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 3, 5, 1, 1

    queries = jnp.ones((batch, query_time, num_heads, units_per_head))
    keys = jnp.ones((batch, kv_time, num_heads, units_per_head))
    values = jnp.ones((batch, kv_time, num_heads, units_per_head))

    # [batch, num_heads, query_time, key_time]
    valid_mask = jnp.asarray([[[
        [True, True, True, True, True],
        [True, True, True, True, True],
        [False, False, False, False, False],
    ]]])

    context_vectors, _ = common.dot_product_attention(
        queries,
        keys,
        values,
        valid_mask,
        logit_bias=None,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        per_dim_key_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_positions=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    # Timestep 2 is fully masked so context vector is zero.
    expected_context_vectors = jnp.asarray([[[[1.0]], [[1.0]], [[0.0]]]])

    self.assertAllEqual(context_vectors, expected_context_vectors)

  @parameterized.product(
      use_per_dim_scale=(False, True),
      zero_fully_masked=(True, False),
  )
  def test_multi_key_value_dot_product_attention(
      self,
      use_per_dim_scale: bool,
      zero_fully_masked: bool,
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 3, 10, 12, 7, 11
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(42), 5)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    valid_mask = (
        jax.random.uniform(k5, (batch, num_heads, query_time, kv_time)) > 0.5
    )

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
    else:
      per_dim_scale = None

    expected_context_vectors, expected_probabilities = (
        common.dot_product_attention(
            queries,
            keys,
            values,
            valid_mask,
            logit_bias=None,
            training=False,
            attention_logits_soft_cap=None,
            attention_probabilities_dropout=None,
            per_dim_scale=per_dim_scale,
            per_dim_key_scale=None,
            query_scale=None,
            precision=None,
            get_logits_fn=None,
            zero_fully_masked=zero_fully_masked,
            compute_dtype=None,
            num_sink_positions=0,
            sink_key_logits=None,
            sink_value_embeddings=None,
        )
    )

    keys1, keys2 = jnp.split(keys, 2, axis=1)
    values1, values2 = jnp.split(values, 2, axis=1)
    valid_mask1, valid_mask2 = jnp.split(valid_mask, 2, axis=3)

    context_vectors, (probabilities1, probabilities2) = (
        common.multi_key_value_dot_product_attention(
            queries,
            (
                (keys1, values1, valid_mask1),
                (keys2, values2, valid_mask2),
            ),
            logit_bias=None,
            training=False,
            attention_logits_soft_cap=None,
            attention_probabilities_dropout=None,
            per_dim_scale=per_dim_scale,
            per_dim_key_scale=None,
            query_scale=None,
            precision=None,
            get_logits_fn=None,
            zero_fully_masked=zero_fully_masked,
            compute_dtype=None,
        )
    )
    probabilities = jnp.concatenate([probabilities1, probabilities2], axis=-1)

    self.assertAllClose(context_vectors, expected_context_vectors)
    self.assertAllClose(probabilities, expected_probabilities)


class SameSegmentTest(test_utils.SequenceLayerTest):

  def test_no_segments(self):
    with self.assertRaises(ValueError):
      common.SegmentMask()(
          common.QBundle(
              queries=None,
              segment_ids=jnp.array([[1, 2, 3], [4, 5, 6]]),
              position=None,
              mask=None,
          ),
          common.KVBundle(
              keys=None,
              values=None,
              segment_ids=None,
              position=None,
              mask=None,
          ),
      )
    with self.assertRaises(ValueError):
      common.SegmentMask()(
          common.QBundle(
              queries=None,
              segment_ids=None,
              position=None,
              mask=None,
          ),
          common.KVBundle(
              keys=None,
              values=None,
              segment_ids=jnp.array([[1, 2, 3], [4, 5, 6]]),
              position=None,
              mask=None,
          ),
      )

  def test_basic(self):
    mask_fn = common.SegmentMask()

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=jnp.array([[1, 2, 3], [4, 5, 6]]),
            position=None,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=jnp.array([[3, 3, 2], [0, 1, 6]]),
            position=None,
            mask=None,
        ),
    )
    self.assertAllEqual(
        mask,
        jnp.array([
            [
                [False, False, False],
                [False, False, True],
                [True, True, False],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, True],
            ],
        ]),
    )


class LocalCausalMaskTest(test_utils.SequenceLayerTest):

  def test_unbounded(self):
    mask_fn = common.LocalCausalMask(None, None)

    query_time = 35
    key_time = 50

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    self.assertIsNone(mask)

  @parameterized.parameters(
      (0, 0),
      (None, 0),
      (0, None),
      (None, 5),
      (3, None),
      (3, 5),
      (3, 0),
      (0, 5),
  )
  def test_bounded(self, max_past_horizon, max_future_horizon):
    mask_fn = common.LocalCausalMask(max_past_horizon, max_future_horizon)

    query_time = 35
    key_time = 50

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    num_lower = max_past_horizon if max_past_horizon is not None else key_time
    num_upper = (
        max_future_horizon if max_future_horizon is not None else key_time
    )

    expected = utils.ones_matrix_band_part(
        query_time,
        key_time,
        num_lower=num_lower,
        num_upper=num_upper,
        out_dtype=jnp.bool_,
        out_shape=(1, query_time, key_time),
    )

    self.assertAllEqual(mask, expected)


class BlockwiseLocalCausalMaskTest(test_utils.SequenceLayerTest):

  def test_no_context(self):
    mask_fn = common.BlockwiseLocalCausalMask(
        block_size=2,
        max_past_horizon_blocks=0,
        max_future_horizon_blocks=0,
    )

    query_time = 10
    key_time = 10

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    expected = jnp.array([[
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False, False, False],
        [False, False, True, True, False, False, False, False, False, False],
        [False, False, True, True, False, False, False, False, False, False],
        [False, False, False, False, True, True, False, False, False, False],
        [False, False, False, False, True, True, False, False, False, False],
        [False, False, False, False, False, False, True, True, False, False],
        [False, False, False, False, False, False, True, True, False, False],
        [False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, True, True],
    ]])

    self.assertAllEqual(mask, expected)

  def test_one_block_past_horizon(self):
    mask_fn = common.BlockwiseLocalCausalMask(
        block_size=2,
        max_past_horizon_blocks=1,
        max_future_horizon_blocks=0,
    )

    query_time = 10
    key_time = 10

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    expected = jnp.array([[
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, True, True, True, True],
    ]])

    self.assertAllEqual(mask, expected)

  def test_one_block_future_horizon(self):
    mask_fn = common.BlockwiseLocalCausalMask(
        block_size=2,
        max_past_horizon_blocks=0,
        max_future_horizon_blocks=1,
    )

    query_time = 10
    key_time = 10

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    expected = jnp.array([[
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False],
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, True, True, True, True, False, False],
        [False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, True, True],
    ]])

    self.assertAllEqual(mask, expected)

  def test_infinite_past(self):
    mask_fn = common.BlockwiseLocalCausalMask(
        block_size=2,
        max_past_horizon_blocks=None,
        max_future_horizon_blocks=0,
    )

    query_time = 10
    key_time = 10

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    expected = jnp.array([[
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True, True, False, False],
        [True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True],
    ]])

    self.assertAllEqual(mask, expected)

  def test_infinite_future(self):
    mask_fn = common.BlockwiseLocalCausalMask(
        block_size=2,
        max_past_horizon_blocks=0,
        max_future_horizon_blocks=None,
    )

    query_time = 10
    key_time = 10

    query_pos = jnp.arange(query_time)[jnp.newaxis, :]
    key_pos = jnp.arange(key_time)[jnp.newaxis, :]

    mask = mask_fn(
        common.QBundle(
            queries=None,
            segment_ids=None,
            position=query_pos,
            mask=None,
        ),
        common.KVBundle(
            keys=None,
            values=None,
            segment_ids=None,
            position=key_pos,
            mask=None,
        ),
    )

    x = True
    o = False

    expected = jnp.array([[
        [x, x, x, x, x, x, x, x, x, x],
        [x, x, x, x, x, x, x, x, x, x],
        [o, o, x, x, x, x, x, x, x, x],
        [o, o, x, x, x, x, x, x, x, x],
        [o, o, o, o, x, x, x, x, x, x],
        [o, o, o, o, x, x, x, x, x, x],
        [o, o, o, o, o, o, x, x, x, x],
        [o, o, o, o, o, o, x, x, x, x],
        [o, o, o, o, o, o, o, o, x, x],
        [o, o, o, o, o, o, o, o, x, x],
    ]])

    self.assertAllEqual(mask, expected)


class OnlineMultiKeyValueDotProductTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      use_per_dim_scale=(False, True),
      horizon=((None, None), (3, 5), (None, 5), (3, None)),
      attention_logits_soft_cap=(None, 5.0),
      compute_dtype=(None, jnp.bfloat16),
      num_query_kv_heads=((7, 7), (8, 4)),
      query_block_size=(7, None),
  )
  def test_online_multi_key_value_dot_product_attention(
      self,
      use_per_dim_scale: bool,
      horizon: tuple[int, int],
      attention_logits_soft_cap: float | None,
      compute_dtype: types.DType | None,
      num_query_kv_heads: tuple[int, int],
      query_block_size: int | None,
  ):
    max_past_horizon, max_future_horizon = horizon
    num_query_heads, num_kv_heads = num_query_kv_heads
    batch, query_time, kv_time, units_per_head = 5, 16, 32, 11
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(42), 5)
    queries = jax.random.normal(
        k1, (batch, query_time, num_query_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_kv_heads, units_per_head))
    values = jax.random.normal(
        k3, (batch, kv_time, num_kv_heads, units_per_head)
    )
    keys_mask = jax.random.uniform(k5, (batch, kv_time)) > 0.5

    # Reset segment after this many timesteps.
    segment_length = 5
    query_pos = jnp.arange(query_time)[jnp.newaxis, :] % segment_length
    key_pos = jnp.arange(kv_time)[jnp.newaxis, :] % segment_length
    query_segment_ids = jnp.arange(query_time)[jnp.newaxis, :] // segment_length
    key_segment_ids = jnp.arange(kv_time)[jnp.newaxis, :] // segment_length

    query_pos = jnp.tile(query_pos, (batch, 1))
    key_pos = jnp.tile(key_pos, (batch, 1))
    query_segment_ids = jnp.tile(query_segment_ids, (batch, 1))
    key_segment_ids = jnp.tile(key_segment_ids, (batch, 1))

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head]) / np.sqrt(
          units_per_head
      )
      query_scale = None
    else:
      per_dim_scale = None
      query_scale = 1 / np.sqrt(units_per_head)

    expected_context_vectors = _dot_product_attention_reference(
        queries,
        keys,
        values,
        query_segment_ids,
        key_segment_ids,
        keys_mask,
        query_pos,
        key_pos,
        max_past_horizon,
        max_future_horizon,
        query_scale,
        per_dim_scale,
        attention_logits_soft_cap,
        compute_dtype,
    )

    if compute_dtype is None:
      atol, rtol = 1e-6, 1e-6
    else:
      atol, rtol = 1e-2, 1e-2

    with self.subTest('single functions.KVBundle'):
      kv_block_sizes = 3
      context_vectors = common.multi_key_value_dot_product_flash_attention(
          queries=common.QBundle(
              queries,
              segment_ids=query_segment_ids,
              position=query_pos,
              mask=None,
          ),
          query_block_size=query_block_size,
          kv_bundles=[
              common.KVBundle(
                  keys=keys,
                  values=values,
                  segment_ids=key_segment_ids,
                  position=key_pos,
                  mask=keys_mask,
              ),
          ],
          kv_block_sizes=kv_block_sizes,
          attention_mask_fns=[
              common.SegmentMask(),
              common.LocalCausalMask(max_past_horizon, max_future_horizon),
          ],
          attention_logits_soft_cap=attention_logits_soft_cap,
          per_dim_scale=per_dim_scale,
          query_scale=query_scale,
          precision=None,
          compute_dtype=compute_dtype,
      )
      self.assertAllClose(
          context_vectors, expected_context_vectors, atol=atol, rtol=rtol
      )

    with self.subTest('no key blocking'):
      kv_block_sizes = None
      context_vectors = common.multi_key_value_dot_product_flash_attention(
          queries=common.QBundle(
              queries,
              segment_ids=query_segment_ids,
              position=query_pos,
              mask=None,
          ),
          query_block_size=query_block_size,
          kv_bundles=[
              common.KVBundle(
                  keys=keys,
                  values=values,
                  segment_ids=key_segment_ids,
                  position=key_pos,
                  mask=keys_mask,
              ),
          ],
          kv_block_sizes=kv_block_sizes,
          attention_mask_fns=[
              common.SegmentMask(),
              common.LocalCausalMask(max_past_horizon, max_future_horizon),
          ],
          attention_logits_soft_cap=attention_logits_soft_cap,
          per_dim_scale=per_dim_scale,
          query_scale=query_scale,
          precision=None,
          compute_dtype=compute_dtype,
      )
      self.assertAllClose(
          context_vectors, expected_context_vectors, atol=atol, rtol=rtol
      )

    with self.subTest('multi functions.KVBundle'):
      kv_block_sizes = (3, 11)
      keys1, keys2 = jnp.split(keys, 2, axis=1)
      key_pos1, key_pos2 = jnp.split(key_pos, 2, axis=1)
      key_segment_ids1, key_segment_ids2 = jnp.split(key_segment_ids, 2, axis=1)
      values1, values2 = jnp.split(values, 2, axis=1)
      keys_mask1, keys_mask2 = jnp.split(keys_mask, 2, axis=1)

      context_vectors = common.multi_key_value_dot_product_flash_attention(
          queries=common.QBundle(
              queries,
              segment_ids=query_segment_ids,
              position=query_pos,
              mask=None,
          ),
          query_block_size=query_block_size,
          kv_bundles=[
              common.KVBundle(
                  keys=keys1,
                  values=values1,
                  segment_ids=key_segment_ids1,
                  position=key_pos1,
                  mask=keys_mask1,
              ),
              common.KVBundle(
                  keys=keys2,
                  values=values2,
                  segment_ids=key_segment_ids2,
                  position=key_pos2,
                  mask=keys_mask2,
              ),
          ],
          kv_block_sizes=kv_block_sizes,
          attention_mask_fns=[
              common.SegmentMask(),
              common.LocalCausalMask(max_past_horizon, max_future_horizon),
          ],
          attention_logits_soft_cap=attention_logits_soft_cap,
          per_dim_scale=per_dim_scale,
          query_scale=query_scale,
          precision=None,
          compute_dtype=compute_dtype,
      )
      self.assertAllClose(
          context_vectors, expected_context_vectors, atol=atol, rtol=rtol
      )


class LocalDotProductAttentionHelperTest(test_utils.SequenceLayerTest):

  @parameterized.product(
      query_scale=(None, 2.0), use_per_dim_scale=(False, True)
  )
  def test_query_scale(
      self, query_scale: float | None, use_per_dim_scale: bool
  ):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 16, 16, 1, 1
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    queries = jax.random.normal(
        k1, (batch, query_time, num_heads, units_per_head)
    )
    keys = jax.random.normal(k2, (batch, kv_time, num_heads, units_per_head))
    values = jax.random.normal(k3, (batch, kv_time, num_heads, units_per_head))
    keys_mask = jnp.ones((batch, kv_time), jnp.bool_)

    if query_scale is None:
      query_scale = 1 / jnp.sqrt(units_per_head)

    if use_per_dim_scale:
      per_dim_scale = jax.random.normal(k4, [units_per_head])
      queries_scaled = (
          queries * query_scale * 1.442695041 * jax.nn.softplus(per_dim_scale)
      )
    else:
      per_dim_scale = None
      queries_scaled = queries * query_scale

    max_past_horizon = 3
    max_future_horizon = 0
    block_size = 2
    context_size = block_size + max_past_horizon + max_future_horizon

    _, probabilities = common.local_dot_product_attention(
        queries,
        keys,
        keys_mask,
        values,
        block_size=block_size,
        max_past_horizon=max_past_horizon,
        max_future_horizon=max_future_horizon,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=per_dim_scale,
        per_dim_key_scale=None,
        query_scale=query_scale,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=False,
        compute_dtype=None,
        num_sink_positions=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    keys_blocks = common._extract_block_context(
        keys,
        block_size=block_size,
        left_context=max_past_horizon,
        right_context=max_future_horizon,
    )

    # [B, T, N, H] -> [B, U, W, N, H]; (U = T/W).
    queries_blocks = common._convert_to_block(
        queries_scaled, block_size=block_size
    )

    valid_mask_blocked = common._extract_block_context(
        keys_mask,
        block_size=block_size,
        left_context=max_past_horizon,
        right_context=max_future_horizon,
        # Mask is False for invalid timesteps.
        padding_val=False,
    )
    # Reshape to [b, h=1, num_blocks, block_size=1, context_size].
    valid_mask_blocked = valid_mask_blocked[:, jnp.newaxis, :, jnp.newaxis, :]

    local_causal_valid_mask = utils.ones_matrix_band_part(
        block_size,
        context_size,
        num_upper=max_past_horizon + max_future_horizon,
        num_lower=0,
        out_dtype=jnp.bool_,
        out_shape=[1, 1, 1, block_size, context_size],
    )

    valid_mask_blocked = jnp.logical_and(
        valid_mask_blocked,
        local_causal_valid_mask,
    )

    logits = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries_blocks,
        keys_blocks,
    )

    logits = jnp.where(
        valid_mask_blocked,
        logits,
        common._INVALID_LOGIT_VALUE,
    )
    logits = logits.transpose((0, 2, 3, 1, 4)).reshape(
        (batch, -1, num_heads, context_size)
    )
    expected_probabilities = jax.nn.softmax(logits)
    self.assertAllClose(probabilities, expected_probabilities)

  def test_zero_fully_masked(self):
    batch, query_time, kv_time, num_heads, units_per_head = 1, 5, 5, 1, 1

    queries = jnp.ones((batch, query_time, num_heads, units_per_head))
    keys = jnp.ones((batch, kv_time, num_heads, units_per_head))
    values = jnp.ones((batch, kv_time, num_heads, units_per_head))

    keys_mask = jnp.asarray([[False, True, True, True, True]])

    context_vectors, _ = common.local_dot_product_attention(
        queries,
        keys,
        keys_mask,
        values,
        block_size=2,
        max_past_horizon=2,
        max_future_horizon=0,
        training=False,
        attention_logits_soft_cap=None,
        attention_probabilities_dropout=None,
        per_dim_scale=None,
        per_dim_key_scale=None,
        query_scale=None,
        precision=None,
        get_logits_fn=None,
        zero_fully_masked=True,
        compute_dtype=None,
        num_sink_positions=0,
        sink_key_logits=None,
        sink_value_embeddings=None,
    )

    # Timestep 0 is fully masked since the first key timestep is masked and the
    # attention is causal.
    expected_context_vectors = jnp.asarray(
        [[[[0.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]]
    )

    self.assertAllEqual(context_vectors, expected_context_vectors)


if __name__ == '__main__':
  test_utils.main()
