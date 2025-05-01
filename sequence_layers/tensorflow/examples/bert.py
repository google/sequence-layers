# Copyright 2024 Google LLC
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

"""BERT using SequenceLayers.

Based on:
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/abs/1810.04805
"""

from typing import Optional, Tuple
from sequence_layers import tensorflow as sl
from sequence_layers.tensorflow.examples import transformer
import tensorflow.compat.v2 as tf


class BERTEncoder(tf.Module):
  """A BERT encoder network."""

  def __init__(
      self,
      vocab_size: int,
      num_layers: int = 12,
      dimension: int = 768,
      num_heads: int = 12,
      dropout_rate: float = 0.1,
      kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
      max_token_length: int = 512,
      name: Optional[str] = None,
  ):
    """Create a BERT encoder.

    Args:
      vocab_size: The vocab size for the token embedding.
      num_layers: Number of Transformer layers in the encoder.
      dimension: The model dimenison.
      num_heads: The number of heads to use in self-attention layers.
      dropout_rate: The dropout rate to use throughout the model (attention
        probabilities, feed-forward network outputs, etc.).
      kernel_initializer: Initializer to use for kernels throughout the model.
        Biases are initialized as zeros.
      max_token_length: The maximum length in tokens the model supports.
      name: An optional name for the module.
    """
    super().__init__(name=name)
    if kernel_initializer is None:
      kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    with self.name_scope:
      self._token_embedding = tf.keras.layers.Embedding(
          vocab_size,
          dimension,
          embeddings_initializer=transformer.CloneInitializer(
              kernel_initializer
          ),
          name='token_embeddings',
      )
      self._source_embedding = tf.keras.layers.Embedding(
          2,
          dimension,
          embeddings_initializer=transformer.CloneInitializer(
              kernel_initializer
          ),
          name='source_embeddings',
      )
      self._position_embedding = tf.Variable(
          transformer.CloneInitializer(kernel_initializer)(
              (max_token_length, dimension)
          ),
          dtype=tf.float32,
          name='position_embeddings',
      )
      self._encoder = sl.Serial([
          sl.LayerNormalization(epsilon=1e-12),
          sl.Dropout(dropout_rate),
          transformer.TransformerEncoder(
              num_layers=num_layers,
              dimension=dimension,
              num_heads=num_heads,
              max_horizon=-1,
              max_future_horizon=-1,
              ffn_activation=tf.nn.gelu,
              ffn_dropout_rate=dropout_rate,
              attention_probabilities_dropout_rate=dropout_rate,
              kernel_initializer=transformer.CloneInitializer(
                  kernel_initializer
              ),
              name='transformer',
          ),
      ])
      self._pooler = sl.Dense(
          dimension,
          kernel_initializer=transformer.CloneInitializer(kernel_initializer),
          activation=tf.nn.tanh,
          name='pooler_transform',
      )

  @tf.Module.with_name_scope
  def encode(
      self, token_ids: sl.Sequence, source_ids: sl.Sequence, training: bool
  ) -> Tuple[sl.Sequence, tf.Tensor]:
    """Encodes the provided sequence.

    Args:
      token_ids: A [batch_size, time] integer sequence of token IDs. The input
        ids are expected to be formatted according to the BERT paper, i.e.:
        [CLS] Token1 Token2 [SEP] Token3 Token4 [SEP] [MASK] [MASK] [MASK] ...
      source_ids: A [batch_size, time] integer sequence of source IDs (source 0
        is sentence A, source 1 is sentence B). source_ids mask is assumed equal
        to token_ids's mask.
      training: Whether we are in training mode.

    Returns:
      - A [batch_size, time, dimension] encoded sequence.
      - A [batch_size, dimension] tensor containing the pooled (CLS token)
        outputs for each batch item.
    """
    token_embeddings = token_ids.apply_values(
        self._token_embedding, training=training
    )
    source_embeddings = source_ids.apply_values(
        self._source_embedding, training=training
    )

    length = token_ids.values.shape[1] or tf.shape(token_ids.values)[1]
    position_embeddings = self._position_embedding[tf.newaxis, :length]

    embeddings = token_embeddings.apply_values(
        lambda v: v + source_embeddings.values + position_embeddings
    ).mask_invalid()
    outputs = self._encoder.layer(embeddings, training=training)
    cls_output = self._pooler.layer(outputs[:, :1], training=training)
    # TODO(rryan): Corner case of zero output.
    return outputs, cls_output.values[:, 0, :]
