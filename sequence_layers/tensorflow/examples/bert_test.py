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

from absl.testing import parameterized
import numpy as np
from sequence_layers import tensorflow as sl
from sequence_layers.tensorflow import test_util
from sequence_layers.tensorflow.examples import bert
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class BERTEncoderTest(test_util.SequenceLayerTest, parameterized.TestCase):

  def test_bert_encoder(self):
    batch_size, time, vocab_size = 2, 4, 10
    l = bert.BERTEncoder(
        vocab_size=vocab_size,
        num_layers=2,
        dimension=8,
        num_heads=2,
        dropout_rate=0.0,  # Disable dropout so train/test are equivalent.
        max_token_length=8,
    )

    token_ids = self.random_sequence(
        batch_size, time, dtype=tf.int32, low=0, high=vocab_size
    )
    source_ids = sl.Sequence(
        np.random.randint(0, 2, size=[batch_size, time]).astype(np.int32),
        token_ids.mask,
    )

    encoded_train, pooled_cls_train = l.encode(
        token_ids, source_ids, training=True
    )
    encoded_test, pooled_cls_test = l.encode(
        token_ids, source_ids, training=False
    )

    self.evaluate(tf1.global_variables_initializer())
    encoded_train, encoded_test, pooled_cls_train, pooled_cls_test = (
        self.evaluate(
            [encoded_train, encoded_test, pooled_cls_train, pooled_cls_test]
        )
    )

    self.assertEqual(encoded_train.values.shape, (batch_size, time, 8))
    self.assertEqual(pooled_cls_train.shape, (batch_size, 8))
    self.assertSequencesClose(encoded_train, encoded_test)
    self.assertAllClose(pooled_cls_train, pooled_cls_test)


if __name__ == '__main__':
  tf.test.main()
