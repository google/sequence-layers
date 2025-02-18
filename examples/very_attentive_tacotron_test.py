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

from absl import logging
from sequence_layers.examples import very_attentive_tacotron
from sequence_layers.tensorflow import test_util
import tensorflow.compat.v2 as tf


class VeryAttentiveTacotronTest(test_util.SequenceLayerTest):

  def test_small_vat_text_encoder(self):
    encoder = very_attentive_tacotron.SmallVATTextEncoder()
    logging.info(encoder)

  def test_large_vat_text_encoder(self):
    encoder = very_attentive_tacotron.LargeVATTextEncoder()
    logging.info(encoder)

  def test_small_t5tts_text_encoder(self):
    encoder = very_attentive_tacotron.SmallT5BaselineTextEncoder()
    logging.info(encoder)

  def test_large_t5tts_text_encoder(self):
    encoder = very_attentive_tacotron.LargeT5BaselineTextEncoder()
    logging.info(encoder)

  def test_small_very_attentive_decoder(self):
    decoder = very_attentive_tacotron.SmallVATDecoder()
    logging.info(decoder)

  def test_large_very_attentive_decoder(self):
    decoder = very_attentive_tacotron.LargeVATDecoder()
    logging.info(decoder)

  def test_small_t5_tts_decoder(self):
    decoder = very_attentive_tacotron.SmallT5BaselineDecoder()
    logging.info(decoder)

  def test_large_t5_tts_decoder(self):
    decoder = very_attentive_tacotron.LargeT5BaselineDecoder()
    logging.info(decoder)


if __name__ == '__main__':
  tf.test.main()
