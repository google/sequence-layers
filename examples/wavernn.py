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

"""WaveRNN using SequenceLayers.

This demonstrates how to train and sample from an autoregressive,
sample-by-sample waveform predictor that is loosely based on WaveRNN. The main
differences between this model and WaveRNN are:
- Uses a single quantized logistic output distribution instead of the "double
  softmax" output distribution.
- Conditions on multiple previous waveform samples to demonstrate how
  SequenceLayers tracks convolution state.
- Adds truncated normal noise in training to avoid overfitting.

Based on:
- Efficient Neural Audio Synthesis
  https://arxiv.org/abs/1802.08435
- PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
  Likelihood and Other Modifications
  https://arxiv.org/abs/1701.05517
"""

from typing import Optional

from sequence_layers import tensorflow as sl
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def WaveRNNBody(
    num_units: int,
    num_distribution_parameters: int,
    conditioning_name: str,
    upsample_ratio: int,
) -> sl.SequenceLayer:
  """Builds a WaveRNN-like SequenceLayer body.

  This is the "body" of the autoregressive model. In sampling, the input is the
  previous output sample, while in training it is teacher forced inputs (i.e.
  the ground truth audio shifted by one timestep). The body doesn't know whether
  the inputs it processes are teacher forced or true samples.

  See the WaveRNN class below for an example of how to teacher force and sample
  from this model.

  Args:
    num_units: The number of units for the WaveRNN.
    num_distribution_parameters: The number of distribution parameters to
      predict per timestep.
    conditioning_name: The name of the conditioning feature provided to the
      layer.
    upsample_ratio: The ratio of waveform timesteps to conditioning feature
      timesteps.

  Returns:
    A SequenceLayer that outputs num_distribution_parameters for each timestep.
  """
  return sl.Serial([
      # Condition on the previous 5 samples.
      sl.Conv1D(filters=num_units, kernel_size=5),
      # Upsample and project the conditioning to num_units and add it to the
      # previous output representation.
      sl.UpsampleConditioning(
          conditioning_name,
          projection=sl.UpsampleConditioning.Projection.LINEAR,
          combination=sl.UpsampleConditioning.Combination.ADD,
          upsample_ratio=upsample_ratio,
      ),
      # Add noise to the GRU inputs in training to avoid overfitting on
      # the conditioning and previous outputs.
      sl.Noise(
          sl.TruncatedNormalSampler(mean=0.0, stddev=0.01), training_only=True
      ),
      # Recurrent transformation.
      sl.RNN(tf.keras.layers.GRUCell(num_units)),
      # Linear projection to the distribution parameters we are modeling.
      sl.Dense(num_distribution_parameters),
  ])


class WaveRNN(tf.Module):
  """A simplified WaveRNN model."""

  def __init__(
      self, num_units: int, upsample_ratio: int, name: Optional[str] = None
  ):
    super().__init__(name=name)
    self._upsample_ratio = upsample_ratio
    # Build the SequenceLayer body that predicts distribution parameters for
    # each waveform sample from conditioning features and previous timesteps.
    with self.name_scope:
      self._body = WaveRNNBody(
          num_units,
          # One location and scale parameter per timestep.
          num_distribution_parameters=2,
          conditioning_name='conditioning',
          upsample_ratio=upsample_ratio,
      )

  def _get_distribution(self, logits: tf.Tensor) -> tfd.Distribution:
    loc, scale_logits = tf.split(logits, 2, axis=-1)
    # Ensure scale is positive.
    scale = 1e-16 + tf.nn.softplus(scale_logits)
    # Model the waveform with a single quantized logistic distribution.
    # This is similar to the output distribution used in the PixelCNN++ paper.
    distribution = tfd.Logistic(loc, scale)
    # Quantize the [-1.0, 1.0] float samples into 2**14 + 1 (for symmetry) bins.
    # TODO(b/157322614): Predict 2**16 + 1 bins once numerically stable to match
    # the actual quantization of typical PCM audio we train WaveRNN on.
    distribution = tfd.QuantizedDistribution(distribution, low=-8192, high=8192)
    return tfd.TransformedDistribution(
        distribution=distribution, bijector=tfb.Scale(scale=1.0 / 8192)
    )

  def log_prob(
      self, waveform: sl.Sequence, conditioning: sl.Sequence, training: bool
  ) -> sl.Sequence:
    """Compute log P(waveform|conditioning) under this model.

    Computes the log probability of waveform under the distribution implied by
    conditioning. Maximizing this value trains the WaveRNN to predict this
    waveform.

    It is typical for waveform and conditioning to be a slice (e.g. 40 ms of
    audio) of a larger waveform sequence, since it's typically infeasible to
    train an RNN unrolled at common speech waveform sample rates.

    Args:
      waveform: The waveform to compute the log probability of, shaped [b, t_w,
        1] with float samples in the range [-1, 1].
      conditioning: The conditioning feature sequence, [b, t_c, ...]. The
        relation t_w = t_c * upsample_ratio must hold.
      training: Whether we are in training mode.

    Returns:
      A Sequence containing per-timestep log probabilities.
    """
    # To teacher force an autoregressive model we pretend that it did a perfect
    # job producing the ground truth on the previous step. This is normally done
    # by padding the ground truth with one sample and trimming one off the end.
    #
    # So for the sequence:
    # A B C D E F
    # the inputs to the layer look like:
    # 0 A B C D E
    # We then optimize the likelihood of the ground truth to train it to predict
    # the ground truth sequence.
    #
    # An easy way to do this is to compose the body with a Delay layer:
    teacher_forcing = sl.Serial([sl.Delay(self._body.block_size), self._body])
    logits = teacher_forcing.layer(
        waveform, training=training, constants={'conditioning': conditioning}
    )
    distribution = self._get_distribution(logits.values)
    return waveform.apply_values(distribution.log_prob).mask_invalid()

  @tf.function
  def sample(self, conditioning: sl.Sequence) -> sl.Sequence:
    """Sample from the distribution P(waveform|conditioning).

    Args:
      conditioning: The conditioning sequence [b, t_c, ...].

    Returns:
      A sampled waveform sequence shaped [b, t_w, 1], where
      t_w = t_c * upsample_ratio.
    """
    batch_size, conditioning_time = conditioning.values.shape.as_list()[:2]

    # Since the ratio between the conditioning features and the waveform
    # sequence is fixed, we know how many samples we are going to produce for
    # conditioning.
    time = conditioning_time * self._upsample_ratio

    # A TensorArray-like object to write the samples to at each timestep.
    sequence_array = sl.SequenceArray.new(tf.float32, size=time)

    # When sampling from the model the input to each timestep is the output from
    # the previous timestep. On the first iteration, we have no output to feed
    # in, so we just feed in a zero vector. This matches the zero padding
    # timestep we insert via the Delay layer in the above log_prob method.
    output = sl.Sequence(
        tf.zeros([batch_size, self._body.block_size, 1]),
        tf.ones([batch_size, self._body.block_size]),
    )

    # We provide the conditioning sequence to the SequenceLayer via a
    # "constants" dictionary. The key matches the name we used when creating the
    # WaveRNN body.
    constants = {'conditioning': conditioning}

    # Get the initial state for the layer. This will be updated for each
    # timestep of the sampling loop.
    state = self._body.get_initial_state(output, constants=constants)

    # For each timestep of the output waveform:
    # - compute the new state and distribution parameters from the previous
    #   state and output.
    # - draw a sample from the distribution.
    # - write the sample to the SequenceArray.
    for i in tf.range(time):
      logits, state = self._body.step(
          output, state, training=False, constants=constants
      )
      distribution = self._get_distribution(logits.values)
      output = sl.Sequence(distribution.sample(), logits.mask)
      sequence_array = sequence_array.write(i, output)

    # Concatenate all the timesteps together into a final Sequence, and mask out
    # the timesteps that were computed from padding.
    return sequence_array.concat().mask_invalid()
