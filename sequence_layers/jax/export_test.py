# Copyright 2025 Google LLC
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
import tempfile

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import sequence_layers.jax as sl
from sequence_layers.jax import export
from sequence_layers.jax import test_utils
import tensorflow as tf


class ExportTest(test_utils.SequenceLayerTest):

  def _check_tflite_step(
      self,
      l: sl.SequenceLayer,
      x: sl.Sequence,
      constants: sl.Constants | None = None,
      allow_custom_ops: bool = False,
      use_flex: bool = False,
      rtol: float = 1e-7,
      atol: float = 1e-7,
      check_state: bool = True,
  ):
    """Builds a tf.lite model for a step and verifies identical outputs."""

    assert l.supports_step, l

    if constants is None:
      constants = {}

    def get_initial_state_fn(params, constants: sl.Constants):
      bound = l.clone().bind(params)
      state = bound.get_initial_state(
          x.shape[0], x.channel_spec, training=False, constants=constants
      )
      return {'state': state}

    def step_fn(
        params, x: sl.Sequence, state: sl.State, constants: sl.Constants
    ):
      bound = l.clone().bind(params)
      y, state = bound.step(x, state, training=False, constants=constants)
      return {'y': y, 'state': state}

    s0 = l.get_initial_state(
        x.shape[0], x.channel_spec, training=False, constants=constants
    )
    y_step, s1 = l.step(x, s0, training=False, constants=constants)

    get_initial_state_inputs = {'constants': constants}
    get_initial_state_outputs_spec = jax.eval_shape(
        get_initial_state_fn, l.variables, **get_initial_state_inputs
    )
    step_inputs = {'x': x, 'state': s0, 'constants': constants}
    step_outputs_spec = jax.eval_shape(step_fn, l.variables, **step_inputs)

    export_dir = tempfile.mkdtemp()
    export.export_to_tf_saved_model(
        l.variables,
        {
            'get_initial_state': export.Signature(
                get_initial_state_fn, get_initial_state_inputs
            ),
            'step': export.Signature(step_fn, step_inputs),
        },
        export_dir,
    )
    tflite_model = export.tflite_convert(
        export_dir,
        allow_custom_ops=allow_custom_ops,
        use_flex=use_flex,
    )

    the_interpreter = tf.lite.Interpreter(model_content=tflite_model)
    the_interpreter.allocate_tensors()

    # If the layer has no state, get_initial_state will not be exported. Do not
    # attempt to invoke it.
    if jax.tree.leaves(s0):
      get_initial_state_inputs_dict = export._tree_to_flat_dict(
          get_initial_state_inputs
      )
      get_initial_state_outputs_tflite = the_interpreter.get_signature_runner(
          'get_initial_state'
      )(**get_initial_state_inputs_dict)
      get_initial_state_outputs_tflite = export._result_dict_to_tree(
          get_initial_state_outputs_tflite, get_initial_state_outputs_spec
      )
      s0_tflite = get_initial_state_outputs_tflite['state']
      if check_state:
        chex.assert_trees_all_close(s0_tflite, s0, rtol=rtol, atol=atol)

    step_inputs_dict = export._tree_to_flat_dict(step_inputs)
    step_outputs_tflite = the_interpreter.get_signature_runner('step')(
        **step_inputs_dict
    )
    step_outputs_tflite = export._result_dict_to_tree(
        step_outputs_tflite, step_outputs_spec
    )

    y_step_tflite = step_outputs_tflite['y']
    s1_tflite = step_outputs_tflite['state']
    self.assertSequencesClose(y_step_tflite, y_step, rtol=rtol, atol=atol)
    chex.assert_trees_all_close(s1_tflite, s1, rtol=rtol, atol=atol)

  @parameterized.parameters(
      [sl.Abs.Config()],
      [sl.Add.Config(5)],
      [sl.AddTimingSignal.Config()],
      [sl.Affine.Config()],
      [sl.ApplyRotaryPositionalEncoding.Config(max_wavelength=10000.0)],
      [sl.ApplySharding.Config((None, 'data', 'model'))],
      [sl.Argmax.Config()],
      [sl.AveragePooling1D.Config(pool_size=3, strides=2, padding='causal')],
      [
          sl.AveragePooling2D.Config(
              pool_size=3,
              strides=2,
              time_padding='causal',
              spatial_padding='same',
          ),
          sl.ShapeDType((1, 16, 7, 11), jnp.float32),
      ],
      [sl.BatchNormalization.Config()],
      [sl.Cast.Config(jnp.int32)],
      [sl.CheckpointName.Config('test')],
      [sl.Conv1D.Config(8, kernel_size=3, strides=2, padding='causal')],
      [
          sl.DepthwiseConv1D.Config(
              kernel_size=3, strides=2, depth_multiplier=2, padding='causal'
          )
      ],
      [sl.Downsample1D.Config(rate=4)],
      [sl.Dropout.Config(rate=0.5)],
      [
          sl.Conv1DTranspose.Config(
              8, kernel_size=3, strides=2, padding='causal'
          )
      ],
      [
          sl.Conv2DTranspose.Config(
              8,
              kernel_size=[3, 2],
              strides=2,
              time_padding='causal',
              spatial_padding='same',
          ),
          sl.ShapeDType((1, 16, 7, 11), jnp.float32),
      ],
      [
          sl.Conv2D.Config(
              8,
              kernel_size=[3, 2],
              strides=2,
              time_padding='causal',
              spatial_padding='same',
          ),
          sl.ShapeDType((1, 16, 7, 11), jnp.float32),
      ],
      [
          sl.Conv3D.Config(
              8,
              kernel_size=[3, 2, 5],
              strides=2,
              time_padding='causal',
              spatial_padding=('same', 'same'),
          ),
          sl.ShapeDType((1, 16, 7, 11, 13), jnp.float32),
      ],
      [sl.Delay.Config(3)],
      [sl.Dense.Config(8)],
      [sl.DenseShaped.Config([8, 3])],
      [
          sl.DotProductAttention.Config(
              'source', num_heads=8, units_per_head=3
          ),
          sl.ShapeDType((1, 16, 8), jnp.float32),
          {'source': sl.ShapeDType((1, 8, 7), jnp.float32)},
      ],
      [
          sl.DotProductSelfAttention.Config(
              num_heads=8,
              units_per_head=3,
              max_past_horizon=7,
              max_future_horizon=0,
          ),
      ],
      [sl.EinopsRearrange.Config('(c d) -> c d', {'c': 2})],
      [sl.EinsumDense.Config('...a,abc->...bc', [8, 3], bias_axes='c')],
      [sl.Elu.Config()],
      [sl.Embedding.Config(8, 16), sl.ShapeDType((1, 16), jnp.int32)],
      # Requires complicated setup.
      # [
      #     sl.EmbeddingTranspose.Config(8, 16),
      #     sl.ShapeDType((1, 16), jnp.int32)
      # ],
      [sl.Emit.Config('test')],
      [sl.Exp.Config()],
      [sl.ExpandDims.Config(-1)],
      # No complex support.
      # [sl.FFT.Config(8), sl.ShapeDType((1, 5, 8), jnp.complex64)],
      [sl.Flatten.Config()],
      [sl.Frame.Config(frame_length=5, frame_step=2, padding='causal')],
      [sl.GatedLinearUnit.Config()],
      [sl.GatedTanhUnit.Config()],
      [sl.GatedUnit.Config(jax.nn.relu, jax.nn.gelu)],
      [sl.Gelu.Config()],
      [
          sl.GmmAttention.Config(
              'source',
              num_heads=8,
              units_per_head=3,
              num_components=5,
              monotonic=True,
          ),
          sl.ShapeDType((1, 16, 8), jnp.float32),
          {'source': sl.ShapeDType((1, 8, 7), jnp.float32)},
      ],
      # GlobalEinopsRearrange not steppable.
      # GlobalReshape not steppable.
      [sl.GradientClipping.Config(1.0)],
      [sl.GroupNormalization.Config(num_groups=2, cumulative=True)],
      # IFFT and IRFFT are not supported.
      # [sl.IFFT.Config(8), sl.ShapeDType((1, 5, 8), jnp.complex64)],
      # [sl.IRFFT.Config(8), sl.ShapeDType((1, 5, 8), jnp.complex64)],
      [sl.Identity.Config()],
      # IRFFT is not supported.
      # [
      #     sl.InverseSTFT.Config(
      #         frame_length=8, frame_step=2, fft_length=8,
      #         time_padding='causal'
      #     ),
      #     sl.ShapeDType((1, 16, 8), jnp.complex64),
      # ],
      [sl.Lambda.Config(lambda x: x + 5.0)],
      [sl.LayerNormalization.Config()],
      [sl.LeakyRelu.Config()],
      [
          sl.LinearToMelSpectrogram.Config(
              num_mel_bins=8,
              sample_rate=240000.0,
              lower_edge_hertz=0.0,
              upper_edge_hertz=12000.0,
          )
      ],
      [
          sl.LocalDotProductSelfAttention.Config(
              num_heads=8,
              units_per_head=3,
              block_size=7,
              max_past_horizon=7,
              max_future_horizon=0,
          ),
      ],
      [sl.Log.Config()],
      [sl.Logging.Config('test')],
      [sl.Lookahead.Config(3)],
      [sl.LSTM.Config(8)],
      [sl.MaskedDense.Config(8, 16)],
      [sl.MaskInvalid.Config()],
      [sl.Maximum.Config(1.0)],
      {
          'config': sl.MaxPooling1D.Config(
              pool_size=3, strides=2, padding='causal'
          ),
          # State contains Inf which tf.lite converts to a large float.
          'check_state': False,
      },
      # MaxPooling2D seems buggy.
      # [
      #     sl.MaxPooling2D.Config(
      #         pool_size=2,
      #         strides=1,
      #         time_padding='causal',
      #         spatial_padding='same',
      #     ),
      #     sl.ShapeDType((1, 16, 7, 11), jnp.float32),
      # ],
      {
          'config': sl.MinPooling1D.Config(
              pool_size=3, strides=2, padding='causal'
          ),
          # State contains Inf which tf.lite converts to a large float.
          'check_state': False,
      },
      # MinPooling2D seems buggy.
      # [
      #     sl.MinPooling2D.Config(
      #         pool_size=2,
      #         strides=1,
      #         time_padding='causal',
      #         spatial_padding='same',
      #     ),
      #     sl.ShapeDType((1, 16, 7, 11), jnp.float32),
      # ],
      [sl.Minimum.Config(-1.0)],
      [sl.Minimum.Config(-1.0)],
      [sl.Mod.Config(5.0)],
      [sl.MoveAxis.Config(2, 3), sl.ShapeDType((1, 16, 7, 11), jnp.float32)],
      [sl.OneHot.Config(10), sl.ShapeDType((1, 16, 10), jnp.int32)],
      [sl.OptimizationBarrier.Config()],
      [sl.OverlapAdd.Config(frame_length=8, frame_step=2, padding='causal')],
      [sl.PRelu.Config()],
      [sl.Power.Config(2)],
      [sl.Relu.Config()],
      [sl.Reshape.Config((2, 1, 4))],
      [sl.RMSNormalization.Config()],
      [sl.RGLRU.Config(units=8, num_heads=2)],
      [sl.RFFT.Config(8)],
      # sl.STFT depends on an unsupported complex64 operation.
      # [
      #     sl.STFT.Config(
      #         frame_length=8,
      #         frame_step=2,
      #         fft_length=8,
      #         time_padding='causal',
      #         output_magnitude=True,
      #     )
      # ],
      [sl.Scale.Config(5.0)],
      [sl.SequenceDense.Config(8, 16)],
      [
          sl.SequenceEmbedding.Config(8, 10, 16),
          sl.ShapeDType((1, 16), jnp.int32),
      ],
      [sl.Sigmoid.Config()],
      [sl.Slice.Config(slices=(4,))],
      [sl.Softmax.Config()],
      [sl.Softplus.Config()],
      [sl.Squeeze.Config(2), sl.ShapeDType((1, 16, 1, 11), jnp.float32)],
      [
          sl.StreamingDotProductAttention.Config(
              'source',
              num_heads=8,
              units_per_head=3,
              max_past_horizon=7,
              max_future_horizon=0,
          ),
          sl.ShapeDType((1, 16, 8), jnp.float32),
          {'source': sl.ShapeDType((1, 16, 7), jnp.float32)},
      ],
      [
          sl.StreamingLocalDotProductAttention.Config(
              'source',
              num_heads=8,
              units_per_head=3,
              block_size=7,
              max_past_horizon=7,
              max_future_horizon=0,
          ),
          sl.ShapeDType((1, 16, 8), jnp.float32),
          {'source': sl.ShapeDType((1, 16, 7), jnp.float32)},
      ],
      [sl.SwapAxes.Config(2, 3), sl.ShapeDType((1, 16, 3, 5), jnp.float32)],
      [sl.Swish.Config()],
      [sl.Tanh.Config()],
      [sl.Transpose.Config((2, 3)), sl.ShapeDType((1, 16, 2, 3), jnp.float32)],
      [sl.Upsample1D.Config(rate=2)],
      [
          sl.Upsample2D.Config(rate=[2, 2]),
          sl.ShapeDType((1, 16, 2, 2), jnp.float32),
      ],
      [sl.Window.Config(axis=-1)],
      # Vanilla transformer:
      [
          sl.Serial.Config(
              [
                  sl.Residual.Config(
                      [
                          sl.DotProductSelfAttention.Config(
                              num_heads=8,
                              units_per_head=8,
                              max_past_horizon=7,
                              max_future_horizon=0,
                              name='attention',
                          ),
                          sl.DenseShaped.Config([8], name='output_projection'),
                      ],
                      name='self_attention',
                  ),
                  sl.Residual.Config(
                      [
                          sl.DotProductAttention.Config(
                              source_name='source',
                              num_heads=8,
                              units_per_head=8,
                              name='attention',
                          ),
                          sl.DenseShaped.Config([8], name='output_projection'),
                      ],
                      name='cross_attention',
                  ),
                  sl.Residual.Config(
                      [
                          sl.Dense.Config(
                              32, activation=jax.nn.gelu, name='dense1'
                          ),
                          sl.Dense.Config(
                              8, activation=jax.nn.gelu, name='dense2'
                          ),
                      ],
                      name='ffn',
                  ),
              ],
              name='model',
          ),
          sl.ShapeDType((1, 16, 8), jnp.float32),
          {'source': sl.ShapeDType((1, 8, 7), jnp.float32)},
      ],
  )
  def test_config(
      self,
      config,
      input_spec=None,
      constants_spec=None,
      rtol=1e-6,
      atol=1e-6,
      check_state: bool = True,
  ):
    if input_spec is None:
      input_spec = sl.ShapeDType((1, 16, 8), jnp.float32)

    x = jax.tree.map(
        lambda s: test_utils.random_sequence(
            *s.shape, dtype=s.dtype, low_length=s.shape[1] // 2
        ),
        input_spec,
    )
    constants = jax.tree.map(
        lambda s: test_utils.random_sequence(
            *s.shape, dtype=s.dtype, low_length=s.shape[1] // 2
        ),
        constants_spec or {},
    )
    l = config.make()
    params = l.init(jax.random.key(0), x, training=False, constants=constants)
    l = l.bind(params)
    self._check_tflite_step(
        l, x, constants=constants, rtol=rtol, atol=atol, check_state=check_state
    )


if __name__ == '__main__':
  test_utils.main()
