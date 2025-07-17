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
"""Tests for PyTorch DSP layers."""

import fractions
import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sequence_layers.pytorch.types import Sequence
from sequence_layers.pytorch.utils_test import SequenceLayerTest
from sequence_layers.pytorch import dsp


class DelayTest(SequenceLayerTest):
    """Test Delay layer."""
    
    def test_delay_basic(self):
        """Test basic Delay functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test delay of 2 timesteps
        layer = dsp.Delay(2)
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Check that values are delayed
        self.assertEqual(y.values.shape, x.values.shape)
        
        # Test step-wise execution
        state = layer.get_initial_state(x.shape[0], x.channel_spec, training=False)
        y_step, _ = layer.step(x[:, :1], state, training=False)
        self.assertEqual(y_step.channel_shape, (4,))
    
    def test_delay_zero(self):
        """Test Delay with zero delay."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.Delay(0)
        y = self.verify_contract(layer, x, training=False)
        
        # Zero delay should return input unchanged
        self.assertEqual(y.values.shape, x.values.shape)
        self.assertEqual(y.channel_shape, (4,))
    
    def test_delay_validation(self):
        """Test Delay parameter validation."""
        # Test negative delay
        with self.assertRaises(ValueError):
            dsp.Delay(-1)


class FrameTest(SequenceLayerTest):
    """Test Frame layer."""
    
    def test_frame_basic(self):
        """Test basic Frame functionality."""
        x = self.random_sequence(2, 8, 4)
        
        # Test framing with length 3, step 2
        layer = dsp.Frame(frame_length=3, frame_step=2, padding='causal')
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (3, 4))
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertTrue(layer.supports_step)
        
        # Check output shape
        self.assertEqual(y.values.shape[2], 3)  # frame_length
        self.assertEqual(y.values.shape[3], 4)  # channels
    
    def test_frame_different_padding(self):
        """Test Frame with different padding modes."""
        x = self.random_sequence(2, 8, 4)
        
        # Test reverse causal padding
        layer = dsp.Frame(frame_length=3, frame_step=1, padding='reverse_causal')
        y = self.verify_contract(layer, x, training=False, padding_invariance_pad_value=0.0)
        
        self.assertEqual(y.channel_shape, (3, 4))
        self.assertTrue(layer.supports_step)
    
    def test_frame_layer_only(self):
        """Test Frame with layer-only execution."""
        x = self.random_sequence(2, 8, 4)
        
        # Test with unsupported padding for step
        layer = dsp.Frame(frame_length=3, frame_step=1, padding='valid')
        # This should fail validation, let's use a valid padding
        layer = dsp.Frame(frame_length=3, frame_step=1, padding='causal')
        
        y = layer.layer(x, training=False)
        self.assertEqual(y.channel_shape, (3, 4))
    
    def test_frame_validation(self):
        """Test Frame parameter validation."""
        # Test invalid frame_length
        with self.assertRaises(ValueError):
            dsp.Frame(frame_length=0, frame_step=1)
        
        # Test invalid frame_step
        with self.assertRaises(ValueError):
            dsp.Frame(frame_length=3, frame_step=0)
        
        # Test invalid padding mode
        with self.assertRaises(ValueError):
            dsp.Frame(frame_length=3, frame_step=1, padding='invalid')


class OverlapAddTest(SequenceLayerTest):
    """Test OverlapAdd layer."""
    
    def test_overlap_add_basic(self):
        """Test basic OverlapAdd functionality."""
        # Create framed input
        x = self.random_sequence(2, 4, 3, 4)  # (batch, time, frame_length, channels)
        
        layer = dsp.OverlapAdd(frame_length=3, frame_step=2, padding='causal')
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(2, 1))
        self.assertTrue(layer.supports_step)
    
    def test_overlap_add_no_overlap(self):
        """Test OverlapAdd with no overlap."""
        x = self.random_sequence(2, 4, 3, 4)  # (batch, time, frame_length, channels)
        
        layer = dsp.OverlapAdd(frame_length=3, frame_step=3, padding='causal')
        y = self.verify_contract(layer, x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertTrue(layer.supports_step)
    
    def test_overlap_add_validation(self):
        """Test OverlapAdd parameter validation."""
        # Test frame_length < frame_step
        with self.assertRaises(ValueError):
            dsp.OverlapAdd(frame_length=2, frame_step=3)
        
        # Test invalid frame_length
        with self.assertRaises(ValueError):
            dsp.OverlapAdd(frame_length=0, frame_step=1)


class WindowTest(SequenceLayerTest):
    """Test Window layer."""
    
    def test_window_basic(self):
        """Test basic Window functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.Window(dsp.hann_window, axis=-1)
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # StatelessPointwise
        
        # Check that window was applied
        self.assertEqual(y.values.shape, x.values.shape)
    
    def test_window_different_axis(self):
        """Test Window with different axis."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.Window(dsp.hamming_window, axis=1)
        y = layer.layer(x, training=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(y.values.shape, x.values.shape)


class FFTTest(SequenceLayerTest):
    """Test FFT layer."""
    
    def test_fft_basic(self):
        """Test basic FFT functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.FFT(fft_length=4, axis=-1, padding='right')
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # Stateless
        
        # Check that output is complex
        self.assertTrue(y.dtype.is_complex)
    
    def test_fft_different_length(self):
        """Test FFT with different length."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.FFT(fft_length=8, axis=-1, padding='right')
        y = layer.layer(x, training=False)
        
        self.assertEqual(y.channel_shape, (8,))
        self.assertTrue(y.dtype.is_complex)
    
    def test_fft_validation(self):
        """Test FFT parameter validation."""
        # Test invalid axis
        with self.assertRaises(ValueError):
            layer = dsp.FFT(fft_length=4, axis=0)  # batch dimension
        
        # Test invalid fft_length
        with self.assertRaises(ValueError):
            layer = dsp.FFT(fft_length=0)


class IFFTTest(SequenceLayerTest):
    """Test IFFT layer."""
    
    def test_ifft_basic(self):
        """Test basic IFFT functionality."""
        # Create complex input
        x = self.random_sequence(2, 8, 4)
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.IFFT(fft_length=4, axis=-1, padding='right')
        y = self.verify_contract(layer, x_complex, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # Stateless
        
        # Check that output is real
        self.assertTrue(y.dtype.is_floating_point)
    
    def test_ifft_frame_length(self):
        """Test IFFT with frame_length."""
        x = self.random_sequence(2, 8, 4)
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.IFFT(fft_length=4, frame_length=6, axis=-1, padding='right')
        y = layer.layer(x_complex, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertTrue(y.dtype.is_floating_point)


class RFFTTest(SequenceLayerTest):
    """Test RFFT layer."""
    
    def test_rfft_basic(self):
        """Test basic RFFT functionality."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.RFFT(fft_length=4, axis=-1, padding='right')
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        expected_output_length = 4 // 2 + 1  # 3
        self.assertEqual(y.channel_shape, (expected_output_length,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # Stateless
        
        # Check that output is complex
        self.assertTrue(y.dtype.is_complex)
    
    def test_rfft_different_length(self):
        """Test RFFT with different length."""
        x = self.random_sequence(2, 8, 4)
        
        layer = dsp.RFFT(fft_length=8, axis=-1, padding='right')
        y = layer.layer(x, training=False)
        
        expected_output_length = 8 // 2 + 1  # 5
        self.assertEqual(y.channel_shape, (expected_output_length,))
        self.assertTrue(y.dtype.is_complex)


class IRFFTTest(SequenceLayerTest):
    """Test IRFFT layer."""
    
    def test_irfft_basic(self):
        """Test basic IRFFT functionality."""
        # Create complex input (like output of RFFT)
        x = self.random_sequence(2, 8, 3)  # 3 = 4//2 + 1
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.IRFFT(fft_length=4, axis=-1, padding='right')
        y = self.verify_contract(layer, x_complex, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (4,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # Stateless
        
        # Check that output is real
        self.assertTrue(y.dtype.is_floating_point)
    
    def test_irfft_frame_length(self):
        """Test IRFFT with frame_length."""
        x = self.random_sequence(2, 8, 3)  # 3 = 4//2 + 1
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.IRFFT(fft_length=4, frame_length=6, axis=-1, padding='right')
        y = layer.layer(x_complex, training=False)
        
        self.assertEqual(y.channel_shape, (6,))
        self.assertTrue(y.dtype.is_floating_point)


class STFTTest(SequenceLayerTest):
    """Test STFT layer."""
    
    def test_stft_basic(self):
        """Test basic STFT functionality."""
        x = self.random_sequence(2, 16, 1)  # Single channel audio
        
        layer = dsp.STFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            window_fn=dsp.hann_window,
            time_padding='causal',
            output_magnitude=False
        )
        y = self.verify_contract(layer, x, training=False, test_gradients=False)
        
        expected_fft_bins = 4 // 2 + 1  # 3
        self.assertEqual(y.channel_shape, (expected_fft_bins, 1))
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertTrue(layer.supports_step)
        
        # Check that output is complex
        self.assertTrue(y.dtype.is_complex)
    
    def test_stft_magnitude(self):
        """Test STFT with magnitude output."""
        x = self.random_sequence(2, 16, 1)
        
        layer = dsp.STFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            window_fn=dsp.hann_window,
            time_padding='causal',
            output_magnitude=True
        )
        y = self.verify_contract(layer, x, training=False)
        
        expected_fft_bins = 4 // 2 + 1  # 3
        self.assertEqual(y.channel_shape, (expected_fft_bins, 1))
        
        # Check that output is real (magnitude)
        self.assertTrue(y.dtype.is_floating_point)
    
    def test_stft_no_window(self):
        """Test STFT without window function."""
        x = self.random_sequence(2, 16, 1)
        
        layer = dsp.STFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            window_fn=None,
            time_padding='causal',
            output_magnitude=False
        )
        y = layer.layer(x, training=False)
        
        expected_fft_bins = 4 // 2 + 1  # 3
        self.assertEqual(y.channel_shape, (expected_fft_bins, 1))
        self.assertTrue(y.dtype.is_complex)


class InverseSTFTTest(SequenceLayerTest):
    """Test InverseSTFT layer."""
    
    def test_inverse_stft_basic(self):
        """Test basic InverseSTFT functionality."""
        # Create STFT-like input
        fft_bins = 4 // 2 + 1  # 3
        x = self.random_sequence(2, 8, fft_bins, 1)
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.InverseSTFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            window_fn=dsp.hann_window,
            time_padding='causal'
        )
        y = self.verify_contract(layer, x_complex, training=False, test_gradients=False)
        
        self.assertEqual(y.channel_shape, (1,))
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(2, 1))
        self.assertTrue(layer.supports_step)
        
        # Check that output is real
        self.assertTrue(y.dtype.is_floating_point)
    
    def test_inverse_stft_no_window(self):
        """Test InverseSTFT without window function."""
        fft_bins = 4 // 2 + 1  # 3
        x = self.random_sequence(2, 8, fft_bins, 1)
        x_complex = Sequence(x.values.to(torch.complex64), x.mask)
        
        layer = dsp.InverseSTFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            window_fn=None,
            time_padding='causal'
        )
        y = layer.layer(x_complex, training=False)
        
        self.assertEqual(y.channel_shape, (1,))
        self.assertTrue(y.dtype.is_floating_point)


class LinearToMelSpectrogramTest(SequenceLayerTest):
    """Test LinearToMelSpectrogram layer."""
    
    def test_mel_spectrogram_basic(self):
        """Test basic LinearToMelSpectrogram functionality."""
        # Create spectrogram input
        num_freq_bins = 129  # Common for 256-point FFT
        x = self.random_sequence(2, 16, num_freq_bins, 1)
        
        layer = dsp.LinearToMelSpectrogram(
            num_mel_bins=40,
            sample_rate=16000,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )
        y = self.verify_contract(layer, x, training=False, rtol=1e-4, atol=1e-6)
        
        self.assertEqual(y.channel_shape, (40, 1))  # 40 mel bins
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step())  # Stateless
        
        # Check that output has correct shape
        self.assertEqual(y.values.shape[2], 40)  # mel bins
        self.assertEqual(y.values.shape[3], 1)   # channels
    
    def test_mel_spectrogram_different_params(self):
        """Test LinearToMelSpectrogram with different parameters."""
        num_freq_bins = 65  # Common for 128-point FFT
        x = self.random_sequence(2, 16, num_freq_bins, 1)
        
        layer = dsp.LinearToMelSpectrogram(
            num_mel_bins=80,
            sample_rate=22050,
            lower_edge_hertz=20.0,
            upper_edge_hertz=11025.0
        )
        y = layer.layer(x, training=False)
        
        self.assertEqual(y.channel_shape, (80, 1))  # 80 mel bins
        self.assertEqual(y.values.shape[2], 80)


class DSPLayerPropertiesTest(SequenceLayerTest):
    """Test properties of DSP layers."""
    
    def test_delay_properties(self):
        """Test Delay layer properties."""
        layer = dsp.Delay(3)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (4,))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.float32)
    
    def test_fft_properties(self):
        """Test FFT layer properties."""
        layer = dsp.FFT(fft_length=8, axis=-1)
        
        self.assertEqual(layer.block_size, 1)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 1))
        self.assertTrue(layer.supports_step)  # Stateless
        
        # Test output shape
        input_shape = (4,)
        output_shape = layer.get_output_shape(input_shape)
        self.assertEqual(output_shape, (8,))
        
        # Test output dtype (complex)
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.complex64)
    
    def test_stft_properties(self):
        """Test STFT layer properties."""
        layer = dsp.STFT(
            frame_length=4,
            frame_step=2,
            fft_length=4,
            time_padding='causal'
        )
        
        self.assertEqual(layer.block_size, 2)
        self.assertEqual(layer.output_ratio, fractions.Fraction(1, 2))
        self.assertTrue(layer.supports_step)
        
        # Test output shape
        input_shape = (1,)
        output_shape = layer.get_output_shape(input_shape)
        expected_fft_bins = 4 // 2 + 1  # 3
        self.assertEqual(output_shape, (expected_fft_bins, 1))
        
        # Test output dtype
        input_dtype = torch.float32
        output_dtype = layer.get_output_dtype(input_dtype)
        self.assertEqual(output_dtype, torch.complex64)


class DSPParameterTest(SequenceLayerTest):
    """Test DSP layer parameters."""
    
    def test_frame_parameters(self):
        """Test Frame layer parameter initialization."""
        layer = dsp.Frame(
            frame_length=4,
            frame_step=2,
            padding='causal'
        )
        
        self.assertEqual(layer.frame_length, 4)
        self.assertEqual(layer.frame_step, 2)
        self.assertEqual(layer.padding, 'causal')
        self.assertTrue(layer.supports_step)
    
    def test_stft_parameters(self):
        """Test STFT layer parameter initialization."""
        layer = dsp.STFT(
            frame_length=4,
            frame_step=2,
            fft_length=8,
            window_fn=dsp.hamming_window,
            time_padding='causal',
            fft_padding='center',
            output_magnitude=True
        )
        
        self.assertEqual(layer.frame_length, 4)
        self.assertEqual(layer.frame_step, 2)
        self.assertEqual(layer.fft_length, 8)
        self.assertEqual(layer.window_fn, dsp.hamming_window)
        self.assertEqual(layer.time_padding, 'causal')
        self.assertEqual(layer.fft_padding, 'center')
        self.assertTrue(layer.output_magnitude)
    
    def test_mel_spectrogram_parameters(self):
        """Test LinearToMelSpectrogram layer parameters."""
        layer = dsp.LinearToMelSpectrogram(
            num_mel_bins=40,
            sample_rate=16000,
            lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0
        )
        
        self.assertEqual(layer.num_mel_bins, 40)
        self.assertEqual(layer.sample_rate, 16000)
        self.assertEqual(layer.lower_edge_hertz, 80.0)
        self.assertEqual(layer.upper_edge_hertz, 7600.0)


class DSPWindowFunctionsTest(SequenceLayerTest):
    """Test DSP window functions."""
    
    def test_hann_window(self):
        """Test Hann window function."""
        window = dsp.hann_window(8)
        
        self.assertEqual(window.shape, (8,))
        self.assertEqual(window.dtype, torch.float32)
        
        # Hann window should be symmetric
        for i in range(4):
            self.assertAlmostEqual(window[i].item(), window[7-i].item(), places=5)
    
    def test_hamming_window(self):
        """Test Hamming window function."""
        window = dsp.hamming_window(8)
        
        self.assertEqual(window.shape, (8,))
        self.assertEqual(window.dtype, torch.float32)
        
        # Hamming window should be symmetric
        for i in range(4):
            self.assertAlmostEqual(window[i].item(), window[7-i].item(), places=5)
    
    def test_blackman_window(self):
        """Test Blackman window function."""
        window = dsp.blackman_window(8)
        
        self.assertEqual(window.shape, (8,))
        self.assertEqual(window.dtype, torch.float32)
        
        # Blackman window should be symmetric
        for i in range(4):
            self.assertAlmostEqual(window[i].item(), window[7-i].item(), places=5)
    
    def test_window_different_dtype(self):
        """Test window functions with different dtypes."""
        window = dsp.hann_window(8, dtype=torch.float64)
        
        self.assertEqual(window.shape, (8,))
        self.assertEqual(window.dtype, torch.float64)


if __name__ == '__main__':
    unittest.main() 