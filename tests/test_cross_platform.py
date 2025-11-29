"""
Cross-platform compatibility tests for EasyWakeWord.

These tests verify that the module behaves correctly across different
platforms and environments:
- Windows development environments
- GitHub Copilot/Actions CI environments  
- Linux/macOS systems

All tests in this file are designed to run without real audio hardware.
"""

import os
import platform
import sys

import numpy as np
import pytest
import soundfile as sf

from tests.test_helpers import (
    WakeWord,
    WordMatcher,
    AudioDeviceManager,
    create_minimal_wakeword_instance,
    PORTAUDIO_AVAILABLE
)


# Skip all tests if imports failed
pytestmark = pytest.mark.skipif(
    WordMatcher is None,
    reason="WakeWord classes not available (import failed)"
)


class TestEnvironmentDetection:
    """Tests for environment detection functionality."""
    
    def test_portaudio_detection(self, portaudio_available):
        """Test that PortAudio availability is correctly detected."""
        # This should be a boolean
        assert isinstance(portaudio_available, bool)
        
    def test_ci_environment_detection(self, in_ci_environment):
        """Test that CI environment is correctly detected."""
        assert isinstance(in_ci_environment, bool)
        
        # If CI env vars are set, should detect as CI
        if os.environ.get('CI') == 'true':
            assert in_ci_environment
            
    def test_platform_detection(self, is_windows, is_linux, is_macos):
        """Test that platform is correctly detected."""
        # Exactly one should be true (or none if unusual platform)
        platforms = [is_windows, is_linux, is_macos]
        assert sum(platforms) <= 1
        
        # Current platform should match
        current = platform.system()
        if current == 'Windows':
            assert is_windows
        elif current == 'Linux':
            assert is_linux
        elif current == 'Darwin':
            assert is_macos


class TestCrossPlatformMFCC:
    """Test MFCC functionality works across all platforms."""
    
    def test_mfcc_extraction_consistent(self, tmp_path):
        """Test that MFCC extraction produces consistent results."""
        # Create test audio
        wavfile = tmp_path / "test.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(wavfile), audio, 16000)
        
        # Create matcher and extract MFCCs
        matcher = WordMatcher(sample_rate=16000)
        mfcc_mean, mfcc_std = matcher.extract_mfcc(audio)
        
        # Should have 20 MFCC coefficients
        assert len(mfcc_mean) == 20
        assert len(mfcc_std) == 20
        
        # Values should be reasonable (not NaN or infinite)
        assert not np.any(np.isnan(mfcc_mean))
        assert not np.any(np.isnan(mfcc_std))
        assert not np.any(np.isinf(mfcc_mean))
        assert not np.any(np.isinf(mfcc_std))
        
    def test_similarity_calculation_deterministic(self, tmp_path):
        """Test that similarity calculation is deterministic."""
        wavfile = tmp_path / "test.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(wavfile), audio, 16000)
        
        matcher = WordMatcher(sample_rate=16000)
        matcher.set_reference(audio, "test")
        
        # Multiple calls should return same result
        sim1 = matcher.calculate_similarity(audio)
        sim2 = matcher.calculate_similarity(audio)
        
        assert sim1 == sim2
        assert sim1 == 100.0  # Self-similarity should be 100%


class TestCrossPlatformFileHandling:
    """Test file operations work across platforms."""
    
    def test_wav_file_creation_and_loading(self, tmp_path):
        """Test WAV file creation and loading works cross-platform."""
        wavfile = tmp_path / "cross_platform_test.wav"
        
        # Create audio data
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Write file
        sf.write(str(wavfile), audio, sample_rate)
        assert wavfile.exists()
        
        # Read file back
        loaded_audio, loaded_sr = sf.read(str(wavfile))
        assert loaded_sr == sample_rate
        assert len(loaded_audio) == len(audio)
        
        # WAV file round-trip precision: default soundfile WAV format uses 16-bit PCM
        # which quantizes to ~1/32768 ≈ 3e-5 precision. Using decimal=4 (~1e-4)
        # provides safe margin for this encoding precision loss.
        np.testing.assert_array_almost_equal(
            audio, loaded_audio.astype(np.float32), decimal=4
        )
        
    def test_path_handling(self, tmp_path):
        """Test that path handling works correctly on all platforms."""
        # Test with pathlib Path
        wavfile = tmp_path / "path_test.wav"
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(wavfile), audio, 16000)
        
        # Should work with string path
        matcher = WordMatcher(sample_rate=16000)
        matcher.load_reference_from_file(str(wavfile), "test")
        assert matcher.reference_word == "test"


class TestCrossPlatformWakeWordCreation:
    """Test WakeWord instance creation across platforms."""
    
    def test_minimal_instance_creation(self, tmp_path):
        """Test minimal WakeWord creation without audio hardware."""
        wavfile = tmp_path / "test.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(wavfile), audio, 16000)
        
        ww = create_minimal_wakeword_instance(
            wavfile=str(wavfile),
            textword="test word",
            numberofwords=2,
            timeout=5
        )
        
        assert ww.textword == "test word"
        assert ww.numberofwords == 2
        assert ww.timeout == 5
        assert ww.wavword == str(wavfile)
        
    def test_syllable_estimation_cross_platform(self, tmp_path):
        """Test syllable estimation works consistently."""
        wavfile = tmp_path / "test.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(wavfile), audio, 16000)
        
        ww = create_minimal_wakeword_instance(wavfile, textword="hello world")
        
        # Test various words
        test_cases = [
            ("hello", 2),  # hel-lo
            ("world", 1),  # world
            ("computer", 3),  # com-pu-ter
            ("hey", 1),  # hey
        ]
        
        for word, min_syllables in test_cases:
            syllables = ww._estimate_syllables(word)
            assert syllables >= min_syllables, f"{word} should have at least {min_syllables} syllables"


class TestAudioDeviceManagerMocked:
    """Test AudioDeviceManager with mocked devices."""
    
    def test_system_capture_device_detection(self):
        """Test detection of system audio capture devices."""
        # These should be detected as system capture devices
        capture_devices = [
            "Stereo Mix",
            "What U Hear",
            "System Audio Capture",
            "Loopback Device",
        ]
        
        for device in capture_devices:
            assert AudioDeviceManager._is_system_audio_capture_device(device), \
                f"'{device}' should be detected as system capture device"
                
    def test_real_microphone_detection(self):
        """Test that real microphones are not flagged as system capture."""
        # These should NOT be detected as system capture devices
        real_devices = [
            "USB Microphone",
            "Built-in Microphone",
            "Realtek HD Audio Input",
            "Blue Yeti",
        ]
        
        for device in real_devices:
            assert not AudioDeviceManager._is_system_audio_capture_device(device), \
                f"'{device}' should NOT be detected as system capture device"
