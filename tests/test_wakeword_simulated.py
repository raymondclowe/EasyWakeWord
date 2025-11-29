"""
Simulated audio tests for EasyWakeWord.

These tests mock audio hardware to allow testing in CI environments
without real microphones. They work in both Windows dev environments
and GitHub Copilot CI environments.
"""

import threading

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

# Skip all tests in this module if classes couldn't be imported
pytestmark = pytest.mark.skipif(
    WordMatcher is None,
    reason="WakeWord classes not available (import failed)"
)


def generate_wav(filename, duration=1.0, freq=440, sr=16000):
    """Generate a test WAV file with a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(filename, audio, sr)
    return filename


def generate_speech_like_audio(filename, duration=1.0, sr=16000):
    """Generate audio that more closely resembles speech patterns."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Combine multiple frequencies to simulate formants
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +  # Fundamental
        0.2 * np.sin(2 * np.pi * 500 * t) +  # First formant
        0.15 * np.sin(2 * np.pi * 1500 * t) +  # Second formant
        0.1 * np.sin(2 * np.pi * 2500 * t)  # Third formant
    )
    # Add amplitude envelope to simulate word boundaries
    envelope = np.sin(np.pi * t / duration) ** 0.5
    audio = (audio * envelope).astype(np.float32)
    sf.write(filename, audio, sr)
    return filename


def create_minimal_wakeword(wavfile, textword="hello", numberofwords=1, timeout=1):
    """
    Create a minimal WakeWord instance without initializing audio hardware.
    
    This is an alias to the shared helper function for backward compatibility.
    """
    return create_minimal_wakeword_instance(wavfile, textword, numberofwords, timeout)


class TestWordMatcher:
    """Tests for the WordMatcher MFCC comparison class."""
    
    def test_self_match(self, tmp_path):
        """A reference should match itself perfectly."""
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "test")
        
        matches, similarity = matcher.matches(audio.astype(np.float32))
        assert matches
        assert similarity == 100.0
    
    def test_same_frequency_match(self, tmp_path):
        """Same frequency audio should match."""
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "test")
        
        # Generate identical audio
        wavfile2 = tmp_path / "test2.wav"
        generate_wav(str(wavfile2), freq=440)
        audio2, _ = sf.read(str(wavfile2))
        
        matches, similarity = matcher.matches(audio2.astype(np.float32))
        assert matches
        assert similarity == 100.0
    
    def test_different_frequency_lower_similarity(self, tmp_path):
        """Different frequency should have lower similarity."""
        wavfile = tmp_path / "ref.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "ref")
        
        # Generate audio at a very different frequency
        wavfile2 = tmp_path / "test.wav"
        generate_wav(str(wavfile2), freq=880)  # One octave higher
        audio2, _ = sf.read(str(wavfile2))
        
        matches, similarity = matcher.matches(audio2.astype(np.float32))
        # Should have lower similarity than self-match
        assert similarity < 100.0
    
    def test_noise_should_not_match_well(self, tmp_path):
        """Random noise should have lower similarity to tonal audio."""
        wavfile = tmp_path / "ref.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "ref")
        
        # Generate random noise
        np.random.seed(42)  # For reproducibility
        noise = np.random.randn(len(audio)).astype(np.float32) * 0.1
        
        matches, similarity = matcher.matches(noise)
        # Noise should still have some similarity due to MFCC properties
        # but it shouldn't be 100%
        assert similarity < 100.0
    
    def test_no_reference_raises_error(self):
        """Calculating similarity without reference should raise error."""
        matcher = WordMatcher(sample_rate=16000)
        audio = np.zeros(16000, dtype=np.float32)
        
        with pytest.raises(ValueError, match="No reference word set"):
            matcher.calculate_similarity(audio)
    
    def test_load_reference_from_file(self, tmp_path):
        """Test loading reference from file."""
        wavfile = tmp_path / "ref.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        matcher.load_reference_from_file(str(wavfile), "test_word")
        
        assert matcher.reference_word == "test_word"
        assert matcher.reference_mfcc_mean is not None
        assert matcher.reference_mfcc_std is not None
    
    def test_speech_like_audio_matching(self, tmp_path):
        """Test matching with speech-like audio patterns."""
        wavfile = tmp_path / "speech.wav"
        generate_speech_like_audio(str(wavfile))
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "speech")
        
        matches, similarity = matcher.matches(audio.astype(np.float32))
        assert matches
        assert similarity == 100.0


class TestSyllableEstimation:
    """Tests for the syllable estimation functionality."""
    
    def test_single_word(self, tmp_path):
        """Test syllable count for single words."""
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile))
        
        ww = create_minimal_wakeword(wavfile, textword="hello")
        
        # Test syllable estimation
        assert ww._estimate_syllables("hello") == 2  # hel-lo
        assert ww._estimate_syllables("hey") == 1
        assert ww._estimate_syllables("computer") >= 2  # com-pu-ter
    
    def test_multi_word_phrases(self, tmp_path):
        """Test syllable count for multi-word phrases."""
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile))
        
        ww = create_minimal_wakeword(wavfile, textword="ok google")
        
        # "ok google" should have 3-4 syllables (o-k goo-gle)
        syllables = ww._estimate_syllables("ok google")
        assert syllables >= 2


class TestAudioDurationAnalysis:
    """Tests for the reference audio duration analysis."""
    
    def test_analyze_reference_audio_duration(self, tmp_path):
        """Test that audio duration is correctly analyzed."""
        # Create a 1-second audio file
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = create_minimal_wakeword(wavfile)
        
        duration = ww._analyze_reference_audio_duration()
        # Duration should be detected (might be less than 1.0 due to envelope)
        assert duration is not None
        assert 0.2 <= duration <= 1.5


class TestThresholdCalculation:
    """Tests for detection threshold calculation."""
    
    def test_threshold_from_audio_duration(self, tmp_path):
        """Test that thresholds are correctly calculated from audio duration."""
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = create_minimal_wakeword(wavfile)
        ww.post_speech_silence = None
        
        # Calculate thresholds from 0.8s audio duration
        ww._set_thresholds_from_audio_duration(0.8)
        
        assert ww.pre_speech_silence is not None
        assert ww.speech_duration_min is not None
        assert ww.speech_duration_max is not None
        assert ww.post_speech_silence is not None
        
        # Check reasonable bounds
        assert ww.pre_speech_silence >= 0.5
        assert ww.speech_duration_min >= 0.3
        assert ww.speech_duration_max <= 3.0
        assert ww.post_speech_silence >= 0.3
    
    def test_manual_thresholds_preserved(self, tmp_path):
        """Test that manually set thresholds are preserved."""
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = create_minimal_wakeword(wavfile)
        ww.pre_speech_silence = 1.5
        ww.speech_duration_min = 0.5
        ww.speech_duration_max = 2.0
        ww.post_speech_silence = 0.8
        
        # Calculate thresholds (should not override manual values)
        ww._calculate_detection_thresholds()
        
        # Manual values should be preserved
        assert ww.pre_speech_silence == 1.5
        assert ww.speech_duration_min == 0.5
        assert ww.speech_duration_max == 2.0
        assert ww.post_speech_silence == 0.8


class TestDetectionPipeline:
    """Tests for the detection pipeline with simulated audio buffer."""
    
    def test_mock_detection_timeout(self, tmp_path):
        """Test that detection times out correctly with silent buffer."""
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        # Create WakeWord without initializing real audio
        ww = create_minimal_wakeword(wavfile)
        ww.pre_speech_silence = 0.5
        ww.speech_duration_min = 0.3
        ww.speech_duration_max = 1.5
        ww.post_speech_silence = 0.3
        
        # Create a mock sound buffer that's always silent
        class MockSilentBuffer:
            def is_buffer_full(self): return True
            def is_silent(self): return True
            def stop(self): pass
            def return_last_n_seconds(self, n): return np.zeros(int(n * 16000), dtype=np.float32)
        
        ww._sound_buffer = MockSilentBuffer()
        
        # Initialize matcher
        ww._matcher = WordMatcher(sample_rate=16000)
        ww._matcher.load_reference_from_file(str(wavfile), "hello")
        
        # Should timeout
        with pytest.raises(TimeoutError):
            ww._detect_word()
    
    def test_matcher_integration_with_similar_audio(self, tmp_path):
        """Test MFCC matching integration with the WakeWord class."""
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        # Create matcher and test with the reference audio
        matcher = WordMatcher(sample_rate=16000)
        matcher.load_reference_from_file(str(wavfile), "test")
        
        # Load the reference audio
        audio, _ = sf.read(str(wavfile))
        
        # Should match itself
        matches, similarity = matcher.matches(audio.astype(np.float32), threshold=75.0)
        assert matches
        assert similarity >= 75.0
    
    def test_audio_normalization(self, tmp_path):
        """Test that audio normalization works correctly."""
        wavfile = tmp_path / "ref.wav"
        generate_wav(str(wavfile), freq=440)
        
        matcher = WordMatcher(sample_rate=16000)
        audio, _ = sf.read(str(wavfile))
        matcher.set_reference(audio.astype(np.float32), "test")
        
        # Test with scaled audio (should still match due to normalization in MFCC)
        scaled_audio = audio.astype(np.float32) * 0.5
        matches, similarity = matcher.matches(scaled_audio, threshold=75.0)
        # MFCC is scale-invariant to some extent
        assert similarity > 50.0  # Should still have reasonable similarity


class TestAudioDeviceManager:
    """Tests for AudioDeviceManager without real hardware."""
    
    def test_is_system_audio_capture_device(self):
        """Test detection of system audio capture devices."""
        # Should be detected as system capture devices
        assert AudioDeviceManager._is_system_audio_capture_device("Stereo Mix")
        assert AudioDeviceManager._is_system_audio_capture_device("What U Hear")
        assert AudioDeviceManager._is_system_audio_capture_device("System Audio Capture")
        assert AudioDeviceManager._is_system_audio_capture_device("Loopback Device")
        assert AudioDeviceManager._is_system_audio_capture_device("Speaker Output")
        
        # Should NOT be detected as system capture devices
        assert not AudioDeviceManager._is_system_audio_capture_device("USB Microphone")
        assert not AudioDeviceManager._is_system_audio_capture_device("Built-in Microphone")
        assert not AudioDeviceManager._is_system_audio_capture_device("Realtek HD Audio Input")
    
    def test_device_selection_with_invalid_index(self):
        """Test that invalid device index returns None gracefully."""
        # Very high index that won't exist
        result = AudioDeviceManager.select_device(9999)
        assert result is None
