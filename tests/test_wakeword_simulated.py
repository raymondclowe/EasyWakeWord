"""
Simulated audio tests for EasyWakeWord.

These tests mock audio hardware to allow testing in CI environments
without real microphones.
"""

import numpy as np
import pytest
import soundfile as sf

from easywakeword.wakeword import WordMatcher


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
        from easywakeword.wakeword import WakeWord
        
        # Create a dummy instance to access the method
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile))
        
        # Create WakeWord but don't initialize audio
        ww = object.__new__(WakeWord)
        ww.textword = "hello"
        ww.wavword = str(wavfile)
        ww.numberofwords = 1
        ww.pre_speech_silence = None
        ww.speech_duration_min = None
        ww.speech_duration_max = None
        ww.post_speech_silence = None
        
        # Test syllable estimation
        assert ww._estimate_syllables("hello") == 2  # hel-lo
        assert ww._estimate_syllables("hey") == 1
        assert ww._estimate_syllables("computer") >= 2  # com-pu-ter
    
    def test_multi_word_phrases(self, tmp_path):
        """Test syllable count for multi-word phrases."""
        from easywakeword.wakeword import WakeWord
        
        wavfile = tmp_path / "test.wav"
        generate_wav(str(wavfile))
        
        ww = object.__new__(WakeWord)
        ww.textword = "ok google"
        ww.wavword = str(wavfile)
        
        # "ok google" should have 3-4 syllables (o-k goo-gle)
        syllables = ww._estimate_syllables("ok google")
        assert syllables >= 2


class TestAudioDurationAnalysis:
    """Tests for the reference audio duration analysis."""
    
    def test_analyze_reference_audio_duration(self, tmp_path):
        """Test that audio duration is correctly analyzed."""
        from easywakeword.wakeword import WakeWord, SoundBuffer
        
        # Create a 1-second audio file
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = object.__new__(WakeWord)
        ww.wavword = str(wavfile)
        
        duration = ww._analyze_reference_audio_duration()
        # Duration should be detected (might be less than 1.0 due to envelope)
        assert duration is not None
        assert 0.2 <= duration <= 1.5


class TestThresholdCalculation:
    """Tests for detection threshold calculation."""
    
    def test_threshold_from_audio_duration(self, tmp_path):
        """Test that thresholds are correctly calculated from audio duration."""
        from easywakeword.wakeword import WakeWord
        
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = object.__new__(WakeWord)
        ww.wavword = str(wavfile)
        ww.textword = "hello"
        ww.pre_speech_silence = None
        ww.speech_duration_min = None
        ww.speech_duration_max = None
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
        from easywakeword.wakeword import WakeWord
        
        wavfile = tmp_path / "ref.wav"
        generate_speech_like_audio(str(wavfile), duration=1.0)
        
        ww = object.__new__(WakeWord)
        ww.wavword = str(wavfile)
        ww.textword = "hello"
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
