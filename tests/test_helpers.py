"""
Test helpers for EasyWakeWord tests.

This module provides safe imports and utilities for testing in environments
without audio hardware.
"""

import sys
import threading
from unittest.mock import MagicMock


# Check if PortAudio is available
PORTAUDIO_AVAILABLE = False
try:
    import sounddevice as sd
    PORTAUDIO_AVAILABLE = True
except OSError:
    pass


def create_mock_sounddevice():
    """
    Create a mock sounddevice module for testing without PortAudio.
    
    This allows tests to import easywakeword.wakeword even when
    PortAudio is not available.
    """
    mock_sd = MagicMock()
    mock_sd.query_devices.return_value = []
    mock_sd.query_hostapis.return_value = [{'name': 'Mock Host API'}]
    mock_sd.default.device = (-1, -1)
    mock_sd.InputStream = MagicMock()
    return mock_sd


def get_wakeword_classes():
    """
    Safely import WakeWord classes, mocking sounddevice if needed.
    
    Returns:
        tuple: (WakeWord, WordMatcher, AudioDeviceManager, SoundBuffer) or None values if import fails
    """
    if not PORTAUDIO_AVAILABLE:
        # Install mock sounddevice before importing wakeword
        sys.modules['sounddevice'] = create_mock_sounddevice()
    
    try:
        from easywakeword.wakeword import (
            WakeWord, 
            WordMatcher, 
            AudioDeviceManager,
            SoundBuffer
        )
        return WakeWord, WordMatcher, AudioDeviceManager, SoundBuffer
    except Exception as e:
        print(f"Warning: Could not import wakeword classes: {e}")
        return None, None, None, None


# Import classes - will use mock if PortAudio not available
WakeWord, WordMatcher, AudioDeviceManager, SoundBuffer = get_wakeword_classes()


def create_minimal_wakeword_instance(wavfile, textword="hello", numberofwords=1, timeout=1,
                                     speech_duration_min=None, speech_duration_max=None,
                                     verbose=False):
    """
    Create a minimal WakeWord instance without initializing audio hardware.
    
    This helper bypasses the audio device initialization to allow testing
    the detection logic in CI environments without real microphones.
    
    Args:
        wavfile: Path to the reference WAV file
        textword: Text of the wake word
        numberofwords: Number of words in the phrase
        timeout: Detection timeout in seconds
        speech_duration_min: Minimum speech duration (None for auto-calculate)
        speech_duration_max: Maximum speech duration (None for auto-calculate)
        verbose: Enable verbose logging
        
    Returns:
        WakeWord instance with mocked audio components
    """
    if WakeWord is None:
        raise ImportError("WakeWord class not available")
    
    ww = object.__new__(WakeWord)
    ww.textword = textword
    ww.wavword = str(wavfile)
    ww.numberofwords = numberofwords
    ww.timeout = timeout
    ww.similarity_threshold = 75.0
    ww.pre_speech_silence = 0.8
    ww.post_speech_silence = 0.4
    ww.buffer_seconds = 10
    ww.verbose = verbose
    ww.retry_count = 3
    ww.retry_backoff = 0.5
    
    # Store user-provided values for auto-calculation
    ww._user_speech_duration_min = speech_duration_min
    ww._user_speech_duration_max = speech_duration_max
    
    # Perform auto-calculation of speech durations
    ww._auto_calculate_speech_durations()
    
    ww.pre_speech_silence = None
    ww.speech_duration_min = None
    ww.speech_duration_max = None
    ww.post_speech_silence = None
    ww.buffer_seconds = 10
    ww.verbose = False
    ww.retry_count = 3
    ww.retry_backoff = 0.5
    ww.callback = None
    ww._stop_event = threading.Event()
    ww._transcriber_url = None
    ww._matcher = None
    ww._sound_buffer = None
    ww._listening = False
    ww._listen_thread = None
    ww._session = None
    ww._bundled_transcriber_process = None
    return ww
