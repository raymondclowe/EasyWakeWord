"""
Basic tests for EasyWakeWord module.

These tests verify basic import and instantiation functionality.
They use helpers to work in environments without PortAudio.
"""

import pytest
from tests.test_helpers import WakeWord, create_minimal_wakeword_instance, PORTAUDIO_AVAILABLE


# Skip all tests if WakeWord couldn't be imported
pytestmark = pytest.mark.skipif(
    WakeWord is None,
    reason="WakeWord class not available (import failed)"
)


def test_wakeword_import():
    """Test that WakeWord class can be imported (with mock if needed)."""
    assert WakeWord is not None


def test_wakeword_minimal_init(tmp_path):
    """Test minimal WakeWord instantiation without audio hardware."""
    import numpy as np
    import soundfile as sf
    
    # Create a simple test WAV file
    wavfile = tmp_path / "test.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(wavfile), audio, 16000)
    
    ww = create_minimal_wakeword_instance(
        wavfile=str(wavfile),
        textword="ok google",
        numberofwords=2,
        timeout=1
    )
    assert ww.textword == "ok google"
    assert ww.numberofwords == 2
    assert ww.timeout == 1


@pytest.mark.requires_portaudio
def test_wakeword_full_init(tmp_path):
    """Test full WakeWord instantiation with audio hardware (skipped if not available)."""
    import numpy as np
    import soundfile as sf
    
    # Create a simple test WAV file
    wavfile = tmp_path / "test.wav"
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(str(wavfile), audio, 16000)
    
    ww = WakeWord(
        textword="ok google",
        wavword=str(wavfile),
        numberofwords=2,
        timeout=1
    )
    assert ww.textword == "ok google"
    assert ww.numberofwords == 2
    assert ww.timeout == 1
    # Clean up
    ww.stop()
