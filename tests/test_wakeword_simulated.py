import numpy as np
import pytest
from easywakeword import WakeWord
import os

# Helper: Generate a fake reference WAV file for testing
import soundfile as sf

def generate_wav(filename, duration=1.0, freq=440, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(filename, audio, sr)
    return filename

# Test: WakeWord can load and match a reference word
@pytest.mark.parametrize("textword, freq", [
    ("ok google", 440),
    ("hey computer", 550),
])
def test_mfcc_match(tmp_path, textword, freq):
    wavfile = tmp_path / f"{textword.replace(' ', '')}.wav"
    generate_wav(str(wavfile), freq=freq)
    ww = WakeWord(
        textword=textword,
        wavword=str(wavfile),
        numberofwords=len(textword.split()),
        timeout=1,
        stt_backend=None
    )
    # Simulate MFCC match (should match itself)
    ww._initialize_audio()
    audio, _ = sf.read(str(wavfile))
    matches, similarity = ww._matcher.matches(audio)
    assert matches
    assert similarity > 70

# Test: Timeout is raised if no audio is detected
@pytest.mark.timeout(3)
def test_timeout_on_no_audio(monkeypatch, tmp_path):
    wavfile = tmp_path / "okgoogle.wav"
    generate_wav(str(wavfile))
    ww = WakeWord(
        textword="ok google",
        wavword=str(wavfile),
        numberofwords=2,
        timeout=1,
        stt_backend=None
    )
    # Monkeypatch SoundBuffer to always be silent
    class DummyBuffer:
        def is_buffer_full(self): return True
        def is_silent(self): return True
        def stop(self): pass
    ww._sound_buffer = DummyBuffer()
    with pytest.raises(TimeoutError):
        ww.waitforit()
