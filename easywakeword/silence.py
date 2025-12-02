"""
Silence and Sound Buffer helpers (migrated from easywakeword.silence)
"""
import collections
import numpy as np
import sounddevice as sd
import time

class SoundBuffer:
    FREQUENCY = 16000
    def __init__(self, device=None, buffer_seconds=3.0):
        self.device = device
        self.buffer_seconds = buffer_seconds
        self.buffer_samples = int(self.FREQUENCY * buffer_seconds)
        self._buffer = collections.deque(maxlen=self.buffer_samples)
        self.sd_stream = None
        self.silence_threshold = 0.01
        self.running = False
        self._start_stream()

    def _start_stream(self):
        try:
            self.running = True
            self.sd_stream = sd.InputStream(samplerate=self.FREQUENCY, channels=1, dtype='float32', device=self.device, callback=self._push)
            self.sd_stream.start()
        except Exception as e:
            print(f"[SoundBuffer] Failed to start InputStream: {e}")

    def _push(self, indata, frames, time_info, status):
        samples = indata[:, 0].tolist()
        self._buffer.extend(samples)

    def is_silent(self, window_seconds=0.1):
        """Check if recent audio is silent. Only checks last window_seconds of audio."""
        if len(self._buffer) == 0:
            return True
        # Only check recent samples, not the entire buffer
        window_samples = int(window_seconds * self.FREQUENCY)
        recent = list(self._buffer)[-window_samples:] if len(self._buffer) >= window_samples else list(self._buffer)
        data = np.array(recent)
        rms = np.sqrt(np.mean(data**2))
        return rms < self.silence_threshold

    def get_recent_rms(self, window_seconds=0.1):
        """Get RMS of recent audio window."""
        if len(self._buffer) == 0:
            return 0.0
        window_samples = int(window_seconds * self.FREQUENCY)
        recent = list(self._buffer)[-window_samples:] if len(self._buffer) >= window_samples else list(self._buffer)
        data = np.array(recent)
        return np.sqrt(np.mean(data**2))

    def return_last_n_seconds(self, seconds: float):
        n = int(seconds * self.FREQUENCY)
        return np.array(list(self._buffer)[-n:]) if n > 0 else np.array(list(self._buffer))

def list_audio_devices():
    print(sd.query_devices())

def test_microphone_level(device_index=0, duration=3):
    s = SoundBuffer(device=device_index, buffer_seconds=duration)
    time.sleep(duration)
    arr = s.return_last_n_seconds(duration)
    rms = np.sqrt(np.mean(arr**2)) if len(arr) > 0 else 0
    print(f"RMS: {rms}")
    return arr
__all__ = ["SoundBuffer", "list_audio_devices", "test_microphone_level"]
