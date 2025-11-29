"""
WakeWord detection class for EasyWakeWord module.

This module provides wake word detection using MFCC-based audio matching
and optional Whisper-based transcription confirmation.
"""

import io
import threading
import time
from typing import Callable, Optional

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine


class SoundBuffer:
    """Circular audio buffer with silence detection."""

    FREQUENCY = 16000
    MIN_THRESHOLD = 0.005

    def __init__(self, seconds: int = 10, device: Optional[int] = None):
        """
        Initialize the sound buffer.

        Args:
            seconds: Buffer length in seconds
            device: Audio input device index (None for default)
        """
        self.buffer_seconds = seconds
        self.buffer_length = self.buffer_seconds * self.FREQUENCY
        self.data = np.zeros(self.buffer_length)
        self.pointer = 0
        self.frame_size = 0
        self.silence_threshold = 0.01
        self.samples_collected = 0
        self._lock = threading.Lock()

        self.sd_stream = sd.InputStream(
            samplerate=self.FREQUENCY,
            channels=1,
            callback=self._add_sound_to_buffer,
            device=device,
        )
        self.sd_stream.start()

    def stop(self) -> None:
        """Stop the audio stream."""
        self.sd_stream.stop()

    def start(self) -> None:
        """Start the audio stream."""
        self.sd_stream.start()

    def _add_sound_to_buffer(self, indata, frames, time_info, status) -> None:
        """Callback to add audio data to the circular buffer."""
        new_data = np.array(indata).flatten()
        if self.frame_size == 0:
            self.frame_size = len(new_data)

        with self._lock:
            for sample in new_data:
                self.data[self.pointer] = sample
                self.pointer = (self.pointer + 1) % self.buffer_length
                if self.samples_collected < self.buffer_length:
                    self.samples_collected += 1

            if self.samples_collected < self.buffer_length:
                return

            self._adjust_silence_threshold()

    def _adjust_silence_threshold(self) -> None:
        """Dynamically adjust the silence threshold."""
        if self.frame_size == 0:
            return

        num_frames = len(self.data) // self.frame_size
        all_rms = []
        for i in range(num_frames):
            frame = self.data[i * self.frame_size : (i + 1) * self.frame_size]
            rms = np.sqrt(np.mean(frame**2))
            all_rms.append(rms)

        if all_rms:
            new_threshold = np.percentile(all_rms, 25) * 1.5
            self.silence_threshold = max(new_threshold, self.MIN_THRESHOLD)

    def is_silent(self) -> bool:
        """Check if the current audio is silent."""
        if len(self.data) == 0 or self.frame_size == 0:
            return True
        recent_samples = self.return_last_n_seconds(0.1)
        if len(recent_samples) == 0:
            return True
        rms = np.sqrt(np.mean(recent_samples**2))
        return rms < self.silence_threshold

    def return_last_n_seconds(self, n: float) -> np.ndarray:
        """Return the last n seconds of audio with wrap-around support."""
        n_samples = int(n * self.FREQUENCY)
        if n_samples > len(self.data):
            n_samples = len(self.data)
        if n_samples == 0:
            return np.array([])

        with self._lock:
            start_index = (self.pointer - n_samples) % self.buffer_length
            if start_index < self.pointer:
                return self.data[start_index : self.pointer].copy()
            else:
                return np.concatenate(
                    (self.data[start_index:], self.data[: self.pointer])
                ).copy()

    def is_buffer_full(self) -> bool:
        """Check if the buffer has been filled at least once."""
        return self.samples_collected >= self.buffer_length


class WordMatcher:
    """Matches audio clips using MFCC similarity."""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the word matcher.

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.reference_mfcc_mean = None
        self.reference_mfcc_std = None
        self.reference_word = None

    def extract_mfcc(self, audio: np.ndarray) -> tuple:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return mfcc_mean, mfcc_std

    def set_reference(self, audio: np.ndarray, word_name: str = "target") -> None:
        """Set the reference word to match against."""
        self.reference_word = word_name
        self.reference_mfcc_mean, self.reference_mfcc_std = self.extract_mfcc(audio)

    def load_reference_from_file(self, filepath: str, word_name: str = "target") -> None:
        """Load reference word from audio file."""
        audio, _ = librosa.load(filepath, sr=self.sample_rate)
        self.set_reference(audio, word_name)

    def calculate_similarity(self, audio: np.ndarray) -> float:
        """Calculate similarity between audio and reference word."""
        if self.reference_mfcc_mean is None:
            raise ValueError("No reference word set. Call set_reference() first.")

        candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)

        sim_mean = 1 - cosine(self.reference_mfcc_mean, candidate_mfcc_mean)
        sim_std = 1 - cosine(self.reference_mfcc_std, candidate_mfcc_std)

        combined_similarity = sim_mean * 0.7 + sim_std * 0.3
        similarity_percent = combined_similarity * 100
        scaled_similarity = (similarity_percent**1.5) / (100**0.5)

        return scaled_similarity

    def matches(self, audio: np.ndarray, threshold: float = 75) -> tuple:
        """Check if audio matches reference word."""
        similarity = self.calculate_similarity(audio)
        return similarity >= threshold, similarity


class WakeWord:
    """
    Wake word detector for speech recognition.

    Listens for a specific wake word using MFCC-based audio matching
    and optional Whisper-based transcription confirmation.
    """

    def __init__(
        self,
        textword: str,
        wavword: str,
        numberofwords: int,
        timeout: int = 30,
        externalwisperurl: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        device: Optional[int] = None,
        similarity_threshold: float = 75.0,
    ):
        """
        Initialize the wake word detector.

        Args:
            textword: The text phrase to detect (e.g., "ok google")
            wavword: Path to reference WAV file for MFCC matching
            numberofwords: Number of words in the wake phrase
            timeout: Timeout in seconds (default: 30)
            externalwisperurl: URL of external Whisper API for transcription
            callback: Callback function for async detection
            device: Audio input device index (None for default)
            similarity_threshold: MFCC similarity threshold (0-100, default: 75)
        """
        self.textword = textword.lower().strip()
        self.wavword = wavword
        self.numberofwords = numberofwords
        self.timeout = timeout
        self.externalwisperurl = externalwisperurl
        self.callback = callback
        self.device = device
        self.similarity_threshold = similarity_threshold

        self._sound_buffer: Optional[SoundBuffer] = None
        self._matcher: Optional[WordMatcher] = None
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._session = requests.Session()

    def _initialize_audio(self) -> None:
        """Initialize audio buffer and word matcher."""
        if self._sound_buffer is None:
            self._sound_buffer = SoundBuffer(seconds=10, device=self.device)
        if self._matcher is None:
            self._matcher = WordMatcher(sample_rate=SoundBuffer.FREQUENCY)
            self._matcher.load_reference_from_file(self.wavword, self.textword)

    def _wait_for_buffer(self) -> None:
        """Wait for the audio buffer to be filled."""
        while not self._sound_buffer.is_buffer_full():
            if self._stop_event.is_set():
                return
            time.sleep(0.1)

    def _transcribe_audio(self, audio_samples: np.ndarray) -> Optional[str]:
        """
        Send audio to STT engine and get transcription.

        Args:
            audio_samples: numpy array of audio samples

        Returns:
            Transcription text or None if failed
        """
        if self.externalwisperurl is None:
            return None

        try:
            # Normalize and boost audio
            audio_samples = audio_samples - np.mean(audio_samples)
            max_val = np.max(np.abs(audio_samples))
            if max_val > 0:
                audio_samples = audio_samples / max_val
            audio_samples = audio_samples * 1.5
            audio_samples = np.clip(audio_samples, -1.0, 1.0)

            # Convert audio to WAV format in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_samples, SoundBuffer.FREQUENCY, format="WAV")
            buffer.seek(0)

            files = {"file": ("audio.wav", buffer, "audio/wav")}
            data = {"model": "tiny", "language": "en", "initial_prompt": f"Wake word: {self.textword}"}

            response = self._session.post(
                f"{self.externalwisperurl}/transcribe", files=files, data=data, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("text", "").strip()
            return None

        except Exception:
            return None

    def _detect_word(self) -> Optional[str]:
        """
        Internal method to detect a single word occurrence.

        Returns:
            Detected text or None if not detected
        """
        state = "waiting"
        silence_start_time = None
        sound_start_time = None
        sound_end_time = None
        start_time = time.time()

        # Initialize state based on current audio
        if self._sound_buffer.is_silent():
            state = "in_silence"
            silence_start_time = time.time()

        while not self._stop_event.is_set():
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Wake word detection timed out after {self.timeout} seconds")

            time.sleep(0.1)

            is_currently_silent = self._sound_buffer.is_silent()
            current_time = time.time()

            if state == "waiting":
                if is_currently_silent:
                    state = "in_silence"
                    silence_start_time = current_time

            elif state == "in_silence":
                if not is_currently_silent:
                    silence_duration = current_time - silence_start_time
                    if silence_duration >= 1.0:
                        state = "in_sound"
                        sound_start_time = current_time
                    else:
                        state = "waiting"

            elif state == "in_sound":
                if not is_currently_silent:
                    sound_duration = current_time - sound_start_time
                    if sound_duration > 1.5:
                        state = "waiting"
                else:
                    sound_duration = current_time - sound_start_time
                    if 0.5 <= sound_duration <= 1.5:
                        state = "after_sound"
                        sound_end_time = current_time
                    else:
                        state = "waiting"

            elif state == "after_sound":
                if is_currently_silent:
                    trailing_silence_duration = current_time - sound_end_time
                    if trailing_silence_duration >= 0.5:
                        # Extract the word
                        padding = 0.05
                        extract_start = sound_start_time - current_time - padding
                        extract_end = sound_end_time - current_time + padding

                        word_samples_with_padding = self._sound_buffer.return_last_n_seconds(
                            abs(extract_start)
                        )
                        word_end_idx = int((abs(extract_end)) * SoundBuffer.FREQUENCY)
                        word_audio = word_samples_with_padding[
                            : len(word_samples_with_padding) - word_end_idx
                        ]

                        # Skip if audio is too long
                        audio_duration = len(word_audio) / SoundBuffer.FREQUENCY
                        if audio_duration > 3.0:
                            state = "waiting"
                            continue

                        # Check MFCC similarity
                        matches, similarity = self._matcher.matches(
                            word_audio, threshold=self.similarity_threshold
                        )

                        if matches:
                            # Try transcription if available
                            if self.externalwisperurl:
                                transcription = self._transcribe_audio(word_audio)
                                if transcription:
                                    transcription_clean = transcription.strip().lower().rstrip(".,!?;:")
                                    target_words = self.textword.split()
                                    transcription_words = transcription_clean.split()

                                    # Check if all target words appear in transcription
                                    if all(word in transcription_words for word in target_words):
                                        return transcription
                            else:
                                # No transcription service, rely on MFCC match
                                return self.textword

                        state = "waiting"
                else:
                    state = "waiting"

        return None

    def waitforit(self) -> str:
        """
        Wait for the wake word to be detected (blocking).

        Returns:
            The transcribed text that was detected

        Raises:
            TimeoutError: If timeout is reached without detection
        """
        self._initialize_audio()
        self._stop_event.clear()
        self._listening = True

        try:
            self._wait_for_buffer()
            result = self._detect_word()
            if result is None:
                raise TimeoutError(f"Wake word detection timed out after {self.timeout} seconds")
            return result
        finally:
            self._listening = False

    def start(self) -> None:
        """
        Start listening for the wake word in a background thread (non-blocking).
        Requires a callback to be set.

        Raises:
            ValueError: If no callback is set
        """
        if self.callback is None:
            raise ValueError("Callback must be set for async operation. Use waitforit() for synchronous operation.")

        if self._listening:
            return

        self._initialize_audio()
        self._stop_event.clear()
        self._listening = True

        def listen_loop():
            try:
                self._wait_for_buffer()
                while not self._stop_event.is_set():
                    try:
                        result = self._detect_word()
                        if result and self.callback:
                            self.callback(result)
                    except TimeoutError:
                        continue
            finally:
                self._listening = False

        self._listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self._listen_thread.start()

    def stop(self) -> None:
        """Stop the background listening thread."""
        self._stop_event.set()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        if self._sound_buffer:
            self._sound_buffer.stop()
        self._listening = False

    def is_listening(self) -> bool:
        """
        Check if the detector is currently listening.

        Returns:
            True if listening, False otherwise
        """
        return self._listening

    def __del__(self):
        """Clean up resources."""
        self.stop()
        if self._session:
            self._session.close()
