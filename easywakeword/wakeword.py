"""
WakeWord detection class for EasyWakeWord module.

This module provides wake word detection using MFCC-based audio matching
and optional Whisper-based transcription confirmation.

Supports multiple STT backends:
- Local mini_transcriber (bundled, auto-downloads on first run)
- External Whisper API (user-provided URL)
- Commercial APIs (e.g., replicate.com)
"""

import io
import os
import subprocess
import sys
import threading
import time
import re
from typing import Callable, Optional, Union, Union

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine


# Constants for STT backend configuration
DEFAULT_MINI_TRANSCRIBER_PORT = 8085
MINI_TRANSCRIBER_REPO = "https://github.com/raymondclowe/mini_transcriber.git"


class AudioDeviceManager:
    """Manages audio device selection with intelligent defaults and name-based matching."""

    @staticmethod
    def list_devices() -> list:
        """
        List all available audio input devices with their details.

        Returns:
            List of device dictionaries with index, name, and capabilities
        """
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:  # Only input devices
                    # Filter out system audio capture/loopback devices
                    device_name = device['name'].lower()
                    if not AudioDeviceManager._is_system_audio_capture_device(device_name):
                        devices.append({
                            'index': i,
                            'name': device['name'],
                            'hostapi': sd.query_hostapis()[device['hostapi']]['name'],
                            'default_samplerate': device['default_samplerate'],
                            'max_input_channels': device['max_input_channels']
                        })
        except Exception as e:
            print(f"Warning: Could not query audio devices: {e}")
        return devices

    @staticmethod
    def _is_system_audio_capture_device(device_name: str) -> bool:
        """
        Check if a device name indicates a system audio capture/loopback device.

        These devices capture system audio output rather than microphone input
        and should not be used for wake word detection.

        Args:
            device_name: Device name to check

        Returns:
            True if this appears to be a system audio capture device
        """
        name_lower = device_name.lower()

        # Common patterns for system audio capture devices
        system_capture_patterns = [
            'stereo mix',
            'what u hear',
            'wave out',
            'loopback',
            'capture',
            'monitor',
            'system audio',
            'audio capture',
            'sound capture'
        ]

        # Check for system capture patterns
        for pattern in system_capture_patterns:
            if pattern in name_lower:
                return True

        # Devices with "speaker" or "output" in name but having input channels
        # are likely system audio capture devices
        output_indicators = ['speaker', 'output', 'headphone']
        has_output_indicator = any(indicator in name_lower for indicator in output_indicators)

        # But allow devices that also have clear microphone indicators
        mic_indicators = ['microphone', 'mic', 'input', 'line-in', 'aux']
        has_mic_indicator = any(indicator in name_lower for indicator in mic_indicators)

        if has_output_indicator and not has_mic_indicator:
            return True

        return False

    @staticmethod
    def select_device(device_spec: Optional[Union[int, str]] = None) -> Optional[int]:
        """
        Select an audio input device based on various criteria.

        Args:
            device_spec: Device selection specification:
                - None: Auto-select best default device
                - int: Device index
                - str: Device name pattern, or magic word:
                    - "best": Test all devices and pick one with highest audio level
                    - "first": Find first working device (with audio signal)
                    - "default": Use system default device
                    - Other strings: Match against device names (case-insensitive)

        Returns:
            Device index or None if no suitable device found
        """
        devices = AudioDeviceManager.list_devices()
        if not devices:
            print("Warning: No audio input devices found")
            return None

        # If no specification, auto-select best device
        if device_spec is None:
            return AudioDeviceManager._auto_select_device(devices)

        # If integer, use as device index
        if isinstance(device_spec, int):
            if 0 <= device_spec < len(sd.query_devices()) and sd.query_devices()[device_spec]['max_input_channels'] > 0:
                return device_spec
            else:
                print(f"Warning: Device index {device_spec} is not valid or not an input device")
                return None

        # If string, handle magic words and name matching
        if isinstance(device_spec, str):
            magic_word = device_spec.lower().strip()

            # Magic word: "best" - test all devices and pick highest audio level
            if magic_word == "best":
                return AudioDeviceManager.find_best_device_by_audio_level()

            # Magic word: "first" - find first working device
            elif magic_word == "first":
                return AudioDeviceManager.find_first_working_device()

            # Magic word: "default" - use system default
            elif magic_word in ["default", "system"]:
                return AudioDeviceManager._select_system_default(devices)

            # Otherwise, treat as name pattern
            else:
                return AudioDeviceManager._select_by_name(devices, device_spec)

        print(f"Warning: Invalid device specification: {device_spec}")
        return None

    @staticmethod
    def _select_system_default(devices: list) -> Optional[int]:
        """
        Select the system default input device.

        Args:
            devices: List of device dictionaries

        Returns:
            System default device index, or None if not found
        """
        try:
            default_device = sd.default.device[0]  # (input, output)
            if default_device >= 0:
                # Verify it's an input device
                device_info = sd.query_devices()[default_device]
                if device_info['max_input_channels'] > 0:
                    return default_device
        except:
            pass

        print("Warning: Could not determine system default device")
        return None

    @staticmethod
    def _auto_select_device(devices: list) -> Optional[int]:
        """
        Auto-select the best available input device using preference hierarchy.

        Priority order:
        1. System default input device
        2. First device with "microphone" in name
        3. First device with "input" in name
        4. First available device
        """
        if not devices:
            return None

        # Try to get system default input device
        try:
            default_device = sd.default.device[0]  # (input, output)
            if default_device >= 0:
                # Verify it's an input device
                device_info = sd.query_devices()[default_device]
                if device_info['max_input_channels'] > 0:
                    return default_device
        except:
            pass

        # Look for microphone devices
        for device in devices:
            if 'microphone' in device['name'].lower():
                return device['index']

        # Look for input devices
        for device in devices:
            if 'input' in device['name'].lower():
                return device['index']

        # Return first available device as fallback
        return devices[0]['index']

    @staticmethod
    def _select_by_name(devices: list, name_pattern: str) -> Optional[int]:
        """
        Select device by name pattern matching.

        Args:
            devices: List of device dictionaries
            name_pattern: Pattern to match against device names (case-insensitive)

        Returns:
            Device index or None if no match found
        """
        pattern = name_pattern.lower()

        # Exact match first
        for device in devices:
            if device['name'].lower() == pattern:
                return device['index']

        # Partial match
        for device in devices:
            if pattern in device['name'].lower():
                return device['index']

        # Regex match
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            for device in devices:
                if regex.search(device['name']):
                    return device['index']
        except re.error:
            pass

        print(f"Warning: No device found matching pattern: {name_pattern}")
        return None

    @staticmethod
    def test_device_audio_level(device_index: int, test_duration: float = 0.1) -> float:
        """
        Test a device's audio level by recording a short sample.

        Args:
            device_index: Device index to test
            test_duration: Duration in seconds to record (default: 100ms)

        Returns:
            RMS audio level (0.0 if no signal or error)
        """
        try:
            import sounddevice as sd
            import numpy as np

            # Record test sample
            sample_rate = 16000
            samples = int(test_duration * sample_rate)

            audio_data = sd.rec(
                samples,
                samplerate=sample_rate,
                channels=1,
                device=device_index,
                dtype=np.float32
            )
            sd.wait()

            # Calculate RMS level
            audio_data = audio_data.flatten()
            rms = np.sqrt(np.mean(audio_data**2))

            return float(rms)

        except Exception:
            return 0.0

    @staticmethod
    def find_best_device_by_audio_level(min_rms_threshold: float = 0.001) -> Optional[int]:
        """
        Find the device with the highest audio level by testing all devices.

        Args:
            min_rms_threshold: Minimum RMS level to consider a device "working"

        Returns:
            Device index with highest audio level, or None if no working devices
        """
        devices = AudioDeviceManager.list_devices()
        if not devices:
            return None

        best_device = None
        best_rms = 0.0

        print("Testing audio levels on all devices...")

        for device in devices:
            rms = AudioDeviceManager.test_device_audio_level(device['index'])
            print(f"{rms:.4f}")

            if rms > min_rms_threshold and rms > best_rms:
                best_rms = rms
                best_device = device['index']

        if best_device is not None:
            print(f"Selected device {best_device} with RMS {best_rms:.4f}")
        else:
            print("No devices with sufficient audio levels found")

        return best_device

    @staticmethod
    def find_first_working_device(min_rms_threshold: float = 0.001) -> Optional[int]:
        """
        Find the first device that appears to be working (has audio signal).

        Args:
            min_rms_threshold: Minimum RMS level to consider a device "working"

        Returns:
            First working device index, or None if none found
        """
        devices = AudioDeviceManager.list_devices()
        if not devices:
            return None

        print("Finding first working device...")

        for device in devices:
            rms = AudioDeviceManager.test_device_audio_level(device['index'])
            print(f"{rms:.4f}")

            if rms > min_rms_threshold:
                print(f"Selected first working device {device['index']} with RMS {rms:.4f}")
                return device['index']

        print("No working devices found")
        return None

    @staticmethod
    def print_device_list():
        """Print a formatted list of available audio input devices."""
        devices = AudioDeviceManager.list_devices()

        if not devices:
            print("No audio input devices found.")
            return

        print("Available audio input devices:")
        print("-" * 60)
        for device in devices:
            default_marker = " (default)" if device['index'] == sd.default.device[0] else ""
            print(f"{device['index']:2d}: {device['name']}{default_marker}")
            print(f"    Host API: {device['hostapi']}")
            print(f"    Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}")
            print()


class SoundBuffer:
    """Circular audio buffer with silence detection."""

    FREQUENCY = 16000
    MIN_THRESHOLD = 0.005

    def __init__(self, seconds: int = 10, device: Optional[Union[int, str]] = None):
        """
        Initialize the sound buffer.

        Args:
            seconds: Buffer length in seconds [not implemented: buffer size is currently hardcoded to 10s]
            device: Audio input device specification:
                    - None: Auto-select best available device
                    - int: Device index
                    - str: Device name pattern to match (case-insensitive)
        """
        self.buffer_seconds = seconds
        self.buffer_length = self.buffer_seconds * self.FREQUENCY
        self.data = np.zeros(self.buffer_length)
        self.pointer = 0
        self.frame_size = 0
        self.silence_threshold = 0.01
        self.samples_collected = 0
        self._lock = threading.Lock()

        # Resolve device specification to index
        device_index = AudioDeviceManager.select_device(device)

        self.sd_stream = sd.InputStream(
            samplerate=self.FREQUENCY,
            channels=1,
            callback=self._add_sound_to_buffer,
            device=device_index,
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
        )  # [not implemented: MFCC parameters (n_mfcc, n_fft, hop_length) are hardcoded]
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

    Supports multiple STT backends:
    - "bundled": Auto-downloads and runs mini_transcriber locally
    - URL string: Uses an external Whisper API at the specified URL
    - None: Relies only on MFCC matching without transcription confirmation

    [not implemented: metrics/telemetry (detection rate, latency, false positives)]
    """

    def __init__(
        self,
        textword: str,
        wavword: str,
        numberofwords: int,
        timeout: int = 30,
        external_whisper_url: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        device: Optional[Union[int, str]] = None,
        similarity_threshold: float = 75.0,
        stt_backend: Optional[str] = None,
        pre_speech_silence: Optional[float] = None,
        speech_duration_min: Optional[float] = None,
        speech_duration_max: Optional[float] = None,
        post_speech_silence: Optional[float] = None,
    ):
        """
        Initialize the wake word detector.

        Args:
            textword: The text phrase to detect (e.g., "ok google")
            wavword: Path to reference WAV file for MFCC matching
            numberofwords: Number of words in the wake phrase
            timeout: Timeout in seconds (default: 30)
            external_whisper_url: URL of external Whisper API for transcription
                                  (e.g., "http://localhost:8085" for mini_transcriber)
            callback: Callback function for async detection
            device: Audio input device specification:
                    - None: Auto-select best available device
                    - int: Device index
                    - str: Device name pattern to match (case-insensitive)
            similarity_threshold: MFCC similarity threshold (0-100, default: 75)
            stt_backend: STT backend to use:
                         - "bundled": Auto-download and run mini_transcriber locally
                         - None: Use external_whisper_url if provided, otherwise MFCC only
            pre_speech_silence: Minimum silence duration before speech starts (seconds).
                                If None, auto-calculated from reference audio or heuristics.
            speech_duration_min: Minimum speech duration (seconds).
                                 If None, auto-calculated from reference audio or heuristics.
            speech_duration_max: Maximum speech duration (seconds).
                                 If None, auto-calculated from reference audio or heuristics.
            post_speech_silence: Minimum silence duration after speech ends (seconds).
                                 If None, auto-calculated from reference audio or heuristics.
        """
        self.textword = textword.lower().strip()
        self.wavword = wavword
        self.numberofwords = numberofwords
        self.timeout = timeout
        self.external_whisper_url = external_whisper_url
        self.callback = callback
        self.device = AudioDeviceManager.select_device(device)  # Resolve device specification
        self.similarity_threshold = similarity_threshold
        self.stt_backend = stt_backend

        # Calculate speech detection thresholds
        self.pre_speech_silence = pre_speech_silence
        self.speech_duration_min = speech_duration_min
        self.speech_duration_max = speech_duration_max
        self.post_speech_silence = post_speech_silence
        self._calculate_detection_thresholds()

        self._sound_buffer: Optional[SoundBuffer] = None
        self._matcher: Optional[WordMatcher] = None
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._session = requests.Session()  # [not implemented: session configuration for Whisper API headers (authentication)]
        self._bundled_transcriber_process: Optional[subprocess.Popen] = None
        self._transcriber_url: Optional[str] = None

        # Set up transcriber URL based on backend configuration
        self._setup_stt_backend()

    def _calculate_detection_thresholds(self) -> None:
        """
        Calculate speech detection thresholds based on reference audio or heuristics.

        Sets self.pre_speech_silence, self.speech_duration_min, self.speech_duration_max,
        and self.post_speech_silence if they are None.
        """
        # If all thresholds are provided, use them as-is
        if (self.pre_speech_silence is not None and
            self.speech_duration_min is not None and
            self.speech_duration_max is not None and
            self.post_speech_silence is not None):
            return

        # Try to analyze reference audio file
        reference_duration = self._analyze_reference_audio_duration()

        if reference_duration is not None:
            # Calculate thresholds based on reference audio
            self._set_thresholds_from_audio_duration(reference_duration)
        else:
            # Fall back to text-based heuristics
            self._set_thresholds_from_text_heuristics()

    def _analyze_reference_audio_duration(self) -> Optional[float]:
        """
        Analyze the reference audio file to determine its actual speech duration.

        Returns:
            Duration of speech in the reference audio (seconds), or None if analysis fails
        """
        try:
            # Load the reference audio
            audio, sample_rate = librosa.load(self.wavword, sr=None)

            # Resample to our working rate if needed
            if sample_rate != SoundBuffer.FREQUENCY:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SoundBuffer.FREQUENCY)

            # Simple voice activity detection using RMS energy
            # Split into small frames
            frame_length = int(0.025 * SoundBuffer.FREQUENCY)  # 25ms frames
            hop_length = int(0.010 * SoundBuffer.FREQUENCY)    # 10ms hop

            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

            # Find frames above a threshold (relative to the max RMS)
            threshold = np.max(rms) * 0.1  # 10% of peak energy
            voice_frames = rms > threshold

            # Find contiguous voice segments
            if np.any(voice_frames):
                # Find the start and end of the main voice segment
                voice_indices = np.where(voice_frames)[0]
                start_frame = voice_indices[0]
                end_frame = voice_indices[-1]

                # Convert to time
                duration = (end_frame - start_frame) * hop_length / SoundBuffer.FREQUENCY
                return max(duration, 0.2)  # Minimum 200ms to avoid too short detections

        except Exception as e:
            print(f"Warning: Could not analyze reference audio duration: {e}")  # [not implemented: verbose logging option (currently hardcoded print statements)]

        return None

    def _set_thresholds_from_audio_duration(self, audio_duration: float) -> None:
        """
        Set detection thresholds based on analyzed reference audio duration.

        Args:
            audio_duration: Duration of speech in reference audio (seconds)
        """
        # Pre-speech silence: slightly longer than audio duration to ensure clean start
        if self.pre_speech_silence is None:
            self.pre_speech_silence = max(0.8, audio_duration * 0.8)

        # Speech duration range: allow some variation around the reference
        if self.speech_duration_min is None:
            self.speech_duration_min = max(0.3, audio_duration * 0.6)

        if self.speech_duration_max is None:
            self.speech_duration_max = min(3.0, audio_duration * 1.8)

        # Post-speech silence: similar to pre-speech
        if self.post_speech_silence is None:
            self.post_speech_silence = max(0.3, audio_duration * 0.4)

    def _set_thresholds_from_text_heuristics(self) -> None:
        """
        Set detection thresholds based on text analysis heuristics.
        """
        # Estimate syllables (rough approximation)
        text = self.textword.lower()
        syllable_estimate = self._estimate_syllables(text)

        # Base duration estimate: ~0.3 seconds per syllable
        estimated_duration = syllable_estimate * 0.3

        # Ensure reasonable bounds
        estimated_duration = max(0.5, min(2.5, estimated_duration))

        # Apply the same logic as audio-based calculation
        self._set_thresholds_from_audio_duration(estimated_duration)

    def _estimate_syllables(self, text: str) -> int:
        """
        Rough syllable estimation for English text.

        Args:
            text: Text to analyze

        Returns:
            Estimated number of syllables
        """
        # Remove punctuation and split into words
        words = ''.join(c for c in text if c.isalnum() or c.isspace()).split()

        total_syllables = 0

        for word in words:
            word = word.lower().strip()
            if not word:
                continue

            # Count vowel groups (very rough approximation)
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel

            # Ensure at least 1 syllable per word
            syllable_count = max(1, syllable_count)

            # Special cases
            if word.endswith('e'):
                syllable_count = max(1, syllable_count - 1)  # Silent e
            if word.endswith(('es', 'ed')) and len(word) > 2:
                syllable_count = max(1, syllable_count - 1)  # Common suffixes

            total_syllables += syllable_count

        return max(1, total_syllables)

    def _setup_stt_backend(self) -> None:
        """Set up the STT backend based on configuration."""
        if self.stt_backend == "bundled":
            # Will set up bundled mini_transcriber on first use
            self._transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"
        elif self.external_whisper_url:
            # Use provided external URL
            self._transcriber_url = self.external_whisper_url
        else:
            # No transcription service, rely on MFCC matching only
            self._transcriber_url = None

    def _ensure_bundled_transcriber(self) -> bool:
        """
        Ensure the bundled mini_transcriber is running.

        Downloads and starts mini_transcriber if not already running.
        Returns True if transcriber is available, False otherwise.
        """
        if self.stt_backend != "bundled":
            return self._transcriber_url is not None

        # Check if transcriber is already running
        try:
            response = self._session.get(
                f"{self._transcriber_url}/health", timeout=2
            )  # [not implemented: dedicated health check method for transcription service]
            if response.status_code == 200:
                return True
        except Exception:
            pass

        # Try to download and start mini_transcriber
        transcriber_dir = os.path.join(
            os.path.expanduser("~"), ".easywakeword", "mini_transcriber"
        )

        if not os.path.exists(transcriber_dir):
            print("Downloading mini_transcriber for first-time setup...")  # [not implemented: verbose logging option (currently hardcoded print statements)]
            os.makedirs(os.path.dirname(transcriber_dir), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", MINI_TRANSCRIBER_REPO, transcriber_dir],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode() if e.stderr else "No error details"
                print("Failed to download mini_transcriber: git clone failed")  # [not implemented: verbose logging option (currently hardcoded print statements)]
                print(f"  Command: git clone {MINI_TRANSCRIBER_REPO} {transcriber_dir}")  # [not implemented: verbose logging option (currently hardcoded print statements)]
                print(f"  Error: {stderr_msg}")  # [not implemented: verbose logging option (currently hardcoded print statements)]
                return False

        # Start mini_transcriber
        try:
            app_path = os.path.join(transcriber_dir, "app.py")
            if not os.path.exists(app_path):
                print(f"mini_transcriber app.py not found at {app_path}")  # [not implemented: verbose logging option (currently hardcoded print statements)]
                return False

            env = os.environ.copy()
            env["PORT"] = str(DEFAULT_MINI_TRANSCRIBER_PORT)
            self._bundled_transcriber_process = subprocess.Popen(
                [sys.executable, app_path],
                cwd=transcriber_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for transcriber to start
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                try:
                    response = self._session.get(
                        f"{self._transcriber_url}/health", timeout=2
                    )
                    if response.status_code == 200:
                        print("mini_transcriber started successfully")  # [not implemented: verbose logging option (currently hardcoded print statements)]
                        return True
                except Exception:
                    pass

            print("Timed out waiting for mini_transcriber to start")  # [not implemented: verbose logging option (currently hardcoded print statements)]
            return False

        except Exception as e:
            print(f"Failed to start mini_transcriber: {e}")  # [not implemented: verbose logging option (currently hardcoded print statements)]
            return False

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
        if self._transcriber_url is None:
            return None

        # Ensure bundled transcriber is running if needed
        if self.stt_backend == "bundled":
            if not self._ensure_bundled_transcriber():
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
                f"{self._transcriber_url}/transcribe", files=files, data=data, timeout=10
            )  # [not implemented: retry logic for transient network failures]

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
                    if silence_duration >= self.pre_speech_silence:
                        state = "in_sound"
                        sound_start_time = current_time
                    else:
                        state = "waiting"

            elif state == "in_sound":
                if not is_currently_silent:
                    sound_duration = current_time - sound_start_time
                    if sound_duration > self.speech_duration_max:
                        state = "waiting"
                else:
                    sound_duration = current_time - sound_start_time
                    if self.speech_duration_min <= sound_duration <= self.speech_duration_max:
                        state = "after_sound"
                        sound_end_time = current_time
                    else:
                        state = "waiting"

            elif state == "after_sound":
                if is_currently_silent:
                    trailing_silence_duration = current_time - sound_end_time
                    if trailing_silence_duration >= self.post_speech_silence:
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
                            if self._transcriber_url:
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
        """Stop the background listening thread and clean up resources."""
        self._stop_event.set()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        if self._sound_buffer:
            self._sound_buffer.stop()
        self._listening = False

        # Stop bundled transcriber if we started it
        if self._bundled_transcriber_process:
            self._bundled_transcriber_process.terminate()
            try:
                self._bundled_transcriber_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._bundled_transcriber_process.kill()
                self._bundled_transcriber_process.wait()
            self._bundled_transcriber_process = None

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
