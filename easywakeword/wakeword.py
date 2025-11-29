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
import logging
import os
import subprocess
import sys
import threading
import time
import re
from typing import Callable, Dict, Optional, Union

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine


# Configure module logger
logger = logging.getLogger(__name__)

# Constants for STT backend configuration
DEFAULT_MINI_TRANSCRIBER_PORT = 8085
MINI_TRANSCRIBER_REPO = "https://github.com/raymondclowe/mini_transcriber.git"
DEFAULT_BUFFER_SECONDS = 10
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_BACKOFF = 0.5  # Initial retry delay in seconds

# Default fixed silence durations (in seconds)
# These are fixed values that work well for typical wake words
# Users can override these if needed
DEFAULT_PRE_SPEECH_SILENCE = 0.8
DEFAULT_SPEECH_DURATION_MIN = 0.3
DEFAULT_SPEECH_DURATION_MAX = 2.0
DEFAULT_POST_SPEECH_SILENCE = 0.4


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

    def __init__(self, seconds: int = DEFAULT_BUFFER_SECONDS, device: Optional[Union[int, str]] = None):
        """
        Initialize the sound buffer.

        Args:
            seconds: Buffer length in seconds (default: 10). Larger buffers use more memory
                     but allow detection of longer wake phrases. Typical values: 5-30 seconds.
            device: Audio input device specification:
                    - None: Auto-select best available device
                    - int: Device index
                    - str: Device name pattern to match (case-insensitive)

        Raises:
            OSError: If no audio device is available or the selected device cannot be opened.
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
    """
    Matches audio clips using MFCC (Mel-Frequency Cepstral Coefficients) similarity.
    
    MFCC features capture the spectral characteristics of audio, making them useful
    for comparing speech patterns. This class compares a candidate audio clip against
    a stored reference audio to determine similarity.
    
    The similarity calculation uses both mean and standard deviation of MFCC features,
    weighted to produce a percentage score (0-100).
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """
        Initialize the word matcher.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
        """
        self.sample_rate: int = sample_rate
        self.reference_mfcc_mean: Optional[np.ndarray] = None
        self.reference_mfcc_std: Optional[np.ndarray] = None
        self.reference_word: Optional[str] = None

    def extract_mfcc(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract MFCC features from audio.
        
        Uses 20 MFCC coefficients with a 512-sample FFT window and 160-sample hop.
        These parameters are optimized for speech at 16kHz.

        Args:
            audio: Audio samples as a 1D numpy array
            
        Returns:
            Tuple of (mfcc_mean, mfcc_std) arrays, each with 20 coefficients
        """
        # Extract MFCC features using librosa
        # n_mfcc=20: number of MFCC coefficients (standard for speech)
        # n_fft=512: FFT window size (~32ms at 16kHz)
        # hop_length=160: hop between windows (~10ms at 16kHz)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=20, n_fft=512, hop_length=160
        )
        # Compute temporal statistics (mean and std across time frames)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return mfcc_mean, mfcc_std

    def set_reference(self, audio: np.ndarray, word_name: str = "target") -> None:
        """
        Set the reference word to match against.
        
        Args:
            audio: Audio samples as a 1D numpy array
            word_name: Optional name for the reference word (for debugging)
        """
        self.reference_word = word_name
        self.reference_mfcc_mean, self.reference_mfcc_std = self.extract_mfcc(audio)

    def load_reference_from_file(self, filepath: str, word_name: str = "target") -> None:
        """
        Load reference word from an audio file.
        
        Args:
            filepath: Path to a WAV file containing the reference audio
            word_name: Optional name for the reference word (for debugging)
        """
        audio, _ = librosa.load(filepath, sr=self.sample_rate)
        self.set_reference(audio, word_name)

    def calculate_similarity(self, audio: np.ndarray) -> float:
        """
        Calculate similarity between audio and reference word.
        
        Uses cosine similarity on MFCC features, combining mean (70% weight)
        and standard deviation (30% weight) for robustness. The result is
        scaled to produce intuitive percentage values.

        Args:
            audio: Audio samples as a 1D numpy array
            
        Returns:
            Similarity percentage (0-100), where 100 is an exact match
            
        Raises:
            ValueError: If no reference word has been set
        """
        if self.reference_mfcc_mean is None:
            raise ValueError("No reference word set. Call set_reference() first.")

        candidate_mfcc_mean, candidate_mfcc_std = self.extract_mfcc(audio)

        # Calculate cosine similarity for mean and std features
        # cosine() returns distance (0=identical), so we subtract from 1
        sim_mean = 1 - cosine(self.reference_mfcc_mean, candidate_mfcc_mean)
        sim_std = 1 - cosine(self.reference_mfcc_std, candidate_mfcc_std)

        # Weighted combination (mean more important than std)
        combined_similarity = sim_mean * 0.7 + sim_std * 0.3
        similarity_percent = combined_similarity * 100
        
        # Non-linear scaling to spread out the similarity values
        scaled_similarity = (similarity_percent**1.5) / (100**0.5)

        return scaled_similarity

    def matches(self, audio: np.ndarray, threshold: float = 75.0) -> tuple[bool, float]:
        """
        Check if audio matches reference word above the threshold.
        
        Args:
            audio: Audio samples as a 1D numpy array
            threshold: Minimum similarity percentage to consider a match (0-100)
            
        Returns:
            Tuple of (is_match, similarity_percentage)
        """
        similarity = self.calculate_similarity(audio)
        return similarity >= threshold, similarity


class WakeWord:
    """
    Wake word detector for speech recognition.

    Listens for a specific wake word using a unified three-level detection approach:
    1. Silence and speech timing - Validates audio segment boundaries
    2. MFCC cosine similarity - Fast acoustic matching against reference WAV
    3. Whisper transcription - Final confirmation with word count validation

    Whisper transcription is always used for reliable detection (bundled by default).
    
    Recommended usage: Use a 2-word phrase (e.g., "ok computer") but record only the
    main/major word (e.g., "computer") for the reference WAV. This provides better
    practical results even though it's suboptimal for MFCC similarity checking.

    STT Backends:
    - "bundled" (default): Auto-downloads and runs mini_transcriber locally
    - "external": Uses an external Whisper API at the specified URL

    Example:
        >>> # Two-word phrase with main word reference (recommended)
        >>> detector = WakeWord(
        ...     textword="ok computer",
        ...     wavword="computer.wav",  # Only the main word
        ...     numberofwords=2
        ... )
        >>> detector.waitforit()  # Blocking detection
        'ok computer'

        >>> # Multi-word detection with external Whisper
        >>> detector = WakeWord(
        ...     textword="hey assistant",
        ...     wavword="assistant.wav",  # Only the main word
        ...     numberofwords=2,
        ...     stt_backend="external",
        ...     external_whisper_url="http://localhost:8085"
        ... )
    """

    @classmethod
    def ensure_bundled_transcriber(cls) -> bool:
        """
        Ensure the bundled mini_transcriber is running.

        This is a class method that can be called before creating WakeWord instances
        to pre-start the bundled transcriber for faster initialization.

        Downloads and starts mini_transcriber if not already running.
        Returns True if transcriber is available, False otherwise.

        Example:
            >>> WakeWord.ensure_bundled_transcriber()  # Pre-start transcriber
            >>> detector = WakeWord("ok computer", "computer.wav", 2)
        """
        transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"

        # Check if transcriber is already running
        try:
            response = requests.get(f"{transcriber_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("Bundled transcriber already running")
                return True
        except Exception:
            pass

        # Try to download and start mini_transcriber
        transcriber_dir = os.path.join(
            os.path.expanduser("~"), ".easywakeword", "mini_transcriber"
        )

        if not os.path.exists(transcriber_dir):
            logger.info("Downloading mini_transcriber for first-time setup...")
            os.makedirs(os.path.dirname(transcriber_dir), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", MINI_TRANSCRIBER_REPO, transcriber_dir],
                    check=True,
                    capture_output=True,
                )
                logger.info("mini_transcriber downloaded successfully")
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode() if e.stderr else "No error details"
                logger.error(f"Failed to download mini_transcriber: git clone failed")
                logger.error(f"  Command: git clone {MINI_TRANSCRIBER_REPO} {transcriber_dir}")
                logger.error(f"  Error: {stderr_msg}")
                return False

        # Install dependencies if requirements.txt exists
        req_file = os.path.join(transcriber_dir, "requirements.txt")
        if os.path.exists(req_file):
            logger.info("Installing dependencies...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_file],
                    cwd=transcriber_dir,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error("Failed to install dependencies")
                return False

        # Start mini_transcriber
        try:
            app_path = os.path.join(transcriber_dir, "app.py")
            if not os.path.exists(app_path):
                logger.error(f"mini_transcriber app.py not found at {app_path}")
                return False

            env = os.environ.copy()
            env["PORT"] = str(DEFAULT_MINI_TRANSCRIBER_PORT)
            process = subprocess.Popen(
                [sys.executable, app_path],
                cwd=transcriber_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for transcriber to start
            logger.info("Waiting for mini_transcriber to start...")
            for i in range(60):  # Wait up to 60 seconds
                time.sleep(1)
                try:
                    response = requests.get(f"{transcriber_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("mini_transcriber started successfully")
                        return True
                except Exception:
                    pass
                if (i + 1) % 10 == 0:
                    logger.info(f"Still waiting for mini_transcriber... ({i+1}/60)")

            logger.error("Timed out waiting for mini_transcriber to start")
            return False

        except Exception as e:
            logger.error(f"Failed to start mini_transcriber: {e}")
            return False

    def __init__(
        self,
        textword: str,
        wavword: str,
        numberofwords: int = 2,
        timeout: int = 30,
        external_whisper_url: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        device: Optional[Union[int, str]] = None,
        similarity_threshold: float = 75.0,
        stt_backend: str = "bundled",
        pre_speech_silence: float = DEFAULT_PRE_SPEECH_SILENCE,
        speech_duration_min: float = DEFAULT_SPEECH_DURATION_MIN,
        speech_duration_max: float = DEFAULT_SPEECH_DURATION_MAX,
        post_speech_silence: float = DEFAULT_POST_SPEECH_SILENCE,
        buffer_seconds: int = DEFAULT_BUFFER_SECONDS,
        verbose: bool = False,
        session_headers: Optional[Dict[str, str]] = None,
        retry_count: int = DEFAULT_RETRY_COUNT,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        """
        Initialize the wake word detector.

        The detector uses three-level checking for reliable detection:
        1. Silence and speech timing - Validates audio segment boundaries
        2. MFCC cosine similarity - Fast acoustic matching against reference
        3. Whisper transcription - Final confirmation with word count validation

        By default, the bundled Whisper backend is used for transcription.
        
        Recommended: Use a 2-word phrase (e.g., "ok computer") but record only the
        main word (e.g., "computer") for the reference WAV. This works better in 
        practice even though it's suboptimal for MFCC similarity checking.

        Args:
            textword: The text phrase to detect (e.g., "ok computer")
            wavword: Path to reference WAV file for MFCC matching. Should contain
                     only the main/major word of the phrase (e.g., "computer.wav"
                     for "ok computer"). This provides better practical results.
            numberofwords: Number of words expected in the wake phrase (default: 2).
                           Used to filter/validate Whisper transcription output.
            timeout: Timeout in seconds (default: 30)
            external_whisper_url: URL of external Whisper API for transcription
                                  (e.g., "http://localhost:8085" for mini_transcriber).
                                  Only used if stt_backend is not "bundled".
            callback: Callback function for async detection. Called with detected text.
            device: Audio input device specification:
                    - None: Auto-select best available device
                    - int: Device index
                    - str: Device name pattern to match (case-insensitive), or
                           magic words: "best", "first", "default"
            similarity_threshold: MFCC similarity threshold (0-100, default: 75).
                                  Higher values reduce false positives but may miss detections.
            stt_backend: STT backend to use (default: "bundled"):
                         - "bundled": Auto-download and run mini_transcriber locally (recommended)
                         - "external": Use external_whisper_url for transcription
            pre_speech_silence: Minimum silence duration before speech starts (seconds).
                                Default: 0.8s. Override for custom tuning.
            speech_duration_min: Minimum speech duration in seconds (default: 0.3s).
                                 Override for custom tuning.
            speech_duration_max: Maximum speech duration in seconds (default: 2.0s).
                                 Override for custom tuning.
            post_speech_silence: Minimum silence duration after speech ends (seconds).
                                 Default: 0.4s. Override for custom tuning.
            buffer_seconds: Audio buffer size in seconds (default: 10). Larger buffers use
                            more memory but allow detection of longer wake phrases.
            verbose: Enable verbose logging output (default: False). When True, logs detailed
                     information about detection process to the module logger.
            session_headers: Optional dictionary of HTTP headers to include in all requests
                             to the transcription API. Useful for authentication tokens.
            retry_count: Number of retries for transient network failures (default: 3).
            retry_backoff: Initial backoff delay in seconds for retries (default: 0.5).
                           Uses exponential backoff: retry_backoff * 2^attempt.

        Raises:
            FileNotFoundError: If the wavword file does not exist.
            ValueError: If numberofwords is less than 1, buffer_seconds is not positive,
                        or retry parameters are invalid.
        """
        # Validate parameters
        if numberofwords < 1:
            raise ValueError("numberofwords must be at least 1")
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive")
        if retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be non-negative")
        if pre_speech_silence <= 0:
            raise ValueError("pre_speech_silence must be positive")
        if speech_duration_min <= 0:
            raise ValueError("speech_duration_min must be positive")
        if speech_duration_max <= 0:
            raise ValueError("speech_duration_max must be positive")
        if speech_duration_min > speech_duration_max:
            raise ValueError("speech_duration_min must be <= speech_duration_max")
        if post_speech_silence <= 0:
            raise ValueError("post_speech_silence must be positive")

        self.textword = textword.lower().strip()
        self.wavword = wavword
        self.numberofwords = numberofwords
        self.timeout = timeout
        self.external_whisper_url = external_whisper_url
        self.callback = callback
        self.device = AudioDeviceManager.select_device(device)  # Resolve device specification
        self.similarity_threshold = similarity_threshold
        self.stt_backend = stt_backend
        self.buffer_seconds = buffer_seconds
        self.verbose = verbose
        self.retry_count = retry_count
        self.retry_backoff = retry_backoff

        # Use the provided fixed silence durations directly
        # (auto-calculation methods are kept for backwards compatibility but not called)
        self.pre_speech_silence = pre_speech_silence
        self.speech_duration_min = speech_duration_min
        self.speech_duration_max = speech_duration_max
        self.post_speech_silence = post_speech_silence

        self._sound_buffer: Optional[SoundBuffer] = None
        self._matcher: Optional[WordMatcher] = None
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._session = requests.Session()
        self._bundled_transcriber_process: Optional[subprocess.Popen] = None
        self._transcriber_url: Optional[str] = None

        # Configure session headers for API authentication
        if session_headers:
            self._session.headers.update(session_headers)

        # Set up transcriber URL based on backend configuration
        self._setup_stt_backend()

        self._log(f"Initialized WakeWord detector for '{self.textword}'")

    def _log(self, message: str, level: int = logging.DEBUG) -> None:
        """
        Log a message if verbose mode is enabled.

        Args:
            message: Message to log
            level: Logging level (default: DEBUG)
        """
        if self.verbose:
            logger.log(level, message)

    def configure_session(self, headers: Optional[Dict[str, str]] = None,
                          auth: Optional[tuple] = None) -> None:
        """
        Configure the HTTP session used for transcription API requests.

        This method allows setting authentication headers or credentials for
        cloud Whisper APIs that require authentication.

        Args:
            headers: Dictionary of HTTP headers to add to the session.
                     Common headers include "Authorization", "X-API-Key", etc.
            auth: Tuple of (username, password) for HTTP Basic authentication.

        Example:
            >>> detector.configure_session(
            ...     headers={"Authorization": "Bearer YOUR_API_KEY"}
            ... )
        """
        if headers:
            self._session.headers.update(headers)
            self._log(f"Updated session headers: {list(headers.keys())}")
        if auth:
            self._session.auth = auth
            self._log("Configured session authentication")

    def check_transcriber_health(self) -> Dict[str, Union[bool, str, float]]:
        """
        Check the health status of the transcription service.

        Returns:
            Dictionary with health status information:
            - "healthy": bool indicating if service is reachable and responding
            - "url": str of the transcriber URL being checked
            - "latency_ms": float response time in milliseconds (if healthy)
            - "error": str error message (if unhealthy)
            - "status": str status message (if healthy but no transcriber configured)

        Example:
            >>> health = detector.check_transcriber_health()
            >>> if health["healthy"]:
            ...     print(f"Service OK, latency: {health['latency_ms']:.1f}ms")
            ... else:
            ...     print(f"Service unhealthy: {health['error']}")
        """
        result: Dict[str, Union[bool, str, float]] = {
            "healthy": False,
            "url": self._transcriber_url or "None (MFCC-only mode)",
        }

        if self._transcriber_url is None:
            result["healthy"] = True
            result["status"] = "No transcriber configured (MFCC-only mode)"
            return result

        try:
            start_time = time.time()
            response = self._session.get(
                f"{self._transcriber_url}/health",
                timeout=5
            )
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                result["healthy"] = True
                result["latency_ms"] = latency_ms
                self._log(f"Transcriber health check OK: {latency_ms:.1f}ms")
            else:
                result["error"] = f"HTTP {response.status_code}"
                self._log(f"Transcriber health check failed: HTTP {response.status_code}",
                          logging.WARNING)

        except requests.exceptions.Timeout:
            result["error"] = "Connection timeout"
            self._log("Transcriber health check failed: timeout", logging.WARNING)
        except requests.exceptions.ConnectionError as e:
            result["error"] = f"Connection error: {str(e)}"
            self._log(f"Transcriber health check failed: {e}", logging.WARNING)
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            self._log(f"Transcriber health check failed: {e}", logging.ERROR)

        return result

    def _calculate_detection_thresholds(self) -> None:
        """
        Calculate speech detection thresholds based on reference audio or heuristics.

        Note: This method is kept for backwards compatibility but is not called
        automatically during initialization. Fixed defaults are now used instead.

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

        Note: This method is kept for backwards compatibility. It can be used
        to analyze reference audio if needed for custom threshold tuning.

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
            self._log(f"Could not analyze reference audio duration: {e}", logging.WARNING)

        return None

    def _set_thresholds_from_audio_duration(self, audio_duration: float) -> None:
        """
        Set detection thresholds based on analyzed reference audio duration.

        Note: This method is kept for backwards compatibility. Fixed defaults
        are now used in the constructor instead.

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

        Note: This method is kept for backwards compatibility. Fixed defaults
        are now used in the constructor instead.
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
        """Set up the STT backend based on configuration.
        
        Whisper transcription is always used for reliable detection.
        The bundled backend is the default and recommended option.
        """
        if self.stt_backend == "bundled":
            # Will set up bundled mini_transcriber on first use
            self._transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"
        elif self.stt_backend == "external" and self.external_whisper_url:
            # Use provided external URL
            self._transcriber_url = self.external_whisper_url
        else:
            # Default to bundled for reliability
            self.stt_backend = "bundled"
            self._transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"
            self._log("No valid STT backend specified, defaulting to bundled", logging.INFO)

    def _ensure_bundled_transcriber(self) -> bool:
        """
        Ensure the bundled mini_transcriber is running.

        Downloads and starts mini_transcriber if not already running.

        Returns:
            True if transcriber is available, False otherwise.
        """
        if self.stt_backend != "bundled":
            return self._transcriber_url is not None

        # Check if transcriber is already running
        try:
            response = self._session.get(
                f"{self._transcriber_url}/health", timeout=2
            )
            if response.status_code == 200:
                self._log("Bundled transcriber already running")
                return True
        except Exception:
            pass

        # Try to download and start mini_transcriber
        transcriber_dir = os.path.join(
            os.path.expanduser("~"), ".easywakeword", "mini_transcriber"
        )

        if not os.path.exists(transcriber_dir):
            self._log("Downloading mini_transcriber for first-time setup...", logging.INFO)
            os.makedirs(os.path.dirname(transcriber_dir), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", MINI_TRANSCRIBER_REPO, transcriber_dir],
                    check=True,
                    capture_output=True,
                )
                self._log("mini_transcriber downloaded successfully")
            except subprocess.CalledProcessError as e:
                stderr_msg = e.stderr.decode() if e.stderr else "No error details"
                self._log(f"Failed to download mini_transcriber: git clone failed", logging.ERROR)
                self._log(f"  Command: git clone {MINI_TRANSCRIBER_REPO} {transcriber_dir}", logging.ERROR)
                self._log(f"  Error: {stderr_msg}", logging.ERROR)
                return False

        # Start mini_transcriber
        try:
            app_path = os.path.join(transcriber_dir, "app.py")
            if not os.path.exists(app_path):
                self._log(f"mini_transcriber app.py not found at {app_path}", logging.ERROR)
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
            self._log("Waiting for mini_transcriber to start...")
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                try:
                    response = self._session.get(
                        f"{self._transcriber_url}/health", timeout=2
                    )
                    if response.status_code == 200:
                        self._log("mini_transcriber started successfully", logging.INFO)
                        return True
                except Exception:
                    pass
                self._log(f"Waiting for mini_transcriber... ({i+1}/30)")

            self._log("Timed out waiting for mini_transcriber to start", logging.ERROR)
            return False

        except Exception as e:
            self._log(f"Failed to start mini_transcriber: {e}", logging.ERROR)
            return False

    def _initialize_audio(self) -> None:
        """Initialize audio buffer and word matcher."""
        if self._sound_buffer is None:
            self._sound_buffer = SoundBuffer(seconds=self.buffer_seconds, device=self.device)
            self._log(f"Audio buffer initialized: {self.buffer_seconds}s")
        if self._matcher is None:
            self._matcher = WordMatcher(sample_rate=SoundBuffer.FREQUENCY)
            self._matcher.load_reference_from_file(self.wavword, self.textword)
            self._log(f"Word matcher initialized with reference: {self.wavword}")

    def _wait_for_buffer(self) -> None:
        """Wait for the audio buffer to be filled."""
        while not self._sound_buffer.is_buffer_full():
            if self._stop_event.is_set():
                return
            time.sleep(0.1)

    def _transcribe_audio(self, audio_samples: np.ndarray) -> Optional[str]:
        """
        Send audio to STT engine and get transcription with retry logic.

        Args:
            audio_samples: numpy array of audio samples

        Returns:
            Transcription text or None if failed after all retries.
        """
        if self._transcriber_url is None:
            return None

        # Ensure bundled transcriber is running if needed
        if self.stt_backend == "bundled":
            if not self._ensure_bundled_transcriber():
                return None

        # Normalize and boost audio
        audio_samples = audio_samples - np.mean(audio_samples)
        max_val = np.max(np.abs(audio_samples))
        if max_val > 0:
            audio_samples = audio_samples / max_val
        audio_samples = audio_samples * 1.5
        audio_samples = np.clip(audio_samples, -1.0, 1.0)

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.retry_count):
            try:
                # Convert audio to WAV format in memory
                buffer = io.BytesIO()
                sf.write(buffer, audio_samples, SoundBuffer.FREQUENCY, format="WAV")
                buffer.seek(0)

                files = {"file": ("audio.wav", buffer, "audio/wav")}
                data = {"model": "tiny", "language": "en", "initial_prompt": f"Wake word: {self.textword}"}

                self._log(f"Sending transcription request (attempt {attempt + 1}/{self.retry_count})")

                response = self._session.post(
                    f"{self._transcriber_url}/transcribe", files=files, data=data, timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    self._log(f"Transcription result: '{text}'")
                    return text
                else:
                    self._log(f"Transcription request failed with HTTP {response.status_code}",
                              logging.WARNING)
                    last_error = f"HTTP {response.status_code}"

            except requests.exceptions.Timeout as e:
                self._log(f"Transcription request timed out (attempt {attempt + 1})", logging.WARNING)
                last_error = "Timeout"
            except requests.exceptions.ConnectionError as e:
                self._log(f"Connection error during transcription (attempt {attempt + 1}): {e}",
                          logging.WARNING)
                last_error = f"Connection error: {e}"
            except Exception as e:
                self._log(f"Unexpected error during transcription (attempt {attempt + 1}): {e}",
                          logging.WARNING)
                last_error = f"Error: {e}"

            # Exponential backoff before retry (capped at 30 seconds)
            if attempt < self.retry_count - 1:
                backoff_delay = min(self.retry_backoff * (2 ** attempt), 30.0)
                self._log(f"Retrying in {backoff_delay:.1f}s...")
                time.sleep(backoff_delay)

        self._log(f"All {self.retry_count} transcription attempts failed. Last error: {last_error}",
                  logging.ERROR)
        return None

    def _detect_word(self) -> Optional[str]:
        """
        Internal method to detect a single word occurrence.

        Uses three-level checking for reliable detection:
        1. Silence and speech timing - Validates audio segment boundaries
        2. MFCC cosine similarity - Fast acoustic matching against reference
        3. Whisper transcription - Final confirmation with word count validation

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
                        # Level 1: Silence/speech timing passed - extract audio
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
                            self._log("Audio segment too long, skipping")
                            state = "waiting"
                            continue

                        # Level 2: MFCC cosine similarity check
                        matches, similarity = self._matcher.matches(
                            word_audio, threshold=self.similarity_threshold
                        )
                        self._log(f"MFCC similarity: {similarity:.1f}%")

                        if matches:
                            # Level 3: Whisper transcription confirmation (required)
                            transcription = self._transcribe_audio(word_audio)
                            if transcription:
                                transcription_clean = transcription.strip().lower().rstrip(".,!?;:")
                                target_words = self.textword.split()
                                transcription_words = transcription_clean.split()

                                # Validate word count matches expectation
                                if len(transcription_words) != self.numberofwords:
                                    self._log(
                                        f"Word count mismatch: expected {self.numberofwords}, "
                                        f"got {len(transcription_words)} ('{transcription_clean}')"
                                    )
                                    state = "waiting"
                                    continue

                                # Check if all target words appear in transcription
                                if all(word in transcription_words for word in target_words):
                                    self._log(f"Wake word detected: '{transcription}'")
                                    return transcription
                                else:
                                    self._log(
                                        f"Target words not found in transcription: "
                                        f"'{transcription_clean}' vs '{self.textword}'"
                                    )
                            else:
                                self._log("Transcription failed, cannot confirm detection")

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
        if hasattr(self, '_stop_event') and self._stop_event:
            self._stop_event.set()
        if hasattr(self, '_listen_thread') and self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)
        if hasattr(self, '_sound_buffer') and self._sound_buffer:
            self._sound_buffer.stop()
        if hasattr(self, '_listening'):
            self._listening = False

        # Stop bundled transcriber if we started it
        if hasattr(self, '_bundled_transcriber_process') and self._bundled_transcriber_process:
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
        if hasattr(self, '_session') and self._session:
            self._session.close()
