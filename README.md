# EasyWakeWord

**Reliable wake word detection for Python applications**

A high-performance wake word detection library that combines MFCC-based audio matching with Whisper transcription for reliable detection. Uses a unified three-level checking approach for production reliability.

## Overview

EasyWakeWord provides robust wake word detection with:

- **Three-level checking** - Silence/speech timing → MFCC matching → Whisper confirmation
- **Bundled Whisper** - Auto-downloads mini_transcriber on first use (no setup required)
- **Two-word phrase focus** - Use phrases like "ok computer" with the main word as reference WAV
- **Fixed defaults** - Sensible silence durations out of the box, user-overridable
- **Cross-platform support** - Windows, Linux, macOS

## Installation

```bash
# Using uv (recommended)
uv add easywakeword

# Or using pip
pip install easywakeword
```

Install from source for development:

```bash
git clone https://github.com/raymondclowe/EasyWakeWord.git
cd EasyWakeWord
pip install -e .
```

## Quick Start

### Basic Example (Two-Word Phrase - Recommended)

```python
from easywakeword import WakeWord

# Two-word phrase with main word as reference WAV (recommended)
# The WAV file should contain only the main word ("computer")
# but the full phrase ("ok computer") is used for Whisper validation
detector = WakeWord(
    textword="ok computer",
    wavword="computer.wav"  # Only the main word
)

print("Listening for 'ok computer'...")
detected = detector.waitforit()
print(f"Detected: {detected}")
```

### Single Word Detection

```python
from easywakeword import WakeWord

# Single word detection (less reliable but simpler)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1
)

print("Listening for 'hello'...")
detected = detector.waitforit()
print(f"Detected: {detected}")
```

### Async Detection with Callback

```python
from easywakeword import WakeWord
import time

def on_wake_word(text):
    print(f"Wake word detected: {text}")

detector = WakeWord(
    textword="ok computer",
    wavword="computer.wav",  # Only the main word
    callback=on_wake_word
)

detector.start()
print("Listening asynchronously...")
try:
    time.sleep(30)
finally:
    detector.stop()
```

## API Reference

### WakeWord Class

```python
WakeWord(
    textword: str,
    wavword: str,
    numberofwords: int = 2,
    timeout: int = 30,
    external_whisper_url: Optional[str] = None,
    callback: Optional[Callable[[str], None]] = None,
    device: Optional[int] = None,
    similarity_threshold: float = 75.0,
    stt_backend: str = "bundled",
    pre_speech_silence: float = 0.8,
    speech_duration_min: float = 0.3,
    speech_duration_max: float = 2.0,
    post_speech_silence: float = 0.4,
    buffer_seconds: int = 10,
    verbose: bool = False,
    session_headers: Optional[Dict[str, str]] = None,
    retry_count: int = 3,
    retry_backoff: float = 0.5
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `textword` | str | Required | The full text phrase to detect (e.g., "ok computer") |
| `wavword` | str | Required | Path to reference WAV file containing only the main word (e.g., "computer.wav" for "ok computer") |
| `numberofwords` | int | 2 | Number of words expected in the transcription. Used to validate Whisper output. |
| `timeout` | int | 30 | Timeout in seconds for detection |
| `stt_backend` | str | "bundled" | STT backend: `"bundled"` (auto mini_transcriber) or `"external"` (use external_whisper_url) |
| `external_whisper_url` | str | None | URL of external Whisper API (only used when stt_backend="external") |
| `callback` | callable | None | Callback function `fn(text: str)` for async detection |
| `device` | int, str, or None | None | Audio device: index, name pattern, or magic word ("best", "first", "default") |
| `similarity_threshold` | float | 75.0 | MFCC similarity threshold (0-100). Higher = fewer false positives |
| `pre_speech_silence` | float | 0.8 | Min silence before speech (seconds). Override for custom tuning. |
| `speech_duration_min` | float | 0.3 | Min speech duration (seconds). Override for custom tuning. |
| `speech_duration_max` | float | 2.0 | Max speech duration (seconds). Override for custom tuning. |
| `post_speech_silence` | float | 0.4 | Min silence after speech (seconds). Override for custom tuning. |
| `buffer_seconds` | int | 10 | Audio buffer size in seconds |
| `verbose` | bool | False | Enable verbose logging |
| `session_headers` | dict | None | HTTP headers for transcription API requests |
| `retry_count` | int | 3 | Number of retries for network failures |
| `retry_backoff` | float | 0.5 | Initial backoff delay (seconds) for retries |

#### Methods

##### `waitforit() -> str`

**Blocking** method that waits for the wake word to be detected using all three levels of checking:
1. Silence/speech timing validation
2. MFCC cosine similarity matching
3. Whisper transcription with word count validation

- **Returns**: `str` - The transcribed text that was detected
- **Raises**: `TimeoutError` - If timeout is reached without detection

```python
detected_text = detector.waitforit()
```

##### `start() -> None`

Start listening for the wake word in a **background thread** (non-blocking).

- **Requires**: `callback` parameter must be set during initialization
- **Raises**: `ValueError` - If no callback is set

```python
detector.start()  # Non-blocking, runs in background
```

##### `stop() -> None`

Stop the background listening thread and clean up resources.

```python
detector.stop()
```

##### `is_listening() -> bool`

Check if the detector is currently listening.

- **Returns**: `bool` - True if listening, False otherwise

```python
if detector.is_listening():
    print("Actively listening...")
```

##### `check_transcriber_health() -> dict`

Check the health status of the transcription service.

- **Returns**: `dict` - Health status with keys:
  - `healthy`: bool - True if service is reachable
  - `url`: str - The transcriber URL being checked
  - `latency_ms`: float - Response time in milliseconds (if healthy)
  - `error`: str - Error message (if unhealthy)

```python
health = detector.check_transcriber_health()
if health["healthy"]:
    print(f"Service OK, latency: {health['latency_ms']:.1f}ms")
else:
    print(f"Service unhealthy: {health['error']}")
```

##### `configure_session(headers: dict = None, auth: tuple = None) -> None`

Configure the HTTP session for transcription API requests.

- **Args**:
  - `headers`: Dictionary of HTTP headers (e.g., `{"Authorization": "Bearer token"}`)
  - `auth`: Tuple of (username, password) for HTTP Basic authentication

```python
# Add API key for cloud Whisper service
detector.configure_session(headers={"X-API-Key": "your-api-key"})

# Or use Bearer token authentication
detector.configure_session(headers={"Authorization": "Bearer your-token"})
```

## Advanced Configuration

### Tuning Detection Sensitivity

Adjust `similarity_threshold` to balance false positives vs. false negatives:

```python
# More permissive - catches more, but more false positives
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    similarity_threshold=65.0  # Lower = more permissive
)

# More strict - fewer false positives, might miss some
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    similarity_threshold=85.0  # Higher = more strict
)
```

**Recommended values**:
- **60-70**: Noisy environments, accent variations
- **75-80**: General purpose (default 75)
- **80-90**: Clean environments, strict false positive control

### Tuning Speech Detection Parameters

EasyWakeWord uses fixed silence durations by default that work well for typical wake words. You can override them for custom tuning:

```python
# Default values (work well for most cases)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav"
)
# pre_speech_silence=0.8, speech_duration_min=0.3, speech_duration_max=2.0, post_speech_silence=0.4

# Custom values for specific needs
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    pre_speech_silence=1.0,    # Longer silence requirement before speech
    speech_duration_min=0.5,   # Minimum speech duration
    speech_duration_max=1.5,   # Maximum speech duration
    post_speech_silence=0.6    # Longer trailing silence
)
```

**Parameter Guide**:
- **`pre_speech_silence`**: Silence required before speech starts (default: 0.8s)
- **`speech_duration_min/max`**: Expected speech duration range (default: 0.3s-2.0s)
- **`post_speech_silence`**: Silence required after speech ends (default: 0.4s)

### Using External Whisper Server

For production deployments, you may want to use a dedicated Whisper server:

```python
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    stt_backend="external",
    external_whisper_url="http://192.168.1.100:8085"
)
```

### Audio Device Selection

EasyWakeWord provides intelligent audio device selection:

```python
# Test all devices and pick the one with highest audio level (recommended!)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    device="best"  # Tests all mics, picks one with strongest signal
)

# Find first working device (with audio signal)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    device="first"  # First device that shows audio activity
)

# Use system default device explicitly
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    device="default"  # System default input device
)
```

**Magic Word Behavior**:
- **`"best"`**: Tests all mics, picks one with strongest signal
- **`"first"`**: First device that shows audio activity
- **`"default"`**: System's default input device

You can also match devices by name pattern or use device indices:

```python
# Match by name pattern (case-insensitive)
detector = WakeWord(textword="hello", wavword="hello.wav", device="USB")
detector = WakeWord(textword="hello", wavword="hello.wav", device="blue")

# By index
detector = WakeWord(textword="hello", wavword="hello.wav", device=2)
```

### Device Utilities

```bash
# List all available devices
python -m easywakeword.device_utils list

# Test a specific device
python -m easywakeword.device_utils test "USB"
python -m easywakeword.device_utils test "best"
```

For advanced Whisper setups (custom models, languages):

```python
# When using external_whisper_url, the detector sends:
# - Audio file (WAV format, 16kHz)
# - Model preference: "tiny" (default)
# - Language: "en"
# - Initial prompt with wake word context

# To customize, you'll need to modify the transcription server
# or implement a custom STT endpoint compatible with mini_transcriber API
```

### Recording Reference Audio

Create high-quality reference WAV files:

```python
import sounddevice as sd
import soundfile as sf
import numpy as np

def record_reference(filename, duration=2.0, sample_rate=16000):
    """Record reference audio for wake word"""
    print(f"Recording for {duration} seconds... Say your wake word!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    # Normalize and save
    audio = audio.flatten()
    audio = audio / np.max(np.abs(audio))
    sf.write(filename, audio, sample_rate)
    print(f"Saved to {filename}")

# Usage
record_reference("hello.wav", duration=2.0)
```

**Tips for quality reference audio**:
- Record in the same environment where detection will occur
- Speak naturally at normal volume
- Use a single word for best results (recommended)
- Include minimal silence before/after the word

## How It Works

### Three-Level Detection Pipeline

EasyWakeWord uses a unified three-level detection approach for reliability:

1. **Silence/Speech Timing**: Monitors audio for valid speech patterns
   - Requires minimum silence before speech starts (default: 0.8s)
   - Speech duration must be within expected range (default: 0.3-2.0s)
   - Requires silence after speech ends (default: 0.4s)

2. **MFCC Cosine Similarity**: Fast acoustic matching
   - Extracts 20 MFCC coefficients from audio segment
   - Compares with reference WAV using cosine similarity
   - Configurable threshold (default: 75%)
   - Runs in < 10ms

3. **Whisper Transcription**: Final confirmation
   - Sends audio to Whisper for transcription
   - Validates word count matches expectation
   - Checks all target words appear in transcription
   - Dramatically reduces false positives (< 1%)

### Performance Characteristics

| Component | Typical Latency |
|-----------|----------------|
| MFCC matching | < 10ms |
| Bundled Whisper | ~500ms |
| External Whisper | ~200ms |

### Resource Usage

- CPU: ~2-5% (detection) + transcription overhead
- RAM: ~50MB (detection) + model size (if bundled)
- Network: ~200KB per detection (audio upload to Whisper)

## System Requirements

- Python 3.10+
- Operating System: Windows, Linux, macOS
- Audio input device (microphone)
- Git (for bundled Whisper auto-download)
- 4GB+ RAM recommended

## Troubleshooting

### Detection Issues

**Wake word not detected**:
- Lower `similarity_threshold` to 65-70 for testing
- Record in same environment as detection occurs
- Use a single word for best results

**Too many false positives**:
- Increase `similarity_threshold` to 80-85
- Verify reference audio is clean
- Whisper confirmation is enabled by default to reduce false positives

### Transcription Issues

**Bundled mode fails to start**:
- Verify Git is installed: `git --version`
- Check disk space (needs ~1GB for models)
- Ensure port 8085 is available

**External Whisper connection fails**:
- Test server reachability: `curl http://server-ip:8085/health`
- Check firewall rules on server

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest

# Run specific test file
pytest tests/test_wakeword_basic.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=easywakeword
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

```bash
# Clone and setup
git clone https://github.com/raymondclowe/EasyWakeWord.git
cd EasyWakeWord
pip install -e .

# Run tests
pytest
```

## FAQ

**Q: Does this work offline?**

A: Yes! The bundled Whisper mode works offline after first-time model download.

**Q: Can I use different wake words on different devices?**

A: Yes! Each `WakeWord` instance is independent. Use different reference audio files and text words for each device.

**Q: Why does bundled mode take so long to start the first time?**

A: First-time setup downloads mini_transcriber and Whisper models. Subsequent starts are fast.

**Q: Can I run this on a Raspberry Pi?**

A: Yes! The bundled mode will work well on Pi 4/5 with 4GB+ RAM.

**Q: How do I debug detection issues?**

A: Enable verbose logging with `verbose=True`, lower `similarity_threshold` for testing, and check your reference audio quality.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and notable changes.

## Related Projects

- [mini_transcriber](https://github.com/raymondclowe/mini_transcriber) - Lightweight Whisper API server
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Alternative ML-based wake word detection
- [Porcupine](https://github.com/Picovoice/porcupine) - Commercial wake word engine with free tier

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/raymondclowe/EasyWakeWord/issues)
- **Discussions**: [GitHub Discussions](https://github.com/raymondclowe/EasyWakeWord/discussions)

---

**Built with ❤️ for voice-enabled applications**
