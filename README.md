# EasyWakeWord

**Production-ready wake word detection for Python applications**

A high-performance, flexible wake word detection library that combines efficient MFCC-based audio matching with optional Whisper transcription for confirmation. Designed for real-world production deployments across edge devices, servers, and distributed systems.

## Overview

EasyWakeWord provides a robust wake word detection system with:

- **Fast, lightweight pre-filtering** using MFCC acoustic matching (runs locally, no network)
- **Optional transcription confirmation** via Whisper (local, LAN, or cloud)
- **Multiple deployment modes** optimized for different production scenarios
- **Async and sync APIs** for easy integration
- **Resource-efficient** circular audio buffering with dynamic silence detection
- **Cross-platform support** (Windows, Linux, macOS)

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

## Production Architecture & STT Backend Options

EasyWakeWord is designed for **three-tier detection**:

1. **Pre-filter (Always Local)**: Fast MFCC matching runs on-device to identify potential matches (< 10ms latency)
2. **Confirmation (Configurable)**: Optional Whisper transcription confirms matches to reduce false positives
3. **Action**: Execute your application logic

### STT Backend Modes

Choose the right backend for your deployment scenario:

| Backend | Use Case | Latency | Privacy | Setup Complexity |
|---------|----------|---------|---------|------------------|
| **None** (MFCC-only) | Ultra-low latency, edge devices, privacy-critical | < 10ms | 100% local | Minimal |
| **Bundled** | Quick start, prototyping, single-machine | ~500ms | 100% local | Auto (downloads on first run) |
| **LAN Whisper** | Production server, single detector (CPU-only) | ~200ms | Network-local | Medium (setup CPU transcription server) |
| **Cloud Whisper** | High accuracy, scalable infrastructure | ~1-2s | Data sent to cloud | Easy (use API key) |

## Usage Examples

### Basic Blocking Example

```python
from easywakeword import WakeWord

# Blocking (synchronous) usage
detector = WakeWord(
    textword="hey device",
    wavword="hey_device.wav",
    numberofwords=2,
    timeout=30,
    stt_backend=None
)

print("Listening for wake word...")
detected = detector.waitforit()
print(f"Detected: {detected}")
```

### Basic Async Example

```python
from easywakeword import WakeWord
import time

def on_wake_word(text):
    print(f"Wake word detected: {text}")

detector = WakeWord(
    textword="hey computer",
    wavword="hey_computer.wav",
    numberofwords=2,
    timeout=30,
    stt_backend=None,
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
    numberofwords: int,
    timeout: int = 30,
    external_whisper_url: Optional[str] = None,
    callback: Optional[Callable[[str], None]] = None,
    device: Optional[int] = None,
    similarity_threshold: float = 75.0,
    stt_backend: Optional[str] = None,
    buffer_seconds: int = 10,
    verbose: bool = False,
    session_headers: Optional[Dict[str, str]] = None,
    retry_count: int = 3,
    retry_backoff: float = 0.5
)
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `textword` | str | ✓ | - | The text phrase to detect (e.g., "ok google") |
| `wavword` | str | ✓ | - | Path to reference WAV file for MFCC matching |
| `numberofwords` | int | ✓ | - | Number of words in the wake phrase |
| `timeout` | int | | 30 | Timeout in seconds for detection |
| `external_whisper_url` | str | | None | URL of external Whisper API (LAN or cloud) |
| `stt_backend` | str or None | | None | STT backend: `"bundled"` for auto mini_transcriber, `None` for external or MFCC-only |
| `callback` | callable | | None | Callback function `fn(text: str)` for async detection |
| `device` | int, str, or None | | None | Audio device: index, name pattern, or magic word ("best", "first", "default") |
| `similarity_threshold` | float | | 75.0 | MFCC similarity threshold (0-100). Higher = fewer false positives |
| `pre_speech_silence` | float or None | | None | Min silence before speech (auto-calculated from reference audio) |
| `speech_duration_min` | float or None | | None | Min speech duration (auto-calculated from reference audio) |
| `speech_duration_max` | float or None | | None | Max speech duration (auto-calculated from reference audio) |
| `post_speech_silence` | float or None | | None | Min silence after speech (auto-calculated from reference audio) |
| `buffer_seconds` | int | | 10 | Audio buffer size in seconds. Larger = more memory, longer phrases |
| `verbose` | bool | | False | Enable verbose logging via Python logging module |
| `session_headers` | dict | | None | HTTP headers for transcription API requests (e.g., for auth) |
| `retry_count` | int | | 3 | Number of retries for transient network failures |
| `retry_backoff` | float | | 0.5 | Initial backoff delay (seconds) for retries (exponential) |

#### Methods

##### `waitforit() -> str`

**Blocking** method that waits for the wake word to be detected.

- **Returns**: `str` - The transcribed text (if STT enabled) or the textword (if MFCC-only)
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
    textword="hey there",
    wavword="hey_there.wav",
    numberofwords=2,
    similarity_threshold=65.0  # Lower = more permissive
)

# More strict - fewer false positives, might miss some
detector = WakeWord(
    textword="hey there",
    wavword="hey_there.wav",
    numberofwords=2,
    similarity_threshold=85.0  # Higher = more strict
)
```

**Recommended values**:
- **60-70**: Noisy environments, accent variations, casual use
- **75-80**: General purpose (default 75)
- **80-90**: Clean environments, strict false positive control

### Tuning Speech Detection Parameters

EasyWakeWord automatically calculates speech detection thresholds based on your reference audio, but you can override them for fine-tuning:

```python
# Auto-calculated thresholds (recommended)
detector = WakeWord(
    textword="ok computer",
    wavword="ok_computer.wav",
    numberofwords=2
)
print(f"Pre-speech silence: {detector.pre_speech_silence:.1f}s")
print(f"Speech duration: {detector.speech_duration_min:.1f}s - {detector.speech_duration_max:.1f}s")
print(f"Post-speech silence: {detector.post_speech_silence:.1f}s")

# Manual override for custom tuning
detector = WakeWord(
    textword="ok computer",
    wavword="ok_computer.wav",
    numberofwords=2,
    pre_speech_silence=0.8,    # 0.8s silence before speech
    speech_duration_min=0.4,   # 0.4-1.2s speech duration
    speech_duration_max=1.2,
    post_speech_silence=0.4    # 0.4s silence after speech
)
```

**Parameter Guide**:
- **`pre_speech_silence`**: How much silence before speech starts. Longer = fewer false starts from noise
- **`speech_duration_min/max`**: Expected speech duration range. Should match your wake phrase length
- **`post_speech_silence`**: How much silence after speech ends. Ensures complete phrase capture

**Auto-calculation Logic**:
1. **From reference audio**: Analyzes actual speech duration in your WAV file
2. **Fallback heuristics**: Estimates based on word count and syllable analysis
3. **Final defaults**: Conservative values that work for most wake phrases

### Audio Device Selection

EasyWakeWord provides intelligent audio device selection with automatic defaults and name-based matching. No more fragile device indices that change between machines!

#### Magic Word Selection

Use intelligent device selection with magic words:

```python
# Test all devices and pick the one with highest audio level (recommended!)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="best"  # Tests all mics, picks one with strongest signal
)

# Find first working device (with audio signal)
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="first"  # First device that shows audio activity
)

# Use system default device explicitly
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="default"  # System default input device
)
```

**Magic Word Behavior**:
- **`"best"`**: Records 100ms from each microphone device, selects the one with highest RMS audio level (indicating strongest/loudest signal). Filters out system audio capture devices (like "Stereo Mix" or "What U Hear").
- **`"first"`**: Tests microphone devices in order, selects the first one that shows any audio activity. Filters out system audio capture devices.
- **`"default"`**: Uses the system's default input device (same as `device=None` but explicit)

#### Device Selection by Name

Match devices by name pattern (case-insensitive):

```python
# Use any device containing "USB"
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="USB"
)

# Use device with "Blue" in name (e.g., "Blue Yeti")
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="blue"
)

# Exact match
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device="Microphone (Realtek High Definition Audio)"
)
```

#### Device Selection by Index

For advanced users, you can still specify device indices directly:

```python
# Use device index 2
detector = WakeWord(
    textword="hello",
    wavword="hello.wav",
    numberofwords=1,
    device=2
)
```

#### Device Utilities

Use the included device utilities to list and test devices:

```bash
# List all available devices
python -m easywakeword.device_utils list

# Test default device
python -m easywakeword.device_utils test

# Test specific device by index
python -m easywakeword.device_utils test 1

# Test device by name pattern
python -m easywakeword.device_utils test "USB"
python -m easywakeword.device_utils test "microphone"

# Test with magic words
python -m easywakeword.device_utils test "best"      # Test all, select best by audio level
python -m easywakeword.device_utils test "first"     # Test in order, select first working
python -m easywakeword.device_utils test "default"   # Test system default
```

**Device Testing Output**:
```
Testing device 1...
Recording 2 seconds of audio... Speak into the microphone.
Recording complete!
RMS Level: 0.1234
Peak Level: 0.5678
SNR Estimate: 15.2 dB  [not implemented: SNR estimation in device testing]
Audio levels look good!
```

#### Manual Device Listing

For programmatic device enumeration:

```python
from easywakeword.wakeword import AudioDeviceManager

# List all devices with details
devices = AudioDeviceManager.list_devices()
for device in devices:
    print(f"{device['index']}: {device['name']} ({device['hostapi']})")

# Pretty-print device list
AudioDeviceManager.print_device_list()
```

**Example Output**:
```
Available audio input devices:
------------------------------------------------------------
0: Microphone (Realtek High Definition Audio) (MME)
    Host API: MME
    Channels: 1, Sample Rate: 44100.0

1: Microphone (USB Audio Device) (MME) (default)
    Host API: MME
    Channels: 2, Sample Rate: 48000.0
```

**Note**: Only true microphone devices are listed. System audio capture devices (like "Stereo Mix", "What U Hear", or devices with "speaker" in the name) are automatically filtered out to prevent selecting output devices for wake word detection.

#### Troubleshooting Device Issues

**Problem**: Wrong device selected
- Use `python -m easywakeword.device_utils list` to see available devices
- Specify device by name pattern instead of index
- Test with `python -m easywakeword.device_utils test "device name"`

**Problem**: No audio input devices found
- Check microphone is connected and enabled
- Verify microphone permissions (especially on Linux/macOS)
- Try different USB ports or audio interfaces

**Problem**: Audio levels too low/high
- Use device testing utility to check levels
- Adjust microphone volume in system settings
- Move closer to microphone or reduce background noise

### Custom Whisper Configuration

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
record_reference("my_wake_word.wav", duration=2.0)
```

**Tips for quality reference audio**:
- Record in the same environment where detection will occur
- Speak naturally at normal volume
- Include minimal silence before/after the word
- Record multiple variations if needed (use the clearest one)

## Production Deployment Patterns

### Pattern 1: Edge Device with Fallback

```python
class EdgeDetector:
    """Edge device with LAN fallback"""
    def __init__(self):
        # Try LAN first, fallback to MFCC-only
        try:
            self.detector = WakeWord(
                textword="device wake",
                wavword="device_wake.wav",
                numberofwords=2,
                external_whisper_url="http://192.168.1.100:8085",
                timeout=60
            )
            # Test connection
            import requests
            requests.get(self.detector._transcriber_url + "/health", timeout=2)
            print("✓ Using LAN Whisper")
        except:
            # Fallback to MFCC-only
            self.detector = WakeWord(
                textword="device wake",
                wavword="device_wake.wav",
                numberofwords=2,
                stt_backend=None,  # MFCC-only
                timeout=60
            )
            print("⚠ Using MFCC-only mode (no LAN connection)")
    
    def start(self):
        detected = self.detector.waitforit()
        return detected
```

### Pattern 2: Distributed System with Load Balancing

```python
import random

class LoadBalancedDetector:
    """Distribute transcription across multiple servers"""
    def __init__(self, transcription_servers):
        self.servers = transcription_servers  # ["http://server1:8085", ...]
        
    def create_detector(self):
        # Round-robin or random selection
        server = random.choice(self.servers)
        return WakeWord(
            textword="system ready",
            wavword="system_ready.wav",
            numberofwords=2,
            external_whisper_url=server,
            timeout=30
        )
    
    def listen(self):
        detector = self.create_detector()
        return detector.waitforit()

# Usage
balancer = LoadBalancedDetector([
    "http://whisper-1.local:8085",
    "http://whisper-2.local:8085",
    "http://whisper-3.local:8085"
])
```

### Pattern 3: Multi-Stage Pipeline

```python
class MultiStageDetector:
    """Fast MFCC filter → Whisper confirmation → Action"""
    def __init__(self):
        # Stage 1: Fast MFCC-only detector
        self.fast_detector = WakeWord(
            textword="assistant",
            wavword="assistant.wav",
            numberofwords=1,
            stt_backend=None,  # No transcription
            similarity_threshold=70.0  # Permissive
        )
        
        # Stage 2: Whisper confirmation
        self.confirming_detector = WakeWord(
            textword="assistant",
            wavword="assistant.wav",
            numberofwords=1,
            external_whisper_url="http://192.168.1.100:8085",
            similarity_threshold=75.0
        )
    
    def listen(self):
        # Fast pre-filter (< 10ms)
        print("Stage 1: Fast MFCC filter...")
        potential_match = self.fast_detector.waitforit()
        
        if potential_match:
            # Confirm with Whisper (~200ms)
            print("Stage 2: Whisper confirmation...")
            confirmed = self.confirming_detector.waitforit()
            return confirmed
        
        return None
```

## Performance Characteristics

### Latency Breakdown

| Component | Typical Latency | Notes |
|-----------|----------------|-------|
| MFCC matching | < 10ms | Always local, CPU-efficient |
| Bundled Whisper | ~500ms | Local inference, model size dependent |
| LAN Whisper | ~200ms | Network + inference, CPU-only |
| Cloud Whisper | 1-2s | Network RTT + queue + inference |

### Resource Usage

**MFCC-only mode** (no transcription):
- CPU: ~2-5% (single core)
- RAM: ~50MB
- Network: 0 bytes

**With transcription**:
- CPU: ~2-5% (detection) + transcription overhead
- RAM: ~50MB (detection) + model size (if bundled)
- Network: ~200KB per detection (audio upload)

### Accuracy Trade-offs

| Mode | False Positive Rate | False Negative Rate | Notes |
|------|-------------------|-------------------|--------|
| MFCC-only (threshold=75) | ~5-10% | ~5% | Tune threshold for your environment |
| MFCC + Whisper | < 1% | ~5% | Whisper confirmation dramatically reduces FP |
| MFCC + Whisper (multi-accent) | < 2% | ~8% | Whisper helps with accents, MFCC may vary |

**Recommendations**:
1. **Start with**: `similarity_threshold=75`, enable Whisper confirmation
2. **Monitor**: False positive rate in your environment
3. **Adjust**: Increase threshold if too many FP, decrease if missing detections
4. **Test**: Record 100+ samples in production environment to establish baseline

## System Requirements

### Core Requirements

- Python 3.10+
- Operating System: Windows, Linux, macOS
- Audio input device (microphone)

### Python Dependencies

```toml
# Automatically installed via pip/uv
numpy >= 1.24.0
sounddevice >= 0.4.6
librosa >= 0.10.0
scipy >= 1.11.0
soundfile >= 0.12.0
requests >= 2.31.0
```

### Optional Components

**For bundled STT mode**:
- Git (to auto-download mini_transcriber)
- ~1GB disk space (Whisper model cache)
- 4GB+ RAM recommended

**For LAN Whisper mode**:
- Separate server running mini_transcriber or compatible API
- Local network connectivity

**For optimal performance**:
- Modern multi-core CPU for Whisper transcription (mini_transcriber is CPU-only)
- Low-latency network for LAN deployments

## How It Works

### Detection Pipeline

1. **Audio Capture**: Continuously captures audio from microphone using efficient circular buffer (10s rolling window)

2. **Silence Detection**: Dynamically adjusts silence threshold based on ambient noise to detect speech boundaries

3. **Word Segmentation**: Identifies audio segments that match expected duration (0.5-1.5s for typical wake words)

4. **MFCC Pre-filter**: Extracts 20 MFCC coefficients from audio segment and compares with reference using cosine similarity
   - Match threshold configurable (default 75%)
   - Runs in < 10ms on modern CPUs
   - Eliminates ~95% of non-matches before transcription

5. **Transcription Confirmation** (Optional): Sends matched audio to Whisper for text confirmation
   - Reduces false positives from ~5-10% to < 1%
   - Adds 200ms-2s latency depending on backend
   - Skipped if `stt_backend=None`

6. **Verification**: Checks if transcribed text contains all target wake words

7. **Callback/Return**: Returns transcribed text or calls callback function with result

### Architecture Diagram

```
┌─────────────────┐
│   Microphone    │
└────────┬────────┘
         │ Continuous audio stream
         ▼
┌─────────────────┐
│ Circular Buffer │ ◄── 10 second rolling window
│  (16kHz, Mono)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Silence Detector│ ◄── Dynamic threshold adjustment
└────────┬────────┘
         │ Speech segments (0.5-1.5s)
         ▼
┌─────────────────┐
│  MFCC Extractor │ ◄── 20 coefficients
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Similarity Check│ ◄── Cosine similarity vs reference
│  (threshold 75) │
└────────┬────────┘
         │ Potential matches only
         ▼
    ┌────────┐
    │ Match? │───── No ──► Discard (95% of segments)
    └───┬────┘
        │ Yes (5% of segments)
        ▼
┌─────────────────┐
│ Whisper API     │ ◄── Optional confirmation
│ (Local/LAN/Cloud│
└────────┬────────┘
         │ Transcribed text
         ▼
┌─────────────────┐
│ Text Matcher    │ ◄── Contains all wake words?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Callback/Return │ ◄── Final detection
└─────────────────┘
```

## Troubleshooting

### Detection Issues

**Problem**: Wake word not detected
- ✓ Verify reference audio quality: `librosa.load("your.wav", sr=16000)`
- ✓ Check audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- ✓ Lower `similarity_threshold` to 65-70 for testing
- ✓ Record in same environment as detection occurs
- ✓ Ensure 0.5-1.5 second word duration (longer/shorter may not trigger)

**Problem**: Too many false positives
- ✓ Increase `similarity_threshold` to 80-85
- ✓ Enable Whisper confirmation (`stt_backend="bundled"` or `external_whisper_url`)
- ✓ Verify reference audio is clean and matches target phrase exactly
- ✓ Use multi-word phrases instead of single words

**Problem**: Inconsistent detection
- ✓ Check ambient noise levels (detector auto-adjusts, but extreme noise affects performance)
- ✓ Ensure consistent speaking volume and speed
- ✓ Record multiple reference samples and use the best one
- ✓ Test `similarity_threshold` values between 70-80

### Transcription Issues

**Problem**: Bundled mode fails to start
- ✓ Verify Git is installed: `git --version`
- ✓ Check disk space (needs ~1GB for models)
- ✓ Ensure port 8085 is available
- ✓ Check firewall settings
- ✓ Review logs in `~/.easywakeword/mini_transcriber/`

**Problem**: LAN Whisper connection fails
- ✓ Test server reachability: `curl http://server-ip:8085/health`
- ✓ Verify mini_transcriber is running on server
- ✓ Check firewall rules on server
- ✓ Ensure correct URL format: `http://ip:port` (no trailing slash)

**Problem**: Transcription returns wrong text
- ✓ Whisper confidence may be low - check audio quality
- ✓ Use `initial_prompt` to guide Whisper (done automatically)
- ✓ Test with clearer pronunciation
- ✓ Consider using larger Whisper model on server (base, small, medium)

### Performance Issues

**Problem**: High CPU usage
- ✓ Normal: 2-5% for detection, spikes during MFCC processing
- ✓ If sustained high CPU: Check for audio device issues
- ✓ Consider reducing buffer size (not currently configurable, but can be modified in code)

**Problem**: High latency
- ✓ MFCC-only: Should be < 50ms total
- ✓ With transcription: Check network latency to Whisper server
- ✓ For bundled mode: Ensure sufficient RAM for model
- ✓ For LAN mode: Use a modern multi-core CPU server for faster inference (mini_transcriber is CPU-only)

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

### Writing Tests

Tests are located in `tests/` directory. Add tests for new features:

```python
# tests/test_my_feature.py
import pytest
from easywakeword import WakeWord

def test_my_feature():
    detector = WakeWord(
        textword="test",
        wavword="test.wav",
        numberofwords=1,
        stt_backend=None
    )
    # Your test assertions
    assert detector.is_listening() == False
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code style guidelines
- Development setup
- Pull request process
- Issue reporting guidelines

Quick start for contributors:

```bash
# Clone and setup
git clone https://github.com/raymondclowe/EasyWakeWord.git
cd EasyWakeWord
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

## FAQ

### General Questions

**Q: What's the difference between MFCC matching and Whisper transcription?**

A: MFCC (Mel-Frequency Cepstral Coefficients) is a fast acoustic fingerprint matching technique that runs locally in < 10ms. It's great for pre-filtering but can have false positives. Whisper is a full speech-to-text model that's slower (~200ms-2s) but much more accurate. EasyWakeWord uses MFCC first to filter, then optionally confirms with Whisper.

**Q: Can I use this for continuous listening (always-on)?**

A: Yes! Use the async API with callbacks. Set a long timeout (e.g., 300 seconds) and restart detection after each trigger. See the "Multi-Room Smart Home" example.

**Q: How accurate is MFCC-only mode?**

A: Typically 90-95% accuracy with default settings. False positive rate is ~5-10%, which can be reduced by increasing `similarity_threshold` or enabling Whisper confirmation.

**Q: Does this work offline?**

A: Yes! MFCC-only mode (`stt_backend=None`) works completely offline with no internet. Bundled mode (`stt_backend="bundled"`) also works offline after first-time model download.

### Deployment Questions

**Q: Should I use bundled, LAN, or cloud Whisper?**

A: 
- **Bundled**: Prototyping, single-user apps, privacy-critical (auto-downloads, runs locally)
- **LAN**: Production with a single detector, best latency/accuracy balance (setup mini_transcriber on local server, CPU-only)
- **Cloud**: Highest scale, managed infrastructure, don't need to manage servers
- **None** (MFCC-only): Ultra-low latency edge devices, no network available

**Q: Can multiple clients or detectors share one LAN Whisper server?**

A: No. Only one detector instance is expected to be running at a time. Multiple concurrent detectors or clients are not supported. mini_transcriber is CPU-only and not designed for concurrent use.

**Q: What's the recommended server spec for LAN Whisper?**

A:
- **Minimum**: 4 CPU cores, 4GB RAM (CPU inference, ~500ms latency)
- **Recommended**: 8+ CPU cores, 8GB+ RAM (CPU inference, ~200ms latency)
- Only one detector/client should connect at a time. For more, use a different architecture.

**Q: Can I use different wake words on different devices?**

A: Yes! Each `WakeWord` instance is independent. Use different reference audio files and text words for each device.

### Technical Questions

**Q: How do I create a good reference audio file?**

A: Record in the actual environment where detection will occur. Speak naturally, at normal volume, with minimal background noise. The recording should be 1-2 seconds long with the wake word centered. Use 16kHz sample rate, mono. See "Recording Reference Audio" section.

**Q: Why 0.5-1.5 seconds for word duration?**

A: The detector is optimized for typical wake phrases (1-3 words). Segments shorter than 0.5s are likely noise; longer than 1.5s are likely full sentences. This can be adjusted in the code if needed.

**Q: Can I detect multiple wake words simultaneously?**

A: No. Only one detector instance is expected to be running at a time. Multiple concurrent detectors are not supported.

**Q: How do I reduce memory usage?**

A: 
1. Use MFCC-only mode (no Whisper model in memory)
2. Use external Whisper API instead of bundled
3. Reduce audio buffer size (requires code modification)

**Q: Does this work with accents or different languages?**

A: MFCC matching is accent-agnostic (acoustic similarity). Whisper supports 90+ languages. Set the language in your mini_transcriber configuration. Note: You'll need to record reference audio in the target accent/language.

### Troubleshooting Questions

**Q: Why does bundled mode take so long to start the first time?**

A: First-time setup downloads mini_transcriber (~100MB) and Whisper models (~500MB). Subsequent starts are fast (~5-10 seconds).

**Q: What if my network is unreliable?**

A: Use a fallback pattern: Try LAN Whisper first, fall back to MFCC-only if network fails. See "Pattern 1: Edge Device with Fallback" example.

**Q: Can I run this on a Raspberry Pi?**

A: Yes! Use MFCC-only mode for best performance, or point to an external LAN Whisper server. Bundled mode will work but may be slow on Pi 3 or earlier (works well on Pi 4/5 with 4GB+ RAM).

**Q: How do I debug detection issues?**

A:
1. Enable verbose logging (add print statements in code)
2. Test MFCC similarity manually: Use `WordMatcher.calculate_similarity()` directly
3. Record failed detections and analyze audio
4. Check silence threshold: `detector._sound_buffer.silence_threshold`
5. Visualize MFCC features with librosa

## License

MIT License - See [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and notable changes.

## Project Documentation

- [LEARNINGS.md](LEARNINGS.md) - Technical discoveries and lessons learned
- [MY-MEMORIES.md](MY-MEMORIES.md) - Project context and goals
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## Related Projects

- [mini_transcriber](https://github.com/raymondclowe/mini_transcriber) - Lightweight Whisper API server
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) - Alternative ML-based wake word detection
- [Porcupine](https://github.com/Picovoice/porcupine) - Commercial wake word engine with free tier

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/raymondclowe/EasyWakeWord/issues)
- **Discussions**: [GitHub Discussions](https://github.com/raymondclowe/EasyWakeWord/discussions)
- **Email**: (Add your preferred contact method)

## Citation

If you use EasyWakeWord in academic work, please cite:

```bibtex
@software{easywakeword,
  title = {EasyWakeWord: Production-Ready Wake Word Detection for Python},
  author = {Raymond Lowe},
  year = {2024},
  url = {https://github.com/raymondclowe/EasyWakeWord}
}
```

---

**Built with ❤️ for voice-enabled applications**
