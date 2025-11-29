# EasyWakeWord

A simple way to do Wake Words in Python. Tested on Windows. YMMV.

## Overview

EasyWakeWord is a Python module that allows you to create wake word detection systems. It uses MFCC (Mel-Frequency Cepstral Coefficients) for audio matching and integrates with Whisper-based speech-to-text services for transcription confirmation.

## Installation

```bash
pip install easywakeword
```

Or install from source:

```bash
git clone https://github.com/raymondclowe/EasyWakeWord.git
cd EasyWakeWord
pip install -e .
```

## Usage

### Basic Usage (Synchronous)

```python
from easywakeword import WakeWord

# Create a wake word detector
my_wake_word = WakeWord(
    textword="ok google",          # The text phrase to detect
    wavword="okgoogle.wav",        # Path to reference audio file
    numberofwords=2,               # Number of words in the wake phrase
    timeout=30,                    # Timeout in seconds
    externalwisperurl="http://localhost:8085"  # Optional: external Whisper API URL
)

# Wait for the wake word (blocking call)
words_received_text = my_wake_word.waitforit()
print(f"Detected: {words_received_text}")
```

### Async Usage (with Callback)

```python
from easywakeword import WakeWord

def on_wake_word_detected(text):
    print(f"Wake word detected: {text}")

# Create a wake word detector with callback
my_wake_word = WakeWord(
    textword="hey computer",
    wavword="heycomputer.wav",
    numberofwords=2,
    timeout=30,
    callback=on_wake_word_detected
)

# Start listening (non-blocking)
my_wake_word.start()

# ... do other things ...

# Stop listening when done
my_wake_word.stop()
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `textword` | str | Yes | The text phrase to detect (e.g., "ok google") |
| `wavword` | str | Yes | Path to reference WAV file for MFCC matching |
| `numberofwords` | int | Yes | Number of words in the wake phrase |
| `timeout` | int | No | Timeout in seconds (default: 30) |
| `externalwisperurl` | str | No | URL of external Whisper API for transcription |
| `callback` | callable | No | Callback function for async detection |

## Methods

### `waitforit()`

Blocking method that waits for the wake word to be detected.

**Returns:** `str` - The transcribed text that was detected

**Raises:** `TimeoutError` - If timeout is reached without detection

### `start()`

Start listening for the wake word in a background thread (non-blocking).
Requires a callback to be set.

### `stop()`

Stop the background listening thread.

### `is_listening()`

Check if the detector is currently listening.

**Returns:** `bool` - True if listening, False otherwise

## Requirements

- Python 3.10+
- numpy
- sounddevice
- librosa
- scipy
- soundfile
- requests

## How It Works

1. **Audio Capture**: Continuously captures audio from the microphone using a circular buffer
2. **Silence Detection**: Dynamically adjusts silence threshold to detect speech boundaries
3. **MFCC Matching**: Extracts MFCC features from captured audio and compares with reference
4. **Transcription Confirmation**: Uses Whisper-based STT to confirm the detected wake word
5. **Callback/Return**: Returns the transcribed text or calls the callback function

## License

MIT License - See [LICENSE](LICENSE) for details
