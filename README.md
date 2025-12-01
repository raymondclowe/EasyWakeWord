# EasyWakeWord

Easy-to-use wake word detection library using MFCC matching and Whisper STT validation.

## Features

- **MFCC-based audio matching** for fast wake word detection
- **Whisper STT validation** for accurate confirmation
- **Multiple reference audio support** for robust matching
- **Configurable parameters** for silence detection, word boundaries, and thresholds
- **Simple API** with sensible defaults

## Installation

```bash
pip install easywakeword
```

Or with uv:

```bash
uv add easywakeword
```

## Quick Start

```python
import easywakeword

# Create a wake word recognizer
wakeword = easywakeword.wakeword(
    wakewordstrings=["computer"],
    wakewordreferenceaudios=["path/to/computer.wav"]
)

# Wait for the wake word
print("Listening for wake word...")
result = wakeword.waitforit()
print(f"Wake word detected: {result}")
```

## Usage with Example Files

This package includes example wake word audio files:

```python
import easywakeword
import pkg_resources

# Get path to example files
example_male = pkg_resources.resource_filename('easywakeword', 'examples/example_computer_male.wav')

# Create recognizer with example
recognizer = easywakeword.wakeword(
    wakewordstrings=["computer"],
    wakewordreferenceaudios=[example_male]
)

# Wait for wake word
result = recognizer.waitforit()
```

## Advanced Configuration

```python
recognizer = easywakeword.Recogniser(
    wakewordstrings=["computer", "alexa"],
    wakewordreferenceaudios=["computer.wav", "alexa.wav"],
    threshold=75,              # MFCC matching threshold (0-100)
    wordsminmax=(1, 3),        # Expected word count range
    whispermodel="tiny",       # Whisper model size
    stt_minscore=85.0,         # STT confidence threshold
    min_silence_before=0.3,    # Silence before speech (seconds)
    min_sound=0.15,            # Minimum speech duration
    max_sound=1.5,             # Maximum speech duration
    min_trailing_silence=0.15, # Silence after speech
    allowed_other_words=[],    # Additional allowed words
    debug=False,               # Enable debug output
    debug_playback=False       # Play detected audio
)
```

## API Reference

### `easywakeword.wakeword()`

Factory function to create a `Recogniser` instance.

**Parameters:**
- `wakewordstrings` (list): List of wake word strings to detect
- `wakewordreferenceaudios` (list): List of paths to reference audio files
- `threshold` (float): MFCC matching threshold (default: 75)
- `wordsminmax` (tuple): Min/max word count (default: (1, 3))
- `whispermodel` (str): Whisper model name (default: "tiny")
- `stt_minscore` (float): STT confidence threshold (default: 85.0)
- `debug` (bool): Enable debug output (default: False)

### `Recogniser.waitforit()`

Wait for and detect the wake word.

**Returns:** Detected transcription string or None

## Requirements

- Python >= 3.8
- numpy
- sounddevice
- librosa
- soundfile
- scipy
- requests
- A running Whisper STT server (see below)

## Whisper STT Server

This library requires a Whisper STT server. The recommended server is **mini_transcriber**:

```bash
# Clone mini_transcriber
git clone https://github.com/raymondclowe/mini_transcriber.git
cd mini_transcriber

# Follow setup instructions in that repo
# The server should be accessible at mini_transcriber.local:8080
```

The default configuration expects:
- Hostname: `mini_transcriber.local`
- Port: `8080`
- Endpoint: POST with `audio` file field

Or configure a custom server URL:

```python
recognizer = easywakeword.Recogniser(
    whisperurl="http://your-server:8080",
    ...
)
```

## Example Files

The package includes three example audio files:
- `example_computer_male.wav`
- `example_computer_male_teen.wav`
- `example_computer_female..wav`

## Demo Script

Run the included demo:

```bash
easywakeword-demo
```

## License

MIT License - see LICENSE file for details

## Author

Raymond C Lowe

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Links

- Repository: https://github.com/raymondclowe/EasyWakeWord
- Issues: https://github.com/raymondclowe/EasyWakeWord/issues
