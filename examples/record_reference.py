#!/usr/bin/env python3
"""
Example: Recording reference audio for wake word detection.

This utility helps you create high-quality reference WAV files for use
with EasyWakeWord. The reference audio should contain only the main word
of your wake phrase (e.g., "computer" for "ok computer").

Tips for quality reference audio:
- Record in the same environment where detection will occur
- Speak naturally at normal volume
- Include minimal silence before/after the word
- Use a single word for best MFCC matching results

Usage:
    python record_reference.py computer.wav      # Record for 2 seconds
    python record_reference.py hello.wav 3       # Record for 3 seconds
"""

import sys
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install sounddevice soundfile")
    sys.exit(1)


def record_reference(filename: str, duration: float = 2.0, sample_rate: int = 16000):
    """
    Record reference audio for wake word detection.
    
    Args:
        filename: Output WAV file path
        duration: Recording duration in seconds
        sample_rate: Audio sample rate (16000 Hz recommended for wake word)
    """
    print(f"Recording for {duration} seconds...")
    print("Say your wake word NOW!")
    print()
    
    # Record audio
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    # Flatten and normalize
    audio = audio.flatten()
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Check audio level
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.01:
        print("Warning: Audio level is very low. Try speaking louder.")
    
    # Save to file
    sf.write(filename, audio, sample_rate)
    print(f"Saved to: {filename}")
    print(f"Duration: {duration:.1f}s, Sample rate: {sample_rate}Hz")
    print(f"Audio level (RMS): {rms:.4f}")


def list_audio_devices():
    """List available audio input devices."""
    print("Available audio input devices:")
    print("-" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  {i}: {device['name']}{default}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print()
        list_audio_devices()
        sys.exit(0)
    
    filename = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    
    if not filename.endswith('.wav'):
        filename += '.wav'
    
    print(f"Recording reference audio: {filename}")
    print(f"Duration: {duration} seconds")
    print()
    
    # Countdown
    print("Get ready...")
    import time
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    record_reference(filename, duration)
    
    print()
    print("Reference audio recorded successfully!")
    print()
    print("Next steps:")
    print(f"1. Test the audio: play {filename}")
    print("2. Use it with EasyWakeWord:")
    print()
    print(f'    detector = WakeWord(')
    print(f'        textword="ok {filename.replace(".wav", "")}",  # Your full phrase')
    print(f'        wavword="{filename}",')
    print(f'        numberofwords=2')
    print(f'    )')


if __name__ == "__main__":
    main()
