"""
Example: Basic synchronous wake word detection using EasyWakeWord.

This example demonstrates the recommended two-word phrase detection pattern.
The reference WAV should contain only the main word (e.g., "google.wav" for "ok google").

Requires:
- A reference WAV file with the main word (e.g., google.wav for "ok google")
- The bundled mini_transcriber will auto-download on first use
"""
from easywakeword import WakeWord

if __name__ == "__main__":
    # Two-word phrase detection (recommended)
    # WAV file contains only the main word "google"
    # Full phrase "ok google" is validated by Whisper
    my_wake_word = WakeWord(
        textword="ok google",      # Full phrase to detect
        wavword="google.wav",      # Reference WAV with main word only
        numberofwords=2,           # Expected word count in transcription
        timeout=30,                # Detection timeout in seconds
        stt_backend="bundled"      # Auto-downloads mini_transcriber
    )
    
    try:
        print("Say 'ok google'...")
        detected = my_wake_word.waitforit()
        print(f"Detected: {detected}")
    except TimeoutError:
        print("Wake word not detected in time.")
    finally:
        my_wake_word.stop()
