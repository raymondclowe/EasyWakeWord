"""
Example: Async wake word detection with callback using EasyWakeWord.

This example shows how to run wake word detection in a background thread
while your application does other work. The callback is called each time
the wake word is detected.

Requires:
- A reference WAV file with the main word (e.g., computer.wav for "hey computer")
- The bundled mini_transcriber will auto-download on first use
"""
import time
from easywakeword import WakeWord


def on_detected(text):
    """Callback function called when wake word is detected."""
    print(f"Wake word detected: {text}")


if __name__ == "__main__":
    # Two-word phrase detection with callback
    my_wake_word = WakeWord(
        textword="hey computer",   # Full phrase to detect
        wavword="computer.wav",    # Reference WAV with main word only
        numberofwords=2,           # Expected word count
        timeout=30,                # Timeout for each detection attempt
        stt_backend="bundled",     # Auto-downloads mini_transcriber
        callback=on_detected       # Called when wake word detected
    )
    
    print("Listening for 'hey computer' (async mode)...")
    print("Press Ctrl+C to stop.")
    
    my_wake_word.start()
    try:
        # Listen for 60 seconds (or until Ctrl+C)
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        my_wake_word.stop()
        print("Stopped listening.")
