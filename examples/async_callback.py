"""
Example: Async wake word detection with callback using EasyWakeWord.

Requires: A reference WAV file (e.g., heycomputer.wav) in the current directory.
"""
import time
from easywakeword import WakeWord

def on_detected(text):
    print(f"Wake word detected: {text}")

if __name__ == "__main__":
    my_wake_word = WakeWord(
        textword="hey computer",
        wavword="heycomputer.wav",
        numberofwords=2,
        timeout=15,
        stt_backend=None,
        callback=on_detected
    )
    print("Say the wake word (async)...")
    my_wake_word.start()
    try:
        # Listen for 15 seconds
        time.sleep(15)
    finally:
        my_wake_word.stop()
        print("Stopped listening.")
