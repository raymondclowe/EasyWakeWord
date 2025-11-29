"""
Example: Basic synchronous wake word detection using EasyWakeWord.

Requires: A reference WAV file (e.g., okgoogle.wav) in the current directory.
"""
from easywakeword import WakeWord

if __name__ == "__main__":
    my_wake_word = WakeWord(
        textword="ok google",
        wavword="okgoogle.wav",
        numberofwords=2,
        timeout=10,
        stt_backend=None
    )
    try:
        print("Say the wake word...")
        detected = my_wake_word.waitforit()
        print(f"Detected: {detected}")
    except TimeoutError:
        print("Wake word not detected in time.")
