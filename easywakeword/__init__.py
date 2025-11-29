"""
EasyWakeWord - A simple wake word detection module for Python.

Example usage:
    from easywakeword import WakeWord

    # Synchronous usage
    my_wake_word = WakeWord(
        textword="ok google",
        wavword="okgoogle.wav",
        numberofwords=2,
        timeout=30
    )
    detected_text = my_wake_word.waitforit()

    # Async usage with callback
    def on_detected(text):
        print(f"Detected: {text}")

    my_wake_word = WakeWord(
        textword="hey computer",
        wavword="heycomputer.wav",
        numberofwords=2,
        callback=on_detected
    )
    my_wake_word.start()
"""

from .wakeword import WakeWord

__all__ = ["WakeWord"]
__version__ = "0.1.0"
