import os
import sys
import pytest
import numpy as np
from easywakeword.transcriber import WhisperTranscriber

def test_transcriber_initialization():
    """Test that the transcriber initializes correctly."""
    print("\nTesting transcriber initialization...")
    transcriber = WhisperTranscriber(model_name="tiny", device="cpu", verbose=True)
    assert transcriber.model_name == "tiny"
    assert transcriber.device == "cpu"
    print("Initialization successful.")

def test_model_loading():
    """Test that the model loads correctly."""
    print("\nTesting model loading (this may take a while)...")
    transcriber = WhisperTranscriber(model_name="tiny", device="cpu", verbose=True)
    transcriber.load_model()
    assert transcriber.model is not None
    print("Model loaded successfully.")

def test_transcription():
    """Test transcription with dummy audio."""
    print("\nTesting transcription with silence...")
    transcriber = WhisperTranscriber(model_name="tiny", device="cpu", verbose=True)
    transcriber.load_model()
    
    # Create 1 second of silence
    audio = np.zeros(16000, dtype=np.float32)
    
    result = transcriber.transcribe(audio)
    print(f"Transcription result: '{result}'")
    # Silence usually results in empty string or hallucinations, but it shouldn't crash
    assert isinstance(result, str)

if __name__ == "__main__":
    # Manual run
    try:
        test_transcriber_initialization()
        test_model_loading()
        test_transcription()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
