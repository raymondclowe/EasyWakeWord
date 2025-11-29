"""
Example: Using a cloud Whisper API for confirmation.
Best for: High-scale applications, distributed systems, managed infrastructure.
"""
from easywakeword import WakeWord

# Connect to cloud Whisper API (e.g., Replicate, OpenAI, etc.)
detector = WakeWord(
    textword="activate",
    wavword="activate.wav",
    numberofwords=1,
    timeout=30,
    external_whisper_url="https://api.replicate.com/v1/whisper",  # Example
    # Note: Add API authentication in headers via session configuration
)

print("Listening (cloud Whisper)...")
detected = detector.waitforit()
print(f"Transcription: {detected}")
