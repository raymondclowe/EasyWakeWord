"""
Example: Using a cloud Whisper API for confirmation.
Best for: High-scale applications, distributed systems, managed infrastructure.

This example shows how to use an external cloud Whisper API
with authentication headers for the transcription step.

Requires:
- A reference WAV file with your wake word
- A valid cloud Whisper API endpoint and API key
"""
from easywakeword import WakeWord

if __name__ == "__main__":
    # Connect to cloud Whisper API (e.g., Replicate, OpenAI, etc.)
    detector = WakeWord(
        textword="activate",
        wavword="activate.wav",
        numberofwords=1,
        timeout=30,
        stt_backend="external",  # Use external Whisper server
        external_whisper_url="https://api.replicate.com/v1/whisper",  # Example URL
    )
    
    # Configure API authentication
    detector.configure_session(
        headers={"Authorization": "Bearer YOUR_API_TOKEN"}
    )
    
    # Check if the API is reachable
    health = detector.check_transcriber_health()
    if not health["healthy"]:
        print(f"Warning: Cloud API not reachable: {health.get('error')}")
        print("Continuing anyway...")

    print("Listening for 'activate' (cloud Whisper)...")
    try:
        detected = detector.waitforit()
        print(f"Detected: {detected}")
    except TimeoutError:
        print("Wake word not detected in time.")
    finally:
        detector.stop()
