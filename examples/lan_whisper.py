"""
Example: Using a dedicated LAN Whisper server for confirmation.
Best for: Production environments, multiple clients, load balancing.

This example connects to a mini_transcriber server running on your local network.
You would typically run mini_transcriber on a machine with a GPU for faster processing.

Requires:
- A reference WAV file with your wake word
- mini_transcriber running on the LAN (e.g., 192.168.1.100:8085)

To set up mini_transcriber:
  git clone https://github.com/raymondclowe/mini_transcriber.git
  cd mini_transcriber
  pip install -r requirements.txt
  python app.py  # Starts server on port 8085
"""
from easywakeword import WakeWord

if __name__ == "__main__":
    # Connect to a dedicated Whisper server on your LAN
    detector = WakeWord(
        textword="ok assistant",
        wavword="assistant.wav",   # Reference WAV with main word only
        numberofwords=2,
        timeout=30,
        stt_backend="external",    # Use external Whisper server
        external_whisper_url="http://192.168.1.100:8085"  # Your server
    )
    
    # Check if the server is reachable
    health = detector.check_transcriber_health()
    if health["healthy"]:
        print(f"Whisper server OK, latency: {health.get('latency_ms', 0):.1f}ms")
    else:
        print(f"Warning: Whisper server not reachable: {health.get('error')}")
        print("Make sure mini_transcriber is running on the server.")
        exit(1)

    print("Listening for 'ok assistant' (LAN Whisper)...")
    try:
        detected = detector.waitforit()
        print(f"Detected: {detected}")
    except TimeoutError:
        print("Wake word not detected in time.")
    finally:
        detector.stop()
