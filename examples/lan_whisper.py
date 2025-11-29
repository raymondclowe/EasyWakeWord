"""
Example: Using a dedicated LAN Whisper server for confirmation.
Best for: Production environments, multiple clients, load balancing.
"""
from easywakeword import WakeWord

# Connect to a dedicated Whisper server on your LAN
# (e.g., mini_transcriber running on a GPU server at 192.168.1.100)
detector = WakeWord(
    textword="ok assistant",
    wavword="ok_assistant.wav",
    numberofwords=2,
    timeout=30,
    external_whisper_url="http://192.168.1.100:8085"  # Your transcription server
)

print("Listening (LAN Whisper)...")
detected = detector.waitforit()
print(f"Transcription: {detected}")
