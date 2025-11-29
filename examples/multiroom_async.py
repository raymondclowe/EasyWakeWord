"""
Example: Multi-room smart home async detection with callbacks.
Best for: Always-on listening, event-driven architectures, smart home systems.
"""
from easywakeword import WakeWord
import time

def handle_wake_word(text):
    print(f"🎤 Wake word detected: {text}")
    # Trigger your action here
    # e.g., start recording, activate assistant, etc.

# Room A: Kitchen - using LAN Whisper
kitchen_detector = WakeWord(
    textword="hey kitchen",
    wavword="hey_kitchen.wav",
    numberofwords=2,
    timeout=60,
    external_whisper_url="http://192.168.1.100:8085",
    callback=handle_wake_word
)

# Room B: Living room - using bundled Whisper
living_room_detector = WakeWord(
    textword="hey living room",
    wavword="hey_living_room.wav",
    numberofwords=3,
    timeout=60,
    stt_backend="bundled",
    callback=handle_wake_word
)

# Start all detectors (non-blocking)
kitchen_detector.start()
living_room_detector.start()

print("🏠 Smart home listening on multiple detectors...")
try:
    # Your main application loop
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    kitchen_detector.stop()
    living_room_detector.stop()
