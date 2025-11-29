"""
Example: Full voice assistant pipeline with async detection and command queue.
Best for: Complete voice assistant systems, commercial products.
"""
from easywakeword import WakeWord
import queue
import threading

class VoiceAssistant:
    def __init__(self):
        self.command_queue = queue.Queue()
        self.detector = WakeWord(
            textword="hey assistant",
            wavword="hey_assistant.wav",
            numberofwords=2,
            timeout=300,  # 5 minutes
            external_whisper_url="http://192.168.1.100:8085",  # LAN Whisper
            similarity_threshold=80.0,  # Tune for your environment
            callback=self._on_wake_word
        )
    
    def _on_wake_word(self, text):
        print(f"🎤 Listening for command...")
        # Here you'd typically:
        # 1. Play a confirmation sound [not implemented: audio feedback]
        # 2. Start recording the full command [not implemented: command recording]
        # 3. Process the command with full Whisper transcription [not implemented: command processing pipeline]
        # 4. Execute the action [not implemented: action execution]
        self.command_queue.put(("wake", text))
    
    def start(self):
        self.detector.start()
        print("🤖 Voice assistant ready")
        
        # Process commands in a separate thread
        while True:
            try:
                event_type, data = self.command_queue.get(timeout=1)
                if event_type == "wake":
                    print(f"Processing wake: {data}")
                    # Process command here
            except queue.Empty:
                continue
    
    def stop(self):
        self.detector.stop()

# Usage
assistant = VoiceAssistant()
try:
    assistant.start()
except KeyboardInterrupt:
    assistant.stop()
