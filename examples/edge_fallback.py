"""
Example: Edge device with LAN fallback for wake word detection.
Tries LAN Whisper first, falls back to MFCC-only if network fails.
"""
from easywakeword import WakeWord

class EdgeDetector:
    """Edge device with LAN fallback"""
    def __init__(self):
        # Try LAN first, fallback to MFCC-only
        try:
            self.detector = WakeWord(
                textword="device wake",
                wavword="device_wake.wav",
                numberofwords=2,
                external_whisper_url="http://192.168.1.100:8085",
                timeout=60
            )
            # Test connection
            import requests
            requests.get(self.detector._transcriber_url + "/health", timeout=2)
            print("✓ Using LAN Whisper")
        except:
            # Fallback to MFCC-only
            self.detector = WakeWord(
                textword="device wake",
                wavword="device_wake.wav",
                numberofwords=2,
                stt_backend=None,  # MFCC-only
                timeout=60
            )
            print("⚠ Using MFCC-only mode (no LAN connection)")
    
    def start(self):
        detected = self.detector.waitforit()
        return detected
