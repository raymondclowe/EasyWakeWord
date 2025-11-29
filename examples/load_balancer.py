"""
Example: Distributed system with load balancing across multiple Whisper servers.
"""
import random
from easywakeword import WakeWord

class LoadBalancedDetector:
    """Distribute transcription across multiple servers"""
    def __init__(self, transcription_servers):
        self.servers = transcription_servers  # ["http://server1:8085", ...]
        
    def create_detector(self):
        # Round-robin or random selection
        server = random.choice(self.servers)
        return WakeWord(
            textword="system ready",
            wavword="system_ready.wav",
            numberofwords=2,
            external_whisper_url=server,
            timeout=30
        )
    
    def listen(self):
        detector = self.create_detector()
        return detector.waitforit()

# Usage
balancer = LoadBalancedDetector([
    "http://whisper-1.local:8085",
    "http://whisper-2.local:8085",
    "http://whisper-3.local:8085"
])
