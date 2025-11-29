"""
Example: Multi-stage pipeline - fast MFCC filter, then Whisper confirmation.
"""
from easywakeword import WakeWord

class MultiStageDetector:
    """Fast MFCC filter → Whisper confirmation → Action"""
    def __init__(self):
        # Stage 1: Fast MFCC-only detector
        self.fast_detector = WakeWord(
            textword="assistant",
            wavword="assistant.wav",
            numberofwords=1,
            stt_backend=None,  # No transcription
            similarity_threshold=70.0  # Permissive
        )
        
        # Stage 2: Whisper confirmation
        self.confirming_detector = WakeWord(
            textword="assistant",
            wavword="assistant.wav",
            numberofwords=1,
            external_whisper_url="http://192.168.1.100:8085",
            similarity_threshold=75.0
        )
    
    def listen(self):
        # Fast pre-filter (< 10ms)
        print("Stage 1: Fast MFCC filter...")
        potential_match = self.fast_detector.waitforit()
        
        if potential_match:
            # Confirm with Whisper (~200ms)
            print("Stage 2: Whisper confirmation...")
            confirmed = self.confirming_detector.waitforit()
            return confirmed
        
        return None
