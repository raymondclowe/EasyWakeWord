"""
EasyWakeWord package — user-facing API.

This module exposes the primary classes and helpers needed by consumers.
"""
from .silence import SoundBuffer
from .matching import WordMatcher
from .recogniser import Recogniser
from .recogniser import Recogniser as WakeWord
from .transcription import transcribe_audio, resolve_stt_ip, STT_HOSTNAME, STT_PORT, STT_API_STYLE, STT_API_KEY

__version__ = "0.1.0"

def wakeword(*args, **kwargs):
    """Factory to create a Recogniser (WakeWord).

    Example:
        import easywakeword
        r = easywakeword.wakeword(wakewordstrings=["computer"], wakewordreferenceaudios=["examples/example_computer_male.wav"])
        r.waitforit()
    """
    return WakeWord(*args, **kwargs)

__all__ = [
    "SoundBuffer",
    "WordMatcher",
    "Recogniser",
    "WakeWord",
    "wakeword",
    "transcribe_audio",
    "resolve_stt_ip",
    "STT_HOSTNAME",
    "STT_PORT",
    "STT_API_STYLE",
    "STT_API_KEY",
]
