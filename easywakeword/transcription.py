"""
STT helpers for easywakeword
"""
import requests
from typing import Optional

STT_HOSTNAME = "localhost"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    try:
        url = stt_url or f"http://{resolve_stt_ip()}:{STT_PORT}"
        files = {'audio': ('audio.wav', audio)}
        resp = requests.post(url, files=files, timeout=5)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None

def close_stt_session():
    global _stt_session
    if _stt_session:
        try:
            _stt_session.close()
        except Exception:
            pass
        _stt_session = None

__all__ = [
    "transcribe_audio",
    "resolve_stt_ip",
    "close_stt_session",
    "STT_HOSTNAME",
    "STT_PORT",
]
