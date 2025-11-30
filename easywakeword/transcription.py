"""
STT helpers for easywakeword
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
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
"""
STT helpers for easywakeword
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    # Simple local resolver; replace with mDNS or caching for production
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    """Send audio bytes to a small STT service and return the transcription on success."""
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
"""
STT helpers for easywakeword
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    # Simple local resolver; replace with mDNS or caching for production
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    """Send audio bytes to a small STT service and return the transcription on success.

    This is intentionally minimal. Use `stt_url` to override the service URL.
    """
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
"""
STT helpers for easywakeword (clean version)
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
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
"""
STT helpers for easywakeword
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    # Currently simple resolver for local dev; may implement mDNS or cached IPs later
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    # Simple POST call to an STT engine; this is intentionally minimal for demonstration
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
"""
STT helpers (migrated from easywakeword.transcription)
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    # Placeholder simple POST for STT engines
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
"""
STT helpers (migrated from easywakeword.transcription)
"""
import requests
from typing import Optional

STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    # Placeholder simple POST for STT engines
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
"""
STT helpers (migrated from easywakeword.transcription)
"""
import requests
from typing import Optional
STT_HOSTNAME = "mini_transcriber.local"
STT_PORT = 8080
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(audio: bytes, rate: int, stt_url: Optional[str] = None, prompt: str = None, model: str = "tiny") -> Optional[str]:
    # Placeholder simple POST for STT engines
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
"""
Proxy wrapper to expose STT helpers from easywakeword.transcription
for the easywakeword public API.
"""
from easywakeword.transcription import (
    transcribe_audio,
    resolve_stt_ip,
    close_stt_session,
    STT_HOSTNAME,
    STT_PORT,
)

__all__ = [
    "transcribe_audio",
    "resolve_stt_ip",
    "close_stt_session",
    "STT_HOSTNAME",
    "STT_PORT",
]
