"""
STT helpers for easywakeword

Supports two API styles:
1. mini_transcriber: Simple POST to root with 'audio' field
2. OpenAI-style: POST to /v1/audio/transcriptions with Bearer auth and additional params
"""
import requests
from typing import Optional
import json

STT_HOSTNAME = "localhost"
STT_PORT = 8080
STT_API_STYLE = "mini_transcriber"  # or "openai"
STT_API_KEY = None  # Set if using OpenAI-style API
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(
    audio: bytes, 
    rate: int, 
    stt_url: Optional[str] = None, 
    prompt: str = None, 
    model: str = "tiny",
    api_style: Optional[str] = None,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    response_format: str = "text",
    temperature: float = 0.0
) -> Optional[str]:
    """
    Transcribe audio using either mini_transcriber or OpenAI-style API.
    
    Args:
        audio: Audio data as bytes
        rate: Sample rate (used for context, not sent)
        stt_url: Custom STT server URL (overrides default)
        prompt: Optional prompt to guide the model
        model: Model name (e.g., "tiny", "whisper-1")
        api_style: "mini_transcriber" or "openai" (defaults to STT_API_STYLE)
        api_key: API key for OpenAI-style APIs (defaults to STT_API_KEY)
        language: ISO-639-1 language code (e.g., "en")
        response_format: "text", "json", "verbose_json", "srt", "vtt"
        temperature: Sampling temperature (0-1)
    
    Returns:
        Transcription text or None on error
    """
    try:
        style = api_style or STT_API_STYLE
        key = api_key or STT_API_KEY
        
        if style == "openai":
            # OpenAI-style API: /v1/audio/transcriptions
            base_url = stt_url or f"http://{resolve_stt_ip()}:{STT_PORT}"
            url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
            
            # Build multipart form data
            files = {'file': ('audio.wav', audio, 'audio/wav')}
            data = {
                'model': model if model != "tiny" else "whisper-1",
                'response_format': response_format,
            }
            
            if prompt:
                data['prompt'] = prompt
            if language:
                data['language'] = language
            if temperature != 0.0:
                data['temperature'] = temperature
            
            # Headers with Bearer token if provided
            headers = {}
            if key:
                headers['Authorization'] = f'Bearer {key}'
            
            resp = requests.post(url, files=files, data=data, headers=headers, timeout=10)
            
            if resp.status_code == 200:
                if response_format == "text":
                    return resp.text.strip()
                else:
                    # For JSON formats, extract the text field
                    try:
                        result = resp.json()
                        return result.get('text', '').strip()
                    except:
                        return resp.text.strip()
        
        else:
            # mini_transcriber style: Simple POST to root
            url = stt_url or f"http://{resolve_stt_ip()}:{STT_PORT}"
            files = {'audio': ('audio.wav', audio, 'audio/wav')}
            resp = requests.post(url, files=files, timeout=5)
            
            if resp.status_code == 200:
                return resp.text.strip()
    
    except Exception as e:
        # Silently fail as before, but could log if debug enabled
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
    "STT_API_STYLE",
    "STT_API_KEY",
]
