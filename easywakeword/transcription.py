"""
STT helpers for easywakeword

Supports two API styles:
1. mini_transcriber: Simple POST to root with 'audio' field
2. OpenAI-style: POST to /v1/audio/transcriptions with Bearer auth and additional params
"""
import requests
from typing import Optional, Union
import json
import io
import os
import numpy as np

STT_HOSTNAME = os.environ.get("STT_HOST", "localhost")
STT_PORT = int(os.environ.get("STT_PORT", "8080"))
STT_API_STYLE = os.environ.get("STT_API_STYLE", "mini_transcriber")  # or "openai"
STT_API_KEY = os.environ.get("STT_API_KEY", None)  # Set if using OpenAI-style API
_stt_session = None

def resolve_stt_ip(hostname: str = STT_HOSTNAME) -> str:
    return hostname

def transcribe_audio(
    audio: Union[bytes, np.ndarray], 
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
        audio: Audio data as bytes (WAV) or numpy array (float32, mono)
        rate: Sample rate (used for WAV conversion if audio is numpy array)
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
        # Convert numpy array to WAV bytes if needed
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            buf = io.BytesIO()
            # Ensure audio is in right format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(buf, audio, rate, format='WAV', subtype='PCM_16')
            audio_bytes = buf.getvalue()
        else:
            audio_bytes = audio
        
        style = api_style or STT_API_STYLE
        key = api_key or STT_API_KEY
        
        if style == "openai":
            # OpenAI-style API: /v1/audio/transcriptions
            base_url = stt_url or f"http://{resolve_stt_ip()}:{STT_PORT}"
            url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"
            
            # Build multipart form data
            files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
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
            # mini_transcriber style: POST to /transcribe endpoint
            base_url = stt_url or f"http://{resolve_stt_ip()}:{STT_PORT}"
            url = f"{base_url.rstrip('/')}/transcribe"
            files = {'file': ('audio.wav', audio_bytes, 'audio/wav')}
            resp = requests.post(url, files=files, timeout=10)
            
            if resp.status_code == 200:
                # mini_transcriber returns JSON with "text" field
                try:
                    result = resp.json()
                    return result.get('text', '').strip()
                except:
                    return resp.text.strip()
    
    except Exception as e:
        # Log errors for debugging
        import sys
        print(f"[STT ERROR] {type(e).__name__}: {e}", file=sys.stderr)
    
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
