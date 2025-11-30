import os
import sys
import subprocess
import logging
import numpy as np
from typing import Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    A direct wrapper around OpenAI Whisper for local transcription.
    Replaces the need for a separate mini_transcriber process.
    """

    def __init__(self, model_name: str = "tiny", download_root: Optional[str] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Name of the Whisper model to use (default: "tiny")
            download_root: Directory to store the model (default: ~/.cache/whisper)
        """
        self.model_name = model_name
        self.download_root = download_root
        self._model = None
        self._available = False
        
        # Check if dependencies are installed
        self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if torch and whisper are importable."""
        try:
            import torch
            import whisper
            self._available = True
            return True
        except ImportError:
            self._available = False
            return False

    def install_dependencies(self) -> bool:
        """
        Attempt to install the required dependencies (torch, whisper).
        Returns True if successful.
        """
        logger.info("Installing PyTorch CPU and Whisper...")
        try:
            # Install CPU-only PyTorch wheels (lighter weight)
            logger.info("Installing PyTorch CPU...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--index-url", "https://download.pytorch.org/whl/cpu",
                "torch==2.2.2+cpu", "torchaudio==2.2.2+cpu",
                "-f", "https://download.pytorch.org/whl/cpu/torch_stable.html"
            ], check=True, capture_output=True, text=True)
            
            # Install OpenAI Whisper
            logger.info("Installing Whisper...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "openai-whisper"
            ], check=True, capture_output=True, text=True)
            
            # Verify imports
            import torch
            import whisper
            self._available = True
            logger.info("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            if e.stdout: logger.error(f"STDOUT: {e.stdout}")
            if e.stderr: logger.error(f"STDERR: {e.stderr}")
            return False
        except ImportError:
            logger.error("Installed packages but could not import them")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing dependencies: {e}")
            return False

    def load_model(self) -> bool:
        """
        Load the Whisper model into memory.
        Returns True if successful.
        """
        if not self._available:
            if not self.install_dependencies():
                return False

        try:
            import whisper
            if self._model is None:
                logger.info(f"Loading Whisper model '{self.model_name}'...")
                self._model = whisper.load_model(self.model_name, download_root=self.download_root)
                logger.info("Whisper model loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio data.
        
        Args:
            audio: Numpy array of audio samples (float32, -1.0 to 1.0)
            sample_rate: Sample rate of the audio (must be 16000 for Whisper)
            
        Returns:
            Transcribed text or None if failed
        """
        if self._model is None:
            if not self.load_model():
                return None

        try:
            # Whisper expects float32 audio at 16kHz
            # If audio is not float32, convert it
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # If we need to resample, we would do it here, but EasyWakeWord 
            # generally works at 16kHz. We'll assume input is correct for now
            # or rely on the caller to provide 16kHz.
            
            # Pad or trim to 30 seconds (Whisper expects 30s or handles it internally)
            # Actually whisper.transcribe handles variable lengths fine.
            
            result = self._model.transcribe(audio, language="en", fp16=False)
            text = result.get("text", "").strip()
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
