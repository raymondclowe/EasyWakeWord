#!/usr/bin/env python3
"""
Sound POC Example - Replicates functionality from raymondclowe/sound repo.

This example demonstrates:
1. Listing audio devices and allowing user selection
2. Transcribing a reference WAV file using a LAN Whisper server
3. Waiting for a two-word wake phrase (e.g., "ok computer", "computer activate")
4. Playing a Star Trek-style confirmation chime on detection

Usage:
    # First, start a mini_transcriber server on your LAN:
    # On server: python -m mini_transcriber  (or: cd mini_transcriber && python app.py)
    
    # Then run this example:
    python examples/sound_poc.py
    
    # Or with custom settings:
    python examples/sound_poc.py --url http://192.168.1.100:8085 --wav reference_word.wav

Requirements:
    - A reference WAV file with your wake word recorded
    - A LAN Whisper server running (mini_transcriber) for transcription
    - Microphone access
"""

import argparse
import io
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from easywakeword import WakeWord
from easywakeword.wakeword import AudioDeviceManager


def play_confirmation_chime():
    """
    Play a pleasant two-tone confirmation sound (Star Trek computer-style).
    
    This creates a distinctive "chirp-chirp" sound similar to the 
    Star Trek computer acknowledgment tone.
    """
    sample_rate = 44100
    duration1 = 0.08  # First tone duration
    duration2 = 0.12  # Second tone duration
    gap = 0.02  # Gap between tones
    
    # Generate first tone (higher frequency - A5 note)
    t1 = np.linspace(0, duration1, int(sample_rate * duration1))
    freq1 = 880  # A5 note
    tone1 = np.sin(2 * np.pi * freq1 * t1)
    # Apply envelope to avoid clicks
    envelope1 = np.exp(-3 * t1 / duration1)
    tone1 = tone1 * envelope1 * 0.3
    
    # Gap (silence)
    silence = np.zeros(int(sample_rate * gap))
    
    # Generate second tone (lower frequency - E5 note)
    t2 = np.linspace(0, duration2, int(sample_rate * duration2))
    freq2 = 659.25  # E5 note
    tone2 = np.sin(2 * np.pi * freq2 * t2)
    # Apply envelope
    envelope2 = np.exp(-2.5 * t2 / duration2)
    tone2 = tone2 * envelope2 * 0.35
    
    # Combine tones
    chime = np.concatenate([tone1, silence, tone2])
    
    # Play the chime
    sd.play(chime, sample_rate)
    sd.wait()


def list_and_select_device():
    """
    List all available audio input devices and let user select one.
    
    Returns:
        Selected device index or None for default
    """
    print("\n" + "=" * 60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("=" * 60)
    
    devices = AudioDeviceManager.list_devices()
    
    if not devices:
        print("No audio input devices found!")
        return None
    
    for device in devices:
        default_marker = " (default)" if device['index'] == sd.default.device[0] else ""
        print(f"  {device['index']:2d}: {device['name']}{default_marker}")
    
    print("=" * 60)
    
    try:
        device_input = input("Enter device number (or press Enter for default): ").strip()
        if device_input:
            device_index = int(device_input)
            # Validate device exists
            if AudioDeviceManager.select_device(device_index) is not None:
                print(f"Selected device {device_index}\n")
                return device_index
            else:
                print(f"Invalid device {device_index}, using default\n")
                return None
        else:
            print("Using default device\n")
            return None
    except ValueError:
        print("Invalid input, using default device\n")
        return None


def transcribe_reference_wav(wav_path: str, whisper_url: str) -> str:
    """
    Transcribe a reference WAV file using the LAN Whisper server.
    
    Args:
        wav_path: Path to the WAV file
        whisper_url: URL of the Whisper transcription server
        
    Returns:
        Transcribed text or None if failed
    """
    print(f"\nTranscribing reference file: {wav_path}")
    print(f"Using server: {whisper_url}")
    
    try:
        # Load the audio file
        audio_data, sample_rate = sf.read(wav_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Normalize audio
        audio_data = audio_data - np.mean(audio_data)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        audio_data = np.clip(audio_data * 1.5, -1.0, 1.0)
        
        # Convert to WAV in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Send to transcription server
        files = {'file': ('audio.wav', buffer, 'audio/wav')}
        data = {
            'model': 'tiny',
            'language': 'en'
        }
        
        response = requests.post(
            f"{whisper_url}/transcribe",
            files=files,
            data=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get('text', '').strip()
            print(f"Transcription: '{transcription}'")
            return transcription
        else:
            print(f"Transcription failed: HTTP {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to transcription server at {whisper_url}")
        print("Make sure mini_transcriber is running on your LAN server.")
        return None
    except FileNotFoundError:
        print(f"WAV file not found: {wav_path}")
        return None
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def check_server_health(whisper_url: str) -> bool:
    """
    Check if the Whisper server is reachable and healthy.
    
    Args:
        whisper_url: URL of the Whisper server
        
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        response = requests.get(f"{whisper_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Whisper server at {whisper_url} is healthy")
            return True
        else:
            print(f"✗ Whisper server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to Whisper server at {whisper_url}")
        return False
    except Exception as e:
        print(f"✗ Error checking server health: {e}")
        return False


def on_wake_word_detected(text: str):
    """Callback for when wake word is detected."""
    print(f"\n✓ WAKE WORD DETECTED: '{text}'")
    play_confirmation_chime()
    print("Ready for next command...\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sound POC - Wake word detection with LAN Whisper transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default settings (server at localhost:8085)
    python examples/sound_poc.py
    
    # Specify custom server URL
    python examples/sound_poc.py --url http://192.168.1.100:8085
    
    # Use a custom reference WAV file
    python examples/sound_poc.py --wav my_wake_word.wav
    
    # Set custom wake word text
    python examples/sound_poc.py --text "hey computer" --words 2
"""
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8085',
        help='URL of the Whisper transcription server (default: http://localhost:8085)'
    )
    parser.add_argument(
        '--wav',
        default='reference_word.wav',
        help='Path to reference WAV file (default: reference_word.wav)'
    )
    parser.add_argument(
        '--text',
        default=None,
        help='Wake word text to detect (auto-detected from WAV if not specified)'
    )
    parser.add_argument(
        '--words',
        type=int,
        default=2,
        help='Number of words in the wake phrase (default: 2)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Detection timeout in seconds (default: 60)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=75.0,
        help='MFCC similarity threshold 0-100 (default: 75.0)'
    )
    parser.add_argument(
        '--skip-server-check',
        action='store_true',
        help='Skip checking if the Whisper server is available'
    )
    parser.add_argument(
        '--mfcc-only',
        action='store_true',
        help='Use MFCC matching only (no transcription confirmation)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("SOUND POC - Wake Word Detection Example")
    print("=" * 60)
    print("Replicates functionality from raymondclowe/sound repo")
    print("=" * 60)
    
    # Check if WAV file exists
    wav_path = Path(args.wav)
    if not wav_path.exists():
        # Try in examples directory
        examples_wav = Path(__file__).parent.parent / args.wav
        if examples_wav.exists():
            wav_path = examples_wav
        else:
            print(f"\nError: Reference WAV file not found: {args.wav}")
            print("Please provide a valid WAV file with --wav option")
            print("Or create one by recording your wake word.")
            sys.exit(1)
    
    whisper_url = args.url if not args.mfcc_only else None
    
    # Check server health (unless skipped or MFCC-only mode)
    if whisper_url and not args.skip_server_check:
        print("\nChecking Whisper server...")
        if not check_server_health(whisper_url):
            print("\nOptions:")
            print("  1. Start mini_transcriber on your server")
            print("  2. Use --skip-server-check to skip this check")
            print("  3. Use --mfcc-only to run without transcription")
            print("\nContinuing without transcription confirmation...")
            whisper_url = None
    
    # Transcribe reference WAV to determine wake word
    wake_word_text = args.text
    if wake_word_text is None:
        if whisper_url:
            transcription = transcribe_reference_wav(str(wav_path), whisper_url)
            if transcription:
                wake_word_text = transcription.strip().lower().rstrip('.,!?;:')
                print(f"Detected wake word: '{wake_word_text}'")
            else:
                print("Could not transcribe reference, using 'computer' as default")
                wake_word_text = "computer"
        else:
            print("No transcription server, using 'computer' as default wake word")
            wake_word_text = "computer"
    
    # List and select audio device
    device_index = list_and_select_device()
    
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Wake word: '{wake_word_text}'")
    print(f"  Word count: {args.words}")
    print(f"  Reference WAV: {wav_path}")
    print(f"  Whisper URL: {whisper_url or 'None (MFCC-only mode)'}")
    print(f"  Device: {device_index if device_index is not None else 'default'}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Timeout: {args.timeout}s")
    print("=" * 60)
    
    # Create wake word detector
    detector = WakeWord(
        textword=wake_word_text,
        wavword=str(wav_path),
        numberofwords=args.words,
        timeout=args.timeout,
        external_whisper_url=whisper_url,
        device=device_index,
        similarity_threshold=args.threshold,
        callback=on_wake_word_detected
    )
    
    print("\n" + "=" * 60)
    print("LISTENING FOR WAKE WORD")
    print("=" * 60)
    print(f"Say '{wake_word_text}' to trigger detection")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Start listening
    detector.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            if not detector.is_listening():
                print("Detector stopped unexpectedly")
                break
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        detector.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()
