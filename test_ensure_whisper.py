#!/usr/bin/env python3
"""
Independent test for WakeWord.ensure_bundled_transcriber functionality.

This script tests:
1. Downloading mini_transcriber if not present
2. Installing dependencies
3. Starting the transcriber server
4. Verifying health endpoint
"""

import sys
import time
import requests
from pathlib import Path

from easywakeword import WakeWord
from easywakeword.wakeword import DEFAULT_MINI_TRANSCRIBER_PORT


def test_ensure_bundled_transcriber():
    """Test the ensure_bundled_transcriber class method."""
    print("=" * 60)
    print("TESTING: WakeWord.ensure_bundled_transcriber()")
    print("=" * 60)
    print()
    
    # Check if mini_transcriber directory already exists
    transcriber_dir = Path.home() / ".easywakeword" / "mini_transcriber"
    print(f"Transcriber directory: {transcriber_dir}")
    print(f"Already exists: {transcriber_dir.exists()}")
    print()
    
    # Try to ensure the bundled transcriber
    print("Calling ensure_bundled_transcriber()...")
    print("-" * 60)
    success = WakeWord.ensure_bundled_transcriber()
    print("-" * 60)
    print()
    
    if success:
        print("✓ ensure_bundled_transcriber() returned True")
    else:
        print("✗ ensure_bundled_transcriber() returned False")
        return False
    
    # Verify the server is running
    print("\nVerifying server health...")
    transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"
    
    try:
        response = requests.get(f"{transcriber_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is healthy")
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {data}")
            
            if data.get('model_loaded', False):
                print("✓ Whisper model is loaded")
                return True
            else:
                print("✗ Whisper model is NOT loaded")
                return False
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {transcriber_url}")
        return False
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return False


def test_transcription():
    """Test actual transcription functionality."""
    print("\n" + "=" * 60)
    print("TESTING: Basic transcription")
    print("=" * 60)
    print()
    
    # Check if reference WAV exists
    wav_path = Path("reference_word.wav")
    if not wav_path.exists():
        print(f"⚠ Reference WAV not found: {wav_path}")
        print("Skipping transcription test")
        return True
    
    print(f"Using reference WAV: {wav_path}")
    
    # Create a simple transcription request
    transcriber_url = f"http://localhost:{DEFAULT_MINI_TRANSCRIBER_PORT}"
    
    try:
        with open(wav_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            data = {'model': 'tiny', 'language': 'en'}
            
            print(f"Sending transcription request to {transcriber_url}/transcribe")
            response = requests.post(
                f"{transcriber_url}/transcribe",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('text', '').strip()
                print(f"✓ Transcription successful")
                print(f"  Result: '{transcription}'")
                return True
            else:
                print(f"✗ Transcription failed: HTTP {response.status_code}")
                print(f"  Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Transcription error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" INDEPENDENT TEST: ensure_bundled_transcriber()")
    print("=" * 70)
    print()
    
    # Test 1: Ensure bundled transcriber
    test1_passed = test_ensure_bundled_transcriber()
    
    # Test 2: Try transcription
    test2_passed = test_transcription()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"ensure_bundled_transcriber: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Basic transcription:         {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("=" * 70)
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
