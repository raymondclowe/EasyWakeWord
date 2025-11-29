#!/usr/bin/env python3
"""
Audio device utilities for EasyWakeWord.

This module provides utilities for listing, testing, and selecting audio devices.
"""

import argparse
import sys
import time
from typing import Optional, Union

import numpy as np

# Add parent directory to path so we can import easywakeword
sys.path.insert(0, '.')
from easywakeword.wakeword import AudioDeviceManager


def list_devices():
    """List all available audio input devices."""
    AudioDeviceManager.print_device_list()


def test_device(device_spec: Optional[Union[int, str]] = None):
    """
    Test an audio device by recording a short sample and analyzing it.

    Args:
        device_spec: Device specification (index, name pattern, or None for auto)
    """
    device_index = AudioDeviceManager.select_device(device_spec)

    if device_index is None:
        print(f"Error: Could not find device matching '{device_spec}'")
        return False

    print(f"Testing device {device_index}...")

    try:
        import sounddevice as sd

        # Record a 2-second test sample
        duration = 2.0
        sample_rate = 16000

        print("Recording 2 seconds of audio... Speak into the microphone.")

        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_index,
            dtype=np.float32
        )
        sd.wait()

        # Analyze the recording
        audio_data = audio_data.flatten()

        # Calculate basic statistics
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))

        print("Recording complete!")
        print(f"RMS Level: {rms:.4f}")
        print(f"Peak Level: {peak:.4f}")
        print("SNR Estimate: N/A (not implemented)")

        # Check if audio levels seem reasonable
        if rms < 0.001:
            print("Warning: Audio levels are very low. Check microphone connection/volume.")
            return False
        elif rms > 0.5:
            print("Warning: Audio levels are very high. May cause distortion.")
            return False
        else:
            print("Audio levels look good!")
            return True

    except Exception as e:
        print(f"Error testing device: {e}")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="EasyWakeWord Audio Device Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python device_utils.py list                    # List all devices
  python device_utils.py test                    # Test default device
  python device_utils.py test 1                  # Test device index 1
  python device_utils.py test "microphone"       # Test device with "microphone" in name
  python device_utils.py test "best"             # Test and select best device by audio level
  python device_utils.py test "first"            # Test and select first working device
  python device_utils.py test "default"          # Test system default device
        """
    )

    parser.add_argument(
        'command',
        choices=['list', 'test'],
        help='Command to run'
    )

    parser.add_argument(
        'device',
        nargs='?',
        help='Device specification (index, name pattern, magic word, or omit for auto-select)'
    )

    args = parser.parse_args()

    if args.command == 'list':
        list_devices()
    elif args.command == 'test':
        device_spec = args.device
        if device_spec is not None:
            # Try to convert to int if it looks like a number
            try:
                device_spec = int(device_spec)
            except ValueError:
                pass  # Keep as string

        success = test_device(device_spec)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
