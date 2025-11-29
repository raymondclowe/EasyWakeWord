#!/usr/bin/env python3
"""
Device Selection Examples for EasyWakeWord

This example demonstrates the improved audio device selection capabilities,
including automatic selection, name-based matching, and device utilities.
"""

import sys
import time
from easywakeword import WakeWord
from easywakeword.wakeword import AudioDeviceManager


def example_device_listing():
    """Example: List all available audio devices."""
    print("=== Device Listing Example ===")
    AudioDeviceManager.print_device_list()
    print()


def example_device_selection():
    """Example: Different ways to select audio devices."""
    print("=== Device Selection Examples ===")

    # Auto-selection (recommended)
    print("1. Auto-selection (recommended):")
    device_auto = AudioDeviceManager.select_device()
    print(f"   Selected device index: {device_auto}")
    print()

    # Name-based selection
    print("2. Name-based selection:")
    device_microphone = AudioDeviceManager.select_device("microphone")
    print(f"   'microphone' matches device: {device_microphone}")

    device_intel = AudioDeviceManager.select_device("intel")
    print(f"   'intel' matches device: {device_intel}")

    device_realtek = AudioDeviceManager.select_device("realtek")
    print(f"   'realtek' matches device: {device_realtek}")
    print()

    # Index-based selection (traditional)
    print("3. Index-based selection (traditional):")
    device_index = AudioDeviceManager.select_device(1)
    print(f"   Device index 1: {device_index}")
    print()


def example_magic_words():
    """Example: Using magic words for intelligent device selection."""
    print("=== Magic Word Selection Examples ===")

    print("Testing magic word device selection...")

    # Magic word: "best" - tests all devices and picks highest audio level
    print("1. 'best' - Test all devices, select highest audio level:")
    device_best = AudioDeviceManager.select_device("best")
    print(f"   Selected device: {device_best}")
    print()

    # Magic word: "first" - find first working device
    print("2. 'first' - Find first device with audio signal:")
    device_first = AudioDeviceManager.select_device("first")
    print(f"   Selected device: {device_first}")
    print()

    # Magic word: "default" - system default
    print("3. 'default' - System default device:")
    device_default = AudioDeviceManager.select_device("default")
    print(f"   Selected device: {device_default}")
    print()


def example_speech_detection_tuning():
    """Example: Tuning speech detection parameters."""
    print("=== Speech Detection Tuning Examples ===")

    print("1. Auto-calculated thresholds (recommended):")
    try:
        detector_auto = WakeWord(
            textword="hello",  # Generic wake word for the reference audio
            wavword="reference_word.wav",  # Use the actual reference audio file
            numberofwords=1,
            stt_backend=None
        )
        print(f"   Pre-speech silence: {detector_auto.pre_speech_silence:.2f}s")
        print(f"   Speech duration: {detector_auto.speech_duration_min:.2f}s - {detector_auto.speech_duration_max:.2f}s")
        print(f"   Post-speech silence: {detector_auto.post_speech_silence:.2f}s")
        print("   ✓ Auto-calculated from reference audio or text heuristics")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n2. Manual tuning for short wake words:")
    try:
        detector_short = WakeWord(
            textword="hey",
            wavword="reference_word.wav",  # Use reference audio for consistency
            numberofwords=1,
            stt_backend=None,
            pre_speech_silence=0.5,    # Shorter silence before
            speech_duration_min=0.2,   # Shorter speech duration
            speech_duration_max=0.8,
            post_speech_silence=0.3
        )
        print(f"   Pre-speech silence: {detector_short.pre_speech_silence:.2f}s")
        print(f"   Speech duration: {detector_short.speech_duration_min:.2f}s - {detector_short.speech_duration_max:.2f}s")
        print(f"   Post-speech silence: {detector_short.post_speech_silence:.2f}s")
        print("   ✓ Optimized for short single words")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n3. Manual tuning for long wake phrases:")
    try:
        detector_long = WakeWord(
            textword="wake up computer assistant",
            wavword="reference_word.wav",  # Use reference audio for consistency
            numberofwords=4,
            stt_backend=None,
            pre_speech_silence=1.0,    # Longer silence before
            speech_duration_min=0.8,   # Longer speech duration
            speech_duration_max=2.5,
            post_speech_silence=0.5
        )
        print(f"   Pre-speech silence: {detector_long.pre_speech_silence:.2f}s")
        print(f"   Speech duration: {detector_long.speech_duration_min:.2f}s - {detector_long.speech_duration_max:.2f}s")
        print(f"   Post-speech silence: {detector_long.post_speech_silence:.2f}s")
        print("   ✓ Optimized for longer multi-word phrases")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print()


def example_wakeword_with_devices():
    """Example: Using different device selection methods with WakeWord."""
    print("=== WakeWord with Device Selection ===")

    # Note: This example doesn't actually run detection to avoid requiring audio files
    # In real usage, you would provide valid wavword paths

    print("Creating WakeWord detectors with different device selections...")

    # Auto-selection
    try:
        detector_auto = WakeWord(
            textword="hello",
            wavword="reference_word.wav",  # Use the actual reference audio file
            numberofwords=1,
            device=None,  # Auto-select
            stt_backend=None  # Skip transcription for demo
        )
        print("✓ Auto-selected device detector created")
    except Exception as e:
        print(f"✗ Auto-selected device detector failed: {e}")

    # Name-based selection
    try:
        detector_name = WakeWord(
            textword="hello",
            wavword="reference_word.wav",  # Use reference audio for consistency
            numberofwords=1,
            device="microphone",  # Name pattern
            stt_backend=None
        )
        print("✓ Name-based device detector created")
    except Exception as e:
        print(f"✗ Name-based device detector failed: {e}")

    # Index-based selection
    try:
        detector_index = WakeWord(
            textword="hello",
            wavword="reference_word.wav",  # Use reference audio for consistency
            numberofwords=1,
            device=1,  # Device index
            stt_backend=None
        )
        print("✓ Index-based device detector created")
    except Exception as e:
        print(f"✗ Index-based device detector failed: {e}")

    print()


def main():
    """Run all examples."""
    print("EasyWakeWord Device Selection Examples")
    print("=" * 40)
    print()

    example_device_listing()
    example_device_selection()
    example_magic_words()
    example_speech_detection_tuning()
    example_wakeword_with_devices()

    print("=== Device Testing Utility ===")
    print("You can also test devices with:")
    print("  uv run python -m easywakeword.device_utils test")
    print("  uv run python -m easywakeword.device_utils test microphone")
    print("  uv run python -m easywakeword.device_utils test 1")
    print("  uv run python -m easywakeword.device_utils test \"best\"")
    print("  uv run python -m easywakeword.device_utils test \"first\"")
    print()

    print("Examples completed!")


if __name__ == "__main__":
    main()