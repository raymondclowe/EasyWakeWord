#!/usr/bin/env python3
"""
Test script for WakeWord.ensure_bundled_transcriber()
"""

from easywakeword import WakeWord

if __name__ == "__main__":
    print("Testing WakeWord.ensure_bundled_transcriber()...")
    success = WakeWord.ensure_bundled_transcriber()
    if success:
        print("✓ Bundled transcriber is ready!")
    else:
        print("✗ Failed to ensure bundled transcriber")