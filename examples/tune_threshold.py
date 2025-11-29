#!/usr/bin/env python3
"""
Example: Tuning the similarity threshold for wake word detection.

The similarity_threshold parameter controls how strict the MFCC matching is.
This example helps you find the optimal threshold for your environment.

Recommended values:
- 60-70: Noisy environments, accent variations (more permissive)
- 75-80: General purpose (default is 75)
- 80-90: Clean environments, strict false positive control

Usage:
    python tune_threshold.py reference.wav
    python tune_threshold.py reference.wav 65  # Test with threshold 65
"""

import sys
import time
import numpy as np

try:
    import librosa
    import soundfile as sf
    import sounddevice as sd
    from easywakeword.wakeword import WordMatcher, SoundBuffer
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Run: pip install easywakeword")
    sys.exit(1)


def load_and_resample_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio from file and resample to target sample rate if needed.
    
    Args:
        filepath: Path to the audio file
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Audio samples as a numpy array at the target sample rate
    """
    audio, sr = sf.read(filepath)
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def test_threshold(reference_file: str, threshold: float = 75.0, test_duration: float = 10.0):
    """
    Test MFCC matching with different thresholds.
    
    This helps you understand how different threshold values affect detection.
    
    Args:
        reference_file: Path to reference WAV file
        threshold: Similarity threshold to test (0-100)
        test_duration: How long to test (seconds)
    """
    print(f"Testing similarity threshold: {threshold}")
    print(f"Reference audio: {reference_file}")
    print()
    
    # Load reference audio using shared helper
    reference_audio = load_and_resample_audio(reference_file, target_sr=16000)
    
    # Create matcher
    matcher = WordMatcher(sample_rate=16000)
    matcher.set_reference(reference_audio, "target")
    
    print(f"Recording {test_duration} seconds of audio...")
    print("Speak your wake word multiple times.")
    print()
    
    # Record test audio in 1-second chunks
    sample_rate = 16000
    chunk_duration = 1.0
    chunks = int(test_duration / chunk_duration)
    
    matches_found = 0
    max_similarity = 0
    min_similarity = 100
    similarities = []
    
    for i in range(chunks):
        audio = sd.rec(
            int(chunk_duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()
        
        # Calculate similarity
        try:
            similarity = matcher.calculate_similarity(audio)
            similarities.append(similarity)
            
            # Track stats
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity < min_similarity:
                min_similarity = similarity
            
            # Check if it matches
            matches = similarity >= threshold
            status = "✓ MATCH" if matches else "  no match"
            if matches:
                matches_found += 1
            
            print(f"  Chunk {i+1}/{chunks}: {similarity:5.1f}% {status}")
            
        except Exception as e:
            print(f"  Chunk {i+1}/{chunks}: Error - {e}")
    
    print()
    print("=" * 40)
    print("Results:")
    print(f"  Threshold tested: {threshold}%")
    print(f"  Matches found: {matches_found}/{chunks}")
    print(f"  Max similarity: {max_similarity:.1f}%")
    print(f"  Min similarity: {min_similarity:.1f}%")
    if similarities:
        avg = sum(similarities) / len(similarities)
        print(f"  Avg similarity: {avg:.1f}%")
    print()
    
    # Recommendations
    if matches_found == 0:
        print("Recommendation: Lower the threshold (try 60-70)")
    elif matches_found == chunks:
        print("Recommendation: This threshold may be too permissive.")
        print("               Increase it to reduce false positives.")
    else:
        print(f"Recommendation: {matches_found}/{chunks} matches seems reasonable.")
        print("               Adjust based on your false positive tolerance.")


def compare_thresholds(reference_file: str):
    """
    Compare multiple threshold values to help find the optimal setting.
    
    Args:
        reference_file: Path to reference WAV file
    """
    print("Comparing multiple threshold values")
    print("=" * 50)
    print()
    print("This will record 5 seconds of audio and show how")
    print("different thresholds would classify each segment.")
    print()
    
    # Load reference audio using shared helper
    reference_audio = load_and_resample_audio(reference_file, target_sr=16000)
    
    # Create matcher
    matcher = WordMatcher(sample_rate=16000)
    matcher.set_reference(reference_audio, "target")
    
    # Thresholds to test
    thresholds = [60, 65, 70, 75, 80, 85, 90]
    
    print("Recording 5 seconds - say your wake word multiple times...")
    print()
    
    # Record test audio
    audio = sd.rec(5 * 16000, samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    
    # Analyze in 1-second chunks
    chunk_size = 16000
    chunks = len(audio) // chunk_size
    
    # Header
    header = "Time  Similarity  " + "  ".join(f"{t:3d}%" for t in thresholds)
    print(header)
    print("-" * len(header))
    
    for i in range(chunks):
        chunk = audio[i * chunk_size:(i + 1) * chunk_size]
        
        try:
            similarity = matcher.calculate_similarity(chunk)
            
            # Check each threshold
            results = []
            for t in thresholds:
                match = "✓" if similarity >= t else "·"
                results.append(f"  {match}  ")
            
            print(f"{i+1}s    {similarity:5.1f}%     {''.join(results)}")
            
        except Exception:
            print(f"{i+1}s    Error")
    
    print()
    print("Legend: ✓ = would match, · = would not match")
    print()
    print("Choose a threshold where:")
    print("  - Wake word segments show ✓")
    print("  - Non-wake-word segments show ·")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    
    reference_file = sys.argv[1]
    
    # Check file exists
    try:
        sf.read(reference_file)
    except Exception as e:
        print(f"Error: Cannot read reference file: {e}")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        # Test specific threshold
        threshold = float(sys.argv[2])
        test_threshold(reference_file, threshold)
    else:
        # Compare multiple thresholds
        compare_thresholds(reference_file)


if __name__ == "__main__":
    main()
