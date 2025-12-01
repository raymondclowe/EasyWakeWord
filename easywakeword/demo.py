"""
Demo script for EasyWakeWord.

This demo uses the included example audio files to demonstrate wake word detection.
"""
import os
import sys


def main():
    """Run the wake word detection demo."""
    # Import here to avoid circular imports
    try:
        import easywakeword
    except ImportError:
        print("ERROR: easywakeword package not found!")
        print("Please install with: pip install easywakeword")
        return 1
    
    print("=" * 60)
    print("EasyWakeWord Demo")
    print("=" * 60)
    
    # Get path to package examples
    package_dir = os.path.dirname(os.path.abspath(easywakeword.__file__))
    examples_dir = os.path.join(package_dir, "examples")
    
    # Path to example audio files
    example_male = os.path.join(examples_dir, "example_computer_male.wav")
    example_male_teen = os.path.join(examples_dir, "example_computer_male_teen.wav")
    example_female = os.path.join(examples_dir, "example_computer_female..wav")
    
    # Use the first available example file
    reference_files = []
    for example_file in [example_male, example_male_teen, example_female]:
        if os.path.exists(example_file):
            reference_files.append(example_file)
            print(f"✓ Found reference: {os.path.basename(example_file)}")
    
    if not reference_files:
        print("ERROR: No example audio files found!")
        print(f"Looking in: {examples_dir}")
        return 1
    
    print(f"\nInitializing wake word detector...")
    print(f"Using {len(reference_files)} reference audio file(s)")
    print(f"Wake word: 'computer'")
    
    try:
        # Create recognizer with example files
        recognizer = easywakeword.wakeword(
            wakewordstrings=["computer"],
            wakewordreferenceaudios=reference_files,
            threshold=75,
            debug=True,
            debug_playback=False
        )
        
        print("\n" + "=" * 60)
        print("Listening for wake word 'computer'...")
        print("Speak clearly into your microphone.")
        print("Press Ctrl+C to exit.")
        print("=" * 60 + "\n")
        
        # Wait for wake word
        result = recognizer.waitforit()
        
        if result:
            print(f"\n✓ SUCCESS: Wake word detected - '{result}'")
            return 0
        else:
            print("\n✗ No wake word detected")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
