import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from datetime import datetime

# Read the WAV file
wav_file = r'easywakeword\examples\example_computer_male.wav'
audio_data, sample_rate = sf.read(wav_file)

print(f"WAV file: {wav_file}")
print(f"Duration: {len(audio_data)/sample_rate:.2f}s")
print(f"Sample rate: {sample_rate}")
print()

# Setup recording
recording = []
def audio_callback(indata, frames, time_info, status):
    recording.extend(indata[:, 0].tolist())

print("Starting microphone (device 1) and playing WAV through speakers...")
print()

# Start recording
stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', device=1, callback=audio_callback)
stream.start()

# Wait a bit, then play
time.sleep(0.5)
print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Playing WAV file now...")
sd.play(audio_data, sample_rate)

# Record for duration of WAV + 1 second
duration = len(audio_data) / sample_rate + 1.0
time.sleep(duration)

stream.stop()
stream.close()

print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Recording stopped")
print()

# Analyze what was recorded
recording_array = np.array(recording)
print(f"Recorded {len(recording_array)} samples ({len(recording_array)/16000:.2f}s)")
print()

# Calculate RMS in 0.1s chunks
chunk_size = 1600  # 0.1s at 16kHz
rms_values = []
for i in range(0, len(recording_array), chunk_size):
    chunk = recording_array[i:i+chunk_size]
    if len(chunk) > 0:
        rms = np.sqrt(np.mean(chunk**2))
        rms_values.append(rms)
        timestamp = i / 16000
        print(f"[{timestamp:.1f}s] RMS: {rms:.6f} {'*** SOUND ***' if rms > 0.01 else ''}")

print()
max_rms = max(rms_values) if rms_values else 0
print(f"Max RMS detected: {max_rms:.6f}")
print(f"Silence threshold: 0.010000")
print(f"Sound detected: {max_rms > 0.01}")
