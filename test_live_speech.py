import sounddevice as sd
import numpy as np
import time
from datetime import datetime

# Setup recording
recording = []
def audio_callback(indata, frames, time_info, status):
    recording.extend(indata[:, 0].tolist())

print("="*60)
print("NOW SPEAK 'COMPUTER' INTO THE MICROPHONE")
print("="*60)
print()

# Start recording
stream = sd.InputStream(samplerate=16000, channels=1, dtype='float32', device=1, callback=audio_callback)
stream.start()

print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Recording for 5 seconds...")
time.sleep(5)

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
        marker = '*** SOUND ***' if rms > 0.01 else ''
        print(f"[{timestamp:.1f}s] RMS: {rms:.6f} {marker}")

print()
max_rms = max(rms_values) if rms_values else 0
print(f"Max RMS detected: {max_rms:.6f}")
print(f"Silence threshold: 0.010000")
print(f"Sound detected: {max_rms > 0.01}")
