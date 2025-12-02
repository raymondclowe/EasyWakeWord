import easywakeword
import time
import numpy as np

print("Setting up recognizer...")
r = easywakeword.Recogniser(
    wakewordstrings=['computer'],
    wakewordreferenceaudios=[r'C:\Users\raymo_wxahrt0\EasyWakeWord\easywakeword\examples\example_computer_male.wav'],
    device=1
)

print('Recording for 5 seconds - please speak COMPUTER loudly...')
time.sleep(5)

audio = r.soundBuffer.return_last_n_seconds(5)
rms_values = []

for i in range(0, len(audio), 1600):
    chunk = audio[i:i+1600]
    if len(chunk) > 0:
        rms_values.append(np.sqrt(np.mean(chunk**2)))

print(f'\nRMS over time (0.1s chunks):')
for i, rms in enumerate(rms_values[::10]):
    print(f'{i*1.0:.1f}s: {rms:.6f}')

max_rms = max(rms_values) if rms_values else 0
print(f'\nMax RMS during speech: {max_rms:.6f}')
print(f'Current silence threshold: {r.soundBuffer.silence_threshold:.6f}')
print(f'Speech detected: {max_rms > r.soundBuffer.silence_threshold}')
