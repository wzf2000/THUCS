import librosa
import soundfile

y, sr = librosa.load("test.wav", sr=None)
print(y, sr)

dur = 30

for i in range(4):
    yi = y[i * dur * sr : (i + 1) * dur * sr]
    soundfile.write(f"input_{i}.wav", yi, sr)
