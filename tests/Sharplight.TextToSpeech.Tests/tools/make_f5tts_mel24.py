"""A 24 kHz copy of the reference recording (librosa's resampler) and the Vocos log-mel of it, so the front-end test needs no resampling."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torchaudio.transforms import MelSpectrogram  # noqa: E402

wav, out = sys.argv[1], sys.argv[2]
audio, rate = sf.read(wav, dtype="float32")
audio24 = librosa.resample(audio, orig_sr=rate, target_sr=24000).astype(np.float32)
sf.write(os.path.join(out, "reference24.wav"), audio24, 24000, subtype="FLOAT")
x = torch.from_numpy(audio24)[None]
rms = torch.sqrt(torch.mean(x ** 2))
if rms < 0.1:
    x = x * 0.1 / rms
mel = MelSpectrogram(sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mels=100, power=1, center=True, normalized=False, norm=None)(x)
mel = torch.log(torch.clip(mel, min=1e-5))
np.ascontiguousarray(mel[0].T.numpy()).tofile(os.path.join(out, "cond24.bin"))
print("frames", mel.shape[-1], "rms", float(rms), flush=True)
