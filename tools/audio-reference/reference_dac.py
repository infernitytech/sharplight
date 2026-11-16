"""Reference tensors for the audio-model tests: run with the venv of the audio references (transformers, onnxruntime); writes into ~/models/audio-models/refs."""
import os
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
SILERO = os.environ.get("SILERO_VAD_PACKAGE", ROOT + "/silero-vad/silero_vad")
os.makedirs(REFS, exist_ok=True)
import numpy as np, soundfile as sf, torch, math
from scipy.signal import resample_poly
from transformers import DacModel
torch.set_num_threads(4)
m = DacModel.from_pretrained(ROOT + '/dac_44khz').eval()
a, sr = sf.read(AUDIO + '/jfk.wav', dtype='float32')
if a.ndim > 1: a = a.mean(1)
g = math.gcd(44100, sr)
x = torch.from_numpy(resample_poly(a, 44100//g, sr//g).astype('float32'))
n = min((x.shape[0] // 512) * 512, 512*430)
x = x[:n]
x.numpy().astype('<f4').tofile(REFS + '/dac_in.f32')
with torch.no_grad():
    out = m.encode(x[None, None, :])
    codes = out.audio_codes[0]
    print(codes.shape)
    codes.numpy().astype('<i4').tofile(REFS + '/dac_codes.i32')
    dec = m.decode(audio_codes=out.audio_codes).audio_values[0]
    dec.numpy().astype('<f4').tofile(REFS + '/dac_dec.f32')
    print(dec.shape, dec.abs().max())
