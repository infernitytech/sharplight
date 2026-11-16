"""Reference tensors for the audio-model tests: run with the venv of the audio references (transformers, onnxruntime); writes into ~/models/audio-models/refs."""
import os
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
SILERO = os.environ.get("SILERO_VAD_PACKAGE", ROOT + "/silero-vad/silero_vad")
os.makedirs(REFS, exist_ok=True)
import numpy as np, soundfile as sf, torch, math
from scipy.signal import resample_poly
from transformers import MimiModel
torch.set_num_threads(4)
m = MimiModel.from_pretrained(ROOT + '/mimi').eval()
a, sr = sf.read(AUDIO + '/jfk.wav', dtype='float32')
if a.ndim > 1: a = a.mean(1)
g = math.gcd(24000, sr)
x = torch.from_numpy(resample_poly(a, 24000//g, sr//g).astype('float32'))
n = min((x.shape[0] // 1920) * 1920, 1920*125)
x = x[:n]
x.numpy().astype('<f4').tofile(REFS + '/mimi_in.f32')
with torch.no_grad():
    for q in (8, 32):
        out = m.encode(x[None, None, :], num_quantizers=q)
        codes = out.audio_codes[0]
        print(q, codes.shape)
        codes.numpy().astype('<i4').tofile(f'{REFS}/mimi_codes_{q}.i32')
        dec = m.decode(out.audio_codes).audio_values[0, 0]
        dec.numpy().astype('<f4').tofile(f'{REFS}/mimi_dec_{q}.f32')
        print(dec.shape, dec.abs().max())
