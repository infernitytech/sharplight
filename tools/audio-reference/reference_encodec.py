"""Reference tensors for the audio-model tests: run with the venv of the audio references (transformers, onnxruntime); writes into ~/models/audio-models/refs."""
import os
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
SILERO = os.environ.get("SILERO_VAD_PACKAGE", ROOT + "/silero-vad/silero_vad")
os.makedirs(REFS, exist_ok=True)
import numpy as np, soundfile as sf, torch, sys
from transformers import EncodecModel
torch.set_num_threads(4)
m = EncodecModel.from_pretrained(ROOT + '/encodec_24khz').eval()
a, sr = sf.read(AUDIO + '/jfk.wav', dtype='float32')
if a.ndim > 1: a = a.mean(1)
from scipy.signal import resample_poly
x = torch.from_numpy(resample_poly(a, 24000//__import__('math').gcd(24000,sr), sr//__import__('math').gcd(24000,sr)).astype('float32'))
n = (x.shape[0] // 320) * 320
x = x[:min(n, 320*750)]  # <= 10 s
x.numpy().astype('<f4').tofile(REFS + '/encodec_in.f32')
with torch.no_grad():
    for bw in (6.0, 24.0):
        out = m.encode(x[None, None, :], bandwidth=bw)
        codes = out.audio_codes[0, 0]  # [Q, T]
        print(bw, codes.shape)
        codes.numpy().astype('<i4').tofile(f'{REFS}/encodec_codes_{int(bw)}.i32')
        dec = m.decode(out.audio_codes, out.audio_scales)[0][0, 0]
        dec.numpy().astype('<f4').tofile(f'{REFS}/encodec_dec_{int(bw)}.f32')
        print(dec.shape, dec.abs().max())
