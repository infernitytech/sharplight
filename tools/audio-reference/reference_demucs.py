"""Demucs (htdemucs) reference stems: the demucs package's HTDemucs with the weights of a safetensors conversion from the Hub (the original .th checkpoint holds pickled classes and is
not loaded), apply_model with split=True, overlap=0.25, shifts=0 on jfk.wav as 44.1 kHz stereo. Writes the mixture and the stems into ~/models/audio-models/refs."""
import os, json, math
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
os.makedirs(REFS, exist_ok=True)
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from safetensors.torch import load_file
from demucs.htdemucs import HTDemucs
from demucs.apply import apply_model
torch.set_num_threads(4)
cfg = json.load(open(ROOT + '/htdemucs/config.json'))
model = HTDemucs(**cfg)
model.load_state_dict(load_file(ROOT + '/htdemucs/model.safetensors'))
model.eval()
a, sr = sf.read(AUDIO + '/jfk.wav', dtype='float32')
if a.ndim == 1:
    a = np.stack([a, a], 1)
g = math.gcd(44100, sr)
x = resample_poly(a, 44100 // g, sr // g, axis=0).astype('float32')
x = x[: 44100 * 10]
mix = torch.from_numpy(x.T.copy())
mix.numpy().astype('<f4').tofile(REFS + '/demucs_in.f32')
with torch.no_grad():
    out = apply_model(model, mix[None], shifts=0, split=True, overlap=0.25, progress=False, device='cpu')[0]
print(out.shape)
out.numpy().astype('<f4').tofile(REFS + '/demucs_out.f32')
