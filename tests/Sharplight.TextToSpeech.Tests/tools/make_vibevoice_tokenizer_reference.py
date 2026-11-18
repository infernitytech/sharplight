"""Reference of the VibeVoice acoustic and semantic tokenizers (transformers 5.x modeling, CPU f32), weights from the original microsoft checkpoint.

Usage: python make_vibevoice_tokenizer_reference.py <original checkpoint dir> <hf-layout config dir> <wav> <output dir> [seconds]
Writes audio.bin (24 kHz, normalized to -25 dBFS, padded to a multiple of 3200), acoustic_latents.bin [frames, 64] (mean, no noise), semantic_latents.bin [frames, 128],
decoded.bin (acoustic decoder of the acoustic latents) and info.json.
"""
import sys, os, json, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibevoice_original_layout import load_tts
torch.set_num_threads(4)
model_dir, hf_meta, wav, out = sys.argv[1:5]
seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
os.makedirs(out, exist_ok=True)
model = load_tts(model_dir, hf_meta)
y, sr = sf.read(wav, dtype='float32')
if y.ndim > 1: y = y.mean(1)
y = librosa.resample(y, orig_sr=sr, target_sr=24000)[: int(seconds * 24000)]
rms = np.sqrt(np.mean(y ** 2)); y = y * (10 ** (-25 / 20) / (rms + 1e-6))      # processor: normalize to -25 dBFS
if np.abs(y).max() > 1: y = y / (np.abs(y).max() + 1e-6)
pad = (-len(y)) % 3200
y = np.pad(y, (0, pad)).astype('<f4')
x = torch.from_numpy(y)[None, None]
with torch.no_grad():
    ac = model.model.audio_tower.encode(x, sample=False).latents          # [1, frames, 64]
    sem = model.model.semantic_tokenizer_encoder(x).latents               # [1, frames, 128]
    dec = model.model.audio_tower.decode(ac).audio                        # [1, 1, samples]
y.tofile(out + '/audio.bin')
ac[0].numpy().astype('<f4').tofile(out + '/acoustic_latents.bin')
sem[0].numpy().astype('<f4').tofile(out + '/semantic_latents.bin')
dec[0, 0].numpy().astype('<f4').tofile(out + '/decoded.bin')
json.dump({"samples": int(len(y)), "frames": int(ac.shape[1]), "acoustic_dim": int(ac.shape[2]), "semantic_dim": int(sem.shape[2]), "sample_rate": 24000, "source_seconds": seconds}, open(out + '/info.json', 'w'), indent=1)
print(ac.shape, sem.shape, dec.shape, float(dec.abs().max()))
