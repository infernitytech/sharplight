"""CLAP reference tensors (transformers ClapModel, CPU): features, audio and text embeddings and zero-shot probabilities for jfk.wav (first 10 s at 48 kHz)."""
import os, math
ROOT = os.path.expanduser("~/models/audio-models")
AUDIO = os.path.expanduser("~/models/audio")
REFS = ROOT + "/refs"
os.makedirs(REFS, exist_ok=True)
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor
torch.set_num_threads(4)
m = ClapModel.from_pretrained(ROOT + '/clap-htsat-unfused').eval()
p = ClapProcessor.from_pretrained(ROOT + '/clap-htsat-unfused')
a, sr = sf.read(AUDIO + '/jfk.wav', dtype='float32')
if a.ndim > 1:
    a = a.mean(1)
g = math.gcd(48000, sr)
x = resample_poly(a, 48000 // g, sr // g).astype('float32')[:480000]
x.astype('<f4').tofile(REFS + '/clap_in.f32')
labels = ["a person speaking", "a dog barking", "music playing", "a car engine", "silence"]
inputs = p(text=labels, audio=x, sampling_rate=48000, return_tensors='pt', padding=True)
feats = inputs['input_features'][0, 0]
print('features', feats.shape)
feats.numpy().astype('<f4').tofile(REFS + '/clap_features.f32')
with torch.no_grad():
    ae = m.get_audio_features(input_features=inputs['input_features'])
    ae = ae.pooler_output if hasattr(ae, 'pooler_output') else ae
    te = []
    for l in labels:
        t = p(text=[l], return_tensors='pt', padding=True)
        r = m.get_text_features(**t)
        te.append((r.pooler_output if hasattr(r, 'pooler_output') else r)[0])
    te = torch.stack(te)
    an = ae / ae.norm(dim=-1, keepdim=True)
    tn = te / te.norm(dim=-1, keepdim=True)
    logits = m.logit_scale_a.exp() * an @ tn.T
    print(logits, logits.softmax(-1))
    ae[0].numpy().astype('<f4').tofile(REFS + '/clap_audio_embed.f32')
    te.numpy().astype('<f4').tofile(REFS + '/clap_text_embed.f32')
    logits[0].numpy().astype('<f4').tofile(REFS + '/clap_logits.f32')
    print(ae.shape, te.shape, m.logit_scale_a.exp().item(), m.logit_scale_t.exp().item())
