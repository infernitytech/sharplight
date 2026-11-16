"""MusicGen small reference (transformers MusicgenForConditionalGeneration, CPU, greedy with classifier-free guidance 3): prompt ids, T5 states, codes and audio."""
import os
ROOT = os.path.expanduser("~/models/audio-models")
REFS = ROOT + "/refs"
os.makedirs(REFS, exist_ok=True)
import numpy as np, torch
from transformers import MusicgenForConditionalGeneration, AutoTokenizer
torch.set_num_threads(4)
m = MusicgenForConditionalGeneration.from_pretrained(ROOT + '/musicgen-small').eval()
tok = AutoTokenizer.from_pretrained(ROOT + '/musicgen-small')
prompt = "80s pop track with bassy drums and synth"
t = tok([prompt], return_tensors='pt', padding=True)
t['input_ids'][0].numpy().astype('<i4').tofile(REFS + '/musicgen_ids.i32')
steps = 60
with torch.no_grad():
    enc = m.text_encoder(**t).last_hidden_state
    enc[0].numpy().astype('<f4').tofile(REFS + '/musicgen_t5.f32')
    captured = {}
    original = m.audio_encoder.decode
    def capture(audio_codes, *a, **k):
        captured['codes'] = audio_codes.clone()
        return original(audio_codes, *a, **k)
    m.audio_encoder.decode = capture
    out = m.generate(**t, do_sample=False, guidance_scale=3.0, max_new_tokens=steps)
    audio = out
    audio = audio.sequences if hasattr(audio, 'sequences') else audio
    print('audio', audio.shape)
    audio[0, 0].numpy().astype('<f4').tofile(REFS + '/musicgen_audio.f32')
    codes = captured['codes']
    print('codes', codes.shape)
    codes.reshape(-1, codes.shape[-1]).numpy().astype('<i4').tofile(REFS + '/musicgen_codes.i32')
    print(codes.reshape(-1, codes.shape[-1])[:, :10])
