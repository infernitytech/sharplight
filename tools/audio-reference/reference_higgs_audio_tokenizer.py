"""Reference tensors of the Higgs Audio v2 tokenizer (OmniVoice's audio_tokenizer folder) with transformers' HiggsAudioV2TokenizerModel.

Inputs: mimi_in.f32 (jfk, 24 kHz mono). Writes higgs_codes.i32 ([8, T] codes at the highest bandwidth, row-major), higgs_dec.f32 (decode of those codes) and
higgs_semantic.f32 (the HuBERT features averaged over layers, [T', 768], before the semantic encoder).
Usage: python reference_higgs_audio_tokenizer.py <audio_tokenizer dir>
"""
import os, sys, numpy as np, torch
from transformers import HiggsAudioV2TokenizerModel

torch.set_num_threads(4)
refs = os.path.expanduser("~/models/audio-models/refs")
model = HiggsAudioV2TokenizerModel.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
x = torch.from_numpy(np.fromfile(refs + "/mimi_in.f32", dtype="<f4"))[None, None, :]
with torch.no_grad():
    semantic = model._extract_semantic_features(x)[0]
    print("semantic", semantic.shape)
    semantic.numpy().astype("<f4").tofile(refs + "/higgs_semantic.f32")
    codes = model.encode(x, return_dict=False)  # [1, 8, T]
    print("codes", codes.shape)
    codes[0].contiguous().numpy().astype("<i4").tofile(refs + "/higgs_codes.i32")
    wav = model.decode(codes, return_dict=False)
    print("wav", wav.shape, float(wav.abs().max()))
    wav[0, 0].numpy().astype("<f4").tofile(refs + "/higgs_dec.f32")
