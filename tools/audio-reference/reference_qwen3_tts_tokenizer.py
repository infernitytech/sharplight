"""Reference tensors of the Qwen3-TTS 12 Hz tokenizer (official qwen_tts modeling code on transformers' MimiModel); inputs: mimi_in.f32 (jfk, 24 kHz, 125 frames)."""
import sys, importlib.util, os, numpy as np, torch
torch.set_num_threads(4)
base = sys.argv[1]  # directory of the qwen_tts tokenizer_12hz sources
model_dir = sys.argv[2]
refs = os.path.expanduser("~/models/audio-models/refs")
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
load('configuration_qwen3_tts_tokenizer_v2', base + '/configuration_qwen3_tts_tokenizer_v2.py')
mdl = load('modeling_qwen3_tts_tokenizer_v2', base + '/modeling_qwen3_tts_tokenizer_v2.py')
model = mdl.Qwen3TTSTokenizerV2Model.from_pretrained(model_dir, dtype=torch.float32).eval()
x = torch.from_numpy(np.fromfile(refs + '/mimi_in.f32', dtype='<f4'))
with torch.no_grad():
    codes = model.encode(x[None, :], padding_mask=torch.ones(1, x.shape[0], dtype=torch.long), return_dict=True).audio_codes[0]  # [T, 16]
    print(codes.shape)
    codes.t().contiguous().numpy().astype('<i4').tofile(refs + '/qtok_codes.i32')
    wav = model.decode(codes[None], return_dict=True).audio_values[0]
    print(wav.shape, wav.abs().max())
    wav.numpy().astype('<f4').tofile(refs + '/qtok_dec.f32')
