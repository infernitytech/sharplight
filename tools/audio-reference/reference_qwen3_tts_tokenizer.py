"""Reference tensors of the Qwen3-TTS 12 Hz tokenizer (official qwen_tts modeling code on transformers' MimiModel); inputs: mimi_in.f32 (jfk, 24 kHz, 125 frames)."""
import sys, importlib.util, os, numpy as np, torch
torch.set_num_threads(4)
base = sys.argv[1]  # directory of the qwen_tts tokenizer_12hz sources
model_dir = sys.argv[2]
refs = os.path.expanduser("~/models/audio-models/refs")
import types
pkg = types.ModuleType('qtok_pkg'); pkg.__path__ = [base]; sys.modules['qtok_pkg'] = pkg
def load(name, path):
    spec = importlib.util.spec_from_file_location('qtok_pkg.' + name, path); m = importlib.util.module_from_spec(spec); sys.modules['qtok_pkg.' + name] = m; spec.loader.exec_module(m); return m
load('configuration_qwen3_tts_tokenizer_v2', base + '/configuration_qwen3_tts_tokenizer_v2.py')
mdl = load('modeling_qwen3_tts_tokenizer_v2', base + '/modeling_qwen3_tts_tokenizer_v2.py')
model = mdl.Qwen3TTSTokenizerV2Model.from_pretrained(model_dir, dtype=torch.float32).eval()
# transformers 5 loads on the meta device and leaves the non-persistent rotary buffer uninitialized: recompute it.
rot = model.decoder.pre_transformer.rotary_emb
dc = model.config.decoder_config
rot.inv_freq = 1.0 / (dc.rope_theta ** (torch.arange(0, dc.head_dim, 2, dtype=torch.float32) / dc.head_dim))
rot.original_inv_freq = rot.inv_freq
x = torch.from_numpy(np.fromfile(refs + '/mimi_in.f32', dtype='<f4'))
# transformers 5 renamed the internals of MimiModel (the official wrapper's encoder no longer loads), so the encoder half is loaded as a plain Mimi checkpoint.
import json, tempfile, shutil
from safetensors.torch import load_file, save_file
from transformers import MimiModel
tmp = os.path.join(os.environ.get('REF_TMP', tempfile.gettempdir()), 'mimi-enc')
os.makedirs(tmp, exist_ok=True)
cfg = json.load(open(model_dir + '/config.json'))['encoder_config']
cfg['model_type'] = 'mimi'; cfg['architectures'] = ['MimiModel']
json.dump(cfg, open(tmp + '/config.json', 'w'))
weights = {k[len('encoder.'):]: v for k, v in load_file(model_dir + '/model.safetensors').items() if k.startswith('encoder.')}
save_file(weights, tmp + '/model.safetensors', metadata={'format': 'pt'})
mimi = MimiModel.from_pretrained(tmp, dtype=torch.float32).eval()
for name, buf in mimi.named_buffers():
    if name.endswith('inv_freq'):
        assert torch.isfinite(buf).all() and abs(float(buf[0]) - 1.0) < 1e-6, name
with torch.no_grad():
    codes = mimi.encode(x[None, None, :], num_quantizers=32).audio_codes[0, :16].t().contiguous()  # [T, 16]
    print(codes.shape)
    codes.t().contiguous().numpy().astype('<i4').tofile(refs + '/qtok_codes.i32')
    wav = model.decode(codes[None], return_dict=True).audio_values[0]
    print(wav.shape, wav.abs().max())
    wav.numpy().astype('<f4').tofile(refs + '/qtok_dec.f32')
