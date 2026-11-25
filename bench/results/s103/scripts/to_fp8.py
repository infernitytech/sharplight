# Converts a bf16 HF checkpoint's linear projection weights to FP8 E4M3 with 128 x 128 block scales (weight_scale_inv), like the
# published *-FP8 checkpoints. Usage: to_fp8.py <src dir> <dst dir>
import json, os, shutil, sys, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith('.safetensors') and not f.startswith('.') and os.path.isfile(os.path.join(src, f)) and f != 'model.safetensors.index.json':
        shutil.copy(os.path.join(src, f), dst)
cfg = json.load(open(os.path.join(src, 'config.json')))
cfg['quantization_config'] = {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]}
json.dump(cfg, open(os.path.join(dst, 'config.json'), 'w'), indent=2)
names = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
index = {'metadata': {}, 'weight_map': {}}
for f in sorted(x for x in os.listdir(src) if x.endswith('.safetensors')):
    t = load_file(os.path.join(src, f))
    out = {}
    for k, w in t.items():
        if k.endswith('.weight') and any(n in k for n in names) and w.dim() == 2 and w.shape[0] % 128 == 0 and w.shape[1] % 128 == 0:
            r, c = w.shape
            b = w.float().reshape(r // 128, 128, c // 128, 128)
            amax = b.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
            scale = amax / 448.0
            q = (b / scale).clamp(-448, 448).reshape(r, c).to(torch.float8_e4m3fn)
            out[k] = q.contiguous()
            out[k[:-len('weight')] + 'weight_scale_inv'] = scale.reshape(r // 128, c // 128).contiguous()
        else:
            out[k] = w
    save_file(out, os.path.join(dst, f), metadata={'format': 'pt'})
    for k in out:
        index['weight_map'][k] = f
json.dump(index, open(os.path.join(dst, 'model.safetensors.index.json'), 'w'), indent=2)
print('done', dst)
