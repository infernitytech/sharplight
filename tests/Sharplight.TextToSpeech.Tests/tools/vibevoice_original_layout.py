"""Loads transformers' VibeVoice models (HF layout) from the ORIGINAL microsoft checkpoints by renaming the tensors (shared by the VibeVoice reference scripts)."""
import re, json, glob, torch
from safetensors import safe_open

def rename_tokenizer(k, enc_name, dec_name):
    """original acoustic/semantic tokenizer key (without model.<x>_tokenizer.) -> HF key under the tower"""
    m = re.match(r'(encoder|decoder)\.(.*)', k)
    side, rest = m.group(1), m.group(2)
    tower = enc_name if side == 'encoder' else dec_name
    if rest.startswith('head.conv.conv.'):
        return f'{tower}.head.conv.' + rest[len('head.conv.conv.'):]
    m = re.match(r'(downsample_layers|upsample_layers)\.(\d+)\.0\.(.*)', rest)
    if m:
        i = int(m.group(2)); tail = m.group(3)
        if i == 0:
            return f'{tower}.stem.' + tail                  # conv.conv.weight
        return f'{tower}.conv_layers.{i-1}.' + tail         # conv.conv.weight / convtr.convtr.weight
    m = re.match(r'stages\.(\d+)\.(\d+)\.(.*)', rest)
    if m:
        i, j, tail = int(m.group(1)), int(m.group(2)), m.group(3)
        tail = tail.replace('mixer.conv.conv.conv.', 'mixer.conv.')
        if i == 0:
            return f'{tower}.stem.stage.{j}.{tail}'
        return f'{tower}.conv_layers.{i-1}.stage.{j}.{tail}'
    raise KeyError(k)

def rename_tts(k):
    if k.startswith('model.language_model.'): return k
    m = re.match(r'model\.acoustic_tokenizer\.(.*)', k)
    if m: return 'model.audio_tower.' + rename_tokenizer(m.group(1), 'encoder', 'decoder')
    m = re.match(r'model\.semantic_tokenizer\.(.*)', k)
    if m: return 'model.semantic_tokenizer_encoder.' + rename_tokenizer(m.group(1), 'X', 'X').replace('X.', '', 1)
    for src, dst in (('acoustic_connector', 'multi_modal_projector'), ('semantic_connector', 'semantic_connector')):
        m = re.match(rf'model\.{src}\.(fc1|norm|fc2)\.(.*)', k)
        if m: return f'model.{dst}.' + {'fc1': 'linear_1', 'norm': 'act', 'fc2': 'linear_2'}[m.group(1)] + '.' + m.group(2)
    if k == 'model.speech_scaling_factor': return 'model.latent_scaling_factor'
    if k == 'model.speech_bias_factor': return 'model.latent_bias_factor'
    m = re.match(r'model\.prediction_head\.(.*)', k)
    if m:
        r = m.group(1)
        r = r.replace('t_embedder.mlp.0.', 'timestep_proj.fc1.').replace('t_embedder.mlp.2.', 'timestep_proj.fc2.')
        r = re.sub(r'^layers\.(\d+)\.adaLN_modulation\.1\.', r'layers.\1.linear.', r)
        r = r.replace('final_layer.adaLN_modulation.1.', 'final_layer.linear_1.').replace('final_layer.linear.', 'final_layer.linear_2.')
        return 'model.diffusion_head.' + r
    raise KeyError(k)

def load_original(model_dir):
    sd = {}
    for f in sorted(glob.glob(model_dir + '/*.safetensors')):
        with safe_open(f, 'pt') as s:
            for k in s.keys():
                sd[k] = s.get_tensor(k)
    return sd

def load_tts(model_dir, hf_meta, dtype=torch.float32):
    from transformers import VibeVoiceConfig, VibeVoiceForConditionalGeneration
    cfg = VibeVoiceConfig.from_pretrained(hf_meta)
    with torch.device('meta'):
        model = VibeVoiceForConditionalGeneration(cfg)
    sd = {rename_tts(k): v.to(dtype) for k, v in load_original(model_dir).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)[:2]
    print('missing', [m for m in missing if 'freq' not in m and m != 'lm_head.weight'][:10], 'unexpected', unexpected[:10])
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    # non-persistent buffers (frequencies) were created on the meta device
    for mod in model.modules():
        if hasattr(mod, 'freq') and mod.freq.is_meta:
            mod.freq = type(mod).compute_default_freq(mod.config)
    lm = model.model.language_model
    lm.rotary_emb = type(lm.rotary_emb)(config=lm.config)
    return model.eval()

def rename_asr(k):
    if k.startswith('model.language_model.') or k == 'lm_head.weight': return k
    m = re.match(r'model\.acoustic_tokenizer\.(.*)', k)
    if m:
        if m.group(1).startswith('decoder.'): return None
        return rename_tokenizer(m.group(1), 'model.acoustic_tokenizer_encoder', 'X')
    m = re.match(r'model\.semantic_tokenizer\.(.*)', k)
    if m: return rename_tokenizer(m.group(1), 'model.semantic_tokenizer_encoder', 'X')
    m = re.match(r'model\.(acoustic|semantic)_connector\.(fc1|norm|fc2)\.(.*)', k)
    if m: return f'model.multi_modal_projector.{m.group(1)}_' + {'fc1': 'linear_1', 'norm': 'norm', 'fc2': 'linear_2'}[m.group(2)] + '.' + m.group(3)
    raise KeyError(k)

def load_asr(model_dir, hf_meta, dtype=torch.float32):
    from transformers import VibeVoiceAsrConfig, VibeVoiceAsrForConditionalGeneration
    cfg = VibeVoiceAsrConfig.from_pretrained(hf_meta)
    with torch.device('meta'):
        model = VibeVoiceAsrForConditionalGeneration(cfg)
    sd = {}
    for k, v in load_original(model_dir).items():
        n = rename_asr(k)
        if n is not None: sd[n] = v.to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)[:2]
    print('missing', missing[:10], 'unexpected', unexpected[:10])
    lm = model.model.language_model
    lm.rotary_emb = type(lm.rotary_emb)(config=lm.config)
    return model.eval()
