"""Reference of VibeVoice-1.5B text to speech (transformers 5.x VibeVoiceForConditionalGeneration, CPU f32, weights of the original Microsoft checkpoint renamed to transformers' layout).

Usage: python make_vibevoice_reference.py <original checkpoint dir> <hf-layout metadata dir (config, tokenizer, chat template)> <output dir> <script> [--voice wav] [--frames N] [--seed S]
<script> is the text with lines "Speaker N: ..." (roles are the labels N). Greedy language model, DPM-Solver 10 steps, guidance 1.3, noise recorded.
Writes input_ids.bin, tokens.bin [steps] (the chosen token per step), logits4.bin [steps, 4] (audio start, audio end, audio, end of text), hidden.bin [steps, hidden] (positive hidden state per
step, recomputed along the generated tokens with the uncached modeling), negative_hidden.bin [frames, hidden], noise.bin [frames, 64], latents.bin [frames, 64] (diffusion output), audio.bin (24 kHz), voice_latents.bin [frames, 64] with --voice (the scaled acoustic
features of the recording, no sampling noise), info.json.
"""
import sys, os, re, json, argparse, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vibevoice_original_layout import load_tts
torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('model_dir'); ap.add_argument('hf_meta'); ap.add_argument('out'); ap.add_argument('script')
ap.add_argument('--voice'); ap.add_argument('--frames', type=int, default=40); ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--voice-seconds', type=float, default=4.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from transformers import AutoProcessor
from diffusers import DPMSolverMultistepScheduler
processor = AutoProcessor.from_pretrained(a.hf_meta)
model = load_tts(a.model_dir, a.hf_meta)
turns = []
for line in a.script.split('\n'):
    m = re.match(r'\s*Speaker\s+(\S+?)\s*:\s*(.*)', line)
    if m: turns.append((m.group(1), m.group(2).strip()))
messages = [{"role": r, "content": [{"type": "text", "text": t}]} for r, t in turns]
audio = None
if a.voice:
    y, sr = sf.read(a.voice, dtype='float32')
    if y.ndim > 1: y = y.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=24000)[: int(a.voice_seconds * 24000)]
    first = turns[0][0]
    messages[0]["content"].insert(0, {"type": "audio", "audio": y})
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
ids = inputs['input_ids'][0]
print('prompt tokens', len(ids))
scheduler = DPMSolverMultistepScheduler(beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
cfg = model.config
four = [cfg.audio_bos_token_id, cfg.audio_eos_token_id, cfg.audio_token_id, cfg.eos_token_id]
pos_logits, pos_hidden, neg_hidden, noises, latents = [], [], [], [], []
state = {'negative': False, 'frame': 0}
def hook(mod, args, kwargs, out):
    h = out.logits  # logits [1, T, V]
    if state['negative']:
        neg_hidden.append(None)  # filled below from the last hidden through a second hook
    else:
        pos_logits.append(out.logits[0, -1, four].detach().clone())
model.register_forward_hook(hook, with_kwargs=True)
lm = model.model.language_model
last = {}
lm.register_forward_hook(lambda m, i, kw, o: last.__setitem__('h', o.last_hidden_state[0, -1].detach().clone()), with_kwargs=True)
def hook2(mod, args, kwargs, out):
    if state['negative']:
        neg_hidden[-1] = last['h']
    else:
        pos_hidden.append(last['h'])
model.register_forward_hook(hook2, with_kwargs=True)
orig_neg = model._step_negative_branch
def neg_wrapper(*args, **kwargs):
    state['negative'] = True
    try: return orig_neg(*args, **kwargs)
    finally: state['negative'] = False
model._step_negative_branch = neg_wrapper
orig_sample = model._sample_audio_latent
def sample_wrapper(positive_condition, negative_condition, noise_scheduler, num_diffusion_steps, guidance_scale):
    seed = a.seed + state['frame']
    torch.manual_seed(seed); noise = torch.randn(2, cfg.audio_config.hidden_size)
    torch.manual_seed(seed)
    out = orig_sample(positive_condition, negative_condition, noise_scheduler, num_diffusion_steps, guidance_scale)
    noises.append(noise[0].clone()); latents.append(out[0, 0].detach().clone()); state['frame'] += 1
    return out
model._sample_audio_latent = sample_wrapper
gen_kwargs = {}
if a.voice: gen_kwargs['sample'] = False
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=a.frames, do_sample=False, guidance_scale=1.3, num_diffusion_steps=10, noise_scheduler=scheduler, return_dict_in_generate=True, **gen_kwargs)
tokens = out.sequences[0, len(ids):]
steps = len(tokens)
print('steps', steps, 'frames', len(latents), 'tokens', tokens.tolist()[:10], '...', tokens.tolist()[-3:])
audio_out = out.audio[0]
# Teacher-forced recomputation along the generated tokens with the full-sequence (uncached) modeling: the language model's hidden state and logits of every step
# (forward hooks of the cached generation loop are not reliable for these).
with torch.no_grad():
    wav = audio_out.reshape(1, 1, -1).float()
    sem = model.model.semantic_tokenizer_encoder(wav).latents[0]
    lat = torch.stack(latents)
    frame_rows = model.model.multi_modal_projector(lat) + model.model.semantic_connector(sem)
    emb = model.model.language_model.embed_tokens(ids[None])[0]
    if a.voice:
        voice_rows = model.model.get_audio_features(inputs['input_values'], inputs['padding_mask'], sample=False).pooler_output
        emb[ids == cfg.audio_token_id] = voice_rows
    seq, f = [emb], 0
    for t in tokens[:-1].tolist():
        if t == cfg.audio_token_id:
            seq.append(frame_rows[f:f + 1]); f += 1
        else:
            seq.append(model.model.language_model.embed_tokens(torch.tensor([t])))
    o = model.model.language_model(inputs_embeds=torch.cat(seq, 0)[None]).last_hidden_state[0]
    T = len(ids)
    pos_hidden = [o[T - 1 + k] for k in range(steps)]
    pos_logits = [model.lm_head(o[T - 1 + k])[four] for k in range(steps)]
np.array(ids.tolist(), dtype='<i4').tofile(a.out + '/input_ids.bin')
np.array(tokens.tolist(), dtype='<i4').tofile(a.out + '/tokens.bin')
torch.stack(pos_logits[:steps]).numpy().astype('<f4').tofile(a.out + '/logits4.bin')
torch.stack(pos_hidden[:steps]).numpy().astype('<f4').tofile(a.out + '/hidden.bin')
torch.stack(neg_hidden).numpy().astype('<f4').tofile(a.out + '/negative_hidden.bin')
torch.stack(noises).numpy().astype('<f4').tofile(a.out + '/noise.bin')
torch.stack(latents).numpy().astype('<f4').tofile(a.out + '/latents.bin')
audio_out.reshape(-1).float().numpy().astype('<f4').tofile(a.out + '/audio.bin')
if a.voice:
    with torch.no_grad():
        x = inputs['input_values']; pm = inputs['padding_mask']
        feats = model.model.get_audio_features(x, pm, sample=False).last_hidden_state
    feats.numpy().astype('<f4').tofile(a.out + '/voice_latents.bin')
json.dump({"script": a.script, "turns": turns, "voice": a.voice, "steps": steps, "frames": len(latents), "guidance": 1.3, "diffusion_steps": 10, "seed": a.seed, "sample_rate": 24000,
           "input_ids": ids.tolist(), "audio_samples": int(audio_out.reshape(-1).shape[0])}, open(a.out + '/info.json', 'w'), indent=1)
