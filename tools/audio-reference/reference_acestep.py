"""ACE-Step 1.5 turbo reference (the checkpoint's own modeling code run on CPU in fp32, with the Qwen3 caption encoder and diffusers' AutoencoderOobleck): a 10 s instrumental
from text only (no LM planner, no reference audio, the silence latent as source, chunk mask 2.0 as in the official pipeline's default "auto" mode), 8 Euler steps with shift 3.
Writes the prompt ids, caption rows, lyric embeddings, condition rows, noise, one velocity at t = 1, the final latents and the decoded audio (channel-major tensors; the
audio is transposed to frame-major) to ~/models/audio-models/refs/acestep_*. Needs the folders of ACE-Step/Ace-Step1.5 in ~/models/audio-models/ace-step15 and the official
modeling code importable from the transformer folder (modeling_acestep_v15_turbo.py, which needs vector-quantize-pytorch, einops, diffusers, transformers)."""
import os, sys, time, torch, json, math
sys.path.insert(0, os.path.expanduser("~/models/audio-models/ace-step15/acestep-v15-turbo"))
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModel
import modeling_acestep_v15_turbo as M
from configuration_acestep_v15 import AceStepConfig
from diffusers import AutoencoderOobleck
torch.set_num_threads(4)
ROOT = os.path.expanduser("~/models/audio-models/ace-step15")
root = ROOT
seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = 42
caption = "A calm acoustic guitar melody with soft piano, warm and relaxed"
lyrics = "[Instrumental]"
language = "unknown"
cfg = AceStepConfig.from_pretrained(root + "/acestep-v15-turbo")
cfg._attn_implementation = "eager"
model = M.AceStepConditionGenerationModel(cfg)
sd = load_file(root + "/acestep-v15-turbo/model.safetensors")
print(model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False))
model.eval().float()
silence = torch.load(root + "/acestep-v15-turbo/silence_latent.pt", weights_only=True).float().transpose(1, 2).contiguous()
print("silence", silence.shape)
tok = AutoTokenizer.from_pretrained(root + "/Qwen3-Embedding-0.6B")
qwen = AutoModel.from_pretrained(root + "/Qwen3-Embedding-0.6B", torch_dtype=torch.float32).eval()
instr = "Fill the audio semantic mask based on the given conditions:"
metas = f"- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: {int(seconds)} seconds\n"
prompt = f"# Instruction\n{instr}\n\n# Caption\n{caption}\n\n# Metas\n{metas}<|endoftext|>\n"
lyr = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"
ti = tok(prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
li = tok(lyr, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
out = {}
with torch.no_grad():
    text_h = qwen(input_ids=ti.input_ids, attention_mask=ti.attention_mask).last_hidden_state
    lyric_h = qwen.embed_tokens(li.input_ids)
    T = int(seconds * 25)
    src = silence[0, :T, :]
    if src.shape[0] < T:
        src = silence[0].repeat((T + silence.shape[1] - 1) // silence.shape[1], 1)[:T]
    src = src[None]
    chunk = torch.full((1, T, 64), 2.0)
    refer = silence[:, :750, :]
    order = torch.zeros(1, dtype=torch.long)
    enc_h, enc_m, ctx = model.prepare_condition(
        text_h, ti.attention_mask.bool(), lyric_h, li.attention_mask.bool(), refer, order, src,
        torch.ones(1, T), silence, src, chunk, torch.zeros(1, dtype=torch.bool))
    timbre, _ = model.encoder.timbre_encoder(refer, order)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn((1, T, 64), generator=g)
    res = model.generate_audio(text_h, ti.attention_mask.bool(), lyric_h, li.attention_mask.bool(), refer, order, src, chunk,
        torch.zeros(1, dtype=torch.bool), silence_latent=silence, seed=seed, fix_nfe=8, shift=3.0)
    lat = res["target_latents"]
    # one decoder evaluation at t = 1 for a per-step check
    t1 = torch.ones(1)
    v1 = model.decoder(hidden_states=noise, timestep=t1, timestep_r=t1, attention_mask=torch.ones(1, T), encoder_hidden_states=enc_h,
        encoder_attention_mask=enc_m, context_latents=ctx, use_cache=False)[0]
    vae = AutoencoderOobleck.from_pretrained(root + "/vae").float().eval()
    wav = vae.decode(lat.transpose(1, 2)).sample
print("wav", wav.shape, wav.abs().max())
import numpy as np
REFS = os.path.expanduser("~/models/audio-models/refs")
os.makedirs(REFS, exist_ok=True)
tensors = {"text_ids": ti.input_ids.int(), "lyric_ids": li.input_ids.int(), "text_hidden": text_h[0], "lyric_emb": lyric_h[0], "timbre": timbre[0], "encoder_hidden": enc_h[0],
    "context": ctx[0], "noise": noise[0], "v_t1": v1[0], "latent": lat[0], "wav": wav[0].T}
for name, t in tensors.items():
    ids = name.endswith("_ids")
    np.ascontiguousarray(t.numpy().astype("<i4" if ids else "<f4")).tofile(f"{REFS}/acestep_{name}.{'i32' if ids else 'f32'}")
print("done")
