"""Reference of Dia-1.6B (nari-labs; transformers DiaForConditionalGeneration, CPU f32): greedy decoding with classifier-free guidance, with and without an audio prompt.

The original checkpoint (the `dia` package's layout: [in, heads, dim] projections, fused gate/up, one embedding table per codebook) is renamed to the transformers layout in memory.

Usage: python make_dia_reference.py <Dia-1.6B dir> <dac_44khz dir> <hf config.json of Dia-1.6B-0626> <output dir> [text] [max steps] [prompt wav] [prompt text]
Writes input_ids.bin (text ids), decoder_input.bin [rows, 9] (prompt rows of the delayed sequence), sequence.bin [L, 9] (the delayed sequence including the generated rows),
logits.bin f32 [steps, 2, 9, 1028] (conditional and unconditional logits of every generated step), codes.bin [frames, 9] (generated frames, delay reverted),
audio.bin (DAC decode of the generated frames) and info.json. With a prompt: prompt_codes.bin [frames, 9] (DAC encode), prompt_audio.bin (44.1 kHz mono f32).
"""
import sys, os, json, math, numpy as np, torch
torch.set_num_threads(8)
from safetensors.torch import load_file
from transformers import DiaConfig, DiaForConditionalGeneration, DiaTokenizer, DiaFeatureExtractor, DiaProcessor, DacModel, GenerationConfig

model_dir, dac_dir, hf_config, out = sys.argv[1:5]
text = sys.argv[5] if len(sys.argv) > 5 else "[S1] Hello there, this is a short test. [S2] Nice to meet you."
max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 200
prompt_wav = sys.argv[7] if len(sys.argv) > 7 else None
prompt_text = sys.argv[8] if len(sys.argv) > 8 else ""
os.makedirs(out, exist_ok=True)

def convert(sd):
    o = {}
    def lin(w):  # [in, heads, dim] -> [heads*dim, in]
        return w.reshape(w.shape[0], -1).t().contiguous()
    def lino(w):  # [heads, dim, out] -> [out, heads*dim]
        return w.reshape(-1, w.shape[-1]).t().contiguous()
    o["model.encoder.embedding.weight"] = sd["encoder.embedding.weight"]
    o["model.encoder.norm.weight"] = sd["encoder.norm.weight"]
    for i in range(12):
        p, q = f"encoder.layers.{i}.", f"model.encoder.layers.{i}."
        o[q + "pre_sa_norm.weight"] = sd[p + "pre_sa_norm.weight"]
        o[q + "post_sa_norm.weight"] = sd[p + "post_sa_norm.weight"]
        for n in "qkv":
            o[q + f"self_attention.{n}_proj.weight"] = lin(sd[p + f"self_attention.{n}_proj.weight"])
        o[q + "self_attention.o_proj.weight"] = lino(sd[p + "self_attention.o_proj.weight"])
        o[q + "mlp.gate_up_proj.weight"] = sd[p + "mlp.wi_fused.weight"].reshape(sd[p + "mlp.wi_fused.weight"].shape[0], -1).t().contiguous()
        o[q + "mlp.down_proj.weight"] = sd[p + "mlp.wo.weight"].t().contiguous()
    o["model.decoder.embeddings.embed.weight"] = torch.cat([sd[f"decoder.embeddings.{c}.weight"] for c in range(9)], 0)
    o["model.decoder.norm.weight"] = sd["decoder.norm.weight"]
    for i in range(18):
        p, q = f"decoder.layers.{i}.", f"model.decoder.layers.{i}."
        for n in ("pre_sa_norm", "pre_ca_norm", "pre_mlp_norm"):
            o[q + n + ".weight"] = sd[p + n + ".weight"]
        for a in ("self_attention", "cross_attention"):
            for n in "qkv":
                o[q + f"{a}.{n}_proj.weight"] = lin(sd[p + f"{a}.{n}_proj.weight"])
            o[q + f"{a}.o_proj.weight"] = lino(sd[p + f"{a}.o_proj.weight"])
        o[q + "mlp.gate_up_proj.weight"] = sd[p + "mlp.wi_fused.weight"].reshape(sd[p + "mlp.wi_fused.weight"].shape[0], -1).t().contiguous()
        o[q + "mlp.down_proj.weight"] = sd[p + "mlp.wo.weight"].t().contiguous()
    o["logits_dense.weight"] = sd["decoder.logits_dense.weight"].reshape(sd["decoder.logits_dense.weight"].shape[0], -1).t().contiguous()
    return o

config = DiaConfig.from_pretrained(os.path.dirname(hf_config)) if os.path.exists(os.path.join(os.path.dirname(hf_config), "config.json")) and False else DiaConfig(**json.load(open(hf_config)))
model = DiaForConditionalGeneration(config)
missing = model.load_state_dict(convert(load_file(os.path.join(model_dir, "model.safetensors"))), strict=False)
print("missing", missing.missing_keys, "unexpected", missing.unexpected_keys)
model = model.float().eval()
dac = DacModel.from_pretrained(dac_dir).float().eval()
tokenizer = DiaTokenizer()
fe = DiaFeatureExtractor(feature_size=1, sampling_rate=44100, hop_length=512)
processor = DiaProcessor(fe, tokenizer, dac)
audio = None
if prompt_wav:
    import soundfile as sf, scipy.signal as ss
    w, sr = sf.read(prompt_wav, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 44100:
        w = ss.resample_poly(w, 44100, sr).astype("float32")
    audio = w
    audio.astype("<f4").tofile(out + "/prompt_audio.bin")
    text = prompt_text + " " + text if prompt_text else text
inputs = processor(text=[text], audio=audio, padding=True, return_tensors="pt")
print({k: v.shape for k, v in inputs.items()})
inputs["input_ids"].numpy().astype("<i4").tofile(out + "/input_ids.bin")
dec_in = inputs["decoder_input_ids"][0]
dec_in.numpy().astype("<i4").tofile(out + "/decoder_input.bin")
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
print("text ids", inputs["input_ids"].shape, "decoder rows", dec_in.shape, "prompt len", prompt_len)
logit_steps = []
def hook(mod, args, kwargs, r):
    logit_steps.append(r.logits[:, -1].detach().clone().reshape(2, 9, -1))
model.register_forward_hook(hook, with_kwargs=True)
gen = GenerationConfig(do_sample=False, guidance_scale=3.0, top_k=50, max_new_tokens=max_steps, bos_token_id=1026, eos_token_id=1024, pad_token_id=1025)
with torch.no_grad():
    outputs = model.generate(**inputs, generation_config=gen)
print("generated", outputs.shape)
seq = outputs[0]
seq.numpy().astype("<i4").tofile(out + "/sequence.bin")
L = torch.stack(logit_steps)  # [calls, 2, 9, V]; the first call is the prefill's last row
L.numpy().astype("<f4").tofile(out + "/logits.bin")
audios = processor.batch_decode(outputs, audio_prompt_len=prompt_len if audio is not None else None)
wav = audios[0].numpy()
wav.astype("<f4").tofile(out + "/audio.bin")
# the reverted frames
sp, ep = prompt_len if audio is not None else int((outputs[:, :, 0] == 1026).sum(-1)[0]), outputs.shape[1] - int((outputs[:, :, 0] == 1025).sum(-1)[0]) - 1
idx = processor.build_indices(1, outputs.shape[1], 9, [0, 8, 9, 10, 11, 12, 13, 14, 15], revert=True)
rev = processor.apply_audio_delay(outputs, -1, -1, idx)[0, sp:ep]
rev.numpy().astype("<i4").tofile(out + "/codes.bin")
if audio is not None:
    with torch.no_grad():
        a = torch.from_numpy(audio)[None, None]
        pad = (-a.shape[-1]) % 512
        a = torch.nn.functional.pad(a, (0, pad))
        pc = dac.encode(a).audio_codes[0].t()  # [frames, 9]
    pc.numpy().astype("<i4").tofile(out + "/prompt_codes.bin")
json.dump({"text": text, "prompt_len": int(prompt_len), "steps": len(logit_steps), "frames": int(rev.shape[0]), "total_rows": int(outputs.shape[1]), "samples": int(wav.shape[0]),
           "start": int(sp), "end": int(ep)}, open(out + "/info.json", "w"))
print("frames", rev.shape, "samples", wav.shape)
