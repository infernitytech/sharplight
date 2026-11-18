"""Reference of Qwen3-TTS CustomVoice (official qwen_tts 0.1.1 package, transformers 4.57.3, CPU f32): greedy talker and code predictor, non-streaming text.

Usage: python make_qwen3_tts_reference.py <dir containing qwen_tts> <model dir> <output dir> [text] [speaker] [language]
Writes prompt.bin (talker prompt rows), codes.bin [frames, 16], top2_{ids,values}.bin (talker logits per step), cp_top2_{ids,values}.bin (code predictor
logits per predicted code), hidden.bin and info.json. audio.bin (the decoded frames) comes from the 12 Hz codec reference (decode of codes.bin, see
tools/audio-reference/reference_qwen3_tts_tokenizer.py; transformers 5 with the rotary buffer recomputed).
"""
import sys, os, json, numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, sys.argv[1])  # directory containing the qwen_tts package
from qwen_tts import Qwen3TTSModel
model_dir, out = sys.argv[2], sys.argv[3]
text = sys.argv[4] if len(sys.argv) > 4 else "Hello! This is a test of the speech synthesis."
speaker = sys.argv[5] if len(sys.argv) > 5 else "ryan"
language = sys.argv[6] if len(sys.argv) > 6 else "english"
os.makedirs(out, exist_ok=True)
tts = Qwen3TTSModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
m = tts.model
for name, buf in m.named_buffers():
    if name.endswith('inv_freq'):
        assert torch.isfinite(buf).all() and abs(float(buf[0]) - 1.0) < 1e-6, name
talker = m.talker
calls = {"prefill": None, "logits": [], "hidden": []}
def hook(mod, args, kwargs, r):
    e = kwargs.get("inputs_embeds")
    if e is not None and e.shape[1] > 1 and calls["prefill"] is None:
        calls["prefill"] = e[0].detach().clone()
    calls["logits"].append(r.logits[0, -1].detach().clone())
    calls["hidden"].append(r.past_hidden[0, -1].detach().clone())
talker.register_forward_hook(hook, with_kwargs=True)
cp_calls = []
talker.code_predictor.register_forward_hook(lambda mod, args, r: cp_calls.append(r.logits[0, -1].detach().clone()))
ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0][0].tolist()
with torch.no_grad():
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    codes, hid = m.generate(input_ids=input_ids, instruct_ids=[None], languages=[language], speakers=[speaker], non_streaming_mode=True,
                            do_sample=False, subtalker_dosample=False, max_new_tokens=600, repetition_penalty=1.05)
    sr = 24000
codes = codes[0]
print("frames", codes.shape, "prompt rows", calls["prefill"].shape)
calls["prefill"].numpy().astype('<f4').tofile(out + '/prompt.bin')
codes.numpy().astype('<i4').tofile(out + '/codes.bin')
n = codes.shape[0] + 1
L = torch.stack(calls["logits"][:n])
top = L.topk(2, dim=-1)
top.values.numpy().astype('<f4').tofile(out + '/top2_values.bin')
top.indices.numpy().astype('<i4').tofile(out + '/top2_ids.bin')
torch.stack(calls["hidden"][:n]).numpy().astype('<f4').tofile(out + '/hidden.bin')
C = torch.stack(cp_calls[:15 * codes.shape[0]])
ct = C.topk(2, dim=-1)
ct.values.numpy().astype('<f4').tofile(out + '/cp_top2_values.bin')
ct.indices.numpy().astype('<i4').tofile(out + '/cp_top2_ids.bin')

json.dump({"text": text, "speaker": speaker, "language": language, "input_ids": ids, "frames": int(codes.shape[0]), "prompt_rows": int(calls["prefill"].shape[0]),
           "sample_rate": sr}, open(out + '/info.json', 'w'), indent=1)
