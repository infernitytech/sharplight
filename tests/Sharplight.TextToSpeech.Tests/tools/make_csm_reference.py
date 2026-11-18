"""Reference of CSM-1B (sesame/csm-1b, transformers CsmForConditionalGeneration, CPU f32): greedy backbone and greedy depth decoder, no audio context.

Usage: python make_csm_reference.py <model dir> <output dir> [text] [speaker] [max frames]
Writes input_ids.bin, codes.bin [frames, 32], top2_{ids,values}.bin (backbone logits per step), depth_top2_{ids,values}.bin (depth decoder logits per code),
hidden.bin (backbone last hidden per step), audio.bin (Mimi decode of the frames, 24 kHz) and info.json.
"""
import sys, os, json, numpy as np, torch
torch.set_num_threads(4)
from transformers import CsmForConditionalGeneration, AutoProcessor
model_dir, out = sys.argv[1], sys.argv[2]
text = sys.argv[3] if len(sys.argv) > 3 else "Hello from Sesame, this is a short test."
speaker = sys.argv[4] if len(sys.argv) > 4 else "0"
max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else 40
os.makedirs(out, exist_ok=True)
processor = AutoProcessor.from_pretrained(model_dir)
model = CsmForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
conversation = [{"role": speaker, "content": [{"type": "text", "text": text}]}]
inputs = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
ids = inputs["input_ids"][0].tolist()
print("ids", ids)
back_logits, back_hidden, depth_logits = [], [], []
def bhook(mod, args, kwargs, r):
    back_logits.append(r.logits[0, -1].detach().clone())
    back_hidden.append(r.hidden_states[-1][0, -1].detach().clone())
model.register_forward_hook(bhook, with_kwargs=True)
model.depth_decoder.register_forward_hook(lambda mod, args, r: depth_logits.append(r.logits[0, -1].detach().clone()))
with torch.no_grad():
    codes = model.generate(**inputs, do_sample=False, depth_decoder_do_sample=False, max_new_tokens=max_frames, output_audio=False)
codes = codes[0]  # [frames, 32]
print("frames", codes.shape)
n = codes.shape[0]
# Frames after the end-of-audio frame are not real: stop at an all-zero frame.
codes.numpy().astype('<i4').tofile(out + '/codes.bin')
np.array(ids, dtype='<i4').tofile(out + '/input_ids.bin')
L = torch.stack(back_logits[:n]); top = L.topk(2, dim=-1)
top.values.numpy().astype('<f4').tofile(out + '/top2_values.bin'); top.indices.numpy().astype('<i4').tofile(out + '/top2_ids.bin')
torch.stack(back_hidden[:n]).numpy().astype('<f4').tofile(out + '/hidden.bin')
D = torch.stack(depth_logits[:31 * n]); dt = D.topk(2, dim=-1)
dt.values.numpy().astype('<f4').tofile(out + '/depth_top2_values.bin'); dt.indices.numpy().astype('<i4').tofile(out + '/depth_top2_ids.bin')
with torch.no_grad():
    audio = model.codec_model.decode(codes.transpose(0, 1)[None]).audio_values[0, 0]
audio.numpy().astype('<f4').tofile(out + '/audio.bin')
json.dump({"text": text, "speaker": speaker, "input_ids": ids, "frames": int(n), "sample_rate": 24000, "audio_samples": int(audio.shape[0])}, open(out + '/info.json', 'w'), indent=1)
