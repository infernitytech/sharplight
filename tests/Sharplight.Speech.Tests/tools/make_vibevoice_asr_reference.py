"""Reference of VibeVoice-ASR (transformers 5.x VibeVoiceAsrForConditionalGeneration, CPU f32, weights of the original Microsoft checkpoint renamed to transformers' layout), greedy.

Usage: python make_vibevoice_asr_reference.py <original checkpoint dir> <hf-layout metadata dir (config, tokenizer, chat template)> <wav> <output dir> [max new tokens] [seconds]
Writes input_ids.bin, tokens.bin [steps] (greedy tokens), top2_ids.bin / top2_values.bin [steps, 2] (logits of the step along the greedy path), audio.bin (24 kHz as given to the model), audio_rows.bin
(the model's audio embedding rows [rows, hidden], acoustic latents without sampling noise) and info.json (the decoded text).
"""
import sys, os, json, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Sharplight.TextToSpeech.Tests", "tools"))
from vibevoice_original_layout import load_asr
torch.set_num_threads(4)
model_dir, hf_meta, wav, out = sys.argv[1:5]
max_new = int(sys.argv[5]) if len(sys.argv) > 5 else 160
seconds = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
os.makedirs(out, exist_ok=True)
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(hf_meta)
model = load_asr(model_dir, hf_meta)
y, sr = sf.read(wav, dtype='float32')
if y.ndim > 1: y = y.mean(1)
y = librosa.resample(y, orig_sr=sr, target_sr=24000)[: int(seconds * 24000)]
inputs = processor.apply_transcription_request(audio=y)
ids = inputs['input_ids'][0]
print('prompt tokens', len(ids))
steps_logits = []
with torch.no_grad():
    audio_rows = model.model.get_audio_features(inputs['input_values'], inputs['padding_mask'], sample=False).pooler_output
    gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, sample=False, return_dict_in_generate=True, output_logits=True)
tokens = gen.sequences[0, len(ids):]
end = next((i for i, t in enumerate(tokens.tolist()) if t in (151645, 151643)), len(tokens))   # <|im_end|> or <|endoftext|>: the end of the turn
L = torch.stack([l[0] for l in gen.logits]); top = L.topk(2, dim=-1)
print('steps', len(tokens), repr(processor.decode(tokens, skip_special_tokens=False)[:400]))
np.array(ids.tolist(), dtype='<i4').tofile(out + '/input_ids.bin')
np.array(tokens.tolist(), dtype='<i4').tofile(out + '/tokens.bin')
top.indices.numpy().astype('<i4').tofile(out + '/top2_ids.bin'); top.values.numpy().astype('<f4').tofile(out + '/top2_values.bin')
y.astype('<f4').tofile(out + '/audio.bin')
audio_rows.numpy().astype('<f4').tofile(out + '/audio_rows.bin')
json.dump({"samples": int(len(y)), "rows": int(audio_rows.shape[0]), "text": processor.decode(tokens[:end], skip_special_tokens=True), "raw": processor.decode(tokens[:end], skip_special_tokens=False), "end": end, "input_ids": ids.tolist(), "steps": len(tokens)}, open(out + '/info.json', 'w'), indent=1)
