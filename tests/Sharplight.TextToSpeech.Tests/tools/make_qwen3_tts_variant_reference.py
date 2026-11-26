"""Reference of the Qwen3-TTS Base (voice cloning) and VoiceDesign models (official qwen_tts 0.1.1 package, transformers 4.57.3, CPU f32): greedy talker and
code predictor with the wrapper's default text mode (generate_voice_clone: streaming text, generate_voice_design: non-streaming text).

Usage: python make_qwen3_tts_variant_reference.py <model dir> <output dir> <mode> [text] [language] [reference wav | instruction] [reference text]
  mode: clone (reference audio and its transcript, in-context prompt), xvector (the speaker embedding of the reference audio only) or design (an instruction).
Writes into the output directory:
- info.json: text, language, mode, instruction, reference text, input_ids, ref_ids, instruct_ids, frames, prompt_rows, trailing_rows, ref_frames;
- prompt.bin (the talker's prompt rows) and trailing.bin (the text rows added to the inputs of the first generation steps);
- codes.bin [frames, 16] (generated; at most 120 frames: greedy in-context cloning often does not stop), top2_{ids,values}.bin (talker logits per step), cp_top2_{ids,values}.bin (code predictor logits per predicted code);
- for clone / xvector: reference.bin (the 24 kHz reference samples), mel.bin [mel frames, 128] (the speaker encoder's log-mel), speaker.bin (the speaker
  embedding), ref_codes.bin [ref frames, 16] (the 12 Hz codec's codes of the reference);
- audio.bin: the wrapper's output audio (for clone: the reference codes and the generated ones decoded together, the reference's part cut off).
"""
import json
import os
import sys

import librosa
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

torch.set_num_threads(8)
model_dir, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
text = sys.argv[4] if len(sys.argv) > 4 else "Hello! This is a test of the speech synthesis."
language = sys.argv[5] if len(sys.argv) > 5 else "english"
extra = sys.argv[6] if len(sys.argv) > 6 else None
ref_text = sys.argv[7] if len(sys.argv) > 7 else None
MAX_FRAMES = 120
os.makedirs(out, exist_ok=True)


def save(name, array, dtype):
    np.ascontiguousarray(array).astype(dtype).tofile(os.path.join(out, name + ".bin"))


tts = Qwen3TTSModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
m = tts.model
talker = m.talker
calls = {"prefill": None, "trailing": None, "logits": []}


def pre_hook(mod, args, kwargs):
    e = kwargs.get("inputs_embeds")
    if e is not None and e.shape[1] > 1 and calls["prefill"] is None:
        calls["prefill"] = e[0].detach().clone()
        calls["trailing"] = kwargs["trailing_text_hidden"][0].detach().clone()


def hook(mod, args, kwargs, r):
    calls["logits"].append(r.logits[0, -1].detach().clone())


talker.register_forward_pre_hook(pre_hook, with_kwargs=True)
talker.register_forward_hook(hook, with_kwargs=True)
cp_calls = []
talker.code_predictor.register_forward_hook(lambda mod, args, r: cp_calls.append(r.logits[0, -1].detach().clone()))
generated_codes = []
original_generate = m.generate


def capture_generate(*args, **kwargs):
    result = original_generate(*args, **kwargs)
    generated_codes.append(result[0][0])
    return result


m.generate = capture_generate
# Greedy in-context cloning often does not stop (the reference recording's voice runs on): the frames are capped to keep the comparison short.
greedy = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=MAX_FRAMES, repetition_penalty=1.05)
info = {"text": text, "language": language, "mode": mode, "input_ids": tts._tokenize_texts([tts._build_assistant_text(text)])[0][0].tolist()}
with torch.no_grad():
    if mode == "design":
        info["instruction"] = extra
        info["instruct_ids"] = tts._tokenize_texts([tts._build_instruct_text(extra)])[0][0].tolist()
        wavs, sr = tts.generate_voice_design(text=text, instruct=extra, language=language, **greedy)
    else:
        audio, sr = librosa.load(extra, sr=None, mono=True)
        assert sr == 24000, "the reference recording must be 24 kHz (no resampling in the comparison)"
        save("reference", audio, "<f4")
        mel = mel_spectrogram(torch.from_numpy(audio).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0,
                              fmax=12000).transpose(1, 2)[0]
        save("mel", mel.numpy(), "<f4")
        items = tts.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=ref_text, x_vector_only_mode=(mode == "xvector"))
        save("speaker", items[0].ref_spk_embedding.numpy(), "<f4")
        codes = tts.model.speech_tokenizer.encode(audio, sr=sr).audio_codes[0]
        save("ref_codes", codes.numpy(), "<i4")
        info["ref_frames"] = int(codes.shape[0])
        if mode == "clone":
            info["reference_text"] = ref_text
            info["ref_ids"] = tts._tokenize_texts([tts._build_ref_text(ref_text)])[0][0].tolist()
        wavs, sr = tts.generate_voice_clone(text=text, language=language, voice_clone_prompt=items, **greedy)

generated = generated_codes[0]
frames = int(generated.shape[0])
L = torch.stack(calls["logits"][:frames + 1])
C = torch.stack(cp_calls[:15 * frames])
save("codes", generated.numpy(), "<i4")
save("prompt", calls["prefill"].numpy(), "<f4")
save("trailing", calls["trailing"].numpy(), "<f4")
top = L.topk(2, dim=-1)
save("top2_values", top.values.numpy(), "<f4")
save("top2_ids", top.indices.numpy(), "<i4")
ct = C.topk(2, dim=-1)
save("cp_top2_values", ct.values.numpy(), "<f4")
save("cp_top2_ids", ct.indices.numpy(), "<i4")
save("audio", wavs[0], "<f4")
info.update(frames=frames, prompt_rows=int(calls["prefill"].shape[0]), trailing_rows=int(calls["trailing"].shape[0]), sample_rate=sr)
print(json.dumps({k: v for k, v in info.items() if not k.endswith("_ids")}))
json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)
