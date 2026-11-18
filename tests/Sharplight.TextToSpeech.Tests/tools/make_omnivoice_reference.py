"""Reference tensors of OmniVoice (k2-fsa) from the official omnivoice package (CPU, float32; deterministic: position and class temperature 0).

Cases: "clone" (voice cloning from jfk.wav with its transcript) and "design" (an instruct, no reference). Per case it writes, under <out>/<case>_*:
  info.json            text, ref_text, language, instruct, steps, target_len, ref_len, c_len, ref_rms, schedule
  ids.i32              the conditional sequence's token rows [8, c_len] (row 0 holds the text ids; audio rows hold the reference codes and the mask id)
  audio_mask.u8        [c_len]
  ref_tokens.i32       [8, ref_len] (clone only: the tokenizer's codes of the reference)
  states.i32           [steps, 8, target_len]: the target tokens before each step
  logits_first.f32 / logits_last.f32   [2, 8, target_len, 1025]: conditional and unconditional logits of the first and the last step
  tokens.i32           [8, target_len]: the final codes
  audio.f32            the decoded waveform (no silence removal, no fades) scaled as the package does
Usage: python make_omnivoice_reference.py <omnivoice package dir> <model dir> <jfk.wav> <out dir>
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import math
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig, _get_time_steps  # noqa: E402
from omnivoice.utils.audio import load_audio  # noqa: E402

torch.set_num_threads(4)
model_dir, jfk, out = sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
model = OmniVoice.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu").eval()
TEXT = "Hello, this is a test of the OmniVoice speech model."
REF_TEXT = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."


def run(case, **kwargs):
    calls = []
    original = model.forward

    def spy(*args, **kw):
        result = original(*args, **kw)
        calls.append((kw["input_ids"].clone(), kw["audio_mask"].clone(), result.logits.clone()))
        return result

    model.forward = spy
    cfg = OmniVoiceGenerationConfig(position_temperature=0.0, class_temperature=0.0, preprocess_prompt=False, postprocess_output=False, pad_duration=0.0, fade_duration=0.0)
    ref_tokens = None
    ref_rms = None
    if "ref_audio" in kwargs:
        path = kwargs.pop("ref_audio")
        wav = load_audio(path, model.sampling_rate)
        wav_rms = float(np.sqrt(np.mean(wav**2)))
        if 0 < wav_rms < 0.1:
            wav = wav * 0.1 / wav_rms
        clip = int(wav.shape[-1] % model.audio_tokenizer.config.hop_length)
        (wav[:, :-clip] if clip > 0 else wav).astype("<f4").tofile(f"{out}/{case}_ref_audio.f32")
        prompt = model.create_voice_clone_prompt(path, ref_text=REF_TEXT, preprocess_prompt=False)
        kwargs["voice_clone_prompt"] = prompt
        ref_tokens = prompt.ref_audio_tokens
        ref_rms = prompt.ref_rms
    task = model._preprocess_all(text=TEXT, language="en", preprocess_prompt=False, **kwargs)
    target_len = task.target_lens[0]
    audio = model.generate(text=TEXT, language="en", generation_config=cfg, **kwargs)[0]
    model.forward = original
    steps = len(calls)
    ids0, mask0, _ = calls[0]
    c_len = int(ids0.shape[-1])
    # Row 0 is the conditional sequence; its target region is the last target_len rows.
    states = np.stack([c[0][0, :, c_len - target_len : c_len].numpy() for c in calls]).astype("<i4")
    tokens = model._generate_iterative(task, cfg)[0].numpy().astype("<i4")
    timesteps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=cfg.num_step, t_shift=cfg.t_shift).tolist()
    total, rem, schedule = target_len * 8, target_len * 8, []
    for step in range(cfg.num_step):
        num = rem if step == cfg.num_step - 1 else min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), rem)
        schedule.append(int(num))
        rem -= int(num)
    json.dump({
        "text": TEXT, "ref_text": REF_TEXT if ref_tokens is not None else None, "language": "en", "instruct": kwargs.get("instruct"), "steps": steps,
        "target_len": target_len, "ref_len": int(ref_tokens.shape[-1]) if ref_tokens is not None else 0, "c_len": c_len, "ref_rms": ref_rms,
        "audio_len": int(audio.shape[-1]), "schedule": schedule,
    }, open(f"{out}/{case}_info.json", "w"), indent=1)
    ids0[0].numpy().astype("<i4").tofile(f"{out}/{case}_ids.i32")
    mask0[0].numpy().astype("u1").tofile(f"{out}/{case}_audio_mask.u8")
    if ref_tokens is not None:
        ref_tokens.numpy().astype("<i4").tofile(f"{out}/{case}_ref_tokens.i32")
    states.tofile(f"{out}/{case}_states.i32")
    tokens.tofile(f"{out}/{case}_tokens.i32")
    for name, call in (("first", calls[0]), ("last", calls[-1])):
        logits = call[2]
        c = logits[0, :, c_len - target_len : c_len, :]
        u = logits[1, :, :target_len, :]
        torch.stack([c, u]).numpy().astype("<f4").tofile(f"{out}/{case}_logits_{name}.f32")
    np.asarray(audio, dtype="<f4").tofile(f"{out}/{case}_audio.f32")
    print(case, "steps", steps, "target", target_len, "c_len", c_len, "audio", audio.shape)


with torch.inference_mode():
    run("clone", ref_audio=jfk)
    run("design", instruct="female, british accent")
