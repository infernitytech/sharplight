"""Writes the reference tensors of the Chatterbox tests (resemble-ai's chatterbox-tts package on torch, float32, CPU).

Usage (in a venv with torch, chatterbox-tts and its dependencies; torchaudio may be a stub, nothing here resamples):
    python make_chatterbox_reference.py <checkpoint dir> <output dir> "<text>"

The built-in voice (conds.pt) speaks the text. Written as raw little-endian files next to chatterbox.json:
- text_tokens.bin (i32): the tokenizer's ids with the start and stop tokens;
- cond.bin (f32 [rows, 1024]): T3's conditioning rows (speaker, perceiver over the prompt speech tokens, emotion);
- speech_tokens.bin (i32): T3's greedy chain with classifier-free guidance (weight 0.5) and repetition penalty 1.2, which leaves the argmax of
  the sampler unchanged by temperature, min-p and top-p; margins.bin (f32): the gap between the best and the second-best guided logit per step;
- mel.bin (f32 [80, frames]): S3Gen's flow output for those tokens with the noise in flow_noise.bin (f32 [80, prompt + frames]);
- f0.bin (f32 [frames]), source_noise.bin (f32 [9, samples]) and source_phase.bin (f32 [9]): the HiFT source with its random draws;
- audio.bin (f32 [samples]): the waveform before the watermark.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs"))

import numpy as np
import torch
import torch.nn.functional as F

import perth

perth.PerthImplicitWatermarker = lambda: None  # the dump stops before the watermark; the watermarker's own dependencies are not needed
from chatterbox.tts import ChatterboxTTS, punc_norm  # noqa: E402


def save(out: str, name: str, tensor, dtype) -> None:
    np.ascontiguousarray(np.asarray(tensor, dtype=dtype)).tofile(os.path.join(out, name + ".bin"))


def main(model_dir: str, out: str, text: str) -> None:
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tts = ChatterboxTTS.from_local(model_dir, "cpu")
    t3 = tts.t3
    hp = t3.hp
    text_tokens = tts.tokenizer.text_to_tokens(punc_norm(text))
    text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token).long()
    save(out, "text_tokens", text_tokens[0].numpy(), np.int32)
    cond = tts.conds.t3
    with torch.inference_mode():
        cond_rows = t3.prepare_conditioning(cond)[0]
        save(out, "cond", cond_rows.numpy(), np.float32)
        tokens2 = torch.cat([text_tokens, text_tokens])
        start = hp.start_speech_token * torch.ones_like(tokens2[:, :1])
        embeds, _ = t3.prepare_input_embeds(t3_cond=cond, text_tokens=tokens2, speech_tokens=start, cfg_weight=0.5)
        bos = t3.speech_emb(torch.tensor([[hp.start_speech_token]])) + t3.speech_pos_emb.get_fixed_embedding(0)
        inputs = torch.cat([embeds, torch.cat([bos, bos])], dim=1)
        output = t3.tfmr(inputs_embeds=inputs, use_cache=True)
        past = output.past_key_values
        hidden = output.last_hidden_state
        generated = [hp.start_speech_token]
        margins = []
        for i in range(1000):
            logits = t3.speech_head(hidden[:, -1, :])
            guided = logits[0:1] + 0.5 * (logits[0:1] - logits[1:2])
            ids = torch.tensor([generated])
            score = torch.gather(guided, 1, ids)
            score = torch.where(score < 0, score * 1.2, score / 1.2)
            guided = guided.scatter(1, ids, score)
            top = torch.topk(guided[0], 2)
            margins.append(float(top.values[0] - top.values[1]))
            token = int(top.indices[0])
            generated.append(token)
            if token == hp.stop_speech_token:
                break
            embed = t3.speech_emb(torch.tensor([[token]])) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            output = t3.tfmr(inputs_embeds=torch.cat([embed, embed]), past_key_values=past, use_cache=True)
            past = output.past_key_values
            hidden = output.last_hidden_state
        speech = torch.tensor(generated[1:])
        save(out, "speech_tokens", speech.numpy(), np.int32)
        save(out, "margins", np.array(margins), np.float32)
        speech = speech[speech < 6561]

        draws = {}
        original_randn_like = torch.randn_like

        def recording_randn_like(t, *args, **kwargs):
            value = original_randn_like(t, *args, **kwargs)
            draws.setdefault("flow_noise" if t.shape[1] == 80 else "source_noise", value.clone())
            return value

        torch.randn_like = recording_randn_like
        from torch.distributions.uniform import Uniform
        original_sample = Uniform.sample

        def recording_sample(self, *args, **kwargs):
            value = original_sample(self, *args, **kwargs)
            draws["source_phase"] = value.clone()
            return value

        Uniform.sample = recording_sample
        s3 = tts.s3gen
        mel = super(type(s3), s3).forward(speech, ref_wav=None, ref_sr=None, ref_dict=tts.conds.gen, finalize=True, n_cfm_timesteps=10)
        f0 = s3.mel2wav.f0_predictor(mel)
        wav, _ = s3.mel2wav.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
        wav[:, :len(s3.trim_fade)] *= s3.trim_fade
        torch.randn_like = original_randn_like
        Uniform.sample = original_sample
    save(out, "mel", mel[0].numpy(), np.float32)
    save(out, "flow_noise", draws["flow_noise"][0].numpy(), np.float32)
    save(out, "f0", f0[0].numpy(), np.float32)
    save(out, "source_noise", draws["source_noise"][0].numpy(), np.float32)
    save(out, "source_phase", draws["source_phase"].reshape(-1).numpy(), np.float32)
    save(out, "audio", wav[0].numpy(), np.float32)
    with open(os.path.join(out, "chatterbox.json"), "w") as f:
        json.dump({"text": text, "normalized": punc_norm(text), "speech_tokens": len(generated) - 1, "mel_frames": mel.shape[-1],
                   "samples": wav.shape[-1], "sample_rate": 24000}, f)
    print("tokens", len(generated) - 1, "mel", tuple(mel.shape), "samples", wav.shape[-1], "min margin", min(margins))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
