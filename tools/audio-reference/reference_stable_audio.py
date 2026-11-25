"""Stable Audio Open 1.0 reference: diffusers' StableAudioPipeline run on CPU in fp32 from the diffusers folders of stabilityai/stable-audio-open-1.0
(~/models/audio-models/stable-audio-open-1.0). Generates a clip with a fixed seed and records what Sharplight needs to reproduce the run: the prompt's token ids, the
masked T5 rows, the two timing rows, the initial noise (the standard normal draw, before the scheduler scales it by init_noise_sigma), the noise of every SDE step (the scheduler's
Brownian-tree samples, recorded by wrapping the sampler), one transformer evaluation of the first step (conditional), the latents after every step and the decoded
audio (trimmed to the duration, transposed to frame-major). Writes ~/models/audio-models/refs/<tag>_* (raw little-endian f32 / i32).

Usage: reference_stable_audio.py [tag] [seconds] [steps] [cfg] [negative prompt]
Needs torch (CPU), diffusers, transformers, torchsde, sentencepiece."""
import os, sys, torch, numpy as np
import diffusers.schedulers.scheduling_cosine_dpmsolver_multistep as S
from diffusers import StableAudioPipeline

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
ROOT = os.path.expanduser("~/models/audio-models/stable-audio-open-1.0")
REFS = os.path.expanduser("~/models/audio-models/refs")
tag = sys.argv[1] if len(sys.argv) > 1 else "sao"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cfg = float(sys.argv[4]) if len(sys.argv) > 4 else 7.0
negative = sys.argv[5] if len(sys.argv) > 5 else None
prompt = "The sound of a hammer hitting a wooden surface, then a short drum loop at 120 BPM"
seed = 42

step_noise = []
Original = S.BrownianTreeNoiseSampler


class Recorder:
    def __init__(self, *args, **kwargs):
        self.inner = Original(*args, **kwargs)

    def __call__(self, sigma, sigma_next):
        noise = self.inner(sigma, sigma_next)
        step_noise.append(noise.detach().clone())
        return noise


S.BrownianTreeNoiseSampler = Recorder

pipe = StableAudioPipeline.from_pretrained(ROOT, torch_dtype=torch.float32)
pipe.set_progress_bar_config(disable=False)
latents_per_step = []
first_eval = {}
transformer_forward = pipe.transformer.forward


def recording_forward(hidden_states, timestep=None, encoder_hidden_states=None, global_hidden_states=None, rotary_embedding=None, **kwargs):
    out = transformer_forward(hidden_states, timestep, encoder_hidden_states=encoder_hidden_states, global_hidden_states=global_hidden_states,
                              rotary_embedding=rotary_embedding, **kwargs)
    if not first_eval:
        b = hidden_states.shape[0] - 1   # the conditional half comes last
        first_eval.update(input=hidden_states[b].T.clone(), timestep=timestep.clone(), condition=encoder_hidden_states[b].clone(),
                          uncondition=encoder_hidden_states[0].clone(), glob=global_hidden_states[b].clone(), output=out[0][b].T.clone())
    return out


pipe.transformer.forward = recording_forward
generator = torch.Generator("cpu").manual_seed(seed)
initial = {}
prepare = pipe.prepare_latents


def recording_prepare(*args, **kwargs):
    latents = prepare(*args, **kwargs)
    initial["noise"] = latents[0].T.clone() / pipe.scheduler.init_noise_sigma   # the standard normal draw
    return latents


pipe.prepare_latents = recording_prepare
ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    t5 = pipe.text_encoder(ids.input_ids, attention_mask=ids.attention_mask)[0][0] * ids.attention_mask[0, :, None]
    timing = pipe.projection_model(start_seconds=torch.tensor([0.0]), end_seconds=torch.tensor([seconds]))
    result = pipe(prompt, negative_prompt=negative, audio_end_in_s=seconds, num_inference_steps=steps, guidance_scale=cfg, generator=generator,
                  callback=lambda i, t, lat: latents_per_step.append(lat[0].T.clone()), callback_steps=1, output_type="pt")
audio = result.audios[0].T
print("audio", audio.shape, audio.abs().max().item(), "steps", len(latents_per_step), "noise", len(step_noise))
os.makedirs(REFS, exist_ok=True)
tensors = {
    "ids": ids.input_ids[0][ids.attention_mask[0].bool()].int(),
    "text": t5,
    "timing": torch.cat([timing.seconds_start_hidden_states[0], timing.seconds_end_hidden_states[0]]),
    "noise": initial["noise"],
    "step_noise": torch.stack([n[0].T for n in step_noise]),
    "dit_in": first_eval["input"],
    "dit_t": first_eval["timestep"].reshape(1),
    "dit_cond": first_eval["condition"],
    "dit_global": first_eval["glob"],
    "dit_out": first_eval["output"],
    "steps": torch.stack(latents_per_step),
    "latent": latents_per_step[-1],
    "wav": audio,
}
for name, t in tensors.items():
    integer = name == "ids"
    np.ascontiguousarray(t.detach().numpy().astype("<i4" if integer else "<f4")).tofile(f"{REFS}/{tag}_{name}.{'i32' if integer else 'f32'}")
print("done", {k: tuple(v.shape) for k, v in tensors.items()})
