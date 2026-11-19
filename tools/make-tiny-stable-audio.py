#!/usr/bin/env python3
"""Creates a tiny random Stable Audio pipeline (diffusers StableAudioPipeline layout) and diffusers reference outputs for the audio-generation tests.

Writes OUT (default ~/models/audio-models/tiny-random-stable-audio):

  diffusers/    model_index.json, transformer/ (StableAudioDiTModel: 2 layers, 4 heads of 16, 2 key-value heads, 8 latent channels, 32 frames),
                projection_model/ (the timing conditioners), vae/ (AutoencoderOobleck: strides 2 and 4, 8 latent channels, stereo at 512 Hz),
                text_encoder/ (a 2-layer T5 encoder of width 32 with T5's real 32128-token vocabulary), tokenizer/, scheduler/
  reference/    raw little-endian f32 tensors of two generations with seed 42, 4 steps, guidance 3 and a duration of 0.4 s: <case>_noise (the
                standard normal initial draw, [frames, channels]), <case>_step_noise ([steps, frames, channels], the scheduler's Brownian-tree samples),
                <case>_latent ([frames, channels]) and <case>_wav ([samples, channels], cut to the duration); case "plain" without and "negative"
                with a negative prompt

The tokenizer comes from --tokenizer (the tokenizer/ folder of a Stable Audio or T5 checkpoint) or google-t5/t5-base on the hub.
Requires torch, diffusers, transformers, torchsde, sentencepiece and safetensors.
"""

import argparse
import os

import numpy as np
import torch

PROMPT = "a dog barking twice in a quiet room"
NEGATIVE = "music, noise"
SECONDS = 0.4
STEPS = 4
GUIDANCE = 3.0


def build(out, tokenizer_source):
    from diffusers import AutoencoderOobleck, CosineDPMSolverMultistepScheduler, StableAudioDiTModel, StableAudioPipeline, StableAudioProjectionModel
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    text = T5EncoderModel(T5Config(vocab_size=32128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, relative_attention_num_buckets=32,
                                   relative_attention_max_distance=128, feed_forward_proj="relu", is_encoder_decoder=False)).eval()
    projection = StableAudioProjectionModel(text_encoder_dim=32, conditioning_dim=32, min_value=0, max_value=512)
    transformer = StableAudioDiTModel(sample_size=32, in_channels=8, num_layers=2, attention_head_dim=16, num_attention_heads=4, num_key_value_attention_heads=2,
                                      out_channels=8, cross_attention_dim=32, time_proj_dim=32, global_states_input_dim=64, cross_attention_input_dim=32)
    vae = AutoencoderOobleck(encoder_hidden_size=8, downsampling_ratios=[2, 4], channel_multiples=[1, 2], decoder_channels=8, decoder_input_channels=8,
                             audio_channels=2, sampling_rate=512)
    # Random weights of a usable scale: the default initializations of the norms and the snake parameters are kept, everything else is drawn small.
    for module in (projection, transformer, vae):
        for name, parameter in module.named_parameters():
            if "norm" not in name and not name.endswith(("alpha", "beta")):
                with torch.no_grad():
                    parameter.normal_(0, 0.2)
    scheduler = CosineDPMSolverMultistepScheduler(sigma_min=0.3, sigma_max=500, sigma_data=1.0, sigma_schedule="exponential", solver_order=2,
                                                  prediction_type="v_prediction", rho=7.0, final_sigmas_type="zero", lower_order_final=True, euler_at_final=False)
    pipe = StableAudioPipeline(vae=vae, text_encoder=text, projection_model=projection, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler)
    pipe.save_pretrained(os.path.join(out, "diffusers"))
    return pipe


def generate(pipe, negative):
    import diffusers.schedulers.scheduling_cosine_dpmsolver_multistep as S

    step_noise = []
    original = S.BrownianTreeNoiseSampler

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __call__(self, sigma, sigma_next):
            noise = self.inner(sigma, sigma_next)
            step_noise.append(noise.detach().clone())
            return noise

    S.BrownianTreeNoiseSampler = Recorder
    initial = {}
    prepare = pipe.prepare_latents

    def recording_prepare(*args, **kwargs):
        latents = prepare(*args, **kwargs)
        initial["noise"] = latents[0].T.clone() / pipe.scheduler.init_noise_sigma
        return latents

    pipe.prepare_latents = recording_prepare
    try:
        with torch.no_grad():
            latents = pipe(PROMPT, negative_prompt=negative, audio_end_in_s=SECONDS, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                           generator=torch.Generator("cpu").manual_seed(42), output_type="latent").audios
            pipe.prepare_latents = recording_prepare
            step_noise.clear()
            audio = pipe(PROMPT, negative_prompt=negative, audio_end_in_s=SECONDS, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         generator=torch.Generator("cpu").manual_seed(42), output_type="pt").audios
    finally:
        S.BrownianTreeNoiseSampler = original
        pipe.prepare_latents = prepare
    return {"noise": initial["noise"], "step_noise": torch.stack([n[0].T for n in step_noise]), "latent": latents[0].T, "wav": audio[0].T}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/models/audio-models/tiny-random-stable-audio"))
    parser.add_argument("--tokenizer", default="google-t5/t5-base")
    args = parser.parse_args()
    pipe = build(args.out, args.tokenizer)
    reference = os.path.join(args.out, "reference")
    os.makedirs(reference, exist_ok=True)
    for case, negative in (("plain", None), ("negative", NEGATIVE)):
        for name, tensor in generate(pipe, negative).items():
            np.ascontiguousarray(tensor.float().numpy().astype("<f4")).tofile(os.path.join(reference, f"{case}_{name}.f32"))
            print(case, name, tuple(tensor.shape), float(tensor.abs().max()))


if __name__ == "__main__":
    main()
