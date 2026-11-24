#!/usr/bin/env python3
"""Reference runs of SeedVR2 (ByteDance-Seed/SeedVR) for Sharplight's parity tests and benchmarks.

Runs the official inference path (projects/inference_seedvr2_{3b,7b}.py: the same transforms, VAE, DiT, one-step Euler sampler and
wavelet colour fix) on one clip, instrumented: every intermediate tensor is saved to an .npz file and every stage is timed with CUDA
events. Needs a checkout of github.com/ByteDance-Seed/SeedVR (--repo) and color_fix.py from the official Hugging Face Space (placed in
the checkout as projects/video_diffusion_sr/color_fix.py, or given with --color-fix-file). apex and flash_attn may be replaced by the
small stand-ins in --shims (apex FusedRMSNorm semantics; flash_attn_varlen_func from vLLM's flash-attention build on CUDA).

Dumped arrays (float32 unless noted; latents in [C, T, h, w]):
  input          the transformed clip [3, F, H, W] in [-1, 1] (before the 4n+1 padding)
  padded         the clip after the 4n+1 padding
  latent_mean    the encoder's mean, scaled by the latent scaling factor
  latent_std     the encoder's standard deviation, scaled alike
  latent_eps     the unit Gaussian sample of the posterior (latent = mean + std * eps)
  noise          the initial noise of the one-step sampler
  prediction     the DiT's velocity prediction at t = 1000
  restored       the restored latent (noise - prediction)
  decoded        the decoded frames [F, 3, H, W] in [-1, 1] (not clamped, before the colour fix)
  output         the final frames uint8 [F, H, W, 3] (colour fix, trimmed to the input length)
The input frames themselves are written as input_frames.u8 [F, H, W, 3] so the test can feed the same pixels.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="SeedVR checkout")
    p.add_argument("--shims", help="directory with apex / flash_attn stand-ins, put first on the path")
    p.add_argument("--color-fix-file", help="color_fix.py of the official Space when the checkout lacks it")
    p.add_argument("--model", choices=["3b", "7b"], default="3b")
    p.add_argument("--dit", required=True, help="DiT checkpoint (.pth)")
    p.add_argument("--vae", required=True, help="ema_vae.pth")
    p.add_argument("--pos-emb", required=True, help="pos_emb.pt")
    p.add_argument("--neg-emb", required=True, help="neg_emb.pt")
    p.add_argument("--input", required=True, help="video file, image file or .npy uint8 frames [F, H, W, 3]")
    p.add_argument("--frames", type=int, default=0, help="use only the first N frames (0 = all)")
    p.add_argument("--res-h", type=int, default=720)
    p.add_argument("--res-w", type=int, default=1280)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dit-dtype", choices=["bf16", "fp32"], default="bf16", help="bf16: the official autocast")
    p.add_argument("--dit-weights", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="dtype the DiT's weights are held in (fp32: official; bf16 for the 7B on a 32 GB GPU); .safetensors DiTs are always cast")
    p.add_argument("--vae-dtype", choices=["bf16", "fp32"], default="bf16", help="bf16: the official VAE dtype")
    p.add_argument("--attention", choices=["flash", "sdpa"], default="flash")
    p.add_argument("--no-memory-limit", action="store_true", help="skip the official conv / norm memory limits (VAE)")
    p.add_argument("--no-offload", action="store_true", help="keep DiT and VAE on the device together (no offload between stages)")
    p.add_argument("--no-color-fix", action="store_true")
    p.add_argument("--out", help="write the dump here: a folder of raw little-endian arrays (<name>.f32, output.u8) with index.json")
    p.add_argument("--repeat", type=int, default=1, help="timed runs after the first (benchmarks)")
    p.add_argument("--timing-json", help="write per-run stage timings here")
    p.add_argument("--save-video", help="write the result as an mp4 (mediapy) here")
    return p.parse_args()


def main():
    args = parse()
    for name in ("shims", "color_fix_file", "dit", "vae", "pos_emb", "neg_emb", "input", "out", "timing_json", "save_video", "repo"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if args.shims:
        sys.path.insert(0, args.shims)
    sys.path.insert(0, args.repo)
    os.environ["SEEDVR2_REF_ATTENTION"] = args.attention
    os.chdir(args.repo)

    import torch
    from einops import rearrange
    from omegaconf import OmegaConf
    from torchvision.transforms import Compose, Lambda, Normalize

    from common.config import load_config
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    if args.color_fix_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location("color_fix", args.color_fix_file)
        color_fix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(color_fix)
        wavelet_reconstruction = color_fix.wavelet_reconstruction
    else:
        from projects.video_diffusion_sr.color_fix import wavelet_reconstruction

    patch_rope_tables()
    device = torch.device(args.device)
    if not torch.distributed.is_initialized():
        # The official decorators synchronize ranks; one process is its own group.
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        torch.distributed.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    torch.manual_seed(args.seed)
    config = load_config(os.path.join(f"./configs_{args.model}", "main.yaml"))
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    runner.config.vae.checkpoint = args.vae
    runner.config.vae.dtype = "bfloat16" if args.vae_dtype == "bf16" else "float32"
    if args.dit.endswith(".safetensors") or args.dit_weights != "fp32":
        runner.dit = load_dit(config, args.dit, getattr(torch, {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}[args.dit_weights]))
    else:
        runner.configure_dit_model(device="cpu", checkpoint=args.dit)
    runner.dit.to(device)
    import common.distributed.basic as basic
    basic.get_device = lambda: device
    import projects.video_diffusion_sr.infer as infer_module
    infer_module.get_device = lambda: device
    runner.configure_vae_model()
    runner.vae.to(device)
    if hasattr(runner.vae, "set_memory_limit") and not args.no_memory_limit:
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    pos = torch.load(args.pos_emb).float().to(device)
    neg = torch.load(args.neg_emb).float().to(device)
    frames = load_frames(args.input, args.frames)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # T C H W

    transform = Compose([
        NaResize(resolution=(args.res_h * args.res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def run_once(dump):
        stages = {}
        start = time.perf_counter()
        # End to end: the input file is decoded on every run (the first decode above only sized the clip).
        clip = torch.from_numpy(load_frames(args.input, args.frames)).permute(0, 3, 1, 2).float() / 255.0
        stages["read"] = time.perf_counter() - start
        t0 = time.perf_counter()
        x = transform(clip.to(device))
        padded = cut_videos(x)
        sync()
        stages["preprocess"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        if not args.no_offload:
            runner.dit.to("cpu")
        runner.vae.to(device)
        vae_dtype = getattr(torch, runner.config.vae.dtype)
        with torch.no_grad():
            sample = padded.unsqueeze(0).to(device, vae_dtype)
            posterior = runner.vae.encode(sample).posterior
            mean, std = posterior.mean, posterior.std
            eps = torch.randn(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
            latent = mean + std * eps
            scale = runner.config.vae.scaling_factor
            cond = rearrange(latent.squeeze(0), "c ... -> ... c") * scale
        if not args.no_offload:
            runner.vae.to("cpu")
            runner.dit.to(device)
        sync()
        stages["vae_encode"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        noise = torch.randn_like(cond)
        condition = runner.get_condition(noise, task="sr", latent_blur=cond)
        captured = {}
        dit_forward = runner.dit.forward

        def capture(*a, **k):
            out = dit_forward(*a, **k)
            captured["prediction"] = out.vid_sample.detach()
            return out

        runner.dit.forward = capture
        autocast = torch.autocast(device.type, torch.bfloat16, enabled=args.dit_dtype == "bf16")
        with torch.no_grad(), autocast:
            latents, shapes = flatten([noise])
            conds, _ = flatten([condition])
            text, text_shape = flatten([pos])
            restored = runner.sampler.sample(
                x=latents,
                f=lambda a: runner.dit(vid=torch.cat([a.x_t, conds], dim=-1), txt=text, vid_shape=shapes, txt_shape=text_shape,
                                       timestep=a.t.repeat(1)).vid_sample)
        runner.dit.forward = dit_forward
        restored = restored.reshape(noise.shape)
        sync()
        stages["dit"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        if not args.no_offload:
            runner.dit.to("cpu")
        runner.vae.to(device)
        with torch.no_grad():
            decoded = runner.vae_decode([restored])[0]  # c t h w
        if not args.no_offload:
            runner.dit.to(device)
        sync()
        stages["vae_decode"] = time.perf_counter() - t3

        t4 = time.perf_counter()
        sample_t = rearrange(decoded, "c t h w -> t c h w")
        length = x.shape[1]
        sample_t = sample_t[:length]
        reference = rearrange(x, "c t h w -> t c h w")
        if args.no_color_fix:
            fixed = sample_t.float()
        else:
            fixed = wavelet_reconstruction(sample_t.float().cpu(), reference[: sample_t.size(0)].float().cpu())
        out = rearrange(fixed, "t c h w -> t h w c").clip(-1, 1).mul(0.5).add(0.5).mul(255).round().to(torch.uint8).cpu().numpy()
        stages["color_fix"] = time.perf_counter() - t4
        stages["enhance"] = time.perf_counter() - t0
        t5 = time.perf_counter()
        if args.save_video:
            save_video(args.save_video, out, 24)
        stages["write"] = time.perf_counter() - t5
        stages["total"] = time.perf_counter() - start
        if dump:
            pred = captured["prediction"].reshape(noise.shape)
            return stages, {
                "input": x.float().cpu().numpy(),
                "padded": padded.float().cpu().numpy(),
                "latent_mean": (rearrange(mean.squeeze(0), "c ... -> c ...") * scale).float().cpu().numpy(),
                "latent_std": (std.squeeze(0) * scale).float().cpu().numpy(),
                "latent_eps": eps.squeeze(0).float().cpu().numpy(),
                "noise": rearrange(noise, "t h w c -> c t h w").float().cpu().numpy(),
                "prediction": rearrange(pred, "t h w c -> c t h w").float().cpu().numpy(),
                "restored": rearrange(restored, "t h w c -> c t h w").float().cpu().numpy(),
                "decoded": rearrange(decoded, "c t h w -> t c h w").float().cpu().numpy(),
                "output": out,
                "input_frames": frames,
            }
        return stages, out

    def flatten(items):
        shape = torch.stack([torch.tensor(t.shape[:-1], device=device) for t in items])
        return torch.cat([t.flatten(0, -2) for t in items]), shape

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stages, dump = run_once(dump=True)
    runs = [stages]
    print(json.dumps(stages), flush=True)
    for _ in range(args.repeat):
        stages, _ = run_once(dump=False)
        runs.append(stages)
        print(json.dumps(stages), flush=True)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        shapes = {}
        for name, value in dump.items():
            suffix = ".u8" if value.dtype == np.uint8 else ".f32"
            value.astype(np.uint8 if suffix == ".u8" else np.float32).tofile(os.path.join(args.out, name + suffix))
            shapes[name] = list(value.shape)
        with open(os.path.join(args.out, "index.json"), "w") as f:
            json.dump({"tensors": shapes, "model": args.model, "res_h": args.res_h, "res_w": args.res_w, "seed": args.seed,
                       "frames": int(frames.shape[0]), "input": os.path.basename(args.input), "dit_dtype": args.dit_dtype,
                       "vae_dtype": args.vae_dtype, "attention": args.attention, "timestep": 1000.0}, f, indent=1)
    if args.timing_json:
        with open(args.timing_json, "w") as f:
            json.dump({"runs": runs, "peak_bytes": peak, "frames": int(frames.shape[0]),
                       "input": list(frames.shape), "output": list(dump["output"].shape), "args": vars(args)}, f, indent=1)
    print(f"peak {peak / 2**30:.2f} GiB, output {dump['output'].shape}")


def patch_rope_tables():
    """
    The 3B's RoPE builds its frequency table for a 1024 × 128 × 128 grid (8.4 GB in f32) and slices the windows' corners out of it, which
    does not fit next to the f32 weights on a 32 GB GPU. The table of positions [0, n) is the prefix of the large one (integer positions,
    the same frequencies), so this builds only the corner each window needs: the same values, without the memory.
    """
    import torch
    from models.dit_v2.rope import NaMMRotaryEmbedding3d

    def get_freqs(self, vid_shape, txt_shape):
        vid_list, txt_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid = self.get_axial_freqs(l + f, h, w)[l:l + f]
            vid_list.append(vid.reshape(-1, vid.size(-1)))
            txt_list.append(self.get_axial_freqs(l)[:l].repeat(1, 3).reshape(-1, vid.size(-1)))
        return torch.cat(vid_list, dim=0), torch.cat(txt_list, dim=0)

    NaMMRotaryEmbedding3d.get_freqs = get_freqs


def load_dit(config, path, dtype):
    """The DiT built on the meta device and assigned the checkpoint's tensors in dtype (the 7B does not fit a 32 GB GPU in f32)."""
    import torch
    from common.config import create_object
    from common.distributed.meta_init_utils import meta_non_persistent_buffer_init_fn
    with torch.device("meta"):
        dit = create_object(config.dit.model)
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            state = {k: f.get_tensor(k).to(dtype) for k in f.keys()}
    else:
        state = {k: v.to(dtype) for k, v in torch.load(path, map_location="cpu", mmap=True).items()}
    print(dit.load_state_dict(state, strict=True, assign=True))
    return meta_non_persistent_buffer_init_fn(dit).eval()


def cut_videos(videos):
    """Pads [C, T, H, W] with copies of the last frame to 4n + 1 frames (the official cut_videos with sp_size 1)."""
    import torch
    t = videos.size(1)
    if t == 1 or (t - 1) % 4 == 0:
        return videos
    if t <= 4:
        missing = 4 - t + 1
    else:
        missing = 4 - ((t - 1) % 4)
    return torch.cat([videos, videos[:, -1:].repeat(1, missing, 1, 1)], dim=1)


def save_video(path, frames, fps):
    """H.264 in MP4 through PyAV (the official script writes with mediapy, which drives the same libx264 through the ffmpeg binary)."""
    import av
    with av.open(path, "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def load_frames(path, limit):
    if path.endswith(".npy"):
        frames = np.load(path)
    elif os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        from PIL import Image
        frames = np.asarray(Image.open(path).convert("RGB"))[None]
    else:
        # PyAV, which the official script's torchvision.io.read_video wraps (newer torchvision releases dropped it).
        import av
        with av.open(path) as container:
            frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    frames = np.ascontiguousarray(frames[:limit] if limit > 0 else frames)
    return frames


if __name__ == "__main__":
    main()
