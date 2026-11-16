#!/usr/bin/env python3
"""Reference output of SwinIR (classical / real-world x4, 1conv residual connection; a compact torch re-implementation of the authors' network, CPU,
whole image) for the Sharplight upscaler tests.

Takes IMAGE, shrinks it to WIDTHxHEIGHT (default 160x120, multiples of the window size), writes `sharplight-sample.png` (the input) and
`sharplight-reference.png` (the 8-bit output: clamp to 0..1, round) into MODEL_DIR, next to the checkpoint. The checkpoint is read with
`weights_only=True` (nothing is unpickled) and loaded strictly, so the re-implementation has exactly the checkpoint's parameters.

Usage: swinir-reference.py MODEL_DIR CHECKPOINT IMAGE [WIDTH HEIGHT] [--real]
  --real  the real-world x4 variant (nearest+conv upsampler)
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


def partition(x, w):
    b, h, wd, c = x.shape
    return x.view(b, h // w, w, wd // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)


def reverse(win, w, h, wd):
    return win.view(-1, h // w, wd // w, w, w, win.shape[-1]).permute(0, 1, 3, 2, 4, 5).reshape(1, h, wd, -1)


class Attention(nn.Module):
    def __init__(self, dim, window, heads):
        super().__init__()
        self.window, self.heads, self.scale = window, heads, (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        n_win, n, c = x.shape
        qkv = self.qkv(x).reshape(n_win, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = (attn.view(n_win // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.heads, n, n)
        attn = attn.softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(n_win, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, window, shift, ratio):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, window, heads)
        self.mlp = Mlp(dim, int(dim * ratio))

    def mask(self, h, w):
        img = torch.zeros(1, h, w, 1)
        s = self.window
        slices = (slice(0, -s), slice(-s, -self.shift), slice(-self.shift, None))
        cnt = 0
        for hs in slices:
            for ws in slices:
                img[:, hs, ws, :] = cnt
                cnt += 1
        win = partition(img, s).squeeze(-1)
        m = win.unsqueeze(1) - win.unsqueeze(2)
        return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)

    def forward(self, x, h, w):
        shortcut = x
        x = self.norm1(x).view(1, h, w, -1)
        mask = None
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            mask = self.mask(h, w)
        x = reverse(self.attn(partition(x, self.window), mask), self.window, h, w)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.view(1, h * w, -1)
        return x + self.mlp(self.norm2(x))


class Group(nn.Module):
    def __init__(self, dim, depth, heads, window, ratio):
        super().__init__()
        self.residual_group = nn.Module()
        self.residual_group.blocks = nn.ModuleList([Block(dim, heads, window, 0 if i % 2 == 0 else window // 2, ratio) for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, h, w):
        y = x
        for blk in self.residual_group.blocks:
            y = blk(y, h, w)
        y = self.conv(y.transpose(1, 2).reshape(1, -1, h, w)).flatten(2).transpose(1, 2)
        return y + x


class SwinIR(nn.Module):
    def __init__(self, scale=4, dim=180, depths=(6,) * 6, heads=(6,) * 6, window=8, ratio=2, real=False, feat=64):
        super().__init__()
        self.window, self.real, self.scale = window, real, scale
        self.register_buffer("mean", torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))
        self.conv_first = nn.Conv2d(3, dim, 3, 1, 1)
        self.patch_embed = nn.Module()
        self.patch_embed.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([Group(dim, d, hd, window, ratio) for d, hd in zip(depths, heads)])
        self.norm = nn.LayerNorm(dim)
        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(dim, feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        if real:
            self.conv_up1, self.conv_up2 = nn.Conv2d(feat, feat, 3, 1, 1), nn.Conv2d(feat, feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(feat, feat, 3, 1, 1)
        else:
            mods = []
            for _ in range({2: 1, 4: 2, 8: 3}[scale]):
                mods += [nn.Conv2d(feat, 4 * feat, 3, 1, 1), nn.PixelShuffle(2)]
            self.upsample = nn.Sequential(*mods)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1)

    def forward(self, x):
        x = x - self.mean
        h, w = x.shape[2:]
        first = self.conv_first(x)
        t = self.patch_embed.norm(first.flatten(2).transpose(1, 2))
        for layer in self.layers:
            t = layer(t, h, w)
        t = self.norm(t).transpose(1, 2).reshape(1, -1, h, w)
        feat = self.conv_after_body(t) + first
        feat = self.conv_before_upsample(feat)
        if self.real:
            feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
            feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")), 0.2)
            out = self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2))
        else:
            out = self.conv_last(self.upsample(feat))
        return out + self.mean


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    real = "--real" in sys.argv
    model_dir, checkpoint, image_path = args[:3]
    width, height = (int(args[3]), int(args[4])) if len(args) >= 5 else (160, 120)
    net = SwinIR(real=real)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("params_ema", state.get("params", state))
    state = {k: v for k, v in state.items() if not k.endswith("attn_mask")}
    missing = net.load_state_dict(state, strict=False)
    assert not missing.unexpected_keys and all(k == "mean" or k.endswith("relative_position_index") for k in missing.missing_keys), missing
    net.eval()
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.BICUBIC)
    image.save(os.path.join(model_dir, "sharplight-sample.png"))
    x = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = net(x)
    out = (y[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(out).save(os.path.join(model_dir, "sharplight-reference.png"))
    print("ok", out.shape)


if __name__ == "__main__":
    main()
