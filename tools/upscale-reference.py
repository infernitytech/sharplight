#!/usr/bin/env python3
"""Reference output of Real-ESRGAN (RRDBNet, the reference architecture, CPU, whole image) for the Sharplight upscaler tests.

Takes IMAGE, shrinks it to WIDTHxHEIGHT (default 160x120), writes `sharplight-sample.png` (the input) and `sharplight-reference.png` (the 8-bit output the
reference code produces: clamp to 0..1, round) into MODEL_DIR, next to the checkpoint.

Usage: upscale-reference.py MODEL_DIR CHECKPOINT IMAGE [WIDTH HEIGHT]
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = DenseBlock(nf, gc), DenseBlock(nf, gc), DenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, blocks=23, nf=64, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def main() -> None:
    model_dir, checkpoint, image = sys.argv[1:4]
    width, height = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (160, 120)
    torch.set_num_threads(2)
    source = Image.open(image).convert("RGB").resize((width, height), Image.BICUBIC)
    source.save(os.path.join(model_dir, "sharplight-sample.png"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = RRDBNet()
    net.load_state_dict(state["params_ema"] if "params_ema" in state else state["params"], strict=True)
    net.eval()
    x = torch.from_numpy(np.asarray(source).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = net(x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((y * 255.0).round().astype(np.uint8)).save(os.path.join(model_dir, "sharplight-reference.png"))
    print("done", y.shape)


if __name__ == "__main__":
    main()
