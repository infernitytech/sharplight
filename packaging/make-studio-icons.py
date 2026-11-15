#!/usr/bin/env python3
"""Renders Sharplight Studio's icon sizes from packaging/linux/sharplight-studio.svg.

Writes apps/Sharplight.Studio/Assets/Icons/sharplight-studio-<size>.png (window, taskbar and sidebar icon) and
sharplight-studio.ico (the Windows executable icon, PNG-encoded entries). Needs ImageMagick (`magick`).
"""
import pathlib
import struct
import subprocess

SIZES = [16, 24, 32, 48, 64, 128, 256]
ICO_SIZES = [16, 24, 32, 48, 64, 256]

repo = pathlib.Path(__file__).resolve().parent.parent
svg = repo / "packaging" / "linux" / "sharplight-studio.svg"
out = repo / "apps" / "Sharplight.Studio" / "Assets" / "Icons"
out.mkdir(parents=True, exist_ok=True)

for size in SIZES:
    subprocess.run(["magick", "-background", "none", "-density", "576", str(svg), "-resize", f"{size}x{size}", "-depth", "8",
                    "-strip", "-define", "png:exclude-chunks=date,time", f"PNG32:{out / f'sharplight-studio-{size}.png'}"], check=True)

images = [(size, (out / f"sharplight-studio-{size}.png").read_bytes()) for size in ICO_SIZES]
header = struct.pack("<HHH", 0, 1, len(images))
offset = len(header) + 16 * len(images)
entries = b""
for size, data in images:
    edge = 0 if size >= 256 else size
    entries += struct.pack("<BBBBHHII", edge, edge, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
(out / "sharplight-studio.ico").write_bytes(header + entries + b"".join(data for _, data in images))
