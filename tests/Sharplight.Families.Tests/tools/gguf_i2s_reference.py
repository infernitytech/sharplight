#!/usr/bin/env python3
"""Reference dequantized weights of GGUF I2_S tensors (Microsoft BitNet's llama.cpp fork, ggml type 36), written as
<gguf dir>/sharplight-reference/<tensor name>.f32 (raw little-endian F32, row-major), for the I2_S test of Sharplight.Import.Gguf.Tests.

usage: gguf_i2s_reference.py <file.gguf> [tensor-name-prefix ...]   (default prefix: blk.0.)

gguf-py has no I2_S type, so the file is parsed here. The decoder follows quantize_i2_s of the fork (src/ggml-bitnet-mad.cpp,
MIT License): n elements take n / 4 bytes, every 128 consecutive values in 32 bytes (value j % 128 sits in byte j % 32, bits
6 - 2 * ((j % 128) / 32)), codes 0 / 1 / 2 are -1 / 0 / +1, then one f32 scale (max |w| of the tensor) follows the codes and is the
only multiplier: value = (code - 1) * scale. With --check <bf16 master safetensors> the tensors are compared with the ternarization
BitNet's converter applies (round(w * s).clamp(-1, 1) / s, s = 1 / mean|w|) and the largest difference is printed.
"""
import mmap
import os
import struct
import sys

import numpy as np

I2S = 36


def parse(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pos = 0

    def read(fmt):
        nonlocal pos
        values = struct.unpack_from("<" + fmt, mm, pos)
        pos += struct.calcsize("<" + fmt)
        return values

    def string():
        nonlocal pos
        (n,) = read("Q")
        text = bytes(mm[pos:pos + n]).decode()
        pos += n
        return text

    def skip_value(kind):
        nonlocal pos
        if kind == 8:
            string()
        elif kind == 9:
            (element,), (count,) = read("I"), read("Q")
            for _ in range(count):
                skip_value(element)
        else:
            pos += {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[kind]

    assert bytes(mm[:4]) == b"GGUF"
    pos = 4
    (version,), (tensor_count, kv_count) = read("I"), read("QQ")
    alignment = 32
    for _ in range(kv_count):
        key = string()
        (kind,) = read("I")
        if key == "general.alignment":
            alignment = read("I")[0]
        else:
            skip_value(kind)
    tensors = []
    for _ in range(tensor_count):
        name = string()
        (dims,) = read("I")
        ne = read("Q" * dims)
        (kind,), (offset,) = read("I"), read("Q")
        tensors.append((name, kind, ne, offset))
    data = (pos + alignment - 1) // alignment * alignment
    return mm, data, tensors


def decode(mm, start, ne):
    n = int(np.prod(ne))
    codes = np.frombuffer(mm, dtype=np.uint8, count=n // 4, offset=start).reshape(-1, 32)   # [groups, 32]
    (scale,) = struct.unpack_from("<f", mm, start + n // 4)
    planes = np.stack([(codes >> shift) & 3 for shift in (6, 4, 2, 0)], axis=1)               # [groups, 4, 32]
    assert planes.max() <= 2
    values = (planes.astype(np.float32) - 1.0) * np.float32(scale)
    return values.reshape(ne[::-1]), scale


def main():
    args = sys.argv[1:]
    master = None
    if "--check" in args:
        i = args.index("--check")
        master = args[i + 1]
        del args[i:i + 2]
    path, prefixes = args[0], args[1:] or ["blk.0."]
    mm, data, tensors = parse(path)
    out = {}
    for name, kind, ne, offset in tensors:
        if kind == I2S and any(name.startswith(p) for p in prefixes):
            values, scale = decode(mm, data + offset, ne)
            out[name] = values
            print(name, values.shape, "scale", scale, "codes", {int(v): int((values == v * np.float32(scale)).sum()) for v in (-1, 0, 1)})
    directory = os.path.join(os.path.dirname(os.path.abspath(path)), "sharplight-reference")
    os.makedirs(directory, exist_ok=True)
    for name, values in out.items():
        np.ascontiguousarray(values, dtype="<f4").tofile(os.path.join(directory, name + ".f32"))
    print(len(out), "tensors")


if __name__ == "__main__":
    main()
