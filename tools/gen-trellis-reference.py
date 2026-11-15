#!/usr/bin/env python3
"""Generates tests/Sharplight.Cpu.Tests/Fixtures/trellis_ref.bin: random blocks of the trellis-coded formats of agentionai's
llama.cpp fork (ggml ids 144-149: TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7; the Gyro TQ*_0 releases) with the values the fork's
decoders give them.

The decoders are a line-by-line transcription of the fork's dequantize_row_tq2_t / dequantize_row_tqk_impl /
dequantize_row_tqp_impl, tqk_state, tqp_state and tq2t_hyb_index (ggml/src/ggml-quants.c, ggml/src/ggml-common.h), with the
f32 products d * codebook computed in float32. The codebook is read from the fork's ggml-common.h.

usage: gen-trellis-reference.py <the fork's ggml-common.h> <output.bin>

Record layout (little endian), as tools/gen-quant-reference.py: i32 type id, i32 rows, i32 cols, byte q[rows * rowBytes],
f32 dequantized[rows * cols]. The file starts with the i32 record count.
"""
import random
import re
import struct
import sys

import numpy as np

# ggml id -> (values per step, new bits per step)
TYPES = {144: (4, 8), 145: (4, 6), 146: (4, 7), 147: (2, 5), 148: (2, 6), 149: (2, 7)}
QK = 128
ROWS, COLS = 3, 384
TQ_STATE_MASK = (1 << 15) - 1


def hyb_index(e):
    x = (((e >> 1) & TQ_STATE_MASK) * 0x9e3779b1) & 0xffffffff
    return ((x >> 10) & 2047) if (e & 1) else (x >> 21)


def stream_state(qs, k, nsteps, t):
    nbytes = nsteps * k // 8
    off = (nsteps - 1 - t) * k
    i0 = off >> 3
    i1 = (i0 + 1) % nbytes
    i2 = (i0 + 2) % nbytes
    w = qs[i0] | (qs[i1] << 8) | (qs[i2] << 16)
    return (w >> (off & 7)) & 0xffff


def decode_block(type_id, block, lut):
    v, k = TYPES[type_id]
    d = np.frombuffer(block[:2], dtype=np.float16).astype(np.float32)[0]
    qs = block[2:]
    out = []
    if type_id == 144:
        nsteps = QK // 4
        for t in range(nsteps):
            s = (qs[(t + nsteps - 1) % nsteps] << 8) | qs[t]
            for h in range(2):
                idx = hyb_index(2 * s + h)
                out += [d * lut[2 * idx], d * lut[2 * idx + 1]]
    elif v == 4:
        nsteps = QK // 4
        for t in range(nsteps):
            s = stream_state(qs, k, nsteps, t)
            for h in range(2):
                idx = hyb_index(2 * s + h)
                out += [d * lut[2 * idx], d * lut[2 * idx + 1]]
    else:
        nsteps = QK // 2
        for t in range(nsteps):
            idx = hyb_index(2 * stream_state(qs, k, nsteps, t))
            out += [d * lut[2 * idx], d * lut[2 * idx + 1]]
    return np.array(out, dtype=np.float32)


def main() -> None:
    src = open(sys.argv[1]).read()
    m = re.search(r'GGML_TABLE_BEGIN\(uint16_t, tq2t_lut_f16, 4096\)(.*?)GGML_TABLE_END', src, re.S)
    bits = np.array([int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', m.group(1))], dtype=np.uint16)
    assert len(bits) == 4096
    lut = bits.view(np.float16).astype(np.float32)

    rng = random.Random(9001)
    records = []
    for type_id, (v, k) in TYPES.items():
        block_bytes = 2 + (QK // v) * k // 8
        for _ in range(2):
            data = bytearray()
            values = []
            for _ in range(ROWS * COLS // QK):
                d = np.float16(rng.uniform(0.001, 0.05) * rng.choice([-1, 1]))
                block = d.tobytes() + bytes(rng.getrandbits(8) for _ in range(block_bytes - 2))
                data += block
                values.append(decode_block(type_id, block, lut))
            records.append((type_id, bytes(data), np.concatenate(values)))

    with open(sys.argv[2], 'wb') as out:
        out.write(struct.pack('<i', len(records)))
        for type_id, data, values in records:
            out.write(struct.pack('<iii', type_id, ROWS, COLS))
            out.write(data)
            out.write(values.astype('<f4').tobytes())


if __name__ == '__main__':
    main()
