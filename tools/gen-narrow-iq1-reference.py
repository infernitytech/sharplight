#!/usr/bin/env python3
"""Generates tests/Sharplight.Gguf.Tests/Fixtures/ggml_narrow_iq1.bin from a build of unsloth's llama.cpp fork (libggml-base.so).

The fork adds the narrow IQ1 types IQ1_XS (64), IQ1_XXS (65) and IQ1_XXXS (66). Every type gets two records: blocks produced by
the fork's quantizer from gaussian data (with an importance matrix), and blocks of random bytes. Each record carries the float
values the fork's `to_float` type trait decodes the blocks to. Sharplight widens the blocks to IQ1_S and is tested against these
values bit for bit.

usage: gen-narrow-iq1-reference.py <dir containing the fork's libggml-base.so> <output.bin>

Record layout (little endian): i32 type id, i32 rows, i32 cols, byte q[rows * rowBytes], f32 dequantized[rows * cols].
The file starts with the i32 record count.
"""
import ctypes
import random
import struct
import sys

TYPES = {64: "iq1_xs", 65: "iq1_xxs", 66: "iq1_xxxs"}
ROWS, COLS = 4, 512


def main() -> None:
    lib = ctypes.CDLL(sys.argv[1].rstrip("/") + "/libggml-base.so")
    lib.ggml_get_type_traits.restype = ctypes.c_void_p
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]

    rng = random.Random(4242)
    records = []
    for type_id in TYPES:
        lib.ggml_quantize_init(type_id)
        traits = lib.ggml_get_type_traits(type_id)
        # struct ggml_type_traits: name, blck_size, blck_size_interleave, type_size, is_quantized, to_float, from_float_ref
        to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(
            ctypes.c_void_p.from_address(traits + 40).value)
        row_bytes = lib.ggml_row_size(type_id, COLS)

        src = (ctypes.c_float * (ROWS * COLS))()
        for i in range(ROWS * COLS):
            src[i] = rng.gauss(0, 1) * (1 + i // COLS) + 0.3 * (i % COLS > 256)
        imatrix = (ctypes.c_float * COLS)(*[0.5 + rng.random() * 2 for _ in range(COLS)])
        quantized = (ctypes.c_ubyte * (row_bytes * ROWS))()
        lib.ggml_quantize_chunk(type_id, src, quantized, 0, ROWS, COLS, imatrix)
        records.append((type_id, bytes(quantized), to_float, row_bytes))

        random_bytes = bytes(rng.getrandbits(8) for _ in range(row_bytes * ROWS))
        records.append((type_id, random_bytes, to_float, row_bytes))

    with open(sys.argv[2], "wb") as out:
        out.write(struct.pack("<i", len(records)))
        for type_id, data, to_float, row_bytes in records:
            out.write(struct.pack("<iii", type_id, ROWS, COLS))
            out.write(data)
            decoded = (ctypes.c_float * (ROWS * COLS))()
            raw = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            for r in range(ROWS):
                to_float(ctypes.addressof(raw) + r * row_bytes, ctypes.addressof(decoded) + r * COLS * 4, COLS)
            out.write(bytes(decoded))


if __name__ == "__main__":
    main()
