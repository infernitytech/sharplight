#!/usr/bin/env python3
"""Generates tests/Sharplight.Cpu.Tests/Fixtures/ggml_quant_ext.bin from a llama.cpp build (libggml-base.so).

Every type gets two records: blocks produced by ggml's own quantizer from gaussian data, and blocks of random bytes. Each
record carries the float values ggml's `to_float` type trait decodes the blocks to. The Sharplight dequantizers are tested
against these values bit for bit.

usage: gen-quant-reference.py <dir containing libggml-base.so> <output.bin>

Record layout (little endian): i32 type id, i32 rows, i32 cols, byte q[rows * rowBytes], f32 dequantized[rows * cols].
The file starts with the i32 record count.
"""
import ctypes
import random
import struct
import sys

# ggml type id -> name
TYPES = {
    2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
    10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
    29: "iq1_m", 34: "tq1_0", 35: "tq2_0", 39: "mxfp4", 40: "nvfp4",
}
ROWS, COLS = 4, 512


def main() -> None:
    lib = ctypes.CDLL(sys.argv[1].rstrip("/") + "/libggml-base.so")
    lib.ggml_get_type_traits.restype = ctypes.c_void_p
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]

    rng = random.Random(12345)
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
        needs_imatrix = lib.ggml_quantize_requires_imatrix(type_id)
        lib.ggml_quantize_chunk(type_id, src, quantized, 0, ROWS, COLS, imatrix if needs_imatrix else None)
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
