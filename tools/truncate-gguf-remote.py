"""Cut a split GGUF on the Hugging Face Hub to its first N blocks, fetching only the header and the kept tensors' byte ranges.

    python -I tools/truncate-gguf-remote.py <repo | local dir> <revision> <first shard path in the repo> <out.gguf> <layers> [--widen-narrow-iq1]

The global tensors (embedding, output norm, head, ...) and blocks 0..N-1 are kept with their stored bytes, every metadata key of the first
shard is kept except the split keys, per-layer arrays of the architecture (at least as long as block_count) keep their first N entries, and block_count becomes N (nextn_predict_layers 0: multi-token-prediction layers are dropped).
--widen-narrow-iq1 rewrites the narrow IQ1 types of unsloth's llama.cpp fork (ggml ids 64-66, Kimi-K3's UD-TQ*/UD-Q1_0 files) as IQ1_S
blocks with the same values (the codebook tables of src/Sharplight.Import.Gguf/GgmlNarrowIq1Grids.cs), so that llama.cpp releases without
those types read the file. The shards of a split are named <prefix>-0000k-of-0000n.gguf. HF_TOKEN is used when set.
"""
import os
import re
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

args = [a for a in sys.argv[1:] if not a.startswith("--")]
widen = "--widen-narrow-iq1" in sys.argv
repo, revision, first, out, layers = args[0], args[1], args[2], args[3], int(args[4])
token = os.environ.get("HF_TOKEN")
base = f"https://huggingface.co/{repo}/resolve/{revision}/"

# ggml type id -> (values per block, bytes per block)
SIZES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36), 10: (256, 84),
         11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
         19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
         28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17), 64: (256, 46), 65: (256, 42), 66: (256, 38)}
IQ1_S = 19
NARROW = {64: "Xs", 65: "Xxs", 66: "Xxxs"}


def get(name, start, end):
    if os.path.isdir(repo):
        # A local directory instead of a repository (re-cutting a file already on disk).
        with open(os.path.join(repo, name), "rb") as f:
            f.seek(start)
            return f.read(end - start + 1)
    request = urllib.request.Request(base + name, headers={"Range": f"bytes={start}-{end}"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except Exception:
            if attempt == 5:
                raise


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return value[0]

    def string(self):
        n = self.take("Q")
        s = self.data[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s


SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def skip_value(r, vtype):
    if vtype == 8:
        r.string()
    elif vtype == 9:
        inner, n = r.take("I"), r.take("Q")
        for _ in range(n):
            skip_value(r, inner)
    else:
        r.take(SCALAR[vtype])


def read_header(name):
    size = 1 << 22
    while True:
        data = get(name, 0, size - 1)
        try:
            r = Reader(data)
            assert r.take("I") == 0x46554747, "not a GGUF file"
            version, n_tensors, n_kv = r.take("I"), r.take("Q"), r.take("Q")
            kvs = []
            alignment = 32
            for _ in range(n_kv):
                start = r.pos
                key = r.string()
                vtype = r.take("I")
                value_start = r.pos
                if key == "general.alignment":
                    alignment = struct.unpack_from("<I", data, r.pos)[0]
                skip_value(r, vtype)
                kvs.append((key, vtype, data[start:r.pos], data[value_start:r.pos]))
            tensors = []
            for _ in range(n_tensors):
                tname = r.string()
                dims = [r.take("Q") for _ in range(r.take("I"))]
                ttype, offset = r.take("I"), r.take("Q")
                tensors.append((tname, dims, ttype, offset))
            data_start = (r.pos + alignment - 1) // alignment * alignment
            return version, kvs, tensors, data_start, alignment
        except (struct.error, IndexError, UnicodeDecodeError):
            size *= 4
            if size > 1 << 30:
                raise


def nbytes(dims, ttype):
    count = 1
    for d in dims:
        count *= d
    block, size = SIZES[ttype]
    return count // block * size


def load_grids():
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "Sharplight.Import.Gguf", "GgmlNarrowIq1Grids.cs")).read()
    grids = {}
    for name in ("Xs", "Xxs", "Xxxs"):
        body = re.search(rf"ReadOnlySpan<ushort> {name} =>\s*\[(.*?)\];", source, re.S).group(1)
        grids[name] = [int(v) for v in re.findall(r"\d+", body)]
    return grids


def widen_blocks(ttype, stored, grid):
    """Narrow IQ1 blocks -> IQ1_S blocks with the same values (GgmlNarrowIq1.Widen, vectorized)."""
    import numpy as np
    sb = SIZES[ttype][1]
    s = np.frombuffer(stored, dtype=np.uint8).reshape(-1, sb)
    grid = np.asarray(grid, dtype=np.uint16)
    qs, tail = s[:, 2:34].astype(np.uint16), s[:, 34:].astype(np.uint16)
    out = np.empty((s.shape[0], 50), dtype=np.uint8)
    out[:, 0:2] = s[:, 0:2]
    qh = np.zeros((s.shape[0], 8), dtype=np.uint16)
    for ib in range(8):
        if ttype == 65:
            scale_sign = tail[:, ib] >> 4
        else:
            scale_sign = (tail[:, 8 + ib // 2 if ttype == 64 else ib // 2] >> (4 * (ib & 1))) & 0xF
        qh[:, ib] = (scale_sign & 7) << 12 | (scale_sign >> 3) << 15
        for l in range(4):
            high = (tail[:, ib] >> (2 * l)) & 3 if ttype == 64 else (tail[:, ib] >> l) & 1 if ttype == 65 else 0
            index = grid[qs[:, 4 * ib + l] | (high << 8)]
            out[:, 2 + 4 * ib + l] = (index & 0xFF).astype(np.uint8)
            qh[:, ib] |= (index >> 8) << (3 * l)
    out[:, 34:50] = qh.astype("<u2").view(np.uint8).reshape(-1, 16)
    return out.tobytes()


m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
shards = [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)] if m else [first]
version, kvs, _, _, alignment = read_header(shards[0])
arch = next(v[8:].decode() for k, t, raw, v in kvs if k == "general.architecture")
kept = []
for shard in shards:
    _, _, tensors, data_start, _ = read_header(shard)
    for tname, dims, ttype, offset in tensors:
        block = re.match(r"blk\.(\d+)\.", tname)
        if block and int(block.group(1)) >= layers:
            continue
        kept.append((shard, data_start + offset, tname, dims, ttype))
grids = load_grids() if widen else None

block_count = next(struct.unpack("<" + SCALAR[t], v)[0] for k, t, raw, v in kvs if k == f"{arch}.block_count")
header = bytearray()
out_kvs = []
for key, vtype, raw, value in kvs:
    if key.startswith("split."):
        continue
    if key in (f"{arch}.block_count", f"{arch}.nextn_predict_layers"):
        new_value = layers if key.endswith("block_count") else 0
        enc = key.encode()
        raw = struct.pack("<Q", len(enc)) + enc + struct.pack("<I", vtype) + struct.pack("<" + SCALAR[vtype], new_value)
    elif vtype == 9:
        inner, count = struct.unpack_from("<IQ", value, 0)
        if inner in SCALAR and key.startswith(arch + ".") and count >= block_count and count > layers:
            width = struct.calcsize("<" + SCALAR[inner])
            enc = key.encode()
            raw = struct.pack("<Q", len(enc)) + enc + struct.pack("<IIQ", vtype, inner, layers) + value[12:12 + layers * width]
    out_kvs.append(raw)
plan = []
offset = 0
infos = bytearray()
for shard, start, tname, dims, ttype in kept:
    stored = nbytes(dims, ttype)
    new_type = IQ1_S if widen and ttype in NARROW else ttype
    size = nbytes(dims, new_type)
    enc = tname.encode()
    infos += struct.pack("<Q", len(enc)) + enc + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
    infos += struct.pack("<IQ", new_type, offset)
    plan.append((shard, start, stored, ttype, new_type, offset, size, tname))
    offset = (offset + size + alignment - 1) // alignment * alignment
header += struct.pack("<IIQQ", 0x46554747, version, len(kept), len(out_kvs)) + b"".join(out_kvs) + infos
data_start = (len(header) + alignment - 1) // alignment * alignment
header += b"\0" * (data_start - len(header))

CHUNK = 64 << 20
with open(out, "wb") as f:
    f.write(header)
    with ThreadPoolExecutor(8) as pool:
        for shard, start, stored, ttype, new_type, toff, size, tname in plan:
            pieces = [(start + p, min(start + stored, start + p + CHUNK) - 1) for p in range(0, stored, CHUNK)]
            # Narrow pieces must hold whole blocks.
            if new_type != ttype:
                sb = SIZES[ttype][1]
                chunk = CHUNK // sb * sb
                pieces = [(start + p, min(start + stored, start + p + chunk) - 1) for p in range(0, stored, chunk)]
            f.seek(data_start + toff)
            for data in pool.map(lambda r: get(shard, r[0], r[1]), pieces):
                f.write(widen_blocks(ttype, data, grids[NARROW[ttype]]) if new_type != ttype else data)
            print(f"{tname} {ttype}->{new_type} {stored / 1e6:.1f} MB", flush=True)
        f.truncate(data_start + offset)
print("wrote", out, len(kept), "tensors", f"{(data_start + offset) / 1e9:.2f} GB")
