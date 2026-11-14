"""Writes a header-only GGUF fixture: the metadata and the complete tensor table of a GGUF file, without tensor data.

Usage: python make-gguf-header-fixture.py <model.gguf> <out.gguf>

The tokenizer arrays (tokens, scores, token types, merges) and the chat template are replaced by empty values, so the fixture of a
multi-gigabyte model is a few tens of kilobytes; every other key and every tensor entry (name, dimensions, type, offset) is copied
unchanged. Compatibility and importer-mapping tests read such fixtures with GgufHeaderProbe, as Studio reads the start of a file
on the Hub before downloading it.
"""
import struct
import sys

source, target = sys.argv[1], sys.argv[2]
EMPTIED = {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.merges", "tokenizer.chat_template"}
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_value(f, kind):
    """Returns the raw bytes of one metadata value of the given type."""
    if kind == 8:
        length = f.read(8)
        return length + f.read(struct.unpack("<Q", length)[0])
    if kind == 9:
        element = f.read(4)
        count = f.read(8)
        element_kind, n = struct.unpack("<I", element)[0], struct.unpack("<Q", count)[0]
        return element + count + b"".join(read_value(f, element_kind) for _ in range(n))
    return f.read(struct.calcsize(SCALAR[kind]))


def empty_value(kind, raw):
    """An empty value of the same type: an empty string, or an empty array of the same element type."""
    if kind == 8:
        return struct.pack("<Q", 0)
    if kind == 9:
        return raw[:4] + struct.pack("<Q", 0)
    return raw


with open(source, "rb") as f:
    magic, version = f.read(4), f.read(4)
    assert magic == b"GGUF", "not a GGUF file"
    tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
    pairs = []
    for _ in range(kv_count):
        key_length = f.read(8)
        key = f.read(struct.unpack("<Q", key_length)[0])
        kind_bytes = f.read(4)
        kind = struct.unpack("<I", kind_bytes)[0]
        raw = read_value(f, kind)
        if key.decode() in EMPTIED:
            raw = empty_value(kind, raw)
        pairs.append(key_length + key + kind_bytes + raw)
    tensors = []
    for _ in range(tensor_count):
        name_length = f.read(8)
        name = f.read(struct.unpack("<Q", name_length)[0])
        dimensions = f.read(4)
        dims = f.read(8 * struct.unpack("<I", dimensions)[0])
        tensors.append(name_length + name + dimensions + dims + f.read(4 + 8))

with open(target, "wb") as out:
    out.write(magic + version + struct.pack("<QQ", tensor_count, kv_count))
    out.write(b"".join(pairs))
    out.write(b"".join(tensors))
print(f"wrote {target}: {kv_count} keys, {tensor_count} tensors")
