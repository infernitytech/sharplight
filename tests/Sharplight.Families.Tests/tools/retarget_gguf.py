"""
Rewrites a GGUF under another llama.cpp architecture name, for families whose graph equals an existing one on text input. Used for
clover_moe, whose layers are those of qwen35moe with plain (1-D) rope: the tiny Qwen3.5-MoE checkpoint is converted by llama.cpp's converter,
then retargeted with its M-RoPE sections and its NextN block dropped.

    PYTHONPATH=<llama.cpp>/gguf-py python retarget_gguf.py <input .gguf> <output .gguf> <new arch> [--drop-key k ...] [--drop-last-blocks n]
        [--set-u32 key=value ...]

Keys and tensors are copied unchanged otherwise (keys are renamed from the old architecture prefix to the new one).
"""
import argparse
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("arch")
parser.add_argument("--drop-key", action="append", default=[])
parser.add_argument("--drop-last-blocks", type=int, default=0)
parser.add_argument("--set-u32", action="append", default=[])
args = parser.parse_args()

reader = GGUFReader(args.input)
old = reader.fields["general.architecture"].contents()
blocks = reader.fields[f"{old}.block_count"].contents()
keep_blocks = blocks - args.drop_last_blocks
writer = GGUFWriter(args.output, args.arch)
for key, field in reader.fields.items():
    if key.startswith("GGUF.") or key == "general.architecture":
        continue
    name = args.arch + key[len(old):] if key.startswith(old + ".") else key
    if key[len(old) + 1:] in args.drop_key:
        continue
    value = field.contents()
    if name == f"{args.arch}.block_count":
        value = keep_blocks
    if name == f"{args.arch}.nextn_predict_layers":
        continue
    kind = field.types[0]
    if kind == GGUFValueType.ARRAY:
        sub = field.types[1]
        if len(value) == blocks and sub != GGUFValueType.STRING:
            value = value[:keep_blocks]
        writer.add_key_value(name, value, kind, sub)
    else:
        writer.add_key_value(name, value, kind)
for item in args.set_u32:
    key, value = item.split("=")
    writer.add_uint32(f"{args.arch}.{key}", int(value))
for tensor in reader.tensors:
    if tensor.name.startswith("blk.") and int(tensor.name.split(".")[1]) >= keep_blocks:
        continue
    data = np.array(tensor.data)
    writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("wrote", args.output)
