"""Writes a heterogeneous ("Puzzle") variant of the tiny latent Nemotron-H GGUF (tools/mk_tiny_nemotron_h_latent.py, converted to f32): the
second MoE layer keeps only its first 6 of 8 experts (router rows, selection bias, expert banks) and selects 3 experts per token, the first
one 2, stored as llama.cpp's Nemotron 3 Puzzle files do (expert_used_count per layer). llama.cpp's greedy text is the reference.

Usage: python mk_tiny_nemotron_h_puzzle.py <tiny-random-nemotron-h-latent-f32.gguf> <out.gguf>   (gguf-py of llama.cpp on PYTHONPATH)
"""
import sys

import gguf
import numpy as np

source, target = sys.argv[1], sys.argv[2]
reader = gguf.GGUFReader(source)
arch = "nemotron_h_moe"
writer = gguf.GGUFWriter(target, arch)
layers = int(reader.fields[f"{arch}.block_count"].contents())
moe_layers = [i for i in range(layers) if f"blk.{i}.ffn_gate_inp.weight" in {t.name for t in reader.tensors}]
used = [0] * layers
used[moe_layers[0]], used[moe_layers[1]] = 2, 3
kept = {moe_layers[1]: 6}

for key, field in reader.fields.items():
    if key.startswith("GGUF.") or key == "general.architecture":
        continue
    if key == f"{arch}.expert_used_count":
        writer.add_array(key, used)
        continue
    kind = field.types[0]
    if kind == gguf.GGUFValueType.ARRAY:
        writer.add_array(key, field.contents())
    else:
        writer.add_key_value(key, field.contents(), kind)

for tensor in reader.tensors:
    data = np.array(tensor.data)
    block = tensor.name.split(".")
    if block[0] == "blk" and int(block[1]) in kept and ("exps" in tensor.name or "ffn_gate_inp" in tensor.name or "exp_probs_b" in tensor.name):
        data = data[: kept[int(block[1])]]   # the expert dimension is the slowest one
    writer.add_tensor(tensor.name, np.ascontiguousarray(data))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
