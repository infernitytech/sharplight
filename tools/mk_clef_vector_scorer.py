"""Rewrites a clef GGUF with decision.scorer_out.weight stored as a vector [W], as the released Clef-Flash GGUFs store it.

Usage: python mk_clef_vector_scorer.py <tiny-random-clef-f32.gguf> <out.gguf>   (gguf-py of llama.cpp on PYTHONPATH)
"""
import sys

import gguf
import numpy as np

source, target = sys.argv[1], sys.argv[2]
reader = gguf.GGUFReader(source)
arch = reader.fields['general.architecture'].contents()
writer = gguf.GGUFWriter(target, arch)
for key, field in reader.fields.items():
    if key.startswith('GGUF.') or key == 'general.architecture':
        continue
    kind = field.types[0]
    if kind == gguf.GGUFValueType.ARRAY:
        writer.add_array(key, field.contents())
    else:
        writer.add_key_value(key, field.contents(), kind)
for t in reader.tensors:
    data = np.array(t.data)
    if t.name == 'decision.scorer_out.weight':
        data = data.reshape(-1)   # the unit output dimension dropped
    writer.add_tensor(t.name, np.ascontiguousarray(data), raw_dtype=None if t.tensor_type == gguf.GGMLQuantizationType.F32 else t.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
