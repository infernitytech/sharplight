"""Cut a (split) GGUF to its first N blocks, keeping the global tensors (embedding, output norm, head) and every metadata key except the
split keys; block_count is set to N. Tensor data is copied as stored (no re-quantization), so the file holds the real weights.

    python truncate-gguf.py <shard dir | single file.gguf> <out.gguf> <layers>

Needs llama.cpp's gguf-py on PYTHONPATH (e.g. PYTHONPATH=$LLAMA_SRC/gguf-py).
"""
import sys, glob, re, os
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(os.path.join(src, '*.gguf'))) if os.path.isdir(src) else [src]
readers = [GGUFReader(f) for f in files]
r0 = readers[0]
arch = r0.fields['general.architecture'].contents()
w = GGUFWriter(out, arch, use_temp_file=False)
for k, f in r0.fields.items():
    if k.startswith('GGUF.') or k.startswith('split.') or k == 'general.architecture':
        continue
    vt = f.types[0]
    v = f.contents()
    if k == f'{arch}.block_count':
        v = n
    if vt == GGUFValueType.ARRAY:
        w.add_key_value(k, v, vt, sub_type=f.types[-1] if len(f.types) > 1 else None)
    else:
        w.add_key_value(k, v, vt)
keep = []
for r in readers:
    for t in r.tensors:
        m = re.match(r'blk\.(\d+)\.', t.name)
        if m and int(m.group(1)) >= n:
            continue
        keep.append(t)
for t in keep:
    w.add_tensor_info(t.name, list(t.data.shape), np.dtype(np.uint8) if t.data.dtype == np.uint8 else t.data.dtype, t.n_bytes, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data)
w.close()
print('wrote', out, len(keep), 'tensors')
