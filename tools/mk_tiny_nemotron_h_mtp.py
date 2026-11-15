"""Appends a multi-token-prediction (NextN) block to a tiny nemotron_h_moe GGUF, laid out as llama.cpp's converter writes the
Nemotron 3.5 MTP head (Lightning): one extra block after the trunk (block_count + 1, nextn_predict_layers 1) holding an attention
sub-layer, a mixture of experts, the NextN input fusion (enorm, hnorm, eh_proj), post_attention_norm and the head norm. The block's
weights are random; the trunk is unchanged, so the trunk's greedy output must stay that of the source file.

Usage: python mk_tiny_nemotron_h_mtp.py <tiny-random-nemotron-h-latent-f32.gguf> <out.gguf>   (gguf-py of llama.cpp on PYTHONPATH)
"""
import sys

import gguf
import numpy as np

source, target = sys.argv[1], sys.argv[2]
reader = gguf.GGUFReader(source)
arch = reader.fields['general.architecture'].contents()
blocks = reader.fields[f'{arch}.block_count'].contents()
writer = gguf.GGUFWriter(target, arch)
for key, field in reader.fields.items():
    if key.startswith('GGUF.') or key == 'general.architecture':
        continue
    value = field.contents()
    if key == f'{arch}.block_count':
        writer.add_uint32(key, blocks + 1)
    elif key == f'{arch}.feed_forward_length':
        writer.add_array(key, value + [max(value)])          # the MTP block's experts
    elif key == f'{arch}.attention.head_count_kv':
        writer.add_array(key, value + [max(value)])          # and its attention
    elif field.types[0] == gguf.GGUFValueType.ARRAY:
        writer.add_array(key, value)
    else:
        writer.add_key_value(key, value, field.types[0])
writer.add_uint32(f'{arch}.nextn_predict_layers', 1)

tensors = {t.name: np.array(t.data) for t in reader.tensors}
for name, data in tensors.items():
    writer.add_tensor(name, np.ascontiguousarray(data))

rng = np.random.default_rng(1234)
hidden = tensors['token_embd.weight'].shape[1]
mtp = f'blk.{blocks}.'
attention = next(i for i in range(blocks) if f'blk.{i}.attn_q.weight' in tensors)
moe = next(i for i in range(blocks) if f'blk.{i}.ffn_gate_inp.weight' in tensors)
extra = {}
for i, kinds in [(attention, ('attn_q', 'attn_k', 'attn_v', 'attn_output')),
                 (moe, ('ffn_up_exps', 'ffn_down_exps', 'ffn_latent_down', 'ffn_latent_up', 'ffn_gate_inp', 'exp_probs_b',
                        'ffn_up_shexp', 'ffn_down_shexp'))]:
    for name, data in tensors.items():
        prefix = f'blk.{i}.'
        if name.startswith(prefix) and name[len(prefix):].split('.')[0] in kinds:
            extra[mtp + name[len(prefix):]] = rng.standard_normal(data.shape).astype(np.float32) * 0.08
for norm in ['attn_norm.weight', 'post_attention_norm.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight']:
    extra[mtp + norm] = np.ones(hidden, dtype=np.float32)
extra[mtp + 'nextn.eh_proj.weight'] = rng.standard_normal((hidden, 2 * hidden)).astype(np.float32) * 0.08
for name, data in extra.items():
    writer.add_tensor(name, data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
