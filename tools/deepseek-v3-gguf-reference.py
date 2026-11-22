"""Float32 transformers reference for a layer-truncated deepseek2 (DeepSeek-V3.x) GGUF (see truncate-gguf.py): transformers' DeepseekV3
on the file's weights dequantized with gguf-py, the routed experts dequantized lazily (only those the router selects), so a few real MoE
layers fit in host memory. The absorbed attn_k_b / attn_v_b tensors are joined back into kv_b_proj. Writes the logits of every prompt row as
float32 and prints the last row's best tokens.

    python deepseek-v3-gguf-reference.py <file.gguf> <ids, comma-separated> <out.f32> [n_group topk_group]

n_group / topk_group default to 1 (no group-limited routing), as llama.cpp and Sharplight read files without expert_group_count.
Needs llama.cpp's gguf-py on PYTHONPATH, torch and transformers >= 5.
"""
import sys, re
import numpy as np, torch
from torch import nn
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

path, ids, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
n_group = int(sys.argv[4]) if len(sys.argv) > 4 else 1
topk_group = int(sys.argv[5]) if len(sys.argv) > 5 else 1
r = GGUFReader(path)
meta = {k: f.contents() for k, f in r.fields.items() if not k.startswith('tokenizer.')}
T = {t.name: t for t in r.tensors}
A = 'deepseek2.'
L = meta[A + 'block_count']
def deq(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

heads = meta[A + 'attention.head_count']
rope = meta[A + 'rope.dimension_count']
qk = meta[A + 'attention.key_length_mla']; vdim = meta[A + 'attention.value_length_mla']
cfg = DeepseekV3Config(
    vocab_size=meta[A + 'vocab_size'] if A + 'vocab_size' in meta else T['token_embd.weight'].shape[1],
    hidden_size=meta[A + 'embedding_length'], intermediate_size=meta[A + 'feed_forward_length'],
    moe_intermediate_size=meta[A + 'expert_feed_forward_length'], num_hidden_layers=L,
    num_attention_heads=heads, num_key_value_heads=heads, n_shared_experts=meta[A + 'expert_shared_count'],
    n_routed_experts=meta[A + 'expert_count'], routed_scaling_factor=meta[A + 'expert_weights_scale'],
    kv_lora_rank=meta[A + 'attention.kv_lora_rank'], q_lora_rank=meta[A + 'attention.q_lora_rank'],
    qk_rope_head_dim=rope, v_head_dim=vdim, qk_nope_head_dim=qk - rope, n_group=n_group, topk_group=topk_group,
    num_experts_per_tok=meta[A + 'expert_used_count'], first_k_dense_replace=meta[A + 'leading_dense_block_count'],
    norm_topk_prob=bool(meta[A + 'expert_weights_norm']), rms_norm_eps=meta[A + 'attention.layer_norm_rms_epsilon'],
    max_position_embeddings=meta[A + 'context_length'], tie_word_embeddings=False, rope_interleave=True,
    rope_parameters={"rope_type": "yarn", "rope_theta": meta[A + 'rope.freq_base'], "factor": meta[A + 'rope.scaling.factor'],
                     "original_max_position_embeddings": meta[A + 'rope.scaling.original_context_length'],
                     "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": meta[A + 'rope.scaling.yarn_log_multiplier'] / 0.1},
    attn_implementation="eager", dtype=torch.float32)
assert meta[A + 'expert_gating_func'] == 2

class LazyExperts(nn.Module):
    def __init__(self, layer):
        super().__init__(); self.layer = layer
    def forward(self, x, idx, w):
        outp = torch.zeros_like(x)
        p = f'blk.{self.layer}.'
        for e in torch.unique(idx).tolist():
            tok, slot = torch.where(idx == e)
            g = torch.from_numpy(dequantize(T[p + 'ffn_gate_exps.weight'].data[e], T[p + 'ffn_gate_exps.weight'].tensor_type).astype(np.float32))
            u = torch.from_numpy(dequantize(T[p + 'ffn_up_exps.weight'].data[e], T[p + 'ffn_up_exps.weight'].tensor_type).astype(np.float32))
            d = torch.from_numpy(dequantize(T[p + 'ffn_down_exps.weight'].data[e], T[p + 'ffn_down_exps.weight'].tensor_type).astype(np.float32))
            h = x[tok]
            y = (nn.functional.silu(h @ g.T) * (h @ u.T)) @ d.T
            outp.index_add_(0, tok, y * w[tok, slot, None])
        return outp

with torch.device('meta'):
    model = DeepseekV3ForCausalLM(cfg)
for i in range(cfg.first_k_dense_replace, L):
    model.model.layers[i].mlp.experts = LazyExperts(i)
model.to_empty(device='cpu')
sd = {}
sd['model.embed_tokens.weight'] = deq('token_embd.weight')
sd['model.norm.weight'] = deq('output_norm.weight')
sd['lm_head.weight'] = deq('output.weight')
nope = qk - rope
for i in range(L):
    p, q = f'blk.{i}.', f'model.layers.{i}.'
    m = {'attn_norm.weight': 'input_layernorm.weight', 'ffn_norm.weight': 'post_attention_layernorm.weight',
         'attn_q_a.weight': 'self_attn.q_a_proj.weight', 'attn_q_a_norm.weight': 'self_attn.q_a_layernorm.weight',
         'attn_q_b.weight': 'self_attn.q_b_proj.weight', 'attn_kv_a_mqa.weight': 'self_attn.kv_a_proj_with_mqa.weight',
         'attn_kv_a_norm.weight': 'self_attn.kv_a_layernorm.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
         'ffn_gate.weight': 'mlp.gate_proj.weight', 'ffn_up.weight': 'mlp.up_proj.weight', 'ffn_down.weight': 'mlp.down_proj.weight',
         'ffn_gate_inp.weight': 'mlp.gate.weight', 'exp_probs_b.bias': 'mlp.gate.e_score_correction_bias',
         'ffn_gate_shexp.weight': 'mlp.shared_experts.gate_proj.weight', 'ffn_up_shexp.weight': 'mlp.shared_experts.up_proj.weight',
         'ffn_down_shexp.weight': 'mlp.shared_experts.down_proj.weight'}
    for g, h in m.items():
        if p + g in T:
            sd[q + h] = deq(p + g)
    kb = deq(p + 'attn_k_b.weight')   # [heads, lora, nope]
    vb = deq(p + 'attn_v_b.weight')   # [heads, v, lora]
    kv_b = torch.cat([kb.transpose(1, 2), vb], dim=1).reshape(heads * (nope + vdim), -1)
    sd[q + 'self_attn.kv_b_proj.weight'] = kv_b
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [k for k in missing if 'rotary' not in k]
print('missing', missing[:5], 'unexpected', unexpected[:5], flush=True)
# rotary buffers were emptied by to_empty: rebuild them
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding
model.model.rotary_emb = DeepseekV3RotaryEmbedding(cfg)
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0]
logits.numpy().astype(np.float32).tofile(out)
last = logits[-1]
lp = torch.log_softmax(last, -1)
top = torch.topk(last, 8)
print('top', [(int(i), round(float(last[i]), 4), round(float(lp[i]), 4)) for i in top.indices])
