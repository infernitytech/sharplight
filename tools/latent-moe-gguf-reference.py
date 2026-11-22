"""Float32 transformers reference for a layer-truncated deepseek2 (DeepSeek-V3.x) or glm-dsa (GLM-5.x) GGUF (see truncate-gguf.py):
transformers' DeepseekV3 / GlmMoeDsa on the file's weights dequantized with gguf-py, the routed experts dequantized lazily (only those the router selects), so a few real MoE
layers fit in host memory. The absorbed attn_k_b / attn_v_b tensors are joined back into kv_b_proj. Writes the logits of every prompt row as
float32 and prints the last row's best tokens.

    python latent-moe-gguf-reference.py <file.gguf> <ids, comma-separated> <out.f32> [n_group topk_group]

An optional sixth argument "llama-eps" normalizes the compressed query and latent with rms_norm_eps as llama.cpp does (transformers' GLM-5
uses 1e-6 there). n_group / topk_group default to 1 (no group-limited routing), as llama.cpp and Sharplight read files without expert_group_count.
LAZY_LAYERS=1 dequantizes each decoder layer only while it runs (deep truncations in little host memory).
Needs llama.cpp's gguf-py ($LLAMA_SRC/gguf-py when LLAMA_SRC is set, else an installed gguf package), torch and transformers >= 5.
"""
import sys, re
import os
if os.environ.get('LLAMA_SRC'):
    sys.path.insert(0, os.path.join(os.environ['LLAMA_SRC'], 'gguf-py'))
import numpy as np, torch
from torch import nn
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM, GlmMoeDsaConfig, GlmMoeDsaForCausalLM

path, ids, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
n_group = int(sys.argv[4]) if len(sys.argv) > 4 else 1
topk_group = int(sys.argv[5]) if len(sys.argv) > 5 else 1
r = GGUFReader(path)
meta = {k: f.contents() for k, f in r.fields.items() if not k.startswith('tokenizer.')}
T = {t.name: t for t in r.tensors}
ARCH = meta['general.architecture']
A = ARCH + '.'
GLM = ARCH == 'glm-dsa'
L = meta[A + 'block_count']
def deq(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type)
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

heads = meta[A + 'attention.head_count']
rope = meta[A + 'rope.dimension_count']
qk = meta[A + 'attention.key_length_mla']; vdim = meta[A + 'attention.value_length_mla']
common = dict(
    vocab_size=meta[A + 'vocab_size'] if A + 'vocab_size' in meta else T['token_embd.weight'].shape[1],
    hidden_size=meta[A + 'embedding_length'], intermediate_size=meta[A + 'feed_forward_length'],
    moe_intermediate_size=meta[A + 'expert_feed_forward_length'], num_hidden_layers=L,
    num_attention_heads=heads, num_key_value_heads=heads, n_shared_experts=meta[A + 'expert_shared_count'],
    n_routed_experts=meta[A + 'expert_count'], routed_scaling_factor=meta[A + 'expert_weights_scale'],
    kv_lora_rank=meta[A + 'attention.kv_lora_rank'], q_lora_rank=meta[A + 'attention.q_lora_rank'],
    qk_rope_head_dim=rope, v_head_dim=vdim, qk_nope_head_dim=qk - rope, n_group=n_group, topk_group=topk_group,
    num_experts_per_tok=meta[A + 'expert_used_count'], first_k_dense_replace=meta[A + 'leading_dense_block_count'],
    norm_topk_prob=bool(meta[A + 'expert_weights_norm']), rms_norm_eps=meta[A + 'attention.layer_norm_rms_epsilon'],
    max_position_embeddings=meta[A + 'context_length'], tie_word_embeddings=False, attn_implementation="eager", dtype=torch.float32)
if GLM:
    # The indexer keeps every visible position while the prompt is shorter than index_topk (the dense regime).
    cfg = GlmMoeDsaConfig(**common, index_topk=meta[A + 'attention.indexer.top_k'], index_head_dim=meta[A + 'attention.indexer.key_length'],
                          index_n_heads=meta[A + 'attention.indexer.head_count'], head_dim=rope,
                          rope_parameters={"rope_type": "default", "rope_theta": meta[A + 'rope.freq_base']})
else:
    cfg = DeepseekV3Config(**common, rope_interleave=True,
        rope_parameters={"rope_type": "yarn", "rope_theta": meta[A + 'rope.freq_base'], "factor": meta[A + 'rope.scaling.factor'],
                         "original_max_position_embeddings": meta[A + 'rope.scaling.original_context_length'],
                         "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": meta[A + 'rope.scaling.yarn_log_multiplier'] / 0.1})
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
    model = (GlmMoeDsaForCausalLM if GLM else DeepseekV3ForCausalLM)(cfg)
for i in range(cfg.first_k_dense_replace, L):
    model.model.layers[i].mlp.experts = LazyExperts(i)
# LAZY_LAYERS=1: every decoder layer's weights are dequantized just before the layer runs and dropped after it, so that deep
# truncations fit in host memory (the head and the embedding stay resident).
lazy = os.environ.get('LAZY_LAYERS') == '1'
nope = qk - rope
NAMES = {'attn_norm.weight': 'input_layernorm.weight', 'ffn_norm.weight': 'post_attention_layernorm.weight',
         'attn_q_a.weight': 'self_attn.q_a_proj.weight', 'attn_q_a_norm.weight': 'self_attn.q_a_layernorm.weight',
         'attn_q_b.weight': 'self_attn.q_b_proj.weight', 'attn_kv_a_mqa.weight': 'self_attn.kv_a_proj_with_mqa.weight',
         'attn_kv_a_norm.weight': 'self_attn.kv_a_layernorm.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
         'ffn_gate.weight': 'mlp.gate_proj.weight', 'ffn_up.weight': 'mlp.up_proj.weight', 'ffn_down.weight': 'mlp.down_proj.weight',
         'ffn_gate_inp.weight': 'mlp.gate.weight', 'exp_probs_b.bias': 'mlp.gate.e_score_correction_bias',
         'ffn_gate_shexp.weight': 'mlp.shared_experts.gate_proj.weight', 'ffn_up_shexp.weight': 'mlp.shared_experts.up_proj.weight',
         'ffn_down_shexp.weight': 'mlp.shared_experts.down_proj.weight',
         'indexer.attn_q_b.weight': 'self_attn.indexer.wq_b.weight', 'indexer.attn_k.weight': 'self_attn.indexer.wk.weight',
         'indexer.k_norm.weight': 'self_attn.indexer.k_norm.weight', 'indexer.k_norm.bias': 'self_attn.indexer.k_norm.bias',
         'indexer.proj.weight': 'self_attn.indexer.weights_proj.weight'}


def layer_state(i):
    p, sd = f'blk.{i}.', {}
    for g, h in NAMES.items():
        if p + g in T:
            sd[h] = deq(p + g)
    kb = deq(p + 'attn_k_b.weight')   # [heads, lora, nope]
    vb = deq(p + 'attn_v_b.weight')   # [heads, v, lora]
    sd['self_attn.kv_b_proj.weight'] = torch.cat([kb.transpose(1, 2), vb], dim=1).reshape(heads * (nope + vdim), -1)
    return sd


def release(layer, args, output):
    layer.to('meta')   # returns None: the layer's output stays as it is


def materialize(layer, i):
    layer.to_empty(device='cpu')
    missing, unexpected = layer.load_state_dict(layer_state(i), strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if 'experts' not in k], missing


for module in [model.model.embed_tokens, model.model.norm, model.lm_head]:
    module.to_empty(device='cpu')
model.model.embed_tokens.weight.data = deq('token_embd.weight')
model.model.norm.weight.data = deq('output_norm.weight')
model.lm_head.weight.data = deq('output.weight')
for i, layer in enumerate(model.model.layers):
    if lazy:
        layer.register_forward_pre_hook(lambda module, args, i=i: materialize(module, i))
        layer.register_forward_hook(release)
    else:
        materialize(layer, i)
# rotary buffers were emptied by to_empty: rebuild them
if GLM:
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding as Rotary
else:
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RotaryEmbedding as Rotary
model.model.rotary_emb = Rotary(cfg)
if len(sys.argv) > 6 and sys.argv[6] == 'llama-eps':
    # llama.cpp's convention: the compressed query and latent are normalized with rms_norm_eps (transformers: 1e-6).
    for layer in model.model.layers:
        layer.self_attn.q_a_layernorm.variance_epsilon = cfg.rms_norm_eps
        layer.self_attn.kv_a_layernorm.variance_epsilon = cfg.rms_norm_eps
model.eval()
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0]
logits.numpy().astype(np.float32).tofile(out)
last = logits[-1]
lp = torch.log_softmax(last, -1)
top = torch.topk(last, 8)
print('top', [(int(i), round(float(last[i]), 4), round(float(lp[i]), 4)) for i in top.indices])
