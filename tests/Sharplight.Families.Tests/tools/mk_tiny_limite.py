"""
Writes a tiny random Limite checkpoint (LimiteForCausalLM, remote code that transformers does not ship) in the published layout, without running
the remote code.

    python mk_tiny_limite.py <models/hf directory>

The checkpoint, <dir>/tiny-random-limite/, stores q/k/v and gate/up separately with the scalar scales next to them (qkv_scale, o_scale,
resid_lambda_* / post_lambda_*), per-head attention gates and value-embedding gates that read only the first channels of the layer input,
value embeddings for more heads than the model has key/value heads and narrower than a head, exclusive self attention on all layers but 2,
and the MUDD dense connections (model.mudd.*) into layers 3 and 5. Tokenizer: the 1024-token GPT-2-style one of tiny-random-GPTNeoXForCausalLM.
The reference is llama.cpp on the GGUF that its converter makes of the checkpoint (convert_tiny_gguf.py), or tools/ref_limite.py on that GGUF:
the remote code computes its rotary tables in bfloat16, which the converted model does not.
"""
import json, os, shutil, sys
import numpy as np
import torch
from safetensors.torch import save_file

root = os.path.expanduser(sys.argv[1])
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, FFN, LAYERS, VOCAB = 64, 4, 2, 16, 128, 6, 1024
VE_HEADS, VE_DIM, VE_CHANNELS, GATE_CHANNELS, MUDD_INTER = 3, 12, 12, 24, 8
TAPS = {3: [0, 1, 3], 5: [0, 2, 5]}
VE_LAYERS, XSA_LAYERS, GLOBAL_LAYERS = [1, 4], [0, 1, 3, 4, 5], [3]
out = os.path.join(root, "tiny-random-limite")
rng = np.random.default_rng(int(os.environ.get("LIMITE_SEED", 60)))


def weight(*shape, scale=0.15):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def scalar(center=1.0, spread=0.3):
    return np.array(center + spread * rng.standard_normal(), dtype=np.float32)


t = {"model.embed_tokens.weight": weight(VOCAB, HIDDEN), "model.value_embeds.weight": weight(VOCAB, VE_HEADS * VE_DIM, scale=0.6)}
for il in range(LAYERS):
    p = f"model.layers.{il}."
    t.update({p + "self_attn.q_proj.weight": weight(HEADS * HEAD_DIM, HIDDEN), p + "self_attn.k_proj.weight": weight(KV_HEADS * HEAD_DIM, HIDDEN),
              p + "self_attn.v_proj.weight": weight(KV_HEADS * HEAD_DIM, HIDDEN), p + "self_attn.o_proj.weight": weight(HIDDEN, HEADS * HEAD_DIM),
              p + "self_attn.qkv_scale": scalar(), p + "self_attn.o_scale": scalar(),
              p + "self_attn.attn_gate": weight(HEADS, GATE_CHANNELS, scale=0.6), p + "self_attn.xsa_alpha": weight(HEADS, scale=1.0),
              p + "mlp.gate_proj.weight": weight(FFN, HIDDEN), p + "mlp.up_proj.weight": weight(FFN, HIDDEN), p + "mlp.down_proj.weight": weight(HIDDEN, FFN),
              p + "resid_lambda_attn": scalar(), p + "post_lambda_attn": scalar(), p + "resid_lambda_mlp": scalar(), p + "post_lambda_mlp": scalar()})
    if il in VE_LAYERS:
        t[p + "self_attn.ve_gate"] = weight(VE_HEADS, VE_CHANNELS, scale=0.6)
t.update({"model.mudd.dense1": weight(MUDD_INTER, HIDDEN, scale=0.6), "model.mudd.dense2": weight(LAYERS, 3, MUDD_INTER, scale=0.6),
          "model.mudd.dense2_mlp": weight(LAYERS, 3, MUDD_INTER, scale=0.6), "model.mudd.bias": (0.5 + weight(LAYERS, 3, scale=0.3)),
          "model.mudd.bias_mlp": (0.5 + weight(LAYERS, 3, scale=0.3))})
os.makedirs(out, exist_ok=True)
save_file({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in t.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

config = {
    "architectures": ["LimiteForCausalLM"], "model_type": "limite", "vocab_size": VOCAB, "padded_vocab_size": VOCAB, "tokenizer_vocab_size": VOCAB,
    "hidden_size": HIDDEN, "intermediate_size": FFN, "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM, "max_position_embeddings": 256, "torch_dtype": "float32", "tie_word_embeddings": True,
    "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0,
    "attention_softmax_scale": 0.3, "attn_gate_applied": "per_head_before_o_proj", "attn_gate_channels": GATE_CHANNELS, "attn_gate_scale": 2.0,
    "global_every": 4, "global_layers": GLOBAL_LAYERS, "global_nope": True, "global_window": -1, "sliding_window": 7,
    "mlp_type": "swiglu", "mudd": True, "mudd_mlp": True, "mudd_r_site": "resid", "mudd_inter": MUDD_INTER, "mudd_taps": 3,
    "mudd_layers": sorted(TAPS), "mudd_at": sorted(TAPS), "mudd_tap_idx": {str(k): v for k, v in TAPS.items()},
    "pos_mode": "rope", "qk_norm": "rms_pre_rope", "rms_norm_eps_mode": "torch_finfo_default", "rms_norm_has_weight": False,
    "rope_base_global": 1024.0, "rope_base_local": 1024.0, "rope_cos_sin_dtype": "bfloat16", "rope_frac": 0.5, "rope_n_pairs": HEAD_DIM // 4,
    "rope_per_layer": False, "rope_style": "interleaved_pairs_odd_lane_sign_flip",
    "softcap_logits": {"a": 23.0, "b": 5.0, "c": 7.5, "kind": "sigmoid"}, "final_softcap": 0.0,
    "ve_applied_before_qk_norm": True, "ve_dim": VE_DIM, "ve_gate_channels": VE_CHANNELS, "ve_gate_scale": 2.0,
    "ve_head_slice": "first_num_key_value_heads", "ve_layers": VE_LAYERS, "ve_stored_heads": VE_HEADS,
    "xsa": True, "xsa_layers": XSA_LAYERS, "xsa_normalize_eps": 1e-4,
}
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    if os.path.exists(os.path.join(tokenizer_dir, f)):
        shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))
print("wrote", out)
