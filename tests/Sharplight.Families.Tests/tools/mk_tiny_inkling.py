"""
Builds a tiny random Inkling checkpoint for Sharplight.Families.Tests in the layout of the published thinkingmachines/Inkling checkpoint
(model_type inkling_mm_model, the text model's config in text_config with the checkpoint's own keys, tensors below model.llm. with fused
interleaved gate/up projections; no vision, audio or MTP weights), in float32 with the GPT-2 style tokenizer of tiny-random-GPTNeoXForCausalLM
(vocabulary 1024), and writes its transformers reference (greedy ids, prompt logits in reference_logits.json):

  tiny-random-inkling   4 layers, sliding-window attention (window 8, one kv head) on layers 1 and 3 and full attention (two kv heads,
                        relative-bias extent 16) on layers 2 and 4; per-head q/k norms with scale 1/head_dim, residual short convolutions of
                        the keys, values, attention output and MLP output, the query-conditioned relative position bias, the log-floor query
                        temperature (floor 8, so the prompt passes it); a dense first layer and MoE layers with 8 experts (top 2) and 2 shared
                        experts scored by the router (sigmoid, selection bias, route scale 2.5, global scales); embedding norm, logits divided
                        by the width multiplier 4 and cut to the unpadded vocabulary of 1000 tokens.

The reference is transformers' InklingForCausalLM (eager attention); the saved checkpoint is checked by renaming it back the way transformers'
inkling_mm_model conversion mapping does and comparing the logits.

    python mk_tiny_inkling.py <models/hf directory> [seed]
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingForCausalLM

root = os.path.expanduser(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3   # the seed with the largest greedy margin among 1-5
out = os.path.join(root, "tiny-random-inkling")
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
NEW_TOKENS = 24

# The checkpoint's own keys (intermediate_size is the experts' width when dense_intermediate_size is present).
text = dict(
    vocab_size=1024, unpadded_vocab_size=1000, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    swa_num_attention_heads=4, swa_num_key_value_heads=1, swa_head_dim=16, sliding_window_size=8, d_rel=4, rel_extent=16,
    log_scaling_n_floor=8, log_scaling_alpha=0.3, local_layer_ids=[0, 2], dense_mlp_idx=1, sconv_kernel_size=4,
    dense_intermediate_size=96, intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=2, route_scale=2.5,
    logits_mup_width_multiplier=4.0, rms_norm_eps=1e-6, max_position_embeddings=4096, hidden_act="silu", bos_token_id=0, eos_token_id=0,
)

config = InklingTextConfig(**dict(text))
config._attn_implementation = "eager"
torch.manual_seed(SEED)
model = InklingForCausalLM(config).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("global_scale"):
            p.copy_(0.8 + 0.4 * torch.rand_like(p))
        elif "norm" in n:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif "conv1d" in n:
            p.copy_(0.3 * torch.randn_like(p))
        elif "rel_logits_proj" in n:
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("mlp.gate.weight"):
            p.copy_(0.5 * torch.randn_like(p))
        elif "embed_tokens" in n or "lm_head" in n:
            p.copy_(0.5 * torch.randn_like(p))
        else:
            p.copy_(0.15 * torch.randn_like(p))
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(0.1 * torch.randn_like(layer.mlp.gate.e_score_correction_bias))


def interleave(gate, up, dim):
    """The checkpoint's fused w13: gate and up rows alternating along dim (inverse of transformers' Interleave)."""
    return torch.stack([gate, up], dim=dim + 1).flatten(dim, dim + 1).contiguous()


def original_state(model):
    """The state dict renamed into the published checkpoint layout."""
    hf = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = {
        "model.llm.embed.weight": hf.pop("model.embed_tokens.weight"),
        "model.llm.embed_norm.weight": hf.pop("model.embed_tokens.embed_norm.weight"),
        "model.llm.norm.weight": hf.pop("model.norm.weight"),
        "model.llm.unembed.weight": hf.pop("lm_head.weight"),
    }
    for i in range(config.num_hidden_layers):
        h, o = f"model.layers.{i}.", f"model.llm.layers.{i}."
        renames = {
            "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "mlp_norm.weight",
            "self_attn.q_proj.weight": "attn.wq_du.weight", "self_attn.k_proj.weight": "attn.wk_dv.weight",
            "self_attn.v_proj.weight": "attn.wv_dv.weight", "self_attn.r_proj.weight": "attn.wr_du.weight",
            "self_attn.o_proj.weight": "attn.wo_ud.weight", "self_attn.q_norm.weight": "attn.q_norm.weight",
            "self_attn.k_norm.weight": "attn.k_norm.weight", "self_attn.k_sconv.conv1d.weight": "attn.k_sconv.weight",
            "self_attn.v_sconv.conv1d.weight": "attn.v_sconv.weight", "self_attn.rel_logits_proj.proj": "attn.rel_logits_proj.proj",
            "attn_sconv.conv1d.weight": "attn_sconv.weight", "mlp_sconv.conv1d.weight": "mlp_sconv.weight",
        }
        for a, b in renames.items():
            state[o + b] = hf.pop(h + a)
        if h + "mlp.gate_proj.weight" in hf:
            state[o + "mlp.w13_dn.weight"] = interleave(hf.pop(h + "mlp.gate_proj.weight"), hf.pop(h + "mlp.up_proj.weight"), 0)
            state[o + "mlp.w2_md.weight"] = hf.pop(h + "mlp.down_proj.weight")
            state[o + "mlp.global_scale"] = hf.pop(h + "mlp.global_scale")
        else:
            state[o + "mlp.gate.weight"] = hf.pop(h + "mlp.gate.weight")
            state[o + "mlp.gate.bias"] = hf.pop(h + "mlp.gate.e_score_correction_bias")
            state[o + "mlp.gate.global_scale"] = hf.pop(h + "mlp.gate.global_scale")
            inner = config.moe_intermediate_size
            gate_up = hf.pop(h + "mlp.experts.gate_up_proj")
            state[o + "mlp.experts.w13_weight"] = interleave(gate_up[:, :inner], gate_up[:, inner:], 1)
            state[o + "mlp.experts.w2_weight"] = hf.pop(h + "mlp.experts.down_proj")
            state[o + "mlp.shared_experts.shared_w13_weight"] = interleave(
                hf.pop(h + "mlp.shared_experts.gate_proj"), hf.pop(h + "mlp.shared_experts.up_proj"), 1)
            state[o + "mlp.shared_experts.shared_w2_weight"] = hf.pop(h + "mlp.shared_experts.down_proj")
    assert not hf, sorted(hf)
    return state


def deinterleave(w, dim):
    """transformers' Interleave(dim): rows 2i + j go to j · N/2 + i."""
    shape = list(w.shape)
    shape[dim:dim + 1] = [shape[dim] // 2, 2]
    return w.reshape(shape).transpose(dim, dim + 1).reshape(w.shape)


def back_to_hf(state):
    """The published layout renamed back the way transformers' inkling_mm_model conversion mapping does (for the round-trip check)."""
    hf = {}
    for name, w in state.items():
        n = name.replace("model.llm.layers", "model.layers")
        n = {"model.llm.embed.weight": "model.embed_tokens.weight", "model.llm.embed_norm.weight": "model.embed_tokens.embed_norm.weight",
             "model.llm.norm.weight": "model.norm.weight", "model.llm.unembed.weight": "lm_head.weight"}.get(n, n)
        if n.endswith("shared_w13_weight"):
            g, u = deinterleave(w, 1).chunk(2, dim=1)
            hf[n.replace("shared_w13_weight", "gate_proj")], hf[n.replace("shared_w13_weight", "up_proj")] = g, u
            continue
        if n.endswith("mlp.w13_dn.weight"):
            g, u = deinterleave(w, 0).chunk(2, dim=0)
            hf[n.replace("w13_dn.weight", "gate_proj.weight")], hf[n.replace("w13_dn.weight", "up_proj.weight")] = g, u
            continue
        if n.endswith("mlp.experts.w13_weight"):
            hf[n.replace("w13_weight", "gate_up_proj")] = deinterleave(w, 1)
            continue
        for a, b in [("shared_w2_weight", "down_proj"), ("mlp.experts.w2_weight", "mlp.experts.down_proj"), ("mlp.w2_md.weight", "mlp.down_proj.weight"),
                     ("mlp.gate.bias", "mlp.gate.e_score_correction_bias"), ("attn.wq_du", "self_attn.q_proj"), ("attn.wk_dv", "self_attn.k_proj"),
                     ("attn.wv_dv", "self_attn.v_proj"), ("attn.wr_du", "self_attn.r_proj"), ("attn.wo_ud", "self_attn.o_proj"),
                     (".attn.q_norm", ".self_attn.q_norm"), (".attn.k_norm", ".self_attn.k_norm"), (".attn.k_sconv", ".self_attn.k_sconv.conv1d"),
                     (".attn.v_sconv", ".self_attn.v_sconv.conv1d"), (".attn.rel_logits_proj", ".self_attn.rel_logits_proj"),
                     ("mlp_norm", "post_attention_layernorm"), ("attn_norm", "input_layernorm")]:
            n = n.replace(a, b)
        if n.endswith("attn_sconv.weight") and "conv1d" not in n:
            n = n.replace("attn_sconv.weight", "attn_sconv.conv1d.weight")
        if n.endswith("mlp_sconv.weight"):
            n = n.replace("mlp_sconv.weight", "mlp_sconv.conv1d.weight")
        hf[n] = w
    return hf


os.makedirs(out, exist_ok=True)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), out)
tokenizer = AutoTokenizer.from_pretrained(out)
prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

state = original_state(model)
check = InklingForCausalLM(config).eval()
missing, unexpected = check.load_state_dict(back_to_hf(state), strict=False)
assert not unexpected and all("e_score" not in m for m in missing) and not missing, (missing, unexpected)
with torch.no_grad():
    assert torch.equal(check(input_ids=prompt_ids).logits, model(input_ids=prompt_ids).logits)

save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
published = dict(architectures=["InklingForConditionalGeneration"], model_type="inkling_mm_model", text_config=text, torch_dtype="float32")
json.dump(published, open(os.path.join(out, "config.json"), "w"), indent=2)

ids = prompt_ids
generated, margins, prompt_logits = [], [], None
with torch.no_grad():
    for _ in range(NEW_TOKENS):
        logits = model(input_ids=ids).logits[0, -1]
        assert logits.shape[0] == text["unpadded_vocab_size"]
        if prompt_logits is None:
            prompt_logits = logits.tolist()
        top = torch.topk(logits, 2)
        margins.append(float(top.values[0] - top.values[1]))
        generated.append(int(top.indices[0]))
        ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
print("reference ids:", ",".join(map(str, generated)))
print("min margin: %.5f" % min(margins))
json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids[0].tolist(), "logits": prompt_logits}, open(os.path.join(out, "reference_logits.json"), "w"))
print("wrote", out)
