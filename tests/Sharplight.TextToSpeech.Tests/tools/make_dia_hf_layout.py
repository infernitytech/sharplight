"""Writes the original Dia-1.6B checkpoint in the transformers layout (as Dia-1.6B-0626 ships it), to check that both layouts import to the same canonical weights.

Usage: python make_dia_hf_layout.py <Dia-1.6B dir> <hf config.json of Dia-1.6B-0626> <output dir>
"""
import sys, os, shutil, torch
from safetensors.torch import load_file, save_file

model_dir, hf_config, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
def convert(sd):
    o = {}
    def lin(w):  # [in, heads, dim] -> [heads*dim, in]
        return w.reshape(w.shape[0], -1).t().contiguous()
    def lino(w):  # [heads, dim, out] -> [out, heads*dim]
        return w.reshape(-1, w.shape[-1]).t().contiguous()
    o["model.encoder.embedding.weight"] = sd["encoder.embedding.weight"]
    o["model.encoder.norm.weight"] = sd["encoder.norm.weight"]
    for i in range(12):
        p, q = f"encoder.layers.{i}.", f"model.encoder.layers.{i}."
        o[q + "pre_sa_norm.weight"] = sd[p + "pre_sa_norm.weight"]
        o[q + "post_sa_norm.weight"] = sd[p + "post_sa_norm.weight"]
        for n in "qkv":
            o[q + f"self_attention.{n}_proj.weight"] = lin(sd[p + f"self_attention.{n}_proj.weight"])
        o[q + "self_attention.o_proj.weight"] = lino(sd[p + "self_attention.o_proj.weight"])
        o[q + "mlp.gate_up_proj.weight"] = sd[p + "mlp.wi_fused.weight"].reshape(sd[p + "mlp.wi_fused.weight"].shape[0], -1).t().contiguous()
        o[q + "mlp.down_proj.weight"] = sd[p + "mlp.wo.weight"].t().contiguous()
    o["model.decoder.embeddings.embed.weight"] = torch.cat([sd[f"decoder.embeddings.{c}.weight"] for c in range(9)], 0)
    o["model.decoder.norm.weight"] = sd["decoder.norm.weight"]
    for i in range(18):
        p, q = f"decoder.layers.{i}.", f"model.decoder.layers.{i}."
        for n in ("pre_sa_norm", "pre_ca_norm", "pre_mlp_norm"):
            o[q + n + ".weight"] = sd[p + n + ".weight"]
        for a in ("self_attention", "cross_attention"):
            for n in "qkv":
                o[q + f"{a}.{n}_proj.weight"] = lin(sd[p + f"{a}.{n}_proj.weight"])
            o[q + f"{a}.o_proj.weight"] = lino(sd[p + f"{a}.o_proj.weight"])
        o[q + "mlp.gate_up_proj.weight"] = sd[p + "mlp.wi_fused.weight"].reshape(sd[p + "mlp.wi_fused.weight"].shape[0], -1).t().contiguous()
        o[q + "mlp.down_proj.weight"] = sd[p + "mlp.wo.weight"].t().contiguous()
    o["logits_dense.weight"] = sd["decoder.logits_dense.weight"].reshape(sd["decoder.logits_dense.weight"].shape[0], -1).t().contiguous()
    return o


tensors = convert(load_file(os.path.join(model_dir, "model.safetensors")))
save_file({k: v.contiguous() for k, v in tensors.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(hf_config, os.path.join(out, "config.json"))
