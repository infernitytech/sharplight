"""
Builds tiny random Kimi K3 and K3.1 checkpoints for Sharplight.Families.Tests, in the layout of the published Kimi-K3 checkpoint (model_type
kimi_k3 with the text model's config in text_config and its tensors below language_model.; no vision tower), in float32 with the GPT-2 style
tokenizer of tiny-random-GPTNeoXForCausalLM (vocabulary 1024), and writes their transformers references (greedy ids, prompt logits in
reference_logits.json) with kimi_k3_reference.py:

  tiny-random-kimi-k3-f32    8 layers: Kimi delta attention (bounded gate, full-rank output gate, decay rate per key channel shared by the heads)
                         with latent attention without rotary embedding and with a sigmoid output gate at layers 4 and 8 (1-based); attention
                         residuals over blocks of 3 layers; a dense first layer and latent MoE layers (experts in a 32-wide latent space with
                         an RMS norm on their sum, sigmoid router with selection bias, shared expert), every MLP with the situ activation.
  tiny-random-kimi-k31-f32   the same with a DSA indexer (16 heads, index_topk 8) on both latent-attention layers, so that the prompt runs in
                         the sparse regime; the indexer weights come from the seed with the largest greedy margin among those whose
                         top-k boundary gap is not tiny.

    python mk_tiny_kimi_k3.py <models/hf directory> <Kimi-K3 remote code directory> [name ...]

The remote code directory holds configuration_kimi_k3.py and modeling_kimi_linear.py (e.g. a Kimi-K3 checkpoint without its weights).
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_k3_reference import load_text_model

root, code_dir = os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2])
only = sys.argv[3:]
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
NEW_TOKENS = 24

text = dict(
    vocab_size=1024, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=8, num_attention_heads=4,
    num_key_value_heads=4, head_dim=16, q_lora_rank=48, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    mla_use_nope=True, mla_use_output_gate=True, num_experts=8, num_experts_per_token=2, num_shared_experts=1, num_expert_group=1,
    topk_group=1, first_k_dense_replace=1, moe_renormalize=True, moe_router_activation_func="sigmoid", routed_scaling_factor=2.5,
    routed_expert_hidden_size=32, latent_moe_use_norm=True, hidden_act="situ", activation_situ_beta=0.8, activation_situ_linear_beta=1.5,
    attn_res_block_size=3, rms_norm_eps=1e-5, max_position_embeddings=256, rope_theta=10000.0, tie_word_embeddings=False,
    bos_token_id=0, eos_token_id=0, pad_token_id=1,
    linear_attn_config=dict(full_attn_layers=[4, 8], kda_layers=[1, 2, 3, 5, 6, 7], head_dim=16, num_heads=4, short_conv_kernel_size=4,
                            gate_lower_bound=-5.0, use_full_rank_gate=True))
dsa = dict(dsa_layers=[4, 8], index_num_heads=16, index_head_dim=16, index_topk=8)
variants = {
    "kimi-k3": ("KimiK3ForConditionalGeneration", text),
    "kimi-k31": ("KimiK31ForConditionalGeneration", dict(text, dsa_config=dsa)),
}
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
prompt_ids = tokenizer(PROMPT).input_ids


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(("norm.weight", "layernorm.weight")):
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            elif n.endswith("norm.bias"):
                p.copy_(0.2 * torch.randn_like(p))
            elif n.endswith("res_proj.weight"):
                # Attention-residual scores of standard deviation about 0.8: clearly not uniform, but soft enough that the f16 rounding of an
                # attention cache does not move the logits beyond the test tolerance (0.4 amplified it to 1 % of the largest logit).
                p.copy_(0.1 * torch.randn_like(p))
            elif n.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 4 + 1))
            elif n.endswith("dt_bias"):
                p.copy_(0.5 * torch.randn_like(p))
            elif n.endswith("conv1d.weight"):
                p.copy_(0.4 * torch.randn_like(p))
            elif n.endswith("e_score_correction_bias"):
                p.copy_(0.2 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.15)


def generate(model):
    """Greedy ids, per-step margins and the prompt logits (full recompute, no cache)."""
    ids = torch.tensor([prompt_ids])
    generated, margins, prompt_logits = [], [], None
    with torch.no_grad():
        for _ in range(NEW_TOKENS):
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]
            prompt_logits = logits.tolist() if prompt_logits is None else prompt_logits
            top = torch.topk(logits, 2)
            margins.append(float(top.values[0] - top.values[1]))
            generated.append(int(top.indices[0]))
            ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
    return generated, margins, prompt_logits


def boundary_gap(model, full_ids, k):
    """Smallest relative gap between the k-th and (k+1)-th index score of any row that sees more than k positions."""
    layers = [layer.self_attn for layer in model.model.layers if getattr(layer.self_attn, "indexer", None) is not None]
    for attention in layers:
        attention.record = []
    with torch.no_grad():
        model(input_ids=torch.tensor([full_ids]), use_cache=False)
    worst = float("inf")
    for attention in layers:
        for scores in attention.record:
            for t in range(k, scores.shape[1]):
                row = scores[0, t, : t + 1].sort(descending=True).values
                worst = min(worst, (row[k - 1] - row[k]).item() / max(row.abs().max().item(), 1e-30))
        attention.record = None
    return worst


for name, (architecture, config) in variants.items():
    if only and name not in only:
        continue
    model = load_text_model(code_dir, config)
    randomize(model, 0)
    if "dsa_config" in config:
        # Indexer weights: the seed with the largest greedy logit margin among those whose top-k boundary gap (on their own greedy sequence)
        # is at least 5e-4, so that neither the selection nor the generated ids hinge on rounding.
        indexer_names = [n for n, _ in model.named_parameters() if ".indexer.wq_b." in n or ".indexer.weights_proj." in n]

        def set_indexer(seed):
            gen = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in indexer_names:
                        p.copy_(torch.randn(p.shape, generator=gen) * 0.3)

        best = (-1.0, None)
        for seed in range(1, 31):
            set_indexer(seed)
            generated, margins, _ = generate(model)
            gap = boundary_gap(model, prompt_ids + generated, config["dsa_config"]["index_topk"])
            print(f"{name} seed {seed}: smallest relative boundary gap {gap:.3g}, smallest margin {min(margins):.4f}")
            if gap >= 5e-4:
                best = max(best, (min(margins), seed))
        set_indexer(best[1])
        print(f"{name}: indexer seed {best[1]} (margin {best[0]:.4f})")

    generated, margins, prompt_logits = generate(model)
    out = os.path.join(root, f"tiny-random-{name}-f32")
    if os.path.realpath(out) == os.path.realpath(code_dir):
        sys.exit("the output directory must not be the remote code directory")
    os.makedirs(out, exist_ok=True)
    state = {"language_model." + k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    text_config = dict(config, model_type="kimi_linear", architectures=["KimiLinearForCausalLM"])
    wrapper = dict(architectures=[architecture], model_type="kimi_k3", text_config=text_config, bos_token_id=0, eos_token_id=0, pad_token_id=1,
                   tie_word_embeddings=False)
    json.dump(wrapper, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        shutil.copy(os.path.join(tokenizer_dir, f), os.path.join(out, f))
    for f in ["configuration_kimi_k3.py", "modeling_kimi_linear.py"]:
        shutil.copy(os.path.join(code_dir, f), os.path.join(out, f))
    json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids, "logits": prompt_logits}, open(os.path.join(out, "reference_logits.json"), "w"))
    print(f"{name}: wrote {out}")
    print("  ids:", ",".join(str(i) for i in generated))
    print("  min margin: %.5f" % min(margins))
    print("  margins:", " ".join("%.4f" % m for m in margins))
