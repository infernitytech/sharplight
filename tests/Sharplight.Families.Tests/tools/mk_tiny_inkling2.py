"""
Builds a tiny random Inkling 2 Mini checkpoint (Thinking Machines, transformers' inkling2_loop) for Sharplight.Families.Tests in the layout of the
published thinkingmachines/Inkling-2-Mini checkpoint (model_type inkling2_loop, the text model's config in text_config with the release's keys,
tensors below model.llm. as transformers' save_pretrained writes them back: fused interleaved expert gate/up projections, the padded unembedding,
the loop's message queries as two [2 * loop layers, hidden] tensors; float32, experts unquantized, no vision or MTP weights), with the GPT-2 style
tokenizer of tiny-random-GPTNeoXForCausalLM (vocabulary 1024), and writes its transformers reference (greedy ids, prompt logits in
reference_logits.json):

  tiny-random-inkling2   8 layers in attention-residual blocks of 2, sliding-window layers (window 8, one kv head) and full layers (two kv
                         heads, relative-bias extent 16) alternating; per-token attention sinks, gated outputs, q / k / relative-state L2
                         clips (thresholds 2 and 1, so some heads are clipped), the log-floor scaling (floor 8, passed by the prompt), static
                         short convolutions on the sliding layers' keys and values and per-token (dynamic) ones on the full layers', residual
                         convolutions of the attention and MLP outputs, MoE with 8 experts (top 2, sigmoid router with a selection bias,
                         route scale 2.5) and softcapped SwiGLU (caps 1.5 and 2), the loop over layers 2-5 run 3 times with message gates,
                         final logit softcap 10 and the unpadded vocabulary of 1000 tokens.

The reference is transformers 5.24's Inkling2LoopForCausalLM (eager attention, float32) loaded from the saved checkpoint through transformers'
conversion mapping (it agrees with the model it was saved from up to float rounding).

    python mk_tiny_inkling2.py <models/hf directory> [seed]
"""
import json, os, shutil, sys
import torch
from transformers import AutoTokenizer
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

root = os.path.expanduser(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # the seed with the largest greedy margin among 1-8
out = os.path.join(root, "tiny-random-inkling2")
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
PROMPT = "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art"
NEW_TOKENS = 24
LAYERS = 8
SLIDING = [i % 2 == 0 for i in range(LAYERS)]

# The release's keys (intermediate_size is the experts' width, loop_config the loop).
text = dict(
    model_type="inkling2_loop_text", vocab_size=1024, unpadded_vocab_size=1000, hidden_size=64, num_hidden_layers=LAYERS, num_attention_heads=4,
    num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=1, swa_head_dim=16, sliding_window_size=8, d_rel=4,
    rel_extent=16, log_scaling_n_floor=8, log_scaling_alpha=0.3, rms_norm_eps=1e-6, use_value_norm_no_gamma=True, qk_l2_clip_threshold=2.0,
    rel_l2_clip_threshold=1.0, local_layer_ids=[i for i in range(LAYERS) if SLIDING[i]],
    layer_types=["hybrid_sliding" if s else "hybrid" for s in SLIDING], sconv_kernel_size=4, final_logit_softcapping=10.0,
    n_layers_per_attn_res_block=2, attn_res_normalize_values=True, n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=0,
    intermediate_size=32, mlp_gate_softcap=1.5, mlp_up_softcap=2.0, route_scale=2.5, use_gate_bias=True, model_max_length=4096,
    bos_token_id=0, eos_token_id=0,
    loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=8, is_varloop=False),
)

config = Inkling2LoopTextConfig(**{k: v for k, v in text.items() if k != "model_type"})
config._attn_implementation = "eager"
assert (config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes) == (2, 6, 3)
torch.manual_seed(SEED)
model = Inkling2LoopForCausalLM(config).eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if n.endswith("norm.weight") or n.endswith(".scale"):
            p.copy_(1.0 + 0.3 * torch.randn_like(p))
        elif "conv1d" in n or "static_weight" in n:
            p.copy_(0.3 * torch.randn_like(p))
        elif n.endswith(".query"):
            p.copy_(2.0 * torch.randn_like(p))
        elif "relative_position_bias.proj" in n:
            p.copy_(0.5 * torch.randn_like(p))
        elif n.endswith("mlp.gate.weight"):
            p.copy_(0.5 * torch.randn_like(p))
        elif "embed_tokens.weight" in n or "lm_head" in n:
            p.copy_(0.5 * torch.randn_like(p))
        elif "experts" in n:
            p.copy_(0.3 * torch.randn_like(p))
        else:
            p.copy_(0.2 * torch.randn_like(p))
    for layer in model.model.layers:
        layer.mlp.gate.e_score_correction_bias.copy_(0.1 * torch.randn_like(layer.mlp.gate.e_score_correction_bias))
    # Damped so that the 20 layer passes do not amplify rounding chaotically (batched decoding rounds differently from single sequences).
    for n, p in model.named_parameters():
        if not (n.endswith("norm.weight") or n.endswith(".scale")):
            p.mul_(0.6)

os.makedirs(out, exist_ok=True)
model.save_pretrained(out)
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(tokenizer_dir, f), out)
# The vision tower is not part of the tiny model; its config only has to agree with the text width.
published = dict(architectures=["Inkling2LoopForConditionalGeneration"], model_type="inkling2_loop", eos_token_id=0, text_config=text,
                 vision_config=dict(model_type="inkling2_vision", text_hidden_size=64), torch_dtype="float32")
json.dump(published, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ["generation_config.json"]:
    if os.path.exists(os.path.join(out, f)):
        os.remove(os.path.join(out, f))

tokenizer = AutoTokenizer.from_pretrained(out)
prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
check = Inkling2LoopForCausalLM.from_pretrained(out, config=config, dtype=torch.float32).eval()
with torch.no_grad():
    # Equal up to float rounding (the reloaded weights are laid out differently); the reference is the reloaded model.
    assert torch.allclose(check(input_ids=prompt_ids).logits, model(input_ids=prompt_ids).logits, atol=1e-2)
model = check

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
    # The same continuation decoded with the cache (one token at a time after the prompt) must agree.
    cached = model.generate(prompt_ids, max_new_tokens=NEW_TOKENS, do_sample=False)[0, prompt_ids.shape[1]:].tolist()
print("reference ids:", ",".join(map(str, generated)))
print("cached ids:   ", ",".join(map(str, cached)))
print("min margin: %.5f" % min(margins))
json.dump({"prompt": PROMPT, "prompt_ids": prompt_ids[0].tolist(), "logits": prompt_logits}, open(os.path.join(out, "reference_logits.json"), "w"))
print("wrote", out)
