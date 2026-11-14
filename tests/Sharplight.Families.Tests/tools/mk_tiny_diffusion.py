"""
Builds tiny random masked-diffusion checkpoints in the original remote-code layouts and their greedy references:

  tiny-random-dream  Dream (DreamModel, model_type "Dream"): Qwen2 layers with bidirectional attention, logits shifted by one position;
                     decoded on the timestep schedule over the whole span (Dream's diffusion_generate, alg "maskgit_plus", temperature 0).
  tiny-random-llada  LLaDA (LLaDAModelLM, model_type "llada"): Llama layers with bidirectional attention under LLaDA's
                     model.transformer.blocks.* names, unshifted logits; decoded block by block with low-confidence remasking
                     (LLaDA's generate.py, temperature 0).

The forward pass is transformers' Qwen2 / Llama model run with an all-visible attention mask; the decoding loop is written out here from the
two reference implementations. Each directory gets reference_diffusion.json with the prompt ids, the settings and the generated ids.

    python mk_tiny_diffusion.py <models/hf directory>
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config

root = os.path.expanduser(sys.argv[1])
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB, MASK = 1024, 1023
PROMPT = [53, 259, 276, 486, 275, 286, 282, 368, 540, 362, 365, 260]
SPAN = 24


def randomize(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.15)


def logits_of(model, ids):
    x = torch.tensor([ids])
    visible = torch.ones(1, 1, len(ids), len(ids), dtype=torch.bool)   # bidirectional: every position sees the whole sequence
    with torch.no_grad():
        return model(input_ids=x, attention_mask=visible).logits[0].double()


def confidence(row):
    p = torch.softmax(row, dim=-1)
    token = int(torch.argmax(row))
    return token, float(p[token])


def reveal(x, candidates, count):
    # Most confident first, ties to the earlier position.
    for conf, pos, token in sorted(candidates, key=lambda c: (-c[0], c[1]))[:count]:
        x[pos] = token


def dream(model, prompt, span, steps, eps):
    x = prompt + [MASK] * span
    for i in range(steps):
        masked = [j for j in range(len(x)) if x[j] == MASK]
        if not masked:
            break
        logits = logits_of(model, x)
        logits = torch.cat([logits[:1], logits[:-1]])                          # Dream: row i predicts position i + 1
        t = 1 - i / steps * (1 - eps)
        s = 1 - (i + 1) / steps * (1 - eps)
        count = int(len(masked) * (1 - s / t)) if i < steps - 1 else len(masked)
        reveal(x, [(c, j, tok) for j in masked for tok, c in [confidence(logits[j])]], count)
    return x[len(prompt):]


def llada(model, prompt, span, steps, block):
    x = prompt + [MASK] * span
    blocks = span // block
    per_block = steps // blocks
    for b in range(blocks):
        start, end = len(prompt) + b * block, len(prompt) + (b + 1) * block
        n = sum(1 for j in range(start, end) if x[j] == MASK)
        plan = [n // per_block + (k < n % per_block) for k in range(per_block)]
        for i in range(per_block):
            masked = [j for j in range(start, end) if x[j] == MASK]
            if not masked:
                break
            logits = logits_of(model, x)
            reveal(x, [(c, j, tok) for j in masked for tok, c in [confidence(logits[j])]], plan[i])
    return x[len(prompt):]


def copy_tokenizer(out):
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))


common = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=256, rms_norm_eps=1e-5, rope_theta=10000.0, bos_token_id=0, eos_token_id=0, pad_token_id=0,
              tie_word_embeddings=False)

# Dream: Qwen2 with q/k/v biases, saved under Qwen2's names with the remote-code class and model type.
qwen = AutoModelForCausalLM.from_config(Qwen2Config(**common)).eval()
randomize(qwen, 1)
out = os.path.join(root, "tiny-random-dream")
qwen.save_pretrained(out, safe_serialization=True)
config = json.load(open(os.path.join(out, "config.json")))
config.update(architectures=["DreamModel"], model_type="Dream", mask_token_id=MASK, rope_theta=10000.0)
json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=1)
copy_tokenizer(out)
settings = dict(steps=12, eps=1e-3)
ids = dream(qwen, PROMPT, SPAN, **settings)
json.dump(dict(prompt=PROMPT, span=SPAN, block_length=0, ids=ids, **settings), open(os.path.join(out, "reference_diffusion.json"), "w"))
print("dream", ids)

# LLaDA: Llama layers under LLaDA's names.
llama = AutoModelForCausalLM.from_config(LlamaConfig(**common)).eval()
randomize(llama, 2)
names = {"model.embed_tokens.weight": "model.transformer.wte.weight", "model.norm.weight": "model.transformer.ln_f.weight",
         "lm_head.weight": "model.transformer.ff_out.weight"}
layer = {"self_attn.q_proj": "q_proj", "self_attn.k_proj": "k_proj", "self_attn.v_proj": "v_proj", "self_attn.o_proj": "attn_out",
         "mlp.gate_proj": "ff_proj", "mlp.up_proj": "up_proj", "mlp.down_proj": "ff_out", "input_layernorm": "attn_norm",
         "post_attention_layernorm": "ff_norm"}
tensors = {}
for n, p in llama.state_dict().items():
    if n in names:
        tensors[names[n]] = p.contiguous()
        continue
    parts = n.split(".")          # model.layers.{i}.{module}.{proj}.weight
    module = ".".join(parts[3:-1])
    tensors[f"model.transformer.blocks.{parts[2]}.{layer[module]}.{parts[-1]}"] = p.contiguous()
out = os.path.join(root, "tiny-random-llada")
os.makedirs(out, exist_ok=True)
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
json.dump(dict(architectures=["LLaDAModelLM"], model_type="llada", d_model=64, n_heads=4, n_kv_heads=2, n_layers=3, mlp_hidden_size=128,
               vocab_size=VOCAB, embedding_size=VOCAB, max_sequence_length=256, rope_theta=10000.0, rms_norm_eps=1e-5, mask_token_id=MASK,
               weight_tying=False, include_bias=False, include_qkv_bias=False, activation_type="silu", block_type="llama",
               layer_norm_type="rms", bos_token_id=0, eos_token_id=0, pad_token_id=0),
          open(os.path.join(out, "config.json"), "w"), indent=1)
copy_tokenizer(out)
settings = dict(steps=12, block=8)
ids = llada(llama, PROMPT, SPAN, **settings)
json.dump(dict(prompt=PROMPT, span=SPAN, block_length=settings["block"], steps=settings["steps"], eps=0.0, ids=ids),
          open(os.path.join(out, "reference_diffusion.json"), "w"))
print("llada", ids)
