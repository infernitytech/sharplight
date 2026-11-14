"""
Builds tiny random Hugging Face checkpoints of bidirectional (embedding) families and their reference embeddings:
EuroBERT (eurobert, saved as the masked-LM model), an EmbeddingGemma-style bidirectional Gemma 3 (gemma3_text with
use_bidirectional_attention, symmetric sliding windows, sentence-transformers Pooling + two Dense modules) and ModernBERT (modernbert,
saved as the base model: GeGLU, layer norms without bias, local layers with a symmetric window and their own rope base) and
jina-embeddings-v3 (jina_embeddings_v3, the base model: XLM-RoBERTa post-norm layers with rope) and EmbeddingGemma 2
(embedding_gemma2_text: per-layer head sizes, projection-only per-layer embeddings, the embedding projection inside the model).

    python mk_tiny_encoders.py <models/hf directory> [family ...]
    python mk_tiny_encoders.py --reference <model directory> [...]   (references for an existing checkpoint)

Every model is written to <dir>/tiny-random-<family>/ in fp32 safetensors with the GPT-2-style tokenizer of tiny-random-GPTNeoXForCausalLM,
plus reference_embeddings.json: a few token-id sequences (longer than the sliding windows) and their mean-pooled final hidden states, passed
through the Dense modules when the model has them, not normalized. Sharplight.Families.Tests (EmbeddingFamilyTests) compares against it.
"""
import json, os, shutil, sys
import torch
import transformers
from transformers import AutoConfig

root = os.path.expanduser(sys.argv[1])
only = sys.argv[2:]
tokenizer_dir = os.path.join(root, "tiny-random-GPTNeoXForCausalLM")
VOCAB = 1024
COMMON = dict(vocab_size=VOCAB, bos_token_id=0, eos_token_id=0, pad_token_id=0)

# name -> (model_type, config arguments, model class, Dense module sizes or None)
FAMILIES = {
    "eurobert": ("eurobert", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                  max_position_embeddings=256), "EuroBertForMaskedLM", None),
    # Window 8 becomes the exclusive bound 5 (keys at most 4 positions away); layers 0-4 slide, layer 5 is global.
    "embeddinggemma": ("gemma3_text", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=1,
                                            head_dim=16, max_position_embeddings=256, sliding_window=8, query_pre_attn_scalar=16,
                                            use_bidirectional_attention=True, rope_local_base_freq=10000.0, rope_theta=1000000.0),
                       "Gemma3TextModel", (96, 64)),
    # Local window 8 (keys at most 4 positions away), every third layer global from the first.
    "modernbert": ("modernbert", dict(hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, max_position_embeddings=256,
                                      local_attention=8, global_attn_every_n_layers=3, global_rope_theta=160000.0, local_rope_theta=10000.0,
                                      hidden_activation="gelu", norm_bias=False), "ModernBertModel", None),
    # jina-embeddings-v3 (native transformers form): XLM-RoBERTa post-norm layers with rope instead of absolute positions.
    "jina-embeddings-v3": ("jina_embeddings_v3", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                                                      max_position_embeddings=256, type_vocab_size=1), "JinaEmbeddingsV3Model", None),
    # EmbeddingGemma 2: radius-4 bidirectional windows on layers 0-4, a wider full-attention layer 5 (head_dim 32, one key/value head),
    # projection-only per-layer embeddings, the embedding projection to 48 dimensions inside the model.
    "embeddinggemma2": ("embedding_gemma2_text", dict(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                                                      num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
                                                      sliding_window=4, hidden_size_per_layer_input=16, embedding_dim=48,
                                                      max_position_embeddings=256), "EmbeddingGemma2TextModel", None),
}

# Token sequences: one shorter and two longer than the windows.
INPUTS = [[5, 77, 301, 12, 999, 4], [17 + 31 * i % 1000 for i in range(23)], [3 + 57 * i % 1000 for i in range(40)]]


def randomize(model, scale=0.15):
    with torch.no_grad():
        for n, b in model.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))   # a buffer, not a parameter: give it a value that matters as well
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.3 * torch.randn_like(p) if n.endswith("weight") else 0.3 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * scale)


def write_dense_modules(out, hidden, sizes):
    """Pooling (mean) + Dense modules (identity activation, no bias) + Normalize, as sentence-transformers stores them."""
    modules = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
               {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
    os.makedirs(os.path.join(out, "1_Pooling"), exist_ok=True)
    with open(os.path.join(out, "1_Pooling", "config.json"), "w") as f:
        json.dump({"word_embedding_dimension": hidden, "pooling_mode_mean_tokens": True}, f)
    layers = []
    dims = [hidden, *sizes]
    from safetensors.torch import save_file
    for i in range(len(sizes)):
        path = f"{i + 2}_Dense"
        os.makedirs(os.path.join(out, path), exist_ok=True)
        w = torch.randn(dims[i + 1], dims[i]) * 0.2
        save_file({"linear.weight": w.contiguous()}, os.path.join(out, path, "model.safetensors"))
        with open(os.path.join(out, path, "config.json"), "w") as f:
            json.dump({"in_features": dims[i], "out_features": dims[i + 1], "bias": False,
                       "activation_function": "torch.nn.modules.linear.Identity"}, f)
        modules.append({"idx": i + 2, "name": str(i + 2), "path": path, "type": "sentence_transformers.models.Dense"})
        layers.append(w)
    modules.append({"idx": len(modules), "name": str(len(modules)), "path": f"{len(modules)}_Normalize", "type": "sentence_transformers.models.Normalize"})
    with open(os.path.join(out, "modules.json"), "w") as f:
        json.dump(modules, f, indent=2)
    return layers


def reference_only(model_dir):
    """--reference <model dir>: reference_embeddings.json for an existing checkpoint (its base model, eager attention, mean pooling)."""
    model = transformers.AutoModel.from_pretrained(model_dir, attn_implementation="eager", dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        embeddings = [model(input_ids=torch.tensor([ids])).last_hidden_state[0].mean(dim=0).tolist() for ids in INPUTS]
    with open(os.path.join(model_dir, "reference_embeddings.json"), "w") as f:
        json.dump({"pooling": "mean", "inputs": INPUTS, "embeddings": embeddings}, f)
    print("wrote", os.path.join(model_dir, "reference_embeddings.json"))


if sys.argv[1] == "--reference":
    for model_dir in sys.argv[2:]:
        reference_only(os.path.expanduser(model_dir))
    sys.exit(0)

for name, (model_type, arguments, cls, dense) in FAMILIES.items():
    if only and name not in only:
        continue
    torch.manual_seed(0)
    config = AutoConfig.for_model(model_type, **{**COMMON, **arguments})
    config._attn_implementation = "eager"
    model = getattr(transformers, cls)._from_config(config)
    randomize(model)
    model.eval()
    out = os.path.join(root, f"tiny-random-{name}")
    model.save_pretrained(out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(tokenizer_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))
    projections = write_dense_modules(out, config.hidden_size, dense) if dense else []
    base = model.model if hasattr(model, "model") else model
    embeddings = []
    with torch.no_grad():
        for ids in INPUTS:
            hidden = base(input_ids=torch.tensor([ids])).last_hidden_state[0]
            e = hidden.mean(dim=0)
            for w in projections:
                e = w @ e
            embeddings.append(e.tolist())
    with open(os.path.join(out, "reference_embeddings.json"), "w") as f:
        json.dump({"pooling": "mean", "inputs": INPUTS, "embeddings": embeddings}, f)
    print("wrote", out)
