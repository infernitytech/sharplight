"""Reference embeddings / reranker scores for Sharplight.Embeddings.Tests.

Run with a Python environment that has torch, transformers and sentence-transformers:
    python make_reference.py [MODELS_DIR] [OUT_DIR]
MODELS_DIR defaults to ~/models/hf, OUT_DIR to the directory of this script. Writes one JSON file per model with the inputs, the
token ids the Hugging Face tokenizer produces and the reference outputs (float32, CPU).
"""
import json
import os
import sys

import torch

MODELS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/hf")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
torch.set_grad_enabled(False)

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the capital of France?",
    "Paris is the capital and most populous city of France.",
    "Machine learning models turn text into vectors (embeddings) for semantic search!",
    "Ünïcödé façade café — naïve résumé; 東京は日本の首都です。",
    "a",
    "Numbers 12345 and symbols #@$%^&* should tokenize consistently, e.g. U.S.A. vs USA.",
    " ".join(["Embedding models map sentences with similar meaning to nearby points in a vector space,"
              " which makes retrieval, clustering and deduplication simple."] * 6),
]

QUERY = "What is the capital of France?"
DOCUMENTS = [
    "Paris is the capital and most populous city of France.",
    "The quick brown fox jumps over the lazy dog.",
    "France's capital city is known for the Eiffel Tower.",
    "Berlin is the capital of Germany.",
    "Die Hauptstadt von Frankreich ist Paris.",
    "",
]


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print("wrote", name)


def embedder(name, prompt_name=None, trust=False, extra_texts=None):
    from sentence_transformers import SentenceTransformer
    path = os.path.join(MODELS, name)
    if not os.path.isdir(path):
        print("skip", name)
        return
    # float32 explicitly: transformers 5 otherwise loads the checkpoint dtype (bf16 for Qwen3-Embedding).
    m = SentenceTransformer(path, device="cpu", trust_remote_code=trust, model_kwargs={"dtype": torch.float32})
    for mod in m.modules():
        # nomic-bert's remote code predates transformers 5, which dropped this helper.
        if not hasattr(mod, "get_extended_attention_mask") and type(mod).__name__.endswith("BertModel"):
            def extended(self, mask, shape, *a, **k):
                return (1.0 - mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
            type(mod).get_extended_attention_mask = extended
    texts = TEXTS + (extra_texts or [])
    tokens = [m.tokenizer(t, truncation=True, max_length=m.max_seq_length)["input_ids"] for t in texts]
    emb = m.encode(texts, convert_to_numpy=True, batch_size=4)
    out = {"model": name, "texts": texts, "tokens": tokens, "embeddings": emb.tolist()}
    if prompt_name:
        prompt = m.prompts[prompt_name]
        out["prompt_name"] = prompt_name
        out["prompt"] = prompt
        out["prompt_tokens"] = [m.tokenizer(prompt + t)["input_ids"] for t in texts[:3]]
        out["prompt_embeddings"] = m.encode(texts[:3], prompt_name=prompt_name, convert_to_numpy=True).tolist()
    save(name, out)


def cross_encoder(name):
    from sentence_transformers import CrossEncoder
    path = os.path.join(MODELS, name)
    if not os.path.isdir(path):
        print("skip", name)
        return
    m = CrossEncoder(path, device="cpu", model_kwargs={"dtype": torch.float32})
    pairs = [[QUERY, d] for d in DOCUMENTS]
    tok = m.tokenizer
    # Batch form: a single call treats an empty text_pair as "no pair", the batched model input does not.
    enc = tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=True)["input_ids"]
    inputs = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, return_tensors="pt")
    logits = m.model(**inputs).logits[:, 0].tolist()
    save(name, {"model": name, "query": QUERY, "documents": DOCUMENTS, "tokens": enc, "logits": logits})


def qwen3_reranker(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = os.path.join(MODELS, name)
    if not os.path.isdir(path):
        print("skip", name)
        return
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre, suf = tok.encode(prefix, add_special_tokens=False), tok.encode(suffix, add_special_tokens=False)
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    seqs = []
    for d in DOCUMENTS:
        body = tok(f"<Instruct>: {instruction}\n<Query>: {QUERY}\n<Document>: {d}", add_special_tokens=False)["input_ids"]
        seqs.append(pre + body + suf)
    logits = []
    for s in seqs:
        out = model(torch.tensor([s])).logits[0, -1]
        logits.append(float(out[yes] - out[no]))
    save(name, {"model": name, "query": QUERY, "documents": DOCUMENTS, "tokens": seqs, "logits": logits})


if __name__ == "__main__":
    which = set(sys.argv[3:])
    def want(n): return not which or n in which
    for n in ["bge-small-en-v1.5", "all-MiniLM-L6-v2", "e5-small-v2", "multilingual-e5-small"]:
        if want(n): embedder(n)
    if want("nomic-embed-text-v1.5"): embedder("nomic-embed-text-v1.5", trust=True)
    if want("Qwen3-Embedding-0.6B"): embedder("Qwen3-Embedding-0.6B", prompt_name="query")
    for n in ["ms-marco-MiniLM-L6-v2", "bge-reranker-base", "bge-reranker-v2-m3"]:
        if want(n): cross_encoder(n)
    if want("Qwen3-Reranker-0.6B"): qwen3_reranker("Qwen3-Reranker-0.6B")
