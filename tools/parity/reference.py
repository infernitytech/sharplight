"""transformers reference for the parity harness (bench/ParitySweep): same JSON layout.

Usage: python -I reference.py <checkpoint dir> <spec.json> <out.json> [--gen N] [--device cpu|cuda] [--dtype float32|bfloat16] [--no-gen]
       [--gguf-file F]   (the model weights from this GGUF, dequantized by transformers)
       [--chat-template-file F]   (render the conversations with this template instead, e.g. the one embedded in a GGUF)
       python -I reference.py --many <spec.json> <dir>=<out.json>... [options]   (one process for several checkpoints)
Remote code is never run. Greedy steps record the top-5 raw logits (no logits processors) so the comparison can tell near-ties from real divergences.
"""
import copy
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    if sys.argv[1] == "--many":
        spec_path = sys.argv[2]
        pairs = [a for a in sys.argv[3:] if "=" in a and not a.startswith("--")]
        opts = [a for a in sys.argv[3:] if a not in pairs]
        for pair in pairs:
            model_dir, out_path = pair.split("=", 1)
            try:
                run(model_dir, spec_path, out_path, opts)
                print("ref ok", model_dir, flush=True)
            except Exception as error:  # noqa: BLE001
                print("ref FAIL", model_dir, type(error).__name__, str(error)[:300], flush=True)
        return
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])


def run(model_dir, spec_path, out_path, opts):
    gen = int(opts[opts.index("--gen") + 1]) if "--gen" in opts else 48
    device = opts[opts.index("--device") + 1] if "--device" in opts else "cpu"
    dtype = getattr(torch, opts[opts.index("--dtype") + 1]) if "--dtype" in opts else torch.float32
    spec = json.load(open(spec_path, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    out = {"model": model_dir.rstrip("/").split("/")[-1], "device": "transformers"}

    texts = []
    for text in spec.get("texts", []):
        try:
            ids = tok.encode(text, add_special_tokens=False)
            texts.append({"ids": ids, "decoded": tok.decode(ids)})
        except Exception as error:  # noqa: BLE001
            texts.append({"error": f"{type(error).__name__}: {error}"})
    out["texts"] = texts

    out["has_template"] = tok.chat_template is not None or "--chat-template-file" in opts
    chats = {}
    chat_ids = {}
    for chat in spec.get("chats", []):
        entry = {}
        try:
            kwargs = {}
            if "--chat-template-file" in opts:
                kwargs["chat_template"] = open(opts[opts.index("--chat-template-file") + 1], encoding="utf-8").read()
            if chat.get("tools"):
                kwargs["tools"] = chat["tools"]
            kwargs.update(chat.get("context", {}))
            messages = copy.deepcopy(chat["messages"])
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=chat.get("add_generation_prompt", True), **kwargs)
            ids = tok.encode(text, add_special_tokens=False)
            entry = {"text": text, "ids": ids}
            chat_ids[chat["name"]] = ids
        except Exception as error:  # noqa: BLE001
            entry = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        chats[chat["name"]] = entry
    out["chats"] = chats

    if "--no-gen" not in opts:
        torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
        extra = {"gguf_file": opts[opts.index("--gguf-file") + 1]} if "--gguf-file" in opts else {}
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, trust_remote_code=False, **extra).to(device).eval()
        greedy = []
        prompts = [(p, tok.encode(p, add_special_tokens=True)) for p in spec.get("prompts", [])]
        prompts += [("chat:" + n, chat_ids[n]) for n in spec.get("chat_prompts", []) if n in chat_ids]
        for name, ids in prompts:
            entry = {"prompt": name, "prompt_ids": ids}
            try:
                with torch.no_grad():
                    # A fresh generation config: the checkpoint's own (repetition_penalty, top_k, ...) would alter the logits.
                    config = GenerationConfig(max_new_tokens=gen, min_new_tokens=gen, do_sample=False, pad_token_id=tok.pad_token_id or 0)
                    res = model.generate(torch.tensor([ids], device=device), generation_config=config, output_logits=True, return_dict_in_generate=True)
                new = res.sequences[0, len(ids):].tolist()
                top = []
                for score in res.logits:
                    v, i = torch.topk(score[0].float(), 5)
                    top.append([[int(a), round(float(b), 4)] for a, b in zip(i.tolist(), v.tolist())])
                entry.update({"ids": new, "text": tok.decode(new), "top5": top})
            except Exception as error:  # noqa: BLE001
                entry["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            greedy.append(entry)
        out["greedy"] = greedy

    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
