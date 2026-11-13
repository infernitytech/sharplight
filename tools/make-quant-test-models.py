#!/usr/bin/env python3
"""Builds the quantization end-to-end test models and their llama.cpp reference distributions.

For every weight format Sharplight decodes this requantizes a small model (Qwen3-0.6B Q8_0) with llama-quantize to that format only
(--pure, so all linear weights use it), then asks llama-server for the next-token distribution after a fixed prompt. The result,
tests/Sharplight.Families.Tests/Fixtures/quant_next_token.json, is what QuantizedModelTests compares Sharplight (CPU and Vulkan) with.

usage: make-quant-test-models.py <llama.cpp runtime dir (CPU build)> <source model .gguf> <output dir> <fixture .json> [calibration text file]

The IQ1/IQ2 formats need an importance matrix; it is computed from the calibration text (default: the repository's docs).
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

PROMPT = "The capital of France is"
TOP_PROBS = 20

# Output name -> llama-quantize arguments. IQ2_S is produced by the IQ2_M type (the IQ2_S type itself maps to IQ2_XS tensors);
# the FP4 formats have no llama-quantize type, so their tensors are selected by name.
FP4_TENSORS = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
CASES = {
    "Q2_K": ["Q2_K"],
    "Q3_K": ["Q3_K_S"],
    "IQ3_XXS": ["IQ3_XXS"],
    "IQ3_S": ["IQ3_S"],
    "IQ2_XXS": ["--token-embedding-type", "q5_k", "IQ2_XXS"],
    "IQ2_XS": ["--token-embedding-type", "q5_k", "IQ2_XS"],
    "IQ2_S": ["--token-embedding-type", "q5_k", "IQ2_M"],
    "IQ1_S": ["--token-embedding-type", "q5_k", "IQ1_S"],
    "IQ1_M": ["IQ1_M"],
    "TQ1_0": ["TQ1_0"],
    "TQ2_0": ["TQ2_0"],
    "MXFP4": [arg for t in FP4_TENSORS for arg in ("--tensor-type", f"{t}=mxfp4")] + ["Q8_0"],
    "NVFP4": [arg for t in FP4_TENSORS for arg in ("--tensor-type", f"{t}=nvfp4")] + ["Q8_0"],
}


def run(runtime: str, tool: str, args: list[str]) -> None:
    env = dict(os.environ, LD_LIBRARY_PATH=runtime)
    subprocess.run([os.path.join(runtime, tool)] + args, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reference_distribution(runtime: str, model: str) -> dict:
    port = 18091
    env = dict(os.environ, LD_LIBRARY_PATH=runtime)
    server = subprocess.Popen([os.path.join(runtime, "llama-server"), "-m", model, "--port", str(port), "-ngl", "0", "-c", "512"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        body = json.dumps({"prompt": PROMPT, "n_predict": 1, "n_probs": TOP_PROBS, "temperature": 0, "cache_prompt": False}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})
        result = json.load(urllib.request.urlopen(request, timeout=300))
    finally:
        server.kill()
    top = result["completion_probabilities"][0]
    entries = top["top_logprobs"] if "top_logprobs" in top else top["top_probs"]
    ids = [e["id"] for e in entries]
    probs = [math.exp(e["logprob"]) if "logprob" in e else e["prob"] for e in entries]
    return {"prompt": PROMPT, "ids": ids, "probs": [round(p, 6) for p in probs]}


def main() -> None:
    runtime, source, out_dir, fixture = sys.argv[1:5]
    calibration = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(out_dir, exist_ok=True)
    imatrix = os.path.join(out_dir, "imatrix.gguf")
    if calibration and not os.path.exists(imatrix):
        run(runtime, "llama-imatrix", ["-m", source, "-f", calibration, "-o", imatrix, "--chunks", "20", "-c", "256"])

    references = {}
    for name, args in CASES.items():
        path = os.path.join(out_dir, f"qwen3-0.6b-{name}.gguf")
        if not os.path.exists(path):
            imatrix_args = ["--imatrix", imatrix] if os.path.exists(imatrix) else []
            options, quant_type = args[:-1], args[-1]
            run(runtime, "llama-quantize", ["--allow-requantize", "--pure"] + imatrix_args + options + [source, path, quant_type])
        references[os.path.basename(path)] = reference_distribution(runtime, path)
        print(name, references[os.path.basename(path)]["ids"][:3])
    with open(fixture, "w") as out:
        json.dump(references, out, indent=1)


if __name__ == "__main__":
    main()
