"""llama.cpp greedy reference for the parity harness: runs llama-completion for each raw prompt of the spec.

Usage: python -I llama_reference.py <model.gguf> <spec.json> <out.json> [--gen N] [--build cuda13|vulkan|cpu] [--fit]
--fit leaves the layer placement to llama.cpp (no -ngl), for models larger than the visible GPUs.
LLAMA_ROOT must point at the llama.cpp binaries (runtime.llama-b11177-<build>/). Text only (no ids).
"""
import glob
import json
import os
import subprocess
import sys


def main():
    gguf, spec_path, out_path = sys.argv[1:4]
    opts = sys.argv[4:]
    gen = int(opts[opts.index("--gen") + 1]) if "--gen" in opts else 48
    build = opts[opts.index("--build") + 1] if "--build" in opts else "cuda13"
    root = sorted(glob.glob(os.path.join(os.environ["LLAMA_ROOT"], f"runtime.llama-*-{build}")))[-1]
    env = dict(os.environ, LD_LIBRARY_PATH=root)
    spec = json.load(open(spec_path, encoding="utf-8"))
    greedy = []
    for prompt in spec.get("prompts", []):
        cmd = [os.path.join(root, "llama-completion"), "-m", gguf, "-p", prompt, "-n", str(gen), "--temp", "0", "--top-k", "1", "--seed", "1",
               "-c", "1024", "-no-cnv", "--no-display-prompt", "-t", "8", "--no-warmup", "--ignore-eos"]
        if "--fit" not in opts:
            cmd += ["-ngl", "99" if build != "cpu" else "0"]
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=env, timeout=900, check=False)
        text = res.stdout.decode("utf-8", errors="replace")
        entry = {"prompt": prompt, "text": text}
        if res.returncode != 0:
            entry["error"] = res.stderr.decode("utf-8", errors="replace")[-400:]
        greedy.append(entry)
    json.dump({"model": os.path.basename(gguf), "device": "llama.cpp-" + build, "greedy": greedy}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
