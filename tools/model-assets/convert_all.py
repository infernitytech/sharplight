"""Converts every tiny Hugging Face checkpoint that a test expects as gguf/tiny-random-<name>[-f32].gguf into f32 GGUF with llama.cpp's converter
(through tests/Sharplight.Families.Tests/tools/convert_tiny_gguf.py), skipping files that exist.

    python convert_all.py [name ...]        (environment: MODELS, default ~/models; LLAMA_SRC, a llama.cpp checkout; run in a venv with
                                             torch, transformers and llama.cpp's requirements)

Families the converter cannot take (llama.cpp-only ones are written by mk_tiny_gguf.py instead; Falcon needs TINY_PRE=falcon, the real
tokenizers TINY_PRE=auto, the MTP draft TINY_CONVERT_ARGS=--mtp) are in the table at the end of this file.
"""
import os, re, subprocess, sys

HOME = os.path.expanduser("~")
MODELS = os.environ.get("MODELS", os.path.join(HOME, "models"))
HF, GG = os.path.join(MODELS, "hf"), os.path.join(MODELS, "gguf")
LLAMA = os.environ["LLAMA_SRC"]
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "tests", "Sharplight.Families.Tests", "tools")
TESTS = os.path.join(HERE, "..", "..", "tests", "Sharplight.Families.Tests")
EXTRA = {"tiny-random-falcon": {"TINY_PRE": "falcon"},          # the tokenizer source of mk_tiny_gguf.py
         "tiny-random-glm4v": {"TINY_PRE": "auto"}, "tiny-random-hunyuan-vl": {"TINY_PRE": "auto"}, "tiny-random-deepseek-ocr": {"TINY_PRE": "auto"},
         "tiny-random-lfm3-hf": {"TINY_PRE": "auto"}, "tiny-random-clef": {"TINY_PRE": "auto"}, "tiny-random-vllm-decision2": {"TINY_PRE": "auto"}}   # real tokenizers: the converter looks the pre-tokenizer up

wanted = set()
for name in os.listdir(TESTS):
    if name.endswith(".cs"):
        wanted.update(re.findall(r'"gguf/(tiny-random-[A-Za-z0-9_.-]+?)(?:-f32)?\.gguf"', open(os.path.join(TESTS, name), encoding="utf-8").read()))
only = sys.argv[1:]
for base in sorted(wanted):
    if only and base not in only:
        continue
    out = os.path.join(GG, base + "-f32.gguf")
    if os.path.exists(out) or os.path.exists(os.path.join(GG, base + ".gguf")):
        continue
    candidates = [base + "-f32", base, base.replace("-llamacpp", ""), base.replace("-hf", "")]
    source = next((os.path.join(HF, c) for c in candidates if os.path.exists(os.path.join(HF, c, "config.json"))), None)
    if source is None:
        print("NOSRC", base, flush=True)
        continue
    env = dict(os.environ, PYTHONPATH=os.path.join(LLAMA, "gguf-py"), CUDA_VISIBLE_DEVICES="", **EXTRA.get(base, {}))
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "convert_tiny_gguf.py"), LLAMA, source, out], capture_output=True, text=True, env=env)
    print("OK" if r.returncode == 0 and os.path.exists(out) else "FAIL", base, "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:200], flush=True)
