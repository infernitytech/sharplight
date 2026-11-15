"""Greedy ids of a GGUF from llama.cpp (llama-server, CPU, return_tokens), the reference of the GGUF cases in Sharplight.Families.Tests.

    python llamacpp_ids.py <runtime dir> <model.gguf> <prompt> <n> [extra llama-server args ...]
"""
import json, os, subprocess, sys, time, urllib.request
runtime, model, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
port = 18000 + os.getpid() % 1000
env = dict(os.environ, LD_LIBRARY_PATH=runtime)
p = subprocess.Popen([os.path.join(runtime, "llama-server"), "-m", model, "--port", str(port), "-c", "512", "-ngl", "0", "-t", "4", "--no-warmup", "--seed", "1", *sys.argv[5:]],
                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            if p.poll() is not None: sys.exit("server exited")
            time.sleep(1)
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "return_tokens": True, "cache_prompt": False, "ignore_eos": True}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"})).read())
    print(json.dumps(r["tokens"]))
finally:
    p.terminate()
