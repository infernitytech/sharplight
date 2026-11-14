"""
Reference embeddings of GGUF models from llama.cpp's llama-server, for Sharplight.Families.Tests (EmbeddingFamilyTests).

    python llamacpp_embeddings.py <llama.cpp binary directory> <pooling> <model.gguf> [model.gguf ...]

For every model a llama-server (CPU, flash attention off so fp32 throughout, --embedding, the given pooling: mean, cls or last) is started on a free local port and fed the
token-id sequences below through /embedding without normalization (embd_normalize = -1). The pooled vectors are written next to the model
as <model.gguf>.reference_embeddings.json ({pooling, inputs, embeddings}), in the format mk_tiny_encoders.py writes for transformers.
"""
import json, os, socket, subprocess, sys, time, urllib.request

# Token sequences: one shorter and two longer than the sliding windows of the tiny models (the same as mk_tiny_encoders.py).
INPUTS = [[5, 77, 301, 12, 999, 4], [17 + 31 * i % 1000 for i in range(23)], [3 + 57 * i % 1000 for i in range(40)]]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def wait_ready(port, server):
    for _ in range(300):
        if server.poll() is not None:
            raise RuntimeError("llama-server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("llama-server did not become ready")


def pooled(result):
    # /embedding returns [{"index": 0, "embedding": [[...]]}] (one pooled row) or a bare object in older builds.
    item = result[0] if isinstance(result, list) else result
    vector = item["embedding"]
    return vector[0] if vector and isinstance(vector[0], list) else vector


def reference(binaries, pooling, model):
    port = free_port()
    env = dict(os.environ, LD_LIBRARY_PATH=binaries + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    server = subprocess.Popen([os.path.join(binaries, "llama-server"), "-m", model, "--embedding", "--pooling", pooling, "--port", str(port),
                               "--host", "127.0.0.1", "-c", "256", "-b", "256", "-ub", "256", "-t", "4", "-ngl", "0", "-fa", "off", "--no-warmup"],
                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        try:
            wait_ready(port, server)
        except RuntimeError:
            server.kill()
            print(server.stderr.read()[-4000:], file=sys.stderr)
            raise
        embeddings = [pooled(post(port, "/embedding", {"content": ids, "embd_normalize": -1})) for ids in INPUTS]
    finally:
        server.terminate()
        server.wait(timeout=30)
    with open(model + ".reference_embeddings.json", "w") as f:
        json.dump({"pooling": pooling, "inputs": INPUTS, "embeddings": embeddings}, f)
    print("wrote", model + ".reference_embeddings.json")


binaries, pooling = os.path.expanduser(sys.argv[1]), sys.argv[2]
for path in sys.argv[3:]:
    reference(binaries, pooling, os.path.expanduser(path))
