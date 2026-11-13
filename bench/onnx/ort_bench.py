"""onnxruntime latency (min of N after warm-up) on the reference manifests: CPU EP and CUDA EP (inputs pre-bound on the GPU, outputs copied to host).
usage: ort_bench.py <out.jsonl> [manifest.json ...]   (default: ref/*.json, ref/zoo/*.json, ref/survey/*.json under ~/models/onnx)
Env: source bench/onnx/env.sh first (CUDA 13 + cuDNN 9 libs)."""
import json, os, sys, glob, time
import numpy as np
import onnx, onnxruntime as ort

ROOT = os.path.expanduser("~/models/onnx")
DT = {1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}

def load(man):
    r = json.load(open(man)); name = os.path.basename(man)[:-5]
    model = os.path.join(ROOT, r["model"]) if "model" in r else os.path.join(ROOT, name + ".onnx")
    b = r["batches"][0]
    m = onnx.load(model, load_external_data=False)
    et = {i.name: i.type.tensor_type.elem_type for i in m.graph.input}
    feeds = {}
    for k, i in enumerate(r["inputs"]):
        raw = np.fromfile(f"{man[:-5]}.b{b}.in{k}.bin", dtype=np.float32)
        shp = [b if d == -1 else d for d in i["shape"]]
        feeds[i["name"]] = raw.reshape(shp).astype(DT.get(et[i["name"]], np.float32))
    return name, model, feeds

def timeit(fn, budget=4.0, maxn=30):
    fn(); ts = []; t0 = time.perf_counter()
    while len(ts) < maxn and (len(ts) < 3 or time.perf_counter() - t0 < budget):
        a = time.perf_counter(); fn(); ts.append((time.perf_counter() - a) * 1000)
    return min(ts), len(ts)

def bench(man):
    out = {"name": os.path.basename(man)[:-5], "manifest": man}
    try: name, model, feeds = load(man)
    except Exception as e: out["error"] = f"load: {e}"; return out
    for ep in ("CPUExecutionProvider", "CUDAExecutionProvider"):
        key = "cpu" if ep.startswith("CPU") else "cuda"
        try:
            so = ort.SessionOptions(); so.log_severity_level = 4
            sess = ort.InferenceSession(model, so, providers=[ep] if key == "cpu" else [ep, "CPUExecutionProvider"])
            if key == "cuda" and sess.get_providers()[0] != ep: out[key] = {"error": "CUDA EP not active"}; continue
            if key == "cpu":
                ms, n = timeit(lambda: sess.run(None, feeds), maxn=20)
            else:
                io = sess.io_binding()
                keep = []
                for k, v in feeds.items():
                    ov = ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0); keep.append(ov)
                    io.bind_ortvalue_input(k, ov)
                for o in sess.get_outputs(): io.bind_output(o.name, "cpu")
                def f(): sess.run_with_iobinding(io); io.copy_outputs_to_cpu()
                try: ms, n = timeit(f)
                except Exception as e1:   # models whose declared output dims are wrong make io_binding validation fail: plain run (includes H2D of the inputs)
                    ms, n = timeit(lambda: sess.run(None, feeds)); out["cudaNote"] = "plain run (incl. H2D copy)"
            out[key] = {"minMs": round(ms, 3), "runs": n}
        except Exception as e:
            out[key] = {"error": str(e).replace("\n", " ")[:300]}
    return out

if __name__ == "__main__":
    outp = sys.argv[1]
    mans = sys.argv[2:] or sorted(glob.glob(f"{ROOT}/ref/*.json") + glob.glob(f"{ROOT}/ref/zoo/*.json") + glob.glob(f"{ROOT}/ref/survey/*.json"))
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)["manifest"])
            except Exception: pass
    with open(outp, "a") as f:
        for m in mans:
            if m in done: continue
            r = bench(m); f.write(json.dumps(r) + "\n"); f.flush(); print(r["name"], r.get("cpu"), r.get("cuda"), flush=True)
