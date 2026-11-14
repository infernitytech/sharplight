"""onnxruntime latency of one reference manifest on one execution provider; prints one "RESULT {json}" line.

usage: ort_latency.py <manifest.json> <cpu|cuda|cuda-strict> [--batch B] [--runs N] [--budget-s S] [--threads T]

  cpu          CPUExecutionProvider, graph optimizations ORT_ENABLE_ALL, intra-op threads = --threads (default 16)
  cuda         CUDAExecutionProvider with its defaults (TF32 tensor-core math allowed for f32 MatMul/Conv)
  cuda-strict  CUDAExecutionProvider with use_tf32=0 (full f32 math, the accuracy class of Sharplight's default)

CUDA timing binds the inputs on the GPU once and copies every output to the host per run (the same work the Sharplight
harness times: Run + read back every output). Model paths in the manifest resolve against $SHARPLIGHT_ONNX_MODELS.
When the manifest has no reference data for the batch, the batch-1 inputs are tiled. Source bench/onnx/env.sh first.
"""
import json
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort

ELEM_TYPES = {1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}
MIN_RUNS = 5


def read_input(prefix, k, batch):
    path = f"{prefix}.b{batch}.in{k}.bin"
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.float32)
    one = np.fromfile(f"{prefix}.b1.in{k}.bin", dtype=np.float32)
    return np.tile(one, batch)


def load_feeds(manifest_path, batch):
    root = os.environ["SHARPLIGHT_ONNX_MODELS"]
    man = json.load(open(manifest_path))
    name = os.path.basename(manifest_path)[:-5]
    model = os.path.join(root, man.get("model", name + ".onnx"))
    graph = onnx.load(model, load_external_data=False).graph
    elem = {i.name: i.type.tensor_type.elem_type for i in graph.input}
    constants = man.get("constants") or {}
    feeds = {}
    for k, inp in enumerate(man["inputs"]):
        shape = [batch if d == -1 else d for d in inp["shape"]]
        raw = read_input(manifest_path[:-5], k, batch)
        if inp["name"] in constants:
            raw = np.array(constants[inp["name"]], dtype=np.float32)
        feeds[inp["name"]] = raw.reshape(shape).astype(ELEM_TYPES.get(elem.get(inp["name"]), np.float32))
    return model, feeds


def time_runs(fn, runs, budget_s):
    fn()
    times = []
    start = time.perf_counter()
    while len(times) < runs and (len(times) < MIN_RUNS or time.perf_counter() - start < budget_s):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times


def session(model, provider, threads):
    so = ort.SessionOptions()
    so.log_severity_level = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "cpu":
        so.intra_op_num_threads = threads
        return ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    options = {"device_id": 0}
    if provider == "cuda-strict":
        options["use_tf32"] = 0
    sess = ort.InferenceSession(model, so, providers=[("CUDAExecutionProvider", options), "CPUExecutionProvider"])
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA EP not active")
    return sess


def bench(manifest_path, provider, batch, runs, budget_s, threads):
    model, feeds = load_feeds(manifest_path, batch)
    sess = session(model, provider, threads)
    if provider == "cpu":
        return time_runs(lambda: sess.run(None, feeds), runs, budget_s), None
    io = sess.io_binding()
    keep = []
    for name, value in feeds.items():
        ov = ort.OrtValue.ortvalue_from_numpy(value, "cuda", 0)
        keep.append(ov)
        io.bind_ortvalue_input(name, ov)
    for out in sess.get_outputs():
        io.bind_output(out.name, "cpu")

    def step():
        sess.run_with_iobinding(io)
        io.copy_outputs_to_cpu()

    try:
        return time_runs(step, runs, budget_s), None
    except Exception:  # declared output dims that disagree with the runtime shapes fail io_binding validation
        return time_runs(lambda: sess.run(None, feeds), runs, budget_s), "plain run (includes the H2D copy of the inputs)"


def main(argv):
    if len(argv) < 2 or argv[1] not in ("cpu", "cuda", "cuda-strict"):
        print(__doc__, file=sys.stderr)
        return 2
    manifest_path, provider = argv[0], argv[1]
    opts = dict(zip(argv[2::2], argv[3::2]))
    man = json.load(open(manifest_path))
    batch = int(opts.get("--batch", man.get("batches", [1])[0]))
    result = {"manifest": os.path.basename(manifest_path), "provider": "ort-" + provider, "batch": batch, "ort": ort.__version__}
    try:
        times, note = bench(manifest_path, provider, batch, int(opts.get("--runs", 30)), float(opts.get("--budget-s", 6)), int(opts.get("--threads", 16)))
        result.update(stage="ok", minMs=round(times[0], 3), medMs=round(times[len(times) // 2], 3), runs=len(times))
        if note:
            result["note"] = note
    except Exception as e:
        result.update(stage="error", error=str(e).replace("\n", " ")[:500])
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
