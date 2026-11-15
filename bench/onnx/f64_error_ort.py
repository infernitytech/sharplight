"""Relative RMS of onnxruntime's outputs (CPU f32, CUDA with TF32, CUDA strict) against the f64 reference of
f64_reference.py (benchmarks section 32). Sharplight's engines are scored by OnnxBench on the same manifest (worstRms).

usage: f64_error_ort.py <f64 manifest.json> <batch>   -> one JSON line per engine"""
import json, os, sys
import numpy as np
import onnx
import onnxruntime as ort

man_path, batch = sys.argv[1], int(sys.argv[2])
man = json.load(open(man_path))
prefix = man_path[:-5]
g = onnx.load(man["model"], load_external_data=False).graph
elem = {i.name: i.type.tensor_type.elem_type for i in g.input}
NP = {1: np.float32, 7: np.int64, 6: np.int32, 9: np.bool_, 10: np.float16}
feeds = {}
for k, inp in enumerate(man["inputs"]):
    raw = np.fromfile(f"{prefix}.b{batch}.in{k}.bin", dtype=np.float32)
    feeds[inp["name"]] = raw.reshape([batch if d == -1 else d for d in inp["shape"]]).astype(NP.get(elem[inp["name"]], np.float32))
names = [o["name"] for o in man["outputs"]]
refs = [np.fromfile(f"{prefix}.b{batch}.out{k}.f64.bin", dtype=np.float64) for k in range(len(names))]
engines = {
    "ort-cpu": (["CPUExecutionProvider"], None),
    "ort-cuda": (["CUDAExecutionProvider"], None),
    "ort-cuda-strict": ([("CUDAExecutionProvider", {"use_tf32": "0"})], None),
}
for label, (providers, _) in engines.items():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    out = ort.InferenceSession(man["model"], so, providers=providers).run(names, feeds)
    rms = []
    for o, r in zip(out, refs):
        a = np.asarray(o, np.float64).ravel()
        rms.append(float(np.sqrt(np.sum((a - r) ** 2) / max(np.sum(r ** 2), 1e-30))))
    print(json.dumps({"engine": label, "model": os.path.basename(man_path)[:-5], "batch": batch, "worstRms": max(rms), "firstRms": rms[0]}), flush=True)
