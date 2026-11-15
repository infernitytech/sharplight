"""f64 reference outputs for a reference manifest, to measure the error of f32 / TF32-class engines (benchmarks section 32).

An f64 copy of the f32 graph (initializers, value types, casts and inputs widened; BatchNormalization rewritten as f64 Mul /
Add) runs on onnx's ReferenceEvaluator (onnxruntime has no f64 Conv or Erf kernels). Writes <out_dir>/<name>.json, the
inputs as .b<B>.in<k>.bin and the outputs as .b<B>.out<k>.bin (f32) and .out<k>.f64.bin, so OnnxBench can score any engine
against it. The temporary f64 model is deleted afterwards.

usage: f64_reference.py <manifest.json> <out_dir> <batch>      (model paths relative to $SHARPLIGHT_ONNX_MODELS)"""
import json, os, sys, copy
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

src, out_dir, batch = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = os.environ["SHARPLIGHT_ONNX_MODELS"]
man = json.load(open(src))
name = os.path.basename(src)[:-5]
model_path = os.path.join(root, man.get("model", name + ".onnx"))
os.makedirs(out_dir, exist_ok=True)


def widen_tensor(t):
    if t.data_type == TensorProto.FLOAT:
        t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float64), t.name))


def widen(g):
    for t in g.initializer:
        widen_tensor(t)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.DOUBLE
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                widen_tensor(a.t)
            elif a.type == onnx.AttributeProto.GRAPH:
                widen(a.g)
            elif a.name == "to" and a.i == TensorProto.FLOAT:
                a.i = TensorProto.DOUBLE
            elif a.name in ("dtype",) and a.type == onnx.AttributeProto.INT and a.i == TensorProto.FLOAT:
                a.i = TensorProto.DOUBLE
            elif n.op_type == "GridSample" and a.name == "mode" and a.s in (b"bilinear", b"bicubic"):
                a.s = {b"bilinear": b"linear", b"bicubic": b"cubic"}[a.s]   # opset 20 names (the reference evaluator's)


def batchnorm_as_affine(g):
    """Inference BatchNormalization with constant parameters -> Mul and Add by f64 per-channel constants (the reference
    evaluator's opset-12 BatchNormalization is wrong)."""
    inits = {t.name: t for t in g.initializer}
    nodes = []
    for n in g.node:
        if n.op_type == "BatchNormalization" and len(n.output) == 1 and all(i in inits for i in n.input[1:]):
            eps = next((a.f for a in n.attribute if a.name == "epsilon"), 1e-5)
            sc, b, mean, var = (numpy_helper.to_array(inits[i]).astype(np.float64) for i in n.input[1:])
            a = sc / np.sqrt(var + np.float64(np.float32(eps)))
            c = b - mean * a
            shape = [-1, 1, 1]
            an, cn = n.output[0] + "_bn_a", n.output[0] + "_bn_c"
            g.initializer.append(numpy_helper.from_array(a.reshape(shape), an))
            g.initializer.append(numpy_helper.from_array(c.reshape(shape), cn))
            nodes.append(onnx.helper.make_node("Mul", [n.input[0], an], [n.output[0] + "_bn_m"]))
            nodes.append(onnx.helper.make_node("Add", [n.output[0] + "_bn_m", cn], [n.output[0]]))
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)


m = onnx.load(model_path)
widen(m.graph)
batchnorm_as_affine(m.graph)
f64_path = os.path.join(out_dir, name + ".f64.onnx")
onnx.save(m, f64_path, save_as_external_data=True, location=name + ".f64.onnx_data")
del m

g = onnx.load(model_path, load_external_data=False).graph
elem = {i.name: i.type.tensor_type.elem_type for i in g.input}
NP = {1: np.float64, 7: np.int64, 6: np.int32, 9: np.bool_, 10: np.float16}
prefix = src[:-5]
feeds = {}
for k, inp in enumerate(man["inputs"]):
    path = f"{prefix}.b{batch}.in{k}.bin"
    raw = np.fromfile(path, dtype=np.float32) if os.path.exists(path) else np.tile(np.fromfile(f"{prefix}.b1.in{k}.bin", dtype=np.float32), batch)
    shape = [batch if d == -1 else d for d in inp["shape"]]
    if inp["name"] in (man.get("constants") or {}):
        raw = np.array(man["constants"][inp["name"]], dtype=np.float32)
    feeds[inp["name"]] = raw.reshape(shape).astype(NP.get(elem[inp["name"]], np.float64))
names = [o["name"] for o in man["outputs"]]
import time
t0 = time.time(); from onnx.reference import ReferenceEvaluator
res = ReferenceEvaluator(f64_path).run(names, feeds); print("reference evaluator", round(time.time() - t0, 1), "s", flush=True)
os.remove(f64_path)
os.remove(os.path.join(out_dir, name + ".f64.onnx_data"))
newman = copy.deepcopy(man)
newman["model"] = model_path
newman["batches"] = [batch]
json.dump(newman, open(os.path.join(out_dir, name + ".json"), "w"))
for k, inp in enumerate(man["inputs"]):
    np.asarray(feeds[inp["name"]], np.float32).tofile(os.path.join(out_dir, f"{name}.b{batch}.in{k}.bin"))
for k, v in enumerate(res):
    np.asarray(v, np.float32).tofile(os.path.join(out_dir, f"{name}.b{batch}.out{k}.bin"))
    np.asarray(v, np.float64).tofile(os.path.join(out_dir, f"{name}.b{batch}.out{k}.f64.bin"))
print("ok", name, batch, [r.shape for r in res])
