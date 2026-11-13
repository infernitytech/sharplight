"""Collects opset / producer / op histogram per reference manifest -> results/meta.json (python onnx, no external data loaded)."""
import json, os, glob, collections
import onnx
ROOT = os.path.expanduser("~/models/onnx")
out = {}
for man in sorted(glob.glob(f"{ROOT}/ref/*.json") + glob.glob(f"{ROOT}/ref/zoo/*.json") + glob.glob(f"{ROOT}/ref/survey/*.json")):
    r = json.load(open(man)); name = os.path.basename(man)[:-5]
    s = "survey" if "/survey/" in man else "zoo" if "/zoo/" in man else "ref"
    path = os.path.join(ROOT, r["model"]) if "model" in r else os.path.join(ROOT, name + ".onnx")
    try:
        m = onnx.load(path, load_external_data=False)
    except Exception as e:
        out[f"{s}/{name}"] = {"error": str(e)[:200]}; continue
    ops = collections.Counter((n.domain or "ai.onnx", n.op_type) for n in m.graph.node)
    # nested subgraph ops
    def walk(g):
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    for k in a.g.node: ops[(k.domain or "ai.onnx", k.op_type)] += 1
                    walk(a.g)
    walk(m.graph)
    ops_set = {f"{d}:{o}": c for (d, o), c in ops.items()}
    dtypes = sorted({onnx.TensorProto.DataType.Name(i.data_type) for i in m.graph.initializer})
    out[f"{s}/{name}"] = {"model": os.path.relpath(path, ROOT), "opset": {(i.domain or "ai.onnx"): i.version for i in m.opset_import}, "producer": m.producer_name,
                          "nodes": len(m.graph.node), "ops": ops_set, "weightTypes": dtypes, "sizeMB": round((os.path.getsize(path) + sum(os.path.getsize(path + e) for e in ("_data", ".data") if os.path.exists(path + e))) / 1e6)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/meta.json"), "w"), indent=0)
print(len(out), "models")
