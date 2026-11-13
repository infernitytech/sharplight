"""onnxruntime reference data for the survey models, in the same format as ZooTests fixtures (tests/Sharplight.Onnx.Tests/scripts/make_zoo_refs.py).

usage: make_survey_refs.py [name ...]    (default: every model file under ~/models/onnx/<dir>/ listed by discover())
Writes ~/models/onnx/ref/survey/<name>.json + <name>.b<B>.in<k>.bin / .out<k>.bin (float32; integer/bool tensors stored as float32 values).
Shapes of dynamic inputs come from heuristics (batch = dim 0 -> -1; sequence 16 (4 with KV cache), past 8, images 224) plus OVERRIDES.
Source: ONNX Runtime CPU (falls back to the CUDA EP when the CPU EP lacks a kernel, e.g. fp16 MatMulNBits; recorded in "refProvider").
"""
import json, os, sys, glob, re, traceback
import numpy as np
import onnx, onnxruntime as ort

ROOT = os.path.expanduser("~/models/onnx")
OUT = os.path.join(ROOT, "ref", "survey")
sys.path.insert(0, os.path.expanduser("~/Documents/work/sharplight/tests/Sharplight.Onnx.Tests/scripts"))

def smooth_image(rs, shape, lo=0.0, hi=1.0):
    n, c, h, w = shape
    coarse = rs.rand(n, c, max(h // 16, 2), max(w // 16, 2)).astype(np.float32)
    ys = np.linspace(0, coarse.shape[2] - 1, h); xs = np.linspace(0, coarse.shape[3] - 1, w)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, coarse.shape[2] - 1); x1 = np.minimum(x0 + 1, coarse.shape[3] - 1)
    fy = (ys - y0)[None, None, :, None]; fx = (xs - x0)[None, None, None, :]
    a = coarse[:, :, y0][:, :, :, x0]; b = coarse[:, :, y0][:, :, :, x1]
    c_ = coarse[:, :, y1][:, :, :, x0]; d = coarse[:, :, y1][:, :, :, x1]
    img = (a * (1 - fx) + b * fx) * (1 - fy) + (c_ * (1 - fx) + d * fx) * fy
    img = 0.9 * img + 0.1 * rs.rand(n, c, h, w)
    return (lo + (hi - lo) * img).astype(np.float32)

# model name -> dict(file=<path relative to ROOT/name>, shapes={input: shape}, gen={input|'*': spec}, pin=[inputs], rms=, rtol=, sym={dim name: value})
# spec: "img" "imgnorm" "img255" "imgpm1" ("randn", s) ("ints", lo, hi) ("const", v) ("vals", [..])
OVERRIDES = {
  "ssd-10": dict(gen={"*": "img255norm"}), "ssd-12-int8": dict(gen={"*": "img255norm"}),
  "retinanet-9": dict(gen={"*": "imgnorm"}), "fasterrcnn-12": dict(gen={"*": "img255"}), "maskrcnn-12": dict(gen={"*": "img255"}),
  "mnist-1": dict(gen={"*": "img"}), "emotion-ferplus-8": dict(gen={"*": "img255"}), "super-resolution-10": dict(gen={"*": "img"}),
  "candy-9": dict(gen={"*": "img255"}), "tinyyolov2-8": dict(gen={"*": "img255"}),
  "version-rfb-320": dict(gen={"*": "imgnorm"}), "arcfaceresnet100-8": dict(gen={"*": "imgnorm"}),
  "resnet101-duc-12": dict(gen={"*": "imgnorm"}, shapes={"data": [-1, 3, 800, 800]}),
  "bertsquad-12": dict(gen={"input_ids:0": ("ints", 1, 1000), "input_mask:0": ("const", 1), "segment_ids:0": ("const", 0), "unique_ids_raw_output___9:0": ("const", 1)}),
  "bertsquad-12-int8": dict(gen={"input_ids:0": ("ints", 1, 1000), "input_mask:0": ("const", 1), "segment_ids:0": ("const", 0), "unique_ids_raw_output___9:0": ("const", 1)}, rms=0.1),
  "fasterrcnn-12": dict(shapes={"image": [3, 480, 640]}), "maskrcnn-12": dict(shapes={"image": [3, 480, 640]}),
  "yolov3-12": dict(gen={"input_1": "img", "image_shape": ("vals", [416, 416])}, shapes={"input_1": [-1, 3, 416, 416], "image_shape": [1, 2]}),
  "vitpose-base": dict(shapes={"pixel_values": [-1, 3, 256, 192]}), "openunmix-vocals": dict(shapes={"spectrogram": [1, 2, 2049, 100]}),
  "ppocrv5-det": dict(shapes={"x": [-1, 3, 320, 320]}), "scrfd-10g": dict(shapes={"input.1": [-1, 3, 640, 640]}), "uvdoc": dict(shapes={"image": [-1, 3, 488, 712]}),
  "kokoro-82m": dict(gen={"input_ids": ("ints", 1, 170), "style": ("randn", 0.3), "speed": ("const", 1.0)}, sym={"L": 24}),
  "trocr-small-printed.encoder_model": dict(shapes={"pixel_values": [-1, 3, 384, 384]}),
  "dpt-hybrid-midas": dict(shapes={"pixel_values": [-1, 3, 384, 384]}),
  "grounding-dino-tiny": dict(sym={"L": 6}, gen={"input_ids": ("vals", [[101, 4937, 1012, 4937, 1012, 102]]), "pixel_values": "imgnorm"}),
  "phi35-mini-q4f16": dict(sym={"L": 1, "past": 8}),
  "zipformer-encoder": dict(sym={"N": 1}),
  "efficientnet-lite4-int8": dict(gen={"*": "img255norm"}, rms=0.25),
}

def _gliner():
    n, w = 8, 12
    spans = [(i, min(i + k, n - 1)) for i in range(n) for k in range(w)]
    mask = [1 if i + k < n else 0 for i in range(n) for k in range(w)]
    return dict(sym={"L": 16}, shapes={"span_idx": [1, len(spans), 2], "span_mask": [1, len(spans)], "text_lengths": [1, 1]},
                gen={"input_ids": ("vals", [[1, 128001, 2000, 128002, 300, 400, 500, 600, 700, 800, 900, 1000, 2, 0, 0, 0]]), "attention_mask": ("vals", [[1] * 13 + [0] * 3]), "words_mask": ("vals", [[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]), "text_lengths": ("vals", [[n]]),
                     "span_idx": ("vals", [[list(x) for x in spans]]), "span_mask": ("vals", [mask])})
OVERRIDES["gliner-small"] = _gliner()

def discover():
    out = {}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from download_models import M   # only the survey downloads (the pre-existing models already have references)
    for d in sorted(M):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p): continue
        fs = sorted(f for f in os.listdir(p) if f.endswith(".onnx"))
        for f in fs:
            n = d if len(fs) == 1 else f"{d}.{f[:-5]}"
            out[n] = (d, f)
    return out

def dims_of(i, sym, has_past, batchname_default=-1):
    res = []
    for k, d in enumerate(i.type.tensor_type.shape.dim):
        if d.dim_value > 0: res.append(d.dim_value); continue
        p = d.dim_param or ""
        if k == 0: res.append(-1); continue
        res.append(eval_dim(p, sym, has_past, i.name))
    return res

def eval_dim(p, sym, has_past, iname):
    if p in sym: return sym[p]
    if "+" in p: return sum(eval_dim(q.strip(), sym, has_past, iname) for q in p.split("+"))
    lp = p.lower()
    if "past" in lp: return sym.get("past", 8)
    if "total" in lp: return sym.get("past", 8) + sym.get("L", 4)
    if lp in ("height", "width", "h", "w", "img_h", "img_w") or "height" in lp or "width" in lp: return sym.get("hw", 224)
    if "num_channels" in lp or lp == "channels": return 3
    if "sequence" in lp or "seq" in lp or "length" in lp or "token" in lp or lp.startswith("unk__"):
        return sym.get("L", 4 if has_past else 16)
    if "encoder" in lp: return sym.get("enc", 16)
    return sym.get("default", 16)

def make(rs, name, spec, shape, elem, ctx):
    kind = spec[0] if isinstance(spec, tuple) else spec
    n_ = shape[0] if shape else 1
    if kind == "auto":
        lname = name.lower()
        if elem in (6, 7):  # ints
            if "mask" in lname: a = np.ones(shape)
            elif "position" in lname:
                a = np.broadcast_to(np.arange(ctx["P"], ctx["P"] + shape[-1]), shape) if shape else np.array(0)
            elif "token_type" in lname or "segment" in lname: a = np.zeros(shape)
            elif "length" in lname or "len" in lname: a = np.full(shape, max(shape[-1] if shape else 4, 1))
            else: a = rs.randint(1, 1000, size=shape)
        elif elem == 9:
            a = np.ones(shape) if "use_cache" not in lname else np.zeros(shape)
        elif "past" in lname or "cache" in lname or "state" in lname: a = rs.randn(*shape) * 0.3
        elif len(shape) == 4 and shape[1] in (1, 3):
            a = smooth_image(rs, tuple(shape))
            a = (a - 0.45) / 0.25 if "pixel" in lname else a
        elif "mask" in lname: a = np.ones(shape)
        else: a = rs.randn(*shape) * 0.5
    elif kind == "img": a = smooth_image(rs, tuple(shape))
    elif kind == "imgnorm": a = (smooth_image(rs, tuple(shape)) - 0.45) / 0.25
    elif kind == "imgpm1": a = smooth_image(rs, tuple(shape)) * 2 - 1
    elif kind == "img255": a = smooth_image(rs, tuple(shape), 0, 255)
    elif kind == "img255norm": a = (smooth_image(rs, tuple(shape), 0, 255) - 127) / 128
    elif kind == "randn": a = rs.randn(*shape) * spec[1]
    elif kind == "ints": a = rs.randint(spec[1], spec[2], size=shape)
    elif kind == "const": a = np.full(shape, spec[1])
    elif kind == "vals": a = np.array(spec[1]).reshape(shape)
    else: raise ValueError(spec)
    return np.asarray(a).astype({1: np.float32, 2: np.uint8, 3: np.int8, 5: np.int16, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64, 16: np.float32}.get(elem, np.float32))

def open_session(path):
    last = None
    for prov in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
        try:
            so = ort.SessionOptions(); so.log_severity_level = 3
            return ort.InferenceSession(path, so, providers=prov), prov[0]
        except Exception as e:
            last = e
    raise last

def run(name, d, f):
    ov = OVERRIDES.get(name, {})
    path = os.path.join(ROOT, d, f)
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ins = [i for i in m.graph.input if i.name not in inits]
    has_past = any(i.name.startswith("past") for i in ins)
    sym = dict(ov.get("sym", {}))
    shapes = [ov.get("shapes", {}).get(i.name) or dims_of(i, sym, has_past) for i in ins]
    P = sym.get("past", 8) if has_past else 0
    ctx = {"P": P}
    gen = ov.get("gen", {}); pin = list(ov.get("pin", []))
    for i in ins:
        if "use_cache_branch" in i.name and i.name not in pin: pin.append(i.name)
    sess, refprov = open_session(path)
    outs = [o.name for o in sess.get_outputs()]
    os.makedirs(OUT, exist_ok=True)
    rs = np.random.RandomState(11)
    b = 1; consts = {}; feeds = {}
    for k, (i, s) in enumerate(zip(ins, shapes)):
        shp = [b if x == -1 else x for x in s]
        et = i.type.tensor_type.elem_type
        a = make(rs, i.name, gen.get(i.name, gen.get("*", ("auto",))), shp, et, ctx)
        feeds[i.name] = a
        if i.name in pin: consts[i.name] = a
        a.astype(np.float32).tofile(f"{OUT}/{name}.b{b}.in{k}.bin")
    res = sess.run(outs, feeds)
    for k, o in enumerate(res):
        np.ascontiguousarray(o, dtype=np.float32).tofile(f"{OUT}/{name}.b{b}.out{k}.bin")
    man = {"model": f"{d}/{f}", "batches": [b], "refProvider": refprov,
           "inputs": [{"name": i.name, "shape": s, "elem": i.type.tensor_type.elem_type} for i, s in zip(ins, shapes)],
           "outputs": [{"name": o} for o in outs], "rtol": ov.get("rtol", 1e-3), "atol": ov.get("atol", 2e-4), "fixbatch": ov.get("fixbatch", False),
           "rms": ov.get("rms"), "mode": None,
           "constants": {k: np.asarray(v).ravel().astype(float).tolist() for k, v in consts.items()}, "hints": {}}
    json.dump(man, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("ok", name, refprov, [(i.name, s) for i, s in zip(ins, shapes)][:4], "->", [(r.shape, str(r.dtype)) for r in res][:3], flush=True)

if __name__ == "__main__":
    disc = discover()
    names = sys.argv[1:] or [n for n in disc if not os.path.exists(f"{OUT}/{n}.json")]
    for n in names:
        if n not in disc: print("unknown", n); continue
        try: run(n, *disc[n])
        except Exception as e: print("FAILED", n, type(e).__name__, str(e).replace("\n", " ")[:400], flush=True)
