"""Merges results/*.jsonl (Sharplight harness + onnxruntime) and results/meta.json into docs/onnx-support.md.
usage: make_report.py [out.md]"""
import json, os, re, sys, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../docs/onnx-support.md")

def jl(path):
    for l in open(path):
        l = l.strip()
        if l:
            try: yield json.loads(l)
            except Exception: pass

def setname(man): return "new" if "/ref/survey/" in man else "zoo" if "/ref/zoo/" in man else "ref"

meta = json.load(open(f"{RES}/meta.json"))
def metakey(s, n): return f"{'survey' if s == 'new' else s}/{n}"

sl = {}
ROOTM = os.path.expanduser("~/models/onnx")
for f in sorted(glob.glob(f"{RES}/sharplight*.jsonl")):
    for r in jl(f):
        if not os.path.exists(r["manifest"]): continue
        k = (setname(r["manifest"]), os.path.basename(r["manifest"])[:-5])
        sl.setdefault(k, {}).setdefault(r["provider"], []).append(r)
ort = {}
for f in sorted(glob.glob(f"{RES}/ort*.jsonl")):
    for r in jl(f):
        if not os.path.exists(r["manifest"]): continue
        k = (setname(r["manifest"]), r["name"]); o = ort.setdefault(k, {})
        for p in ("cpu", "cuda"):
            if p in r and "minMs" in r[p]:
                if "minMs" not in o.get(p, {}) or r[p]["minMs"] < o[p]["minMs"]: o[p] = r[p]
            elif p in r and p not in o: o[p] = r[p]

def best(runs):
    oks = [r for r in runs if r.get("stage") == "ok"]
    if oks:
        b = dict(min(oks, key=lambda r: r["minMs"]))            # latency: min over sessions
        b["compileMs"] = min(r["compileMs"] for r in oks)
        last = oks[-1]                                          # correctness: newest harness run
        for k in ("outputs", "parityOk", "worstRms", "fixedBatch", "note"):
            b[k] = last.get(k)
        return b
    return runs[-1]

def parity(r):
    if r is None or r.get("stage") != "ok": return None
    w = r.get("worstRms", 0)
    if r.get("parityOk"): return "pass"
    return "close" if w == w and w <= 0.05 else "FAIL"

def worst_abs(r):
    return max((o.get("maxAbs", 0) for o in r.get("outputs", []) if "maxAbs" in o and o["maxAbs"] == o["maxAbs"]), default=0)

def status(r):
    if r is None: return "n/a"
    st = r.get("stage")
    if st == "ok": return {"pass": "OK", "close": "OK*", "FAIL": "WRONG"}[parity(r)]
    return {"import": "IMPORT-FAIL", "compile": "COMPILE-FAIL", "run": "RUN-FAIL", "timeout": "TIMEOUT", "crash": "CRASH"}.get(st, st)

# ---------------- feature classification of failing results
OPRE = re.compile(r"unsupported operator type\(s\): (.+?)\.?$")
PATTERNS = [
    (r"MatMulInteger B must be a compile-time constant", "MatMulInteger with runtime (activation) B operand"),
    (r"inputs must be f32 .*got i32", "CPU backend: i32 tensor through f32-only op (Scale)"),
    (r"reshape of batched .* fixes the batch dimension", "symbolic batch through Reshape that merges/fixes the batch dim (needs fixed batch)"),
    (r"Conv weight must be a compile-time constant", "Conv with computed (non-foldable) weight, e.g. weight-standardized conv (BiT/NFNet)"),
    (r"NonZero of a runtime-dependent tensor", "NonZero / data-dependent output shapes (runtime-valued)"),
    (r"Range on a runtime-dependent tensor", "Range with runtime-valued limit (data-dependent shape)"),
    (r"Arithmetic on the symbolic batch", "arithmetic on symbolic batch size"),
]
def features(err):
    if not err: return []
    first = err.split("\n")[0]
    m = OPRE.search(first)
    if m: return [("op", x.strip()) for x in m.group(1).split(",") if x.strip()]
    for pat, label in PATTERNS:
        if re.search(pat, err, re.S): return [("feature", label)]
    return [("other", re.sub(r"\s+", " ", err)[:160])]

def fmt(x, d=2):
    if x is None: return "-"
    return f"{x:.{d}f}" if x < 100 else f"{x:.0f}"
def sci(x): return "-" if x is None else f"{x:.1e}"

rows = []
for k in sorted(sl, key=lambda k: ({"ref": 0, "zoo": 1, "new": 2}[k[0]], k[1])):
    s, name = k; pr = sl[k]
    cpu = best(pr["cpu"]) if "cpu" in pr else None
    vk = best(pr["vulkan"]) if "vulkan" in pr else None
    o = ort.get(k, {}); mt = meta.get(metakey(s, name), {})
    rows.append(dict(set=s, name=name, cpu=cpu, vk=vk, ort_cpu=o.get("cpu", {}).get("minMs"), ort_cuda=o.get("cuda", {}).get("minMs"),
                     cuda_note=ort.get(k, {}).get("cuda", {}).get("error"), meta=mt))
for r in rows:
    r["vk_ms"] = r["vk"]["minMs"] if r["vk"] and r["vk"].get("stage") == "ok" else None
    r["cpu_ms"] = r["cpu"]["minMs"] if r["cpu"] and r["cpu"].get("stage") == "ok" else None
    r["opset"] = (r["meta"].get("opset") or {}).get("ai.onnx") or (r["vk"] or r["cpu"] or {}).get("opset")
    r["ratio_cuda"] = r["vk_ms"] / r["ort_cuda"] if r["vk_ms"] and r["ort_cuda"] else None
    r["ratio_cpu"] = r["vk_ms"] / r["ort_cpu"] if r["vk_ms"] and r["ort_cpu"] else None

L = []; P = L.append
sets = {"ref": "existing (ref/)", "zoo": "existing (ref/zoo)", "new": "new (ref/survey)"}
def counts(sel):
    c = collections.Counter(); 
    for r in rows:
        if sel(r): c[status(r["vk"])] += 1
    return c
def counts_cpu(sel):
    c = collections.Counter()
    for r in rows:
        if sel(r): c[status(r["cpu"])] += 1
    return c

P("# ONNX support survey\n")
P("Generated by `bench/onnx/make_report.py` from `bench/onnx/results/*.jsonl` (see *Reproducing* at the end). Survey only: nothing under `src/` or `tests/` was changed.\n")
P("## Method\n")
P("* Models live in `~/models/onnx/<name>/`; every model has an onnxruntime reference (CPU EP; the CUDA EP only where the CPU EP lacks a kernel) in the ZooTests fixture format: "
  "`~/models/onnx/ref/{,zoo/,survey/}<name>.json` + `.b1.in<k>.bin` / `.b1.out<k>.bin`. Inputs are fixed synthetic data (smooth images, random ids, ones masks); dynamic dims are pinned "
  "(sequence 16, KV past 8, images 224 unless the model needs another size) because the importer needs static non-batch dims. Merged decoders run with `use_cache_branch` pinned to 0.")
P("* Sharplight: `bench/onnx/SurveyRunner` (import via `OnnxImporter`, `CompiledModel.Compile`, one process per model and provider). If a symbolic batch fails the harness retries with the concrete batch (that is noted). "
  "Parity = existing ZooTests criteria (manifest rtol/atol, or relative RMS for quantized models); `OK*` = outside the strict tolerance but relative RMS <= 5e-2 (quantized / fp16 models whose reference is itself noisy), `WRONG` = above that.")
P("* Latency = min over N (<=30, >=3, 4 s budget) of `Run` + read back all outputs, batch 1. Sharplight Vulkan device 0 = RTX PRO 6000 Blackwell Max-Q (shared with the desktop: min times only), CPU = Ryzen 9 9950X (32 threads). "
  "ORT CPU = onnxruntime-gpu 1.31 CPU EP, ORT CUDA = CUDA EP (CUDA 13.5 / cuDNN 9.28 pip wheels), inputs pre-bound on the GPU and outputs copied to host, so both sides include the output read-back. "
  "TensorRT EP was not tried (the pip wheels lack libnvinfer; the CUDA EP works on this Blackwell GPU). Latencies that were measured in several sessions are reported as the minimum.\n")

P("## Summary\n")
tot = {s: sum(1 for r in rows if r["set"] == s) for s in sets}
P("| set | models | Vulkan OK | Vulkan OK* | Vulkan WRONG | import fail | compile/run fail | CPU OK (incl. OK*) | CPU fails |")
P("|---|---|---|---|---|---|---|---|---|")
def line(label, sel):
    c = counts(sel); cc = counts_cpu(sel); n = sum(1 for r in rows if sel(r))
    P(f"| {label} | {n} | {c['OK']} | {c['OK*']} | {c['WRONG']} | {c['IMPORT-FAIL']} | {c['COMPILE-FAIL'] + c['RUN-FAIL'] + c['CRASH'] + c['TIMEOUT']} | {cc['OK'] + cc['OK*']} | {n - cc['OK'] - cc['OK*']} |")
for s, lab in sets.items(): line(lab, lambda r, s=s: r["set"] == s)
line("**all**", lambda r: True)
P("")
nf = [r for r in rows if r["vk"] and r["vk"].get("stage") != "ok"]

FIND = os.path.join(HERE, "report_findings.md")
if os.path.exists(FIND): L.append(open(FIND).read().rstrip() + "\n")

# ---------------- ranked features
feat = collections.defaultdict(lambda: {"models": set(), "cpu_only": set()})
for r in rows:
    for prov, key in (("vk", "models"), ("cpu", "cpu_only")):
        x = r[prov]
        if x and x.get("stage") != "ok":
            for kind, f in features(x.get("error", "")):
                if prov == "cpu" and r["vk"] and r["vk"].get("stage") != "ok" and any(f == g for _, g in features(r["vk"].get("error", ""))): continue
                feat[(kind, f)][key].add(r["name"])
P("## Missing ops / features, ranked by models blocked\n")
P("An importer reports all unsupported op *types* of a model at once, but a model that is fixed for those ops can still hit a later limitation, so counts are lower bounds on work, upper bounds on what is unblocked by one item. "
  "Vulkan failures first; items that only fail on the CPU backend are marked.\n")
P("| rank | missing op / feature | kind | models blocked (Vulkan) | models blocked (CPU only) | models |")
P("|---|---|---|---|---|---|")
rk = sorted(feat.items(), key=lambda kv: (-len(kv[1]["models"]), -len(kv[1]["cpu_only"]), kv[0][1]))
for i, ((kind, f), d) in enumerate(rk, 1):
    P(f"| {i} | `{f}` | {kind} | {len(d['models'])} | {len(d['cpu_only'])} | {', '.join(sorted(d['models'] | d['cpu_only']))} |")
P("")

# ---------------- slow
P("## Where Sharplight is much slower than onnxruntime\n")
P("Ratios are Sharplight Vulkan min latency divided by ORT min latency (> 1 = Sharplight slower). Shown: models with ratio vs ORT CUDA >= 4 or Vulkan at least half as slow as ORT CPU, worst first. Cause comes from `SHARPLIGHT_VK_PROFILE=2` GPU timestamps (`results/profile.jsonl`) where measured.\n")
P("| model | Sharplight Vulkan ms | ORT CUDA ms | Vulkan / CUDA | ORT CPU ms | Vulkan / ORT CPU | Sharplight CPU ms | cause |")
P("|---|---|---|---|---|---|---|---|")
slow = [r for r in rows if r["ratio_cuda"] and (r["ratio_cuda"] >= 4 or (r["ratio_cpu"] and r["ratio_cpu"] >= 0.5))]
CAUSES = {
    "yolov10n": "`topk` kernel: 2 launches x ~38 ms = 99% of GPU time",
    "yolo26n": "`topk` kernel: ~40 ms per launch = 88% of GPU time",
    "rtdetr-r18": "`topk` kernel: ~40 ms per launch = 62% of GPU time",
    "mobilevitv2": "`scatter_nd`: 336 launches x ~170 us per run = 95% of GPU time (unrolled unfold/fold graph, 13.8k nodes)",
    "animegan-v2": "`groupnorm` (InstanceNorm 512x512 maps): 24 launches x ~2 ms = 92% of GPU time",
    "trunc-animegan-v2": "same InstanceNorm / `groupnorm` kernel as animegan-v2 (truncated copy)",
    "resnet101-duc-12": "3x3 convs 100-700 us each and a 21 ms `softmax` over [1,19,160000]",
    "sdxl-vae": "3x3 convs (512 ch at 64x64: 600 us each) dominate",
    "roberta-seqcls-9": "f32 `matmul` at 16 tokens: 768->768 44 us, 3072->768 170 us (~55 GB/s of weights)",
    "vit-base-patch16-224": "f32 `matmul` 768->768 44 us, 3072->768 168 us (~5 TFLOPS) vs ORT CUDA cuBLAS",
    "depth-anything-v2-base": "f32 `matmul` 768/3072 (same as ViT)",
}
for r in sorted(slow, key=lambda r: -(r["ratio_cuda"] or 0)):
    cp = f"{r['ratio_cpu']:.2f}x" if r["ratio_cpu"] else "-"
    P(f"| {r['name']} | {fmt(r['vk_ms'])} | {fmt(r['ort_cuda'])} | {r['ratio_cuda']:.1f}x | {fmt(r['ort_cpu'])} | {cp} | {fmt(r['cpu_ms'])} | {CAUSES.get(r['name'], '')} |")
P("")
rs = [r["ratio_cuda"] for r in rows if r["ratio_cuda"]]
if rs:
    rs.sort(); P(f"Over the {len(rs)} models that run on Vulkan and have a CUDA reference: median Vulkan/CUDA = {rs[len(rs)//2]:.1f}x, min {rs[0]:.2f}x, max {rs[-1]:.1f}x; "
                 f"Vulkan faster than ORT CPU on {sum(1 for r in rows if r['ratio_cpu'] and r['ratio_cpu'] < 1)} of {sum(1 for r in rows if r['ratio_cpu'])}.\n")

# ---------------- cpu vs ort cpu
cr = [(r["cpu_ms"] / r["ort_cpu"], r) for r in rows if r["cpu_ms"] and r["ort_cpu"] and r["ort_cpu"] > 0.5]
if cr:
    cr.sort(key=lambda x: x[0])
    P("### Sharplight CPU backend vs onnxruntime CPU\n")
    P(f"Over {len(cr)} models with ORT CPU latency > 0.5 ms: median Sharplight-CPU / ORT-CPU = {cr[len(cr)//2][0]:.2f}x, Sharplight faster on {sum(1 for x in cr if x[0] < 1)}; worst:\n")
    P("| model | Sharplight CPU ms | ORT CPU ms | ratio |")
    P("|---|---|---|---|")
    for q, r in cr[::-1][:10]: P(f"| {r['name']} | {fmt(r['cpu_ms'])} | {fmt(r['ort_cpu'])} | {q:.1f}x |")
    P("")

# ---------------- opsets
P("## Opset coverage\n")
P("| ai.onnx opset | models | Vulkan OK / OK* | failing |")
P("|---|---|---|---|")
byop = collections.defaultdict(list)
for r in rows:
    if r["opset"]: byop[int(r["opset"])].append(r)
for o in sorted(byop):
    rs_ = byop[o]; okc = sum(1 for r in rs_ if status(r["vk"]) in ("OK", "OK*"))
    P(f"| {o} | {len(rs_)} | {okc} | {', '.join(r['name'] for r in rs_ if status(r['vk']) not in ('OK', 'OK*'))} |")
P("")

# ---------------- matrix
P("## Per-model matrix\n")
P("Status: `OK` parity within tolerance, `OK*` approx (rel. RMS <= 5e-2), `WRONG`, `IMPORT-FAIL` / `COMPILE-FAIL` / `RUN-FAIL` (stage that failed). Parity columns are the worst output over all outputs: max abs error and relative RMS "
  "(sqrt(sum diff^2 / sum ref^2)) on Vulkan. `fb` = needed the concrete batch (symbolic batch failed).\n")
P("| model | opset | MB | Vulkan | CPU | max abs (VK) | rel RMS (VK) | rel RMS (CPU) | SL Vulkan ms | SL CPU ms | ORT CUDA ms | ORT CPU ms | VK/CUDA | VK/ORTcpu | VK compile s | note |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for s, lab in sets.items():
    P(f"| **{lab}** | | | | | | | | | | | | | | | |")
    for r in [x for x in rows if x["set"] == s]:
        vk, cpu = r["vk"], r["cpu"]
        note = []
        if vk and vk.get("fixedBatch") and vk.get("note"): note.append("fb")
        for x, tag in ((vk, "VK"), (cpu, "CPU")):
            if x and x.get("stage") != "ok":
                note.append(f"{tag}: {re.sub(chr(10), ' ', x.get('error', ''))[:140]}")
        if parity(vk) == "FAIL":
            bad = [o["name"] for o in vk["outputs"] if not o.get("ok")][:2]; note.append("bad: " + ",".join(bad))
        P(f"| {r['name']} | {r['opset'] or '-'} | {r['meta'].get('sizeMB', '-')} | {status(vk)} | {status(cpu)} | "
          f"{sci(worst_abs(vk)) if vk and vk.get('stage') == 'ok' else '-'} | {sci(vk.get('worstRms')) if vk and vk.get('stage') == 'ok' else '-'} | {sci(cpu.get('worstRms')) if cpu and cpu.get('stage') == 'ok' else '-'} | "
          f"{fmt(r['vk_ms'])} | {fmt(r['cpu_ms'])} | {fmt(r['ort_cuda'])} | {fmt(r['ort_cpu'])} | {'-' if r['ratio_cuda'] is None else f'{r[chr(114)+chr(97)+chr(116)+chr(105)+chr(111)+chr(95)+chr(99)+chr(117)+chr(100)+chr(97)]:.1f}x'} | "
          f"{'-' if r['ratio_cpu'] is None else f'{r[chr(114)+chr(97)+chr(116)+chr(105)+chr(111)+chr(95)+chr(99)+chr(112)+chr(117)]:.2f}x'} | {'-' if not (vk and vk.get('compileMs')) else f"{vk['compileMs'] / 1000:.1f}"} | {' ; '.join(note).replace('|', '/')} |")
P("")

# ---------------- failure details
P("## Exact errors (failing models)\n")
P("| model | provider | stage | error |")
P("|---|---|---|---|")
for r in rows:
    for prov, x in (("Vulkan", r["vk"]), ("CPU", r["cpu"])):
        if x and x.get("stage") != "ok":
            P(f"| {r['name']} | {prov} | {x.get('stage')} | `{re.sub(chr(10), ' ', x.get('error', ''))[:600]}` |".replace("|`", "| `"))
P("")
# ---------------- models + reproduction
sys.path.insert(0, HERE)
try:
    from download_models import M
    P("## New models and references\n")
    P("Downloaded by `bench/onnx/download_models.py` into `~/models/onnx/<name>/` (source repo in the second column); references (inputs + ORT outputs) in `~/models/onnx/ref/survey/<model>.json` + `.b1.in<k>.bin` / `.b1.out<k>.bin`, generated by `bench/onnx/make_survey_refs.py`.\n")
    P("| directory | Hugging Face repo | files |")
    P("|---|---|---|")
    for n, (repo, fs) in M.items():
        P(f"| `~/models/onnx/{n}/` | `{repo}` | {', '.join(os.path.basename(f) for f in fs)} |")
    P("")
except Exception as e:
    P(f"(model list unavailable: {e})")
P("## Reproducing\n")
P("```sh")
P("uv venv ~/venvs/ort --python 3.12 && source ~/venvs/ort/bin/activate")
P("uv pip install onnx onnxruntime-gpu numpy huggingface_hub nvidia-cublas nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cuda-nvrtc")
P("source bench/onnx/env.sh                       # LD_LIBRARY_PATH for the CUDA 13 / cuDNN 9 wheels")
P("HF_HUB_DISABLE_XET=1 python bench/onnx/download_models.py")
P("python bench/onnx/make_survey_refs.py          # ORT reference outputs -> ~/models/onnx/ref/survey")
P("dotnet build bench/onnx/SurveyRunner -c Release")
P("python bench/onnx/run_survey.py bench/onnx/results/sharplight_passN.jsonl     # Sharplight cpu+vulkan, one process per model")
P("python bench/onnx/ort_bench.py bench/onnx/results/ort_passN.jsonl             # ORT CPU + CUDA latency")
P("python bench/onnx/model_meta.py && python bench/onnx/make_report.py          # -> docs/onnx-support.md")
P("# per-kernel GPU time of one model:  dotnet bench/onnx/SurveyRunner/bin/Release/net11.0/SurveyRunner.dll <manifest.json> vulkan --profile")
P("```")
P("Existing models are measured from their existing fixtures (`ref/*.json`, `ref/zoo/*.json`) without regenerating them. The harness reads the same manifest format as `ZooTests`.\n")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, len(rows), "models")
