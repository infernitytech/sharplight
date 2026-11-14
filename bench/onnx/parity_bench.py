"""Sharplight vs onnxruntime latency on a fixed set of representative ONNX models (CPU and GPU).

usage:
  parity_bench.py run <out.jsonl> [--rounds R] [--engines e1,e2,...] [--models m1,m2,...] [--runs N]
  parity_bench.py table <out.jsonl> [<before.jsonl>]

Engines: sl-cpu, ort-cpu, sl-vulkan, sl-vulkan-rp (AllowReducedPrecision), ort-cuda (TF32 allowed, the EP default),
ort-cuda-strict (use_tf32=0). Every measurement is its own process. The engines of one model run back to back and the
rounds repeat the whole set, so A/B pairs are interleaved in time; the table reports the minimum over the rounds of each
process's min-of-N and the median over the rounds of its median.

CPU engines are pinned to cores 0-(T-1) with T = $BENCH_THREADS (default 16) threads (Sharplight: SHARPLIGHT_CPU_THREADS,
onnxruntime: intra_op_num_threads). Environment:
  SHARPLIGHT_ONNX_MODELS  directory with the models and ref/ manifests (required)
  ORT_PYTHON              python with onnxruntime-gpu (default: python3; source bench/onnx/env.sh for the CUDA libs)
  ONNX_BENCH_DLL          OnnxBench.dll (default: bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll)
  BENCH_MIN_FREE_GB       wait before each run until this much memory is available (default 20)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (manifest under $SHARPLIGHT_ONNX_MODELS/ref, batches)
MODELS = {
    "resnet50": ("resnet50-v1-12.json", [1, 8]),
    "convnext-tiny": ("zoo/convnext-tiny.json", [1, 8]),
    "efficientnet-lite4": ("efficientnet-lite4-11.json", [1]),
    "mobilenetv2": ("mobilenetv2-12.json", [1, 8]),
    "yolov8n": ("zoo/yolov8n.json", [1]),
    "yolov10n": ("yolov10n.json", [1]),
    "ssd-10 (NMS)": ("survey/ssd-10.json", [1]),
    "bert-base (16 tok)": ("survey/bert-base-uncased.model.json", [1, 8]),
    "vit-base": ("vit-base-patch16-224.json", [1, 8]),
    "whisper-tiny-encoder": ("whisper-tiny-encoder.json", [1]),
    "qwen2.5-0.5b q4 (MatMulNBits)": ("survey/qwen2.5-0.5b.model_q4.json", [1]),
    "phi3.5-mini q4f16 (GQA)": ("survey/phi35-mini-q4f16.json", [1]),
    "segformer-b0": ("segformer-b0.json", [1, 8]),
    "mobilesam-encoder": ("zoo/mobilesam-encoder.json", [1]),
    "raft-large": ("zoo/raft-large.json", [1]),
}
ENGINES = ["sl-cpu", "ort-cpu", "sl-vulkan", "sl-vulkan-rp", "ort-cuda", "ort-cuda-strict"]
CPU_ENGINES = {"sl-cpu", "ort-cpu"}


def wait_for_memory():
    need = float(os.environ.get("BENCH_MIN_FREE_GB", 20))
    while True:
        with open("/proc/meminfo") as f:
            avail = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2**20
        if avail >= need:
            return
        print(f"  waiting: {avail:.1f} GB available < {need} GB", flush=True)
        time.sleep(30)


def command(engine, manifest, batch, runs, threads):
    if engine.startswith("sl-"):
        dll = os.environ.get("ONNX_BENCH_DLL", os.path.join(HERE, "OnnxBench/bin/Release/net11.0/OnnxBench.dll"))
        provider = "cpu" if engine == "sl-cpu" else "vulkan"
        cmd = ["dotnet", dll, manifest, provider, "--runs", str(runs), "--batch", str(batch)]
        if engine == "sl-vulkan-rp":
            cmd.append("--reduced-precision")
    else:
        python = os.environ.get("ORT_PYTHON", "python3")
        cmd = [python, os.path.join(HERE, "ort_latency.py"), manifest, engine[4:], "--runs", str(runs), "--batch", str(batch), "--threads", str(threads)]
    if engine in CPU_ENGINES:
        cmd = ["taskset", "-c", f"0-{threads - 1}"] + cmd
    return cmd


def run(out, rounds, engines, models, runs):
    root = os.environ["SHARPLIGHT_ONNX_MODELS"]
    threads = int(os.environ.get("BENCH_THREADS", 16))
    env = dict(os.environ, SHARPLIGHT_CPU_THREADS=str(threads))
    done = set()
    if os.path.exists(out):
        for line in open(out):
            j = json.loads(line)
            done.add((j["round"], j["model"], j["engine"], j["batch"]))
    with open(out, "a") as f:
        for r in range(rounds):
            for name in models:
                manifest, batches = MODELS[name]
                for batch in batches:
                    for engine in engines:
                        if (r, name, engine, batch) in done:
                            continue
                        wait_for_memory()
                        t0 = time.time()
                        try:
                            p = subprocess.run(command(engine, os.path.join(root, "ref", manifest), batch, runs, threads),
                                               capture_output=True, text=True, timeout=1800, env=env)
                            lines = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
                            res = json.loads(lines[-1][7:]) if lines else {"stage": "crash", "error": (p.stderr or p.stdout)[-600:]}
                        except subprocess.TimeoutExpired:
                            res = {"stage": "timeout"}
                        res.pop("profile", None)
                        res.update(round=r, model=name, engine=engine, batch=batch, wallS=round(time.time() - t0, 1))
                        f.write(json.dumps(res) + "\n")
                        f.flush()
                        print(f"r{r} {name} b{batch} {engine}: {res.get('stage')} min {res.get('minMs')} med {res.get('medMs')}"
                              f" parity {res.get('parityOk')} {str(res.get('error', ''))[:150]}", flush=True)


def aggregate(path):
    agg = {}
    for line in open(path):
        j = json.loads(line)
        key = (j["model"], j["batch"], j["engine"])
        a = agg.setdefault(key, {"min": [], "med": [], "parity": [], "error": None})
        if j.get("stage") == "ok":
            a["min"].append(j["minMs"])
            a["med"].append(j["medMs"])
            if "parityOk" in j:
                a["parity"].append(j["parityOk"])
        else:
            a["error"] = j.get("stage")
    out = {}
    for key, a in agg.items():
        if a["min"]:
            meds = sorted(a["med"])
            out[key] = (min(a["min"]), meds[len(meds) // 2], all(a["parity"]) if a["parity"] else None)
        else:
            out[key] = (None, None, a["error"])
    return out


def cell(v):
    if v is None or v[0] is None:
        return "fail" if v else "-"
    mark = "" if v[2] in (True, None) else " (!)"
    return f"{v[0]:.2f} / {v[1]:.2f}{mark}"


def table(path, before=None):
    now = aggregate(path)
    old = aggregate(before) if before else {}
    engines = [e for e in ENGINES if any(k[2] == e for k in now)]
    print("| model | batch | " + " | ".join(engines) + " | SL/ORT CPU | SL/ORT CUDA |")
    print("|---|---|" + "---|" * (len(engines) + 2))
    for name, (_, batches) in MODELS.items():
        for batch in batches:
            if not any((name, batch, e) in now for e in engines):
                continue
            cells = []
            for e in engines:
                c = cell(now.get((name, batch, e)))
                if (name, batch, e) in old and e.startswith("sl-"):
                    c = f"{cell(old[(name, batch, e)])} → {c}"
                cells.append(c)

            def ratio(a, b):
                x, y = now.get((name, batch, a)), now.get((name, batch, b))
                return f"{x[0] / y[0]:.2f}x" if x and y and x[0] and y[0] else "-"
            print(f"| {name} | {batch} | " + " | ".join(cells) + f" | {ratio('sl-cpu', 'ort-cpu')} | {ratio('sl-vulkan', 'ort-cuda')} |")


def main(argv):
    if len(argv) < 2 or argv[0] not in ("run", "table"):
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "table":
        table(argv[1], argv[2] if len(argv) > 2 else None)
        return 0
    opts = dict(zip(argv[2::2], argv[3::2]))
    engines = opts.get("--engines", ",".join(ENGINES)).split(",")
    models = opts["--models"].split(",") if "--models" in opts else list(MODELS)
    run(argv[1], int(opts.get("--rounds", 2)), engines, models, int(opts.get("--runs", 30)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
