"""Runs the SurveyRunner harness (one process per model x provider) over reference manifests and appends one JSON line each.
usage: run_survey.py <out.jsonl> [--providers cpu,vulkan] [--timeout 900] [manifest.json ...]
Build the harness first:  dotnet build bench/onnx/SurveyRunner -c Release"""
import json, os, sys, glob, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "SurveyRunner/bin/Release/net11.0/SurveyRunner.dll")
ROOT = os.path.expanduser("~/models/onnx")

def main():
    a = sys.argv[1:]; out = a.pop(0); provs = ["cpu", "vulkan"]; timeout = 900; mans = []
    while a:
        x = a.pop(0)
        if x == "--providers": provs = a.pop(0).split(",")
        elif x == "--timeout": timeout = int(a.pop(0))
        else: mans.append(x)
    if not mans: mans = sorted(glob.glob(f"{ROOT}/ref/*.json") + glob.glob(f"{ROOT}/ref/zoo/*.json") + glob.glob(f"{ROOT}/ref/survey/*.json"))
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: j = json.loads(l); done.add((j["manifest"], j["provider"]))
            except Exception: pass
    with open(out, "a") as f:
        for m in mans:
            for p in provs:
                if (m, p) in done: continue
                t0 = time.time()
                try:
                    pr = subprocess.run(["dotnet", DLL, m, p], capture_output=True, text=True, timeout=timeout)
                    lines = [l for l in pr.stdout.splitlines() if l.startswith("RESULT ")]
                    if lines: r = json.loads(lines[-1][7:])
                    else: r = {"manifest": m, "provider": p, "stage": "crash", "error": f"exit {pr.returncode}: " + (pr.stderr or pr.stdout)[-800:]}
                except subprocess.TimeoutExpired:
                    r = {"manifest": m, "provider": p, "stage": "timeout", "error": f"timeout after {timeout}s"}
                r["wallS"] = round(time.time() - t0, 1)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(os.path.basename(m)[:-5], p, r.get("stage"), r.get("minMs"), (r.get("error") or "")[:120], flush=True)

main()
