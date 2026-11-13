#!/usr/bin/env python3
"""Tables for docs/benchmarks.md section 16 from the files written by s16_single.sh / s16_serve.sh.
usage: s16_summarize.py [results dir, default bench/results/s16]"""
import glob, json, os, re, statistics, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "s16")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rng(xs, fmt):
    return f"{fmt.format(med(xs))} ({fmt.format(min(xs))}..{fmt.format(max(xs))})" if xs else "-"


def single():
    rows = {}
    for f in sorted(glob.glob(os.path.join(D, "single_sharplight_*.txt"))):
        name = os.path.basename(f)[len("single_sharplight_"):-4]
        pp, tg, gpu = {}, {}, {}
        for line in open(f):
            m = re.match(r"\[N=(\d+) run=\d+\] prefill \d+ tok: best [\d.]+ ms = ([\d.]+) tok/s \| decode: best ([\d.]+), median ([\d.]+).*GPU ms/step: best ([\d.]+), median ([\d.]+)", line)
            if m:
                n = int(m.group(1))
                pp.setdefault(n, []).append(float(m.group(2))); tg.setdefault(n, []).append(float(m.group(4))); gpu.setdefault(n, []).append(float(m.group(6)))
        r = rows.setdefault(name, {})
        for n in pp:
            if n >= 512: r[f"pp{n}"] = ("sharplight", pp[n])
        for n in tg:
            key = "tg@d0" if n <= 16 else f"tg@d{n}"
            r.setdefault(key, {})
        rows[name]["_sl"] = {"pp": pp, "tg": tg, "gpu": gpu}
    print("### Sharplight single stream (median (min..max) over process runs)\n")
    print("| model | N | prefill tok/s (best of 4 per run) | decode tok/s @ depth N (median step) | GPU ms / decode step | decode tok/s from GPU time |")
    print("|---|---|---|---|---|---|")
    for name, r in rows.items():
        s = r["_sl"]
        for n in sorted(s["tg"]):
            print(f"| {name} | {n} | {rng(s['pp'][n], '{:.0f}') if n >= 512 else '-'} | {rng(s['tg'][n], '{:.1f}')} | {rng(s['gpu'][n], '{:.2f}')} | {1000 / med(s['gpu'][n]):.1f} |")
    print()
    for f in sorted(glob.glob(os.path.join(D, "single_llama-*.jsonl"))):
        print(f"### {os.path.basename(f)}\n\n| test | tok/s median (min..max) |\n|---|---|")
        for line in open(f):
            try: j = json.loads(line)
            except Exception: continue
            t = f"pp{j['n_prompt']}" if j["n_prompt"] else f"tg{j['n_gen']}"
            print(f"| {t} @d{j['n_depth']} | {rng(j['samples_ts'], '{:.1f}')} |")
        print()
    for f in sorted(glob.glob(os.path.join(D, "single_vllm_*.json"))):
        j = json.load(open(f))
        print(f"### {os.path.basename(f)} (load {j['load_s']:.0f} s, spec {j.get('spec')})\n\n| point | tok/s median (min..max) | acceptance | mean accepted length |\n|---|---|---|---|")
        for k, v in j["results"].items():
            print(f"| {k} | {v['tok_s_median']:.1f} ({v['tok_s_min']:.1f}..{v['tok_s_max']:.1f}) | {v.get('acceptance_rate', float('nan')):.3f} | {v.get('mean_accepted_length', float('nan')):.2f} |")
        print()


def serve():
    runs = {}
    for f in sorted(glob.glob(os.path.join(D, "serve_*_c*_r*.json"))):
        m = re.match(r"serve_(.+?)_(.+)_c(\d+)_r(\d+)\.json", os.path.basename(f))
        eng, name, c = m.group(1), m.group(2), int(m.group(3))
        runs.setdefault((name, c, eng), []).append(json.load(open(f)))
    print("### Serving (median over rounds; agg tok/s min..max)\n")
    print("| model | conc | engine | agg tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p99 ms | steps / out tok | rows / step | step ms (engine) | step busy % of wall | spec accept | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, c, eng) in sorted(runs, key=lambda k: (k[0], k[1], k[2])):
        rs = runs[(name, c, eng)]
        g = lambda k: med([r[k] for r in rs])
        agg = [r["agg_tok_s"] for r in rs]
        md = [r.get("metrics_delta", {}) for r in rs]
        extra = ["-"] * 5
        if eng == "sharplight" and md[0]:
            steps = med([m.get("sharplight_steps_total", 0) for m in md])
            toks = med([m.get("sharplight_completion_tokens_total", 1) for m in md])
            rows = med([m.get("sharplight_step_rows_sum", 0) / max(m.get("sharplight_step_rows_count", 1), 1) for m in md])
            sms = med([1000 * m.get("sharplight_step_duration_seconds_sum", 0) / max(m.get("sharplight_step_duration_seconds_count", 1), 1) for m in md])
            busy = med([100 * m.get("sharplight_step_duration_seconds_sum", 0) / r["wall_s"] for m, r in zip(md, rs)])
            extra = [f"{steps / toks:.3f}", f"{rows:.1f}", f"{sms:.2f}", f"{busy:.0f}", "-"]
        elif eng.startswith("vllm") and md[0]:
            it = med([m.get("vllm:iteration_tokens_total_sum", 0) / max(m.get("vllm:iteration_tokens_total_count", 1), 1) for m in md])
            steps = med([m.get("vllm:iteration_tokens_total_count", 0) for m in md])
            toks = med([r["out_tokens"] for r in rs])
            dr = sum(m.get("vllm:spec_decode_num_draft_tokens_total", 0) for m in md)
            ac = sum(m.get("vllm:spec_decode_num_accepted_tokens_total", 0) for m in md)
            extra = [f"{steps / toks:.3f}" if steps else "-", f"{it:.1f}" if steps else "-", "-", "-", f"{ac / dr:.2f}" if dr else "-"]
        print(f"| {name} | {c} | {eng} | {med(agg):.0f} ({min(agg):.0f}..{max(agg):.0f}) | {g('ttft_p50_ms'):.0f} | {g('ttft_p95_ms'):.0f} | {g('itl_mean_ms'):.2f} | {g('itl_p99_ms'):.1f} | {' | '.join(extra)} | {sum(r['errors'] for r in rs)} |")
    print()


single()
serve()
