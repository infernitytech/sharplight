"""Summarizes sat/<round>/<model>_<engine>_<load>/ runs: tok/s, TTFT / ITL percentiles, gauges, preemptions, KV capacity, text check."""
import glob, json, os, re, sys

root = sys.argv[1]
for d in sorted(glob.glob(root + "/*_*_*")):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    ready = open(glob.glob(d + "/*.ready")[0]).read().strip() if glob.glob(d + "/*.ready") else "?"
    tcs = "FAIL" if glob.glob(d + "/*.textcheck.FAIL") else ("clean" if glob.glob(d + "/*.textcheck.json") else "none")
    cap = ""
    for log in glob.glob(d + "/*.server.log"):
        t = open(log, errors="replace").read()
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t) or re.search(r"paged KV cache of (\d+) tokens", t)
        if m:
            cap = m.group(1).replace(",", "")
    print(f"{name:26s} {ready:20s} text {tcs:5s} kv {cap}")
    for f in sorted(glob.glob(d + "/serve_*_c*_r*.json"), key=lambda p: (int(re.search(r"_c(\d+)_", p).group(1)), p)):
        j = json.load(open(f))
        g = j.get("gauges", {})
        md = j.get("metrics_delta", {})
        run = g.get("sharplight_active_sequences") or g.get("vllm:num_requests_running") or {}
        wait = g.get("sharplight_queue_depth") or g.get("vllm:num_requests_waiting") or {}
        kvu = g.get("sharplight_kv_utilization") or g.get("vllm:kv_cache_usage_perc") or {}
        pre = sum(v for k, v in md.items() if "preempt" in k and (k.endswith("_total") or k.endswith("_sum")))
        qn = md.get("sharplight_queue_wait_seconds_count") or md.get("vllm:request_queue_time_seconds_count") or 0
        qs = md.get("sharplight_queue_wait_seconds_sum") or md.get("vllm:request_queue_time_seconds_sum") or 0
        steps = md.get("sharplight_steps_total") or md.get("vllm:iteration_tokens_total_count") or 0
        print(f"   c{j['concurrency']:<3}: {j['agg_tok_s']:7.0f} tok/s  ttft p50 {j['ttft_p50_ms']:6.0f} p90 {j.get('ttft_p90_ms', float('nan')):6.0f} p99 {j['ttft_p99_ms']:6.0f}"
              f"  itl mean {j['itl_mean_ms']:6.2f} p50 {j['itl_p50_ms']:6.1f} p90 {j.get('itl_p90_ms', float('nan')):6.1f} p99 {j['itl_p99_ms']:6.1f}"
              f"  run {run.get('mean', float('nan')):5.1f}/{run.get('max', float('nan')):3.0f} wait {wait.get('mean', float('nan')):5.1f}/{wait.get('max', float('nan')):3.0f}"
              f" kv {kvu.get('mean', float('nan')):.2f}/{kvu.get('max', float('nan')):.2f} preempt {pre:.0f} queue {1000 * qs / qn if qn else float('nan'):5.0f} ms steps {steps:.0f}  in {j['avg_prompt_tokens']:.0f} out {j['avg_out_tokens']:.0f} n {j['requests']} err {j['errors']}")
