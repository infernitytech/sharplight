import json, glob, os, sys
root = sys.argv[1]
for d in sorted(glob.glob(root + '/*_*_*')):
    name = os.path.basename(d)
    if not os.path.isdir(d):
        continue
    ready = open(glob.glob(d + '/*.ready')[0]).read().strip() if glob.glob(d + '/*.ready') else '?'
    tc = glob.glob(d + '/*.textcheck.json')
    tcs = 'FAIL' if glob.glob(d + '/*.textcheck.FAIL') else ('clean' if tc else 'none')
    firsts = []
    for f in sorted(glob.glob(d + '/serve_*_first*.json')):
        j = json.load(open(f)); firsts.append(f"{j['ttft_p50_ms']:.0f}")
    print(f"{name:14s} {ready:18s} text {tcs:5s} first ttft {'/'.join(firsts)} ms")
    for f in sorted(glob.glob(d + '/serve_*_c*_r*.json')):
        j = json.load(open(f)); c = j['concurrency']
        print(f"   c{c:<3}: {j['agg_tok_s']:7.0f} tok/s  ttft p50 {j['ttft_p50_ms']:5.0f} p99 {j['ttft_p99_ms']:5.0f}  itl mean {j['itl_mean_ms']:6.2f} p95 {j['itl_p95_ms']:5.1f} p99 {j['itl_p99_ms']:6.1f} max {j['itl_max_ms']:6.1f}  err {j['errors']}")
