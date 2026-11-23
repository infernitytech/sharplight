import json, glob, os, sys
root = sys.argv[1]
rows = {}
for d in sorted(glob.glob(root + '/*_*_*')):
    name = os.path.basename(d)
    for f in sorted(glob.glob(d + '/serve_*_c*_r*.json')):
        j = json.load(open(f)); c = j['concurrency']; r = f.split('_r')[-1].split('.')[0]
        print(f"{name:16s} c{c:<3} r{r}: {j['agg_tok_s']:7.0f} tok/s  ttft p50 {j['ttft_p50_ms']:6.0f}  itl mean {j['itl_mean_ms']:6.2f} p95 {j['itl_p95_ms']:6.1f} p99 {j['itl_p99_ms']:6.1f}  err {j['errors']}")
