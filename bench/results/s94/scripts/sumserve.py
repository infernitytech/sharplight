import json,sys,glob,os
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d+'/**/*_c*_r*.json',recursive=True)):
        x=json.load(open(f))
        print(f"{os.path.relpath(f,d):60s} tok/s {x['agg_tok_s']:7.1f} ttft50 {x['ttft_p50_ms']:6.1f} itl mean {x['itl_mean_ms']:5.2f} p50 {x['itl_p50_ms']:5.2f} p95 {x['itl_p95_ms']:5.1f} p99 {x['itl_p99_ms']:5.1f} max {x['itl_max_ms']:6.1f} ttft95 {x['ttft_p95_ms']:6.1f} {x.get('arrival','closed')}")
