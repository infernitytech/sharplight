import glob,json,os,sys,re
for d in sys.argv[1:]:
  for p in sorted(glob.glob(d+"/serve_*_c*_r*.json")):
    j=json.load(open(p)); k=os.path.basename(p)[6:-5]
    g=lambda *n: next((j[x] for x in n if x in j),float('nan'))
    print(f"{d.split('/')[-1]:14} {k:40} tok/s {g('agg_tok_s'):7.0f} ttft50 {g('ttft_p50_ms'):5.0f} itl mean {g('itl_mean_ms'):6.2f} p95 {g('itl_p95_ms'):6.1f} p99 {g('itl_p99_ms'):6.1f} max {g('itl_max_ms'):6.1f}")
  for p in glob.glob(d+"/*.textcheck.FAIL"): print("TEXTCHECK FAIL",p)
