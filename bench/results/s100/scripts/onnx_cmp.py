import json, collections, sys
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1]):
    j=json.loads(l)
    if j.get('stage')!='ok': continue
    d[(j['model'],j['batch'])][j['engine']].append((j['minMs'], j.get('gpuMinMs'), j.get('parityOk'), j.get('worstRms')))
print(f"{'model':34s} b  | wall new / base | GPU new / base | change GPU | parity / rms new, base")
for (m,b),e in d.items():
    n=e.get('sl-cuda',[]); o=e.get('sl-cuda@base',[])
    if not n or not o: continue
    wn=min(x[0] for x in n); wo=min(x[0] for x in o); gn=min(x[1] for x in n); go=min(x[1] for x in o)
    print(f"{m:34s} {b} | {wn:.3f} / {wo:.3f} | {gn:.3f} / {go:.3f} | {100*(gn/go-1):+.1f} % | {n[0][2]} {n[0][3]:.3g}, {o[0][3]:.3g}")
