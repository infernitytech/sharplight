import json, sys, collections
rows = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    j = json.loads(line)
    k = (j['model'], j['batch']); e = j['engine']
    if j.get('stage') != 'ok': rows[k].setdefault(e + '_err', j.get('stage')); continue
    for f, n in (('minMs', 'w'), ('gpuMinMs', 'g')):
        v = j.get(f)
        if v is not None: rows[k][e + n] = min(rows[k].get(e + n, 1e9), v)
    if e.startswith('sl-') and j.get('parityOk') is not None: rows[k][e + 'p'] = rows[k].get(e + 'p', True) and j['parityOk']; rows[k][e + 'r'] = j.get('worstRms')
trt = {}
for line in open(sys.argv[2]):
    j = json.loads(line)
    if j.get('stage') == 'ok': trt[(j['model'], j['batch'])] = min(trt.get((j['model'], j['batch']), 1e9), j['minMs'])
order = []
for line in open(sys.argv[1]):
    j = json.loads(line); k = (j['model'], j['batch'])
    if k not in order: order.append(k)
def f(v, d=2): return '-' if v is None else f'{v:.{d}f}'
def pct(a, b): return '-' if a is None or b is None else f'{(a / b - 1) * 100:+.0f} %'
def x(a, b): return '-' if a is None or b is None else f'{a / b:.2f}x'
print('| model | batch | strict | `develop` | change | GPU strict r39 / `develop` | fast | `develop` | change | ORT TF32 | ORT strict | ORT TensorRT (§98) | fast / TF32 | strict / strict | fast / TRT | parity |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for k in order:
    r = rows[k]; t = trt.get(k)
    s, sb, fa, fb = r.get('sl-cudaw'), r.get('sl-cuda@basew'), r.get('sl-cuda-fastw'), r.get('sl-cuda-fast@basew')
    g, gb = r.get('sl-cudag'), r.get('sl-cuda@baseg')
    par = 'ok' if r.get('sl-cudap') else f"rms {r.get('sl-cudar')}"
    print(f"| {k[0]} | {k[1]} | {f(s)} | {f(sb)} | {pct(s, sb)} | {f(g)} / {f(gb)} ({pct(g, gb)}) | {f(fa)} | {f(fb)} | {pct(fa, fb)} | {f(r.get('ort-cudaw'))} | {f(r.get('ort-cuda-strictw'))} | {f(t)} | {x(fa, r.get('ort-cudaw'))} | {x(s, r.get('ort-cuda-strictw'))} | {x(fa, t)} | {par} |")
