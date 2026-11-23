#!/usr/bin/env python3
"""nsys_cmp.py a.nsys-rep b.nsys-rep [...]: per kernel name, the mean 'end - previous end' (incremental GPU time in the stream) and the
mean duration (start to end) over the kernels of the last half of the trace; prints the names whose incremental time differs most."""
import sqlite3, subprocess, sys, os, collections
def load(rep):
    db = rep.replace('.nsys-rep', '.sqlite')
    if not os.path.exists(db):
        subprocess.run(['nsys', 'export', '-t', 'sqlite', '-o', db, rep], check=True, capture_output=True)
    c = sqlite3.connect(db)
    rows = c.execute('select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName = s.id order by k.start').fetchall()
    rows = rows[len(rows) // 2:]
    inc, dur, cnt = collections.defaultdict(float), collections.defaultdict(float), collections.Counter()
    prev_end = None
    for st, en, name in rows:
        if prev_end is not None and en - prev_end < 1e6:
            inc[name] += (en - max(prev_end, 0)) / 1e3
            dur[name] += (en - st) / 1e3
            cnt[name] += 1
        prev_end = max(prev_end or 0, en)
    return inc, dur, cnt
data = [load(r) for r in sys.argv[1:]]
names = set().union(*[d[2].keys() for d in data])
tot = [sum(d[0].values()) / max(1, sum(d[2].values())) for d in data]
rows = []
for n in names:
    c0 = data[0][2][n]
    if c0 == 0: continue
    incs = [d[0][n] / max(1, d[2][n]) * (c0 / max(1, d[2][n])) * max(1, d[2][n]) / c0 for d in data]
    incs = [d[0][n] / c0 for d in data]   # per launch of the first trace, totals comparable when counts are equal
    rows.append((incs[1] - incs[0] if len(incs) > 1 else 0, n, incs, [d[1][n] / max(1, d[2][n]) for d in data], [d[2][n] for d in data]))
rows.sort(key=lambda r: -abs(r[0]))
print('kernels', [sum(d[2].values()) for d in data], 'incremental us total', [round(sum(d[0].values()), 1) for d in data])
for diff, n, incs, durs, cnts in rows[:int(os.environ.get('TOP', 25))]:
    print(f"{n[:60]:60s} cnt {cnts} inc/launch {['%.2f' % x for x in incs]} dur {['%.2f' % x for x in durs]}")
