#!/usr/bin/env python3
"""nsys_step.py <rep> <kernels per step> [steps]: per kernel name over the last <steps> steps: incremental time (end - previous end) and
duration per step, and the step span; heaviest first."""
import sqlite3, subprocess, sys, os, collections
rep, per = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
db = rep.replace('.nsys-rep', '.sqlite')
if not os.path.exists(db):
    subprocess.run(['nsys', 'export', '-t', 'sqlite', '-o', db, rep], check=True, capture_output=True)
c = sqlite3.connect(db)
rows = c.execute('select k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName = s.id order by k.start').fetchall()
rows = rows[-per * steps - per:]
inc, dur, cnt, grid = collections.defaultdict(float), collections.defaultdict(float), collections.Counter(), {}
gaps = 0.0
for i in range(1, len(rows)):
    st, en, name, g = rows[i]
    name = name.split('(')[0][:70]
    prev = max(r[1] for r in rows[max(0, i - 4):i])
    inc[name] += max(0, en - prev) / 1e3
    dur[name] += (en - st) / 1e3
    gaps += max(0, st - prev) / 1e3
    cnt[name] += 1
    grid[name] = g
n = (len(rows) - 1) / per
span = (rows[-1][1] - rows[1][0]) / 1e3 / n
print(f"steps {n:.1f}, span per step {span:.1f} us, incremental sum {sum(inc.values()) / n:.1f} us, idle gaps {gaps / n:.1f} us")
for k in sorted(inc, key=lambda k: -inc[k])[:int(os.environ.get('TOP', 40))]:
    print(f"{inc[k] / n:8.1f} us inc {dur[k] / n:8.1f} us dur {cnt[k] / n:5.1f}x grid {grid[k]:6d}  {k}")
