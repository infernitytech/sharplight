#!/usr/bin/env python3
"""GPU busy / idle from an nsys report (.nsys-rep -> sqlite): union of kernel, memcpy, memset and graph-trace intervals on the GPU over the
captured window; prints idle share, gap statistics (gaps >= 2 us) and busy-period lengths."""
import sqlite3, subprocess, sys, os, statistics as st
for rep in sys.argv[1:]:
    db = rep.replace('.nsys-rep', '.sqlite')
    if not os.path.exists(db):
        subprocess.run(['nsys', 'export', '--type', 'sqlite', '--force-overwrite', 'true', '-o', db, rep], check=True, capture_output=True)
    c = sqlite3.connect(db)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    iv = []
    used = []
    for t in ['CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET', 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE']:
        if t in tables:
            rows = c.execute(f'select start, end from {t}').fetchall()
            used.append(f'{t.split("_")[-1].lower()}={len(rows)}')
            iv += rows
    iv.sort()
    if not iv:
        print(rep, 'no GPU activity'); continue
    merged = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    t0, t1 = merged[0][0], merged[-1][1]
    busy = sum(e - s for s, e in merged)
    gaps = [(b[0] - a[1]) / 1e3 for a, b in zip(merged, merged[1:])]
    big = [g for g in gaps if g >= 2]
    print(f'{os.path.basename(rep)}: window {(t1-t0)/1e6:.1f} ms, busy {busy/(t1-t0)*100:.2f} %, idle {(t1-t0-busy)/1e6:.2f} ms; {" ".join(used)}')
    if big:
        q = sorted(big)
        print(f'  gaps >= 2 us: {len(big)}, sum {sum(big)/1e3:.2f} ms, median {st.median(big):.1f} us, p90 {q[int(len(q)*0.9)]:.1f} us, max {q[-1]:.1f} us')
        print(f'  gaps >= 100 us: {sum(1 for g in big if g >= 100)}, sum {sum(g for g in big if g >= 100)/1e3:.2f} ms')
