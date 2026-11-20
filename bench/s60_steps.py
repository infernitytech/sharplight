#!/usr/bin/env python3
# Summarizes a serving step profile (SHARPLIGHT_STEP_PROFILE=<file>, docs/benchmarks.md section 60): per step kind the period between
# step starts, device time, idle time and the scheduler-thread phases; a fit of device ms over prompt and decode rows.
# usage: python3 s60_steps.py <steps.csv>   (needs numpy for the fit)
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
f = lambda r, k: float(r[k])
# period = time to next step start
for i in range(len(rows) - 1):
    rows[i]['period'] = f(rows[i+1], 'start_ms') - f(rows[i], 'start_ms')
rows = rows[:-1]
rows = [r for r in rows if r['period'] < 1000]  # drop idle gaps between rounds
def summ(sel, name):
    if not sel: return
    n = len(sel); m = lambda k: sum(f(r, k) if k != 'period' else r['period'] for r in sel) / n
    gpu = m('gpu_ms'); per = m('period')
    print(f"{name}: steps {n}  decode {m('decode_rows'):.1f} prompt {m('prompt_rows'):.1f}  period {per:.3f}  gpu {gpu:.3f}  idle {per-gpu:.3f} ms"
          f" | admit {m('admit_ms'):.3f} sched {m('schedule_ms'):.3f} pack {m('pack_ms'):.3f} upload {m('upload_ms'):.3f} run {m('run_ms'):.3f}"
          f" (record {m('record_ms'):.3f} submit {m('submit_ms'):.3f}) overlap {m('overlap_ms'):.3f} read {m('read_ms'):.3f} (wait {m('wait_ms'):.3f}) process {m('process_ms'):.3f}"
          f" | recorded {sum(int(r['recorded']) for r in sel)} replayed {sum(int(r['replayed']) for r in sel)}")
dec = [r for r in rows if int(r['prompt_rows']) == 0]
mix = [r for r in rows if int(r['prompt_rows']) > 0]
summ(rows, 'all'); summ(dec, 'decode-only'); summ(mix, 'mixed')
full = [r for r in dec if int(r['decode_rows']) >= 60]
summ(full, 'decode>=60')
# regression of gpu ms on prompt rows among steps with decode rows >= 48
import itertools
sel = [r for r in rows if int(r['decode_rows']) >= 48]
if len(sel) > 10:
    xs = [f(r,'prompt_rows') for r in sel]; ys = [f(r,'gpu_ms') - 0.0 for r in sel]; ds=[f(r,'decode_rows') for r in sel]
    # two-variable least squares y = a + b*prompt + c*decode
    import numpy as np
    A = np.array([[1, x, d] for x, d in zip(xs, ds)]); y = np.array(ys)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"gpu ms ~ {c[0]:.2f} + {c[1]*1000:.1f} us/prompt row + {c[2]*1000:.1f} us/decode row (steps with >=48 decode rows)")
tot = sum(r['period'] for r in rows); g = sum(f(r,'gpu_ms') for r in rows)
pm = sum(f(r,'gpu_ms') for r in mix)
print(f"total {tot/1000:.1f} s, gpu {g/1000:.1f} s ({100*g/tot:.1f}%), mixed-step gpu {pm/1000:.1f} s; prompt rows {sum(int(r['prompt_rows']) for r in rows)}, decode rows {sum(int(r['decode_rows']) for r in rows)}")
