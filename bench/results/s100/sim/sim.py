"""Closed-loop serving simulator with a mixed-step cost: decode D(d) + (P>0) * (F + c P). Policies give the step's token budget."""
import random, statistics, sys
def D(d): return 0 if d == 0 else 5.2 + (9.6 - 5.2) * min(d, 16) / 16 if d <= 16 else 9.6 + (16.0 - 9.6) * (d - 16) / 48
def cost(d, p, F=2.4, c=0.051, P0=8.8):
    if p == 0: return D(d) + 0.4
    return (D(d) + F if d else P0) + c * p + 0.4
def run(N, per, policy, prompt=250, out=128, seed=1, **kw):
    rng = random.Random(seed); t = 0.0
    pending = [[0.0, w, prompt + rng.randint(-5, 5)] for w in range(N)]
    active = []; done = [0] * N; ttft = []; itl = []; tokens = 0
    while pending or active:
        for p in [p for p in pending if p[0] <= t]:
            pending.remove(p); active.append(dict(arr=p[0], w=p[1], plen=p[2], pos=0, gen=0, last=None))
        if not active: t = min(p[0] for p in pending); continue
        dec = [s for s in active if s['pos'] >= s['plen']]; pre = [s for s in active if s['pos'] < s['plen']]
        B = policy(len(dec), len(active), pre)
        budget = B - len(dec); chunks = []
        for s in pre:
            if budget <= 0: break
            c = min(budget, s['plen'] - s['pos']); chunks.append((s, c)); budget -= c
        p = sum(c for _, c in chunks)
        t += cost(len(dec), p, **kw)
        for s in dec:
            s['gen'] += 1; tokens += 1; itl.append(t - s['last']); s['last'] = t
        for s, c in chunks:
            s['pos'] += c
            if s['pos'] >= s['plen']: s['gen'] = 1; tokens += 1; ttft.append(t - s['arr']); s['last'] = t
        for s in [s for s in active if s['gen'] >= out]:
            active.remove(s); done[s['w']] += 1
            if done[s['w']] < per: pending.append([t + 0.3, s['w'], prompt + rng.randint(-5, 5)])
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    return tokens / t * 1000, statistics.median(ttft), statistics.mean(itl), q(itl, .95), q(itl, .99)
def waves(per_seq=32, floor=768, full=2048, gran=128):
    def f(d, n, pre):
        if d == 0: return max(floor, min(full, -(-(n * per_seq) // gran) * gran))
        return min(full, max(gran, -(-(max(n * per_seq, n + (pre[0]['plen'] - pre[0]['pos'] if pre else 0))) // gran) * gran))
    return f
POL = {'waves': waves(), 'full': lambda d, n, pre: 2048, 'waves64': waves(64), 'waves16': waves(16)}
if __name__ == '__main__':
    for N, per in ((16, 8), (64, 4)):
        for name, pol in POL.items():
            r = run(N, per, pol)
            print(f"N={N} {name:8s} tok/s {r[0]:6.0f} ttft p50 {r[1]:6.0f} itl mean {r[2]:5.2f} p95 {r[3]:5.1f} p99 {r[4]:5.1f}")
