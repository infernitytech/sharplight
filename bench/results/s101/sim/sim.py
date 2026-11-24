"""Closed-loop serving simulator of the Waves scheduler with KV-block-aligned prompt cuts (as ClipPrefillChunk) and a tail policy.
cost(d, p): decode-only D(d); prompt-bearing steps take a fixed prompt overhead plus a per-row or per-token-tile cost (model table)."""
import random, statistics, sys, math

MODELS = {
    # decode-only ms at d rows (linear between 16 and 64), prompt step = base(d) + per tile / per row
    'qwen': dict(dec=lambda d: 0 if d == 0 else 11.0 + 1.2 * min(d, 16) / 16 + 3.4 * max(0, d - 16) / 48, tile=None),
    'gptoss': dict(dec=lambda d: 0 if d == 0 else 5.2 + 3.9 * min(d, 16) / 16 + 7.0 * max(0, d - 16) / 48, tile=None),
}
TAB = {}  # model -> callable(d, p) ms for prompt-bearing steps, filled from measurements


def run(N, per, cost, rows_per_slot=48, gran=128, budget_max=2048, floor=768, bs=16, tail=None, prompt=250, out=128, seed=1):
    rng = random.Random(seed); t = 0.0
    pending = [[0.0, w, prompt] for w in range(N)]
    active = []; done = [0] * N; ttft = []; itl = []; tokens = 0; steps = 0; psteps = 0
    while pending or active:
        for p in [p for p in pending if p[0] <= t]:
            pending.remove(p); active.append(dict(arr=p[0], w=p[1], plen=p[2], pos=0, gen=0, last=None))
        if not active: t = min(p[0] for p in pending); continue
        dec = [s for s in active if s['pos'] >= s['plen']]; pre = [s for s in active if s['pos'] < s['plen']]
        n = len(active)
        oldest = pre[0]['plen'] - pre[0]['pos'] if pre else 0
        rows = max(n * rows_per_slot, n + oldest)
        rows = -(-rows // gran) * gran
        B = min(budget_max, max(min(gran, budget_max) if dec else floor, rows))
        budget = B - len(dec); chunks = []; used = len(dec)
        for i, s in enumerate(pre):
            if budget <= 0: break
            rest = s['plen'] - s['pos']
            c = min(budget, rest)
            if c < rest and c > bs: c = (s['pos'] + c) // bs * bs - s['pos']
            if c < rest and tail is not None:
                c = tail(len(dec), used, c, rest, others=len(pre) - i - 1 + len(pending), budget_max=budget_max)
            if c <= 0: continue
            chunks.append((s, c)); budget -= c; used += c
        p = sum(c for _, c in chunks)
        t += cost(len(dec), p) + 0.3; steps += 1; psteps += p > 0
        for s in dec:
            s['gen'] += 1; tokens += 1; itl.append(t - s['last']); s['last'] = t
        for s, c in chunks:
            s['pos'] += c
            if s['pos'] >= s['plen']: s['gen'] = 1; tokens += 1; ttft.append(t - s['arr']); s['last'] = t
        for s in [s for s in active if s['gen'] >= out]:
            active.remove(s); done[s['w']] += 1
            if done[s['w']] < per: pending.append([t + 0.3, s['w'], prompt + rng.randint(-5, 5)])
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    return tokens / t * 1000, statistics.median(ttft), statistics.mean(itl), q(itl, .95), q(itl, .99), psteps, steps


def absorb(maxleft=16, tilecost=None):
    """Take the whole rest when what would be left is at most maxleft rows and no other prompt is waiting."""
    def f(d, used, c, rest, others, budget_max):
        left = rest - c
        if others == 0 and left <= maxleft and used + rest <= budget_max:
            return rest
        return c
    return f


def qwen_cost(d, p):
    dec = MODELS['qwen']['dec'](d)
    if p == 0: return dec
    r = d + p; tiles = -(-r // 128)
    return max(dec + 4.8, {1: 17.3, 2: 17.8}.get(tiles, 17.8 + 9.0 * (tiles - 2)) + 0.05 * max(0, d - 16))


def oss_cost(d, p):
    dec = MODELS['gptoss']['dec'](d)
    if p == 0: return dec
    return (dec + 2.6 if d else 8.8) + 0.051 * p


if __name__ == '__main__':
    for mname, cost in (('qwen', qwen_cost), ('gptoss', oss_cost)):
        for N, per in ((16, 8), (64, 4)):
            for rps in (32, 48):
                for tname, tl in (('cut', None), ('absorb16', absorb(16))):
                    r = run(N, per, cost, rows_per_slot=rps, tail=tl)
                    print(f"{mname:6s} N={N} rps={rps} {tname:8s} tok/s {r[0]:6.0f} ttft p50 {r[1]:6.0f} itl mean {r[2]:5.2f} p95 {r[3]:5.1f} p99 {r[4]:5.1f} prompt steps {r[5]}/{r[6]}")
