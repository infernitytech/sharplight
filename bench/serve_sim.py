"""Closed-loop serving simulator of docs/benchmarks.md section 50.1.

Step time is interpolated from BatchBench step times by rows (gpt-oss-20b, section 49.5; the qwen entry is a rough guess),
plus a fixed host overhead. The engine schedules decode rows first and fills the step budget with prompt chunks (oldest
first); the load generator is the closed loop of bench/loadgen.py (N workers, ~250-token prompts, 128 output tokens).
Prints throughput and TTFT for fixed step budgets; run() takes any budget policy f(decoding, waiting prompt rows, prompts)."""
import bisect, random, statistics, sys

# gpt-oss-20b step GPU ms by total rows (BatchBench s49) + host overhead
PTS = {'gptoss': [(1, 6.5), (16, 10.0), (64, 17.1), (171, 28.8), (512, 51.5), (2048, 170.0), (4096, 340.0), (8192, 680.0)],
       # Qwen3-8B-FP8: rough (ITL 15 at 16 streams, 18 at 64); prompt per row ~0.06 ms (to be calibrated)
       'qwen': [(1, 13.0), (16, 14.0), (64, 17.5), (171, 26.0), (512, 45.0), (2048, 150.0), (4096, 300.0)]}
OVH = 0.7


def cost(model, n):
    p = PTS[model]
    for i in range(1, len(p)):
        if n <= p[i][0]:
            (x0, y0), (x1, y1) = p[i - 1], p[i]
            return y0 + (y1 - y0) * (n - x0) / (x1 - x0) + OVH
    return p[-1][1] + OVH


def run(model, N, per, policy, prompt=250, out=128, seed=1):
    rng = random.Random(seed)
    t = 0.0
    pending = []  # (arrival, worker, plen)
    for w in range(N):
        pending.append([0.0, w, prompt + rng.randint(-5, 5)])
    active = []  # dict: arr, pos, plen, gen
    done_w = [0] * N
    ttft = []
    tokens = 0
    steps = []
    while pending or active:
        # admit arrivals up to now
        arrived = [p for p in pending if p[0] <= t]
        if not arrived and not active:
            t = min(p[0] for p in pending)
            continue
        for p in arrived:
            pending.remove(p)
            active.append({'arr': p[0], 'w': p[1], 'plen': p[2], 'pos': 0, 'gen': 0})
        dec = [s for s in active if s['pos'] >= s['plen']]
        pre = [s for s in active if s['pos'] < s['plen']]
        waiting = sum(s['plen'] - s['pos'] for s in pre)
        B = policy(len(dec), waiting, len(pre))
        rows = len(dec)
        budget = B - rows
        chunks = []
        for s in pre:
            if budget <= 0:
                break
            c = min(budget, s['plen'] - s['pos'])
            chunks.append((s, c))
            budget -= c
            rows += c
        dt = cost(model, rows)
        t += dt
        steps.append(rows)
        for s in dec:
            s['gen'] += 1
            tokens += 1
        for s, c in chunks:
            s['pos'] += c
            if s['pos'] >= s['plen']:
                s['gen'] = 1
                tokens += 1
                ttft.append(t - s['arr'])
        for s in [s for s in active if s['gen'] >= out]:
            active.remove(s)
            done_w[s['w']] += 1
            if done_w[s['w']] < per:
                pending.append([t + 0.5, s['w'], prompt + rng.randint(-5, 5)])
    return tokens / t * 1000, statistics.median(ttft), sorted(ttft)[int(0.95 * len(ttft))], len(steps)


def fixed(b):
    return lambda d, w, p: b


POL = {
    'b512': fixed(512), 'b768': fixed(768), 'b1024': fixed(1024), 'b1536': fixed(1536), 'b2048': fixed(2048),
}
if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'gptoss'
    for N, per in ((16, 4), (64, 3)):
        for name, pol in POL.items():
            r = run(model, N, per, pol)
            print(f'{model} N={N} {name:8s} tok/s {r[0]:7.0f} ttft p50 {r[1]:6.0f} p95 {r[2]:6.0f} steps {r[3]}')
