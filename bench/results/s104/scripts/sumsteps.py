# sumsteps.py <tag> <labA> <labB>: per model and shape, GPU ms min of the four ABBA runs and the change B vs A (mean of each pair)
import glob, os, re, sys
tag = sys.argv[1]; la = sys.argv[2] if len(sys.argv) > 2 else 'A'; lb = sys.argv[3] if len(sys.argv) > 3 else 'B'
root = '<path>'
models = sorted({re.match(rf'{tag}_(.+)_[AB][1-4]\.txt', os.path.basename(f)).group(1) for f in glob.glob(f'{root}/{tag}_*_[AB][1-4].txt')})
for m in models:
    runs = {}
    for f in sorted(glob.glob(f'{root}/{tag}_{m}_*.txt')):
        lab = os.path.basename(f)[len(tag) + len(m) + 2:-4]
        d = {}
        for line in open(f):
            mm = re.match(r'^\| (\d+) \| (\d+) \(\d+\) \|[^|]*\|[^|]*\| ([\d.]+) / ([\d.]+), ([\d.]+) \|', line)
            if mm:
                d[f'{mm.group(1)}+{mm.group(2)}'] = float(mm.group(5))
        runs[lab] = d
    print(f'| {m} | {la} | {lb} | change |')
    print('|---|---|---|---:|')
    shapes = list(runs.get('A1', {}).keys())
    for s in shapes:
        a = [runs[l][s] for l in ('A1', 'A4') if s in runs.get(l, {})]
        b = [runs[l][s] for l in ('B2', 'B3') if s in runs.get(l, {})]
        if a and b:
            ch = (sum(b) / len(b)) / (sum(a) / len(a)) - 1
            print(f"| {s} | {', '.join(f'{v:.2f}' for v in a)} | {', '.join(f'{v:.2f}' for v in b)} | {ch * 100:+.1f} % |")
    print()
