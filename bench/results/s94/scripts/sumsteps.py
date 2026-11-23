import sys, re, collections
d = collections.defaultdict(list); order = []
cur = None
for line in open(sys.argv[1]):
    m = re.match(r"== (\S+) (\S+) round", line)
    if m: cur = (m.group(1), m.group(2)); continue
    m = re.match(r"\| (\d+) \| (\d+) \((\d+)\) \|.*\|.*\| [\d.]+ / [\d.]+, ([\d.]+) \|", line)
    if m and cur:
        k = (cur[0], m.group(1) + "+" + m.group(2))
        if k not in order: order.append(k)
        d[k + (cur[1],)].append(float(m.group(4)))
cfgs = sorted({k[2] for k in d}, key=lambda c: list(dict.fromkeys(k[2] for k in d)).index(c))
print("model step " + " ".join(cfgs))
for k in order:
    print(k[0], k[1], "  ".join("/".join(f"{v:.2f}" for v in d[k + (c,)]) for c in cfgs))
