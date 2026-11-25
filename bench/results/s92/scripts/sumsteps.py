import sys,re,collections
d=collections.defaultdict(lambda: collections.defaultdict(list)); cur=None
for l in open(sys.argv[1]):
    m=re.match(r'== (\S+) (\S+) round',l)
    if m: cur=m.group(2); continue
    c=[x.strip() for x in l.split('|')]
    if cur and len(c)>6 and c[1].isdigit() and '(' in c[2]:
        g=float(c[5].split(',')[-1]); d[c[1]+'+'+c[2]][cur].append(g)
forms=list(next(iter(d.values())).keys())
print('step', *forms)
for k,v in d.items(): print(f"{k:16s}", *[ '/'.join(f'{x:.2f}' for x in v[f]) for f in forms], *[f"{min(v[f])/min(v[forms[0]]):.3f}" for f in forms[1:]])
