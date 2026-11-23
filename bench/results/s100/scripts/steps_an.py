import sys, re, collections
def ours(f):
    st=[]
    for l in open(f):
        m=re.match(r'\[sched\]\s+([\d.]+) ms rows\s+(\d+) decode\s+(\d+) active\s+(\d+)(.*)', l)
        if m: st.append((float(m[1]), int(m[2]), int(m[3]), len(re.findall(r' P\d+', m[5]))))
    return st
def vllm(f):
    rows=[l.split() for l in open(f)]
    st=[]
    for a,b in zip(rows, rows[1:]):
        st.append(((float(b[0])-float(a[0]))*1000, int(a[1]), int(a[2]), int(a[4])))
    return st
def report(name, st):
    st=st[len(st)//10:]  # skip warmup
    mixed=[s for s in st if s[3]>0]; dec=[s for s in st if s[3]==0]
    tot=sum(s[0] for s in st)
    print(f"{name}: steps {len(st)}, with prompt rows {len(mixed)} ({100*len(mixed)/len(st):.1f}%), time {tot:.0f} ms, in prompt steps {sum(s[0] for s in mixed):.0f} ms")
    h=collections.Counter((s[1]-s[2])//128*128 for s in mixed)
    print("  prompt rows hist (128 buckets):", sorted(h.items()))
    hp=collections.Counter(s[3] for s in mixed); print("  prompts per prompt step:", sorted(hp.items()))
    over=[s for s in mixed if 0 < s[1] % 128 <= 32 and s[1] > 128]; print(f"  steps <=32 rows past a 128 tile: {len(over)}, mean ms {sum(s[0] for s in over)/max(1,len(over)):.1f}")
    import statistics
    if dec: print(f"  decode-only step ms median {statistics.median(s[0] for s in dec):.2f}, mean rows {statistics.mean(s[1] for s in dec):.1f}")
for d,kind in (('gptoss_strace_1/serve_sharplight_gptoss.server.log','o'),('qwen_strace_1/serve_sharplight_qwen.server.log','o')):
    report(d, ours(d))
import glob
for f in glob.glob('*_vtrace_2/steps.*'): report(f, vllm(f))
