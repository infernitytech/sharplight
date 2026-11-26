import csv,statistics as st,collections,sys
for f in sys.argv[1:]:
    rows=[r for r in csv.DictReader(open(f)) if r.get('gc2') not in (None,'')]
    rows=rows[len(rows)//2:]
    g=collections.defaultdict(list)
    for r in rows:
        k=('prompt' if int(r['prompt_rows'])>0 else 'decode', 'rec' if r['recorded']=='1' else ('replay' if r['replayed']=='1' else '2nd'))
        g[k].append((float(r['gpu_ms']),float(r['run_ms'])+float(r['wait_ms'])))
    print(f, len(rows))
    for k,v in sorted(g.items()): print('  ',k,len(v),'gpu med %.1f  run+wait med %.1f'%(st.median(x[0] for x in v),st.median(x[1] for x in v)))
    pr=[r for r in rows if int(r['prompt_rows'])>0]
    print('   prompt step rows', collections.Counter(int(r['decode_rows'])+int(r['prompt_rows']) for r in pr).most_common(8))
