import csv,sys,statistics as st
for f in sys.argv[1:]:
    rows=list(csv.DictReader(open(f)))
    rows=rows[len(rows)//2:]
    dec=[]
    for a,b in zip(rows,rows[1:]):
        if int(a['prompt_rows'])>0 or a['replayed']!='1': continue
        gap=float(b['start_ms'])-float(a['start_ms'])
        d={k:float(a[k]) for k in ['admit_ms','schedule_ms','pack_ms','upload_ms','run_ms','gpu_ms','submit_ms','wait_ms','overlap_ms','read_ms','process_ms','gc_pause_ms']}
        d['step']=gap; d['idle']=gap-d['gpu_ms']; d['rows']=int(a['decode_rows'])
        dec.append(d)
    print(f,len(dec),'decode replayed steps; medians:')
    print('  '+' '.join(f"{k}={st.median(x[k] for x in dec):.3f}" for k in dec[0]))
