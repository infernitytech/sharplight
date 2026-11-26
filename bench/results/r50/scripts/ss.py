import json,sys,collections
d=json.load(open(sys.argv[1])); fr=[f['name'] for f in d['shared']['frames']]
filt=sys.argv[2] if len(sys.argv)>2 else 'PrepareDecodeSteps'
incl=collections.Counter(); selfc=collections.Counter(); tot=0
for p in d['profiles']:
    st=[]; last=None
    for e in p['events']:
        t=e['at']
        if last is not None and st and any(filt in fr[i] for i in st):
            w=t-last; tot+=w
            for n in set(fr[i] for i in st): incl[n]+=w
            selfc[fr[st[-1]]]+=w
        last=t
        if e['type']=='O': st.append(e['frame'])
        else:
            if st: st.pop()
def short(n): return n.split('(')[0][-110:]
print('total ms',tot)
for n,w in incl.most_common(int(sys.argv[3]) if len(sys.argv)>3 else 45): print(f'{w/tot*100:6.1f}% {short(n)}')
print('--- self')
for n,w in selfc.most_common(25): print(f'{w/tot*100:6.1f}% {short(n)}')
