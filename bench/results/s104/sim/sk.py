# Mirror of the stream-K / config estimates (GraphLowerer.StreamK.cs, NativeGemm.cs) for checking test cases and shapes.
import math, sys
SMS=82
def div(a,b): return -(-a//b)
def blocks(tt): return 0 if tt<2 or tt>16 or tt>SMS else SMS-SMS%tt
def walk(tiles,tt,ks,sk,cost):
    if not sk: return div(tiles,SMS)*ks
    b=blocks(tt)
    if b==0: return math.inf
    whole= (tiles//b-1)*b if tiles>=2*b else 0
    shared=(tiles-whole)*ks
    if shared<b*8: return math.inf
    return whole//b*ks + div(shared,b)*cost[0] + cost[1]
def use(tiles,tt,ks,cost): return walk(tiles,tt,ks,True,cost)<walk(tiles,tt,ks,False,cost)*0.95
def best(tiles,tt,ks,cost): return walk(tiles,tt,ks,use(tiles,tt,ks,cost),cost)
F8=(1.12,4); F16=(1.0,8)
def fp8(rows,n,k):
    ks=k//128; cols=div(n,128); wide=div(div(rows,64)*cols,SMS); tt=div(rows,128); tall=best(tt*cols,tt,ks,F8)/ks
    if wide<=tall*1.3: return "64-row native"
    return "tma SK" if use(tt*cols,tt,ks,F8) else "tma"
TR=[128,128,0,64]; TC=[256,128,0,256]; SP=[1.0,0.94,0,0.85]
def f16(rows,n,k,dual=False):
    ks=k//64; bw=(None,1e18); bs=(None,1e18)
    for c in (0,1,3):
        cols=TC[c]//2 if dual else TC[c]; tt=div(rows,TR[c]); tiles=tt*div(n,cols); area=TR[c]*TC[c]/SP[c]
        dp=div(tiles,SMS)*area; sk=walk(tiles,tt,ks,True,F16)/ks*area
        if dp<bw[1]: bw=(c,dp)
        if sk<bs[1]: bs=(c,sk)
    return (bs[0],True,round(bs[1])) if bs[1]<bw[1]*0.95 else (bw[0],False,round(bw[1]))
for a in sys.argv[1:]:
    kind,r,n,k=a.split(':'); r,n,k=int(r),int(n),int(k)
    print(a, fp8(r,n,k) if kind=='fp8' else f16(r,n,k,kind=='dual'))
def hybrid_fp8(r,n,k):
    tt=div(r,128); tiles=tt*div(n,128); b=blocks(tt)
    return b>0 and tiles>=2*b and fp8(r,n,k)=="tma SK"
def hybrid_f16(r,n,k):
    c,sk,_=f16(r,n,k)
    if not sk: return False
    tt=div(r,TR[c]); tiles=tt*div(n,TC[c]); b=blocks(tt); return tiles>=2*b
if __name__=='__main__' and len(sys.argv)==1:
    res=[]
    for r in range(130,2100,8):
        for n in range(1024,12000,256):
            for k in (1024,2048,4096):
                if hybrid_f16(r,n,k): res.append((r*n*k,r,n,k,'f16',f16(r,n,k)))
                if n%128==0 and hybrid_fp8(r,n,k): res.append((r*n*k,r,n,k,'fp8'))
    res.sort(); [print(x) for x in res[:8]]
    print([x for x in res if x[4]=='f16'][:5])
