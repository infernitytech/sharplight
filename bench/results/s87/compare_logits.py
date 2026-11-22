# compare_logits.py <reference.f32> <vocab> <other.f32> ...: last-row logits of float32 dumps (our --logits, lldump.c, the reference script) against a reference.
import sys, numpy as np
ref = np.fromfile(sys.argv[1], dtype=np.float32)
V = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 129280
ref = ref.reshape(-1, V)[-1]
def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
R = lsm(ref.astype(np.float64))
top = np.argsort(-ref)[:20]
for f in sys.argv[3:]:
    x = np.fromfile(f, dtype=np.float32).reshape(-1, V)[-1].astype(np.float64)
    X = lsm(x)
    p = np.exp(R)
    kl = float((p * (R - X)).sum())
    d = x - ref
    print(f"{f.split('/')[-1]:28s} maxabs {np.abs(d).max():.4f} rms {np.sqrt((d - d.mean())**2).mean():.4f} top20 maxabs logprob {np.abs(X[top]-R[top]).max():.4f} KL {kl:.2e} corr {np.corrcoef(x, ref)[0,1]:.6f} argmax {x.argmax()} (ref {ref.argmax()})")
