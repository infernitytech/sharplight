using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// c[..., M, N] = alpha * a[..., M, K] · b[..., K, N] on strided f32 views (trans flags are expressed as strides) with
/// broadcast batch dims. Goto-style: per task (batch, M block, N block) the operands are packed into MR/NR panels
/// (thread-local scratch, K blocked by <see cref="Gemm.KC"/>) and fed to the micro-kernel.
/// </summary>
internal sealed unsafe class BatchMatMulOp : CpuOp
{
    private readonly TRef _a, _b, _y;
    private readonly bool _ta, _tb; private readonly float _alpha;
    private readonly Action<int> _body;
    [ThreadStatic] private static float[]? s_scratch;

    // per-run state
    private int _batchRank; private long[] _bdims = [], _sa = [], _sb = [];
    private long _m, _n, _k, _batch, _asm, _ask, _bsk, _bsn;
    private int _mpt, _ntt, _mBlocks, _nBlocks;

    public override string Name => "BatchMatMul";

    public BatchMatMulOp(TRef a, TRef b, TRef y, bool transA, bool transB, float alpha)
    {
        _a = a; _b = b; _y = y; _ta = transA; _tb = transB; _alpha = alpha; _body = Body;
    }

    private static long[] AlignedBatchStrides(TRef t, long[] od, RunParameters p)
    {
        int br = od.Length, tr = t.Rank - 2;
        var s = new long[br];
        for (int k = 0; k < br; k++)
        {
            int ik = k - (br - tr);
            if (ik < 0)
            {
                continue;
            }

            long d = t.Dim(ik, p);
            s[k] = d == 1 && od[k] != 1 ? 0 : t.Strides[ik];
        }
        return s;
    }

    public override void Run(RunParameters p)
    {
        int ra = _a.Rank, rb = _b.Rank;
        _batchRank = Math.Max(ra, rb) - 2;
        _bdims = new long[_batchRank];
        for (int k = 0; k < _batchRank; k++)
        {
            int ia = k - (_batchRank - (ra - 2)), ib = k - (_batchRank - (rb - 2));
            long da = ia >= 0 ? _a.Dim(ia, p) : 1, db = ib >= 0 ? _b.Dim(ib, p) : 1;
            _bdims[k] = Math.Max(da, db);
        }
        _sa = AlignedBatchStrides(_a, _bdims, p); _sb = AlignedBatchStrides(_b, _bdims, p);
        long ar = _a.Dim(ra - 2, p), ac = _a.Dim(ra - 1, p), br2 = _b.Dim(rb - 2, p), bc = _b.Dim(rb - 1, p);
        _m = _ta ? ac : ar; _k = _ta ? ar : ac; _n = _tb ? br2 : bc;
        long ssm = _a.Strides[ra - 2], ssk = _a.Strides[ra - 1];
        _asm = _ta ? ssk : ssm; _ask = _ta ? ssm : ssk;
        long bs0 = _b.Strides[rb - 2], bs1 = _b.Strides[rb - 1];
        _bsk = _tb ? bs1 : bs0; _bsn = _tb ? bs0 : bs1;
        _batch = 1; foreach (var d in _bdims)
        {
            _batch *= d;
        }

        if (_batch * _m * _n == 0)
        {
            return;
        }

        if (_k == 0) { new Span<float>(_y.P, (int)(_batch * _m * _n)).Clear(); return; }

        int MR = Gemm.MR, NR = Gemm.NR;
        int mp = (int)((_m + MR - 1) / MR), nt = (int)((_n + NR - 1) / NR);
        int mpt = Math.Min(mp, 8), ntt = Math.Min(nt, 4);
        long target = 2L * Par.Threads;
        while (_batch * ((mp + mpt - 1) / mpt) * (long)((nt + ntt - 1) / ntt) < target && (mpt > 1 || ntt > 1))
        {
            if (mpt >= ntt && mpt > 1)
            {
                mpt = (mpt + 1) / 2;
            }
            else
            {
                ntt = (ntt + 1) / 2;
            }
        }
        _mpt = mpt; _ntt = ntt;
        _mBlocks = (mp + mpt - 1) / mpt; _nBlocks = (nt + ntt - 1) / ntt;
        long tasks = _batch * _mBlocks * _nBlocks;
        Par.For((int)tasks, _body);
    }

    private void Body(int task)
    {
        int MR = Gemm.MR, NR = Gemm.NR, KC = Gemm.KC;
        int nb = task % _nBlocks; int rest = task / _nBlocks;
        int mb = rest % _mBlocks; long bi = rest / _mBlocks;
        // batch offsets
        long ao = 0, bo = 0, rem = bi;
        for (int k = _batchRank - 1; k >= 0; k--)
        {
            long d = _bdims[k]; long ix = rem % d; rem /= d;
            ao += ix * _sa[k]; bo += ix * _sb[k];
        }
        float* a = (float*)_a.P + ao; float* b = (float*)_b.P + bo;
        float* c = (float*)_y.P + bi * _m * _n;

        int need = _mpt * MR * KC + _ntt * NR * KC;
        var sc = s_scratch;
        if (sc is null || sc.Length < need)
        {
            s_scratch = sc = GC.AllocateUninitializedArray<float>(Math.Max(need, 1 << 16), pinned: true);
        }

        fixed (float* scp = sc)
        {
            float* apack = scp; float* bpack = scp + _mpt * MR * KC;
            long m0 = (long)mb * _mpt * MR, m1 = Math.Min(_m, m0 + (long)_mpt * MR);
            long n0 = (long)nb * _ntt * NR, n1 = Math.Min(_n, n0 + (long)_ntt * NR);
            int np = (int)((m1 - m0 + MR - 1) / MR), ntl = (int)((n1 - n0 + NR - 1) / NR);
            for (long kb = 0; kb < _k; kb += KC)
            {
                int kc = (int)Math.Min(KC, _k - kb);
                for (int pi = 0; pi < np; pi++)
                {
                    long r0 = m0 + (long)pi * MR;
                    Gemm.PackA(apack + (long)pi * MR * KC, a + r0 * _asm + kb * _ask, _asm, _ask, (int)Math.Min(MR, m1 - r0), kc, _alpha);
                }
                for (int ti = 0; ti < ntl; ti++)
                {
                    long c0 = n0 + (long)ti * NR;
                    Gemm.PackB(bpack + (long)ti * NR * KC, b + kb * _bsk + c0 * _bsn, _bsk, _bsn, kc, (int)Math.Min(NR, n1 - c0));
                }
                for (int ti = 0; ti < ntl; ti++)
                {
                    long c0 = n0 + (long)ti * NR; int cols = (int)Math.Min(NR, n1 - c0);
                    for (int pi = 0; pi < np; pi++)
                    {
                        long r0 = m0 + (long)pi * MR; int rows = (int)Math.Min(MR, m1 - r0);
                        float* ap = apack + (long)pi * MR * KC, bp = bpack + (long)ti * NR * KC;
                        float* cp = c + r0 * _n + c0;
                        if (rows == MR && cols == NR)
                        {
                            Gemm.Tile(ap, bp, NR, cp, _n, kc, null, kb > 0);
                        }
                        else
                        {
                            Gemm.TileEdge(ap, bp, NR, cp, _n, kc, null, kb > 0, rows, cols);
                        }
                    }
                }
            }
        }
    }
}
