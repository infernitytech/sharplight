using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Transposed 2-D convolution as GEMM (W^T · X into a column buffer) followed by col2im accumulation.</summary>
internal sealed unsafe class ConvTransposeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly int _c;
    private readonly int _h;
    private readonly int _w;
    private readonly int _o;
    private readonly int _kh;
    private readonly int _kw;
    private readonly int _sh;
    private readonly int _sw;
    private readonly int _pt;
    private readonly int _pl;
    private readonly int _dh;
    private readonly int _dw;
    private readonly int _g;
    private readonly int _ho;
    private readonly int _wo;
    private readonly int _cg;
    private readonly int _og;
    private readonly int _m;
    private readonly int _hw;
    private readonly int _mp;
    private readonly int _nt;
    private readonly float* _apack;
    private readonly float* _col;
    private readonly Action<int> _gemmBody;
    private readonly Action<int> _colBody;
    [ThreadStatic] private static float[]? s_scratch;
    private long _n;
    private int _img;
    private int _grp;
    private int _mpt;
    private float* _xg;
    private float* _bias0;
    private long _per;
    public override string Name => "ConvTranspose";

    public ConvTransposeOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int sh, int sw, int pt, int pl, int dh, int dw)
    {
        _x = x;
        _y = y;
        _bias = bias;
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _o = (int)y.Dims[1];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _kh = kh;
        _kw = kw;
        _g = groups;
        _sh = sh;
        _sw = sw;
        _pt = pt;
        _pl = pl;
        _dh = dh;
        _dw = dw;
        _cg = _c / _g;
        _og = _o / _g;
        _m = _og * kh * kw;
        _hw = _h * _w;
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        _mp = (_m + MR - 1) / MR;
        _nt = (_hw + NR - 1) / NR;
        _apack = (float*)Mem.Alloc((long)_g * _mp * _cg * MR * 4, true);
        for (int g = 0; g < _g; g++)
        {
            for (int p = 0; p < _mp; p++)
            {
                int m0 = p * MR;
                int rows = Math.Min(MR, _m - m0);
                Gemm.PackA(_apack + ((long)(g * _mp + p) * _cg) * MR, wDense + (long)g * _cg * _m + m0, 1, _m, rows, _cg, 1f);
            }
        }

        _col = (float*)Mem.Alloc((long)_m * _hw * 4, false);
        _gemmBody = GemmBody;
        _colBody = ColBody;
    }

    public override void Dispose()
    {
        Mem.Free((byte*)_apack);
        Mem.Free((byte*)_col);
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _bias0 = _bias is null ? null : (float*)_bias.P;
        for (int img = 0; img < _n; img++)
        {
            for (int g = 0; g < _g; g++)
            {
                _img = img;
                _grp = g;
                _xg = (float*)_x.P + ((long)img * _c + (long)g * _cg) * _hw;
                int mpt = Math.Min(_mp, 16);
                long target = 4L * CpuParallel.Threads;
                while (((_mp + mpt - 1) / mpt) * (long)_nt < target && mpt > 1)
                {
                    mpt = (mpt + 1) / 2;
                }

                _mpt = mpt;
                int mBlocks = (_mp + mpt - 1) / mpt;
                CpuParallel.For(mBlocks * _nt, _gemmBody);
                int chunks = (int)Math.Min(_og, CpuParallel.Chunks((long)_og * _kh * _kw * _hw, 16384));
                _per = (_og + chunks - 1) / chunks;
                CpuParallel.For(chunks, _colBody);
            }
        }
    }

    private void GemmBody(int task)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        int KC = Gemm.KC;
        int mb = task / _nt;
        int t = task % _nt;
        int p0 = mb * _mpt;
        int p1 = Math.Min(_mp, p0 + _mpt);
        var sc = s_scratch;
        if (sc is null || sc.Length < KC * NR)
        {
            s_scratch = sc = GC.AllocateUninitializedArray<float>(KC * NR, pinned: true);
        }

        long n0 = (long)t * NR;
        int cols = (int)Math.Min(NR, _hw - n0);
        fixed (float* bp = sc)
        {
            for (int kb = 0; kb < _cg; kb += KC)
            {
                int kc = Math.Min(KC, _cg - kb);
                Gemm.PackB(bp, _xg + (long)kb * _hw + n0, _hw, 1, kc, cols);
                for (int p = p0; p < p1; p++)
                {
                    float* a = _apack + ((long)(_grp * _mp + p) * _cg + kb) * MR;
                    float* c = _col + (long)p * MR * _hw + n0;
                    int rows = Math.Min(MR, _m - p * MR);
                    if (rows == MR && cols == NR)
                    {
                        Gemm.Tile(a, bp, NR, c, _hw, kc, null, kb > 0);
                    }
                    else
                    {
                        Gemm.TileEdge(a, bp, NR, c, _hw, kc, null, kb > 0, rows, cols);
                    }
                }
            }
        }
    }

    private void ColBody(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_og, s + _per);
        for (long o = s; o < e; o++)
        {
            long oc = (long)_grp * _og + o;
            float* yp = (float*)_y.P + ((long)_img * _o + oc) * _ho * _wo;
            new Span<float>(yp, _ho * _wo).Fill(_bias0 == null ? 0f : _bias0[oc]);
            for (int kh = 0; kh < _kh; kh++)
            {
                for (int kw = 0; kw < _kw; kw++)
                {
                    float* crow = _col + ((o * _kh + kh) * _kw + kw) * _hw;
                    int off = kw * _dw - _pl;
                    for (int ih = 0; ih < _h; ih++)
                    {
                        int oh = ih * _sh - _pt + kh * _dh;
                        if ((uint)oh >= (uint)_ho)
                        {
                            continue;
                        }

                        float* yr = yp + (long)oh * _wo;
                        float* cr = crow + (long)ih * _w;
                        if (_sw == 1)
                        {
                            int lo = Math.Max(0, -off);
                            int hi = Math.Min(_w, _wo - off);
                            if (hi > lo)
                            {
                                Simd.Axpy(yr + lo + off, cr + lo, 1f, hi - lo);
                            }
                        }
                        else
                        {
                            for (int iw = 0; iw < _w; iw++)
                            {
                                int ow = iw * _sw + off;
                                if ((uint)ow < (uint)_wo)
                                {
                                    yr[ow] += cr[iw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
