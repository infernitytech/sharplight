using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D convolution. Grouped/dense convs run as GEMM per (image, group): the weights are packed once into
/// MR-row panels (A), the input is lowered by im2col into NR-column panels (B, [tile][k][NR], built in parallel;
/// 1x1/stride-1 convs read the input in place) and the register-blocked micro-kernel of <see cref="Gemm"/> produces the
/// output directly in NCHW (C = [O, HO*WO]). Depthwise convs (groups == C) use a dedicated direct kernel.
/// </summary>
internal sealed unsafe class ConvOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly OutputEpilogue? _epilogue;
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
    private readonly int _mg;
    private readonly int _kg;
    private readonly int _hw;
    private readonly int _mp;
    private readonly int _nt;
    private readonly bool _depthwise;
    private readonly bool _direct;
    private readonly bool _tilePack;
    private readonly float* _apack;     // packed weights [g][panel][k][MR]
    private readonly float* _wdw;       // depthwise: dense f32 weights [O, KH*KW]
    private readonly float* _bpack;     // packed im2col [tile][k][NR]
    private readonly Action<int> _packBody;
    private readonly Action<int> _gemmBody;
    private readonly Action<int> _dwBody;
    private readonly Action<int> _tileBody;

    // per-run state
    private long _n;
    private int _img;
    private int _grp;
    private int _kSplit;
    private int _chunkC;
    private int _mpt;
    private int _ntt;
    private int _nBlocks;
    private int _mBlocks;
    private float* _xg;
    private float* _yg;
    private float* _bias0;
    private long _planes;
    private long _perChunk;

    public override string Name => (_depthwise ? "Conv.depthwise" : _direct ? "Conv.1x1" : "Conv") + _epilogue?.Label;

    private ConvOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int sh, int sw, int pt, int pl, int dh, int dw, OutputEpilogue? epilogue)
    {
        _x = x;
        _y = y;
        _bias = bias;
        _epilogue = epilogue;
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
        _mg = _o / _g;
        _kg = _cg * kh * kw;
        _hw = _ho * _wo;
        _packBody = PackBody;
        _gemmBody = GemmBody;
        _dwBody = DwBody;
        _tileBody = TileBody;
        _depthwise = _g == _c && _cg == 1;
        if (_depthwise)
        {
            long n = (long)_o * kh * kw;
            _wdw = (float*)Mem.Alloc(n * 4, false);
            new ReadOnlySpan<float>(wDense, (int)n).CopyTo(new Span<float>(_wdw, (int)n));
            return;
        }
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        _mp = (_mg + MR - 1) / MR;
        _nt = (_hw + NR - 1) / NR;
        _direct = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && _hw == _h * _w && ((long)_hw * 4) % 4096 != 0;
        // Small weights and K (early high-resolution layers): every task packs its own B tiles into thread scratch and runs all
        // weight panels over them, so the im2col buffer never leaves the cache.
        _tilePack = !_direct && (long)_mg * _kg * 4 <= TilePackMaxWeightBytes && _kg <= TilePackMaxK;
        long aBytes = (long)_g * _mp * _kg * MR * 4;
        _apack = (float*)Mem.Alloc(aBytes, true);
        for (int g = 0; g < _g; g++)
        {
            for (int p = 0; p < _mp; p++)
            {
                int r0 = g * _mg + p * MR;
                int rows = Math.Min(MR, _mg - p * MR);
                Gemm.PackA(_apack + ((long)(g * _mp + p) * _kg) * MR, wDense + (long)r0 * _kg, _kg, 1, rows, _kg, 1f);
            }
        }

        long tiles = _direct ? 1 : _nt;
        _bpack = _tilePack ? null : (float*)Mem.Alloc(tiles * _kg * NR * 4, false);
    }

    public static ConvOp Create(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int[] strides, int[] pads, int[] dil, OutputEpilogue? epilogue)
        => new(x, wDense, kh, kw, bias, y, groups, strides[0], strides[1], pads[0], pads[1], dil[0], dil[1], epilogue);

    public override void Dispose()
    {
        Mem.Free((byte*)_apack);
        Mem.Free((byte*)_wdw);
        Mem.Free((byte*)_bpack);
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        if (_n == 0 || _hw == 0)
        {
            return;
        }

        _bias0 = _bias is null ? null : (float*)_bias.P;
        if (_depthwise)
        {
            _planes = _n * _o;
            long work = (long)_hw * _kh * _kw;
            int chunks = (int)Math.Min(_planes, CpuParallel.Chunks(_planes * work, 16384));
            _perChunk = (_planes + chunks - 1) / chunks;
            CpuParallel.For(chunks, _dwBody);
            return;
        }
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        for (int img = 0; img < _n; img++)
        {
            for (int g = 0; g < _g; g++)
            {
                _img = img;
                _grp = g;
                _xg = (float*)_x.P + ((long)img * _c + (long)g * _cg) * _h * _w;
                _yg = (float*)_y.P + ((long)img * _o + (long)g * _mg) * _hw;
                if (_tilePack)
                {
                    _ntt = Math.Max(1, (int)Math.Min(TilePackMaxTilesPerTask, _nt / (4L * CpuParallel.Threads)));
                    _nBlocks = (_nt + _ntt - 1) / _ntt;
                    CpuParallel.For(_nBlocks, _tileBody);
                    continue;
                }

                // phase 1: im2col into packed panels (only the partial tail tile when the input is read in place)
                bool tail = _hw % NR != 0;
                if (!_direct || tail)
                {
                    int tiles = _direct ? 1 : _nt;
                    int ks = Math.Clamp((2 * CpuParallel.Threads + tiles - 1) / tiles, 1, _cg);
                    _kSplit = ks;
                    _chunkC = (_cg + ks - 1) / ks;
                    CpuParallel.For(tiles * ks, _packBody);
                }
                // phase 2: blocked GEMM over (M block, N block) tasks
                int mpt = Math.Min(_mp, 16);
                int ntt = _nt;
                long target = 4L * CpuParallel.Threads;
                while (((_mp + mpt - 1) / mpt) * (long)((_nt + ntt - 1) / ntt) < target && (ntt > 1 || mpt > 1))
                {
                    if (ntt > 1)
                    {
                        ntt = (ntt + 1) / 2;
                    }
                    else
                    {
                        mpt = (mpt + 1) / 2;
                    }
                }
                _mpt = mpt;
                _ntt = ntt;
                _mBlocks = (_mp + mpt - 1) / mpt;
                _nBlocks = (_nt + ntt - 1) / ntt;
                CpuParallel.For(_mBlocks * _nBlocks, _gemmBody);
            }
        }
    }

    /// <summary>Task = (output tile, channel range): fills the B panel rows of those channels.</summary>
    private void PackBody(int task)
    {
        int ks = _kSplit;
        int tileSel = task / ks;
        int part = task % ks;
        int tile = _direct ? _nt - 1 : tileSel;
        int c0 = part * _chunkC;
        int c1 = Math.Min(_cg, c0 + _chunkC);
        if (c0 >= c1)
        {
            return;
        }

        PackTile(tile, c0, c1, _bpack + (_direct ? 0 : (long)tile * _kg * Gemm.NR));
    }

    /// <summary>Fills the im2col rows of channels [c0, c1) of output tile <paramref name="tile"/> into the [k][NR] panel.</summary>
    private void PackTile(int tile, int c0, int c1, float* panel)
    {
        int NR = Gemm.NR;
        long p0 = (long)tile * NR;
        int ncols = (int)Math.Min(NR, _hw - p0);
        int* segOh = stackalloc int[NR];
        int* segOw = stackalloc int[NR];
        int* segJ = stackalloc int[NR];
        int* segLen = stackalloc int[NR];
        int ns = 0;
        for (int j = 0; j < ncols;)
        {
            long pp = p0 + j;
            int oh = (int)(pp / _wo);
            int ow = (int)(pp % _wo);
            int len = Math.Min(_wo - ow, ncols - j);
            segOh[ns] = oh;
            segOw[ns] = ow;
            segJ[ns] = j;
            segLen[ns] = len;
            ns++;
            j += len;
        }
        int W = _w;
        int H = _h;
        int kh = _kh;
        int kw = _kw;
        for (int c = c0; c < c1; c++)
        {
            float* xc = _xg + (long)c * H * W;
            for (int fh = 0; fh < kh; fh++)
            {
                for (int fw = 0; fw < kw; fw++)
                {
                    float* row = panel + (long)((c * kh + fh) * kw + fw) * NR;
                    for (int s = 0; s < ns; s++)
                    {
                        float* d = row + segJ[s];
                        int len = segLen[s];
                        int ih = segOh[s] * _sh - _pt + fh * _dh;
                        if ((uint)ih >= (uint)H)
                        {
                            new Span<float>(d, len).Clear();
                            continue;
                        }
                        float* xr = xc + (long)ih * W;
                        int off = segOw[s] * _sw - _pl + fw * _dw;
                        if (_sw == 1)
                        {
                            int lo = Math.Clamp(-off, 0, len);
                            int hi = Math.Clamp(W - off, lo, len);
                            if (lo > 0)
                            {
                                new Span<float>(d, lo).Clear();
                            }

                            if (hi > lo)
                            {
                                new ReadOnlySpan<float>(xr + off + lo, hi - lo).CopyTo(new Span<float>(d + lo, hi - lo));
                            }

                            if (hi < len)
                            {
                                new Span<float>(d + hi, len - hi).Clear();
                            }
                        }
                        else
                        {
                            for (int t = 0; t < len; t++)
                            {
                                int iw = off + t * _sw;
                                d[t] = (uint)iw < (uint)W ? xr[iw] : 0f;
                            }
                        }
                    }
                    if (ncols < NR)
                    {
                        new Span<float>(row + ncols, NR - ncols).Clear();
                    }
                }
            }
        }
    }

    /// <summary>Largest per-group weight size and reduction length of the tile-packed path (A about the size of L2, a B tile in L1/L2; tuned on Zen 5).</summary>
    private const long TilePackMaxWeightBytes = 1024 * 1024;
    private const int TilePackMaxK = 1152;

    /// <summary>Most output tiles per task of the tile-packed path.</summary>
    private const int TilePackMaxTilesPerTask = 8;

    /// <summary>Tile-packed path: task = a run of output tiles; each tile is packed into thread scratch, then every weight panel runs over it.</summary>
    private void TileBody(int task)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        int t0 = task * _ntt;
        int t1 = Math.Min(_nt, t0 + _ntt);
        float* panel = (float*)ThreadScratch.Get(ThreadScratch.Worker, (long)_kg * NR * 4);
        float* bias = stackalloc float[MR];
        long kg = _kg;
        for (int t = t0; t < t1; t++)
        {
            PackTile(t, 0, _cg, panel);
            long n0 = (long)t * NR;
            int cols = (int)Math.Min(NR, _hw - n0);
            for (int p = 0; p < _mp; p++)
            {
                int rows = Math.Min(MR, _mg - p * MR);
                float* bi = null;
                if (_bias0 != null)
                {
                    for (int i = 0; i < MR; i++)
                    {
                        bias[i] = i < rows ? _bias0[_grp * _mg + p * MR + i] : 0f;
                    }

                    bi = bias;
                }

                float* c = _yg + (long)p * MR * _hw + n0;
                for (int kb = 0; kb < kg; kb += Gemm.KC)
                {
                    int kc = (int)Math.Min(Gemm.KC, kg - kb);
                    float* a = _apack + ((long)(_grp * _mp + p) * kg + kb) * MR;
                    float* b = panel + (long)kb * NR;
                    if (rows == MR && cols == NR)
                    {
                        Gemm.Tile(a, b, NR, c, _hw, kc, bi, kb > 0);
                    }
                    else
                    {
                        Gemm.TileEdge(a, b, NR, c, _hw, kc, bi, kb > 0, rows, cols);
                    }
                }

                if (_epilogue is { } epilogue)
                {
                    long offset = c - (float*)_y.P;
                    for (int i = 0; i < rows; i++)
                    {
                        epilogue.Apply(c + (long)i * _hw, offset + (long)i * _hw, cols);
                    }
                }
            }
        }
    }

    private void GemmBody(int task)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        int mb = task / _nBlocks;
        int nb = task % _nBlocks;
        int pm0 = mb * _mpt;
        int pm1 = Math.Min(_mp, pm0 + _mpt);
        int t0 = nb * _ntt;
        int t1 = Math.Min(_nt, t0 + _ntt);
        float* bias = stackalloc float[MR * _mpt];
        if (_bias0 != null)
        {
            for (int p = pm0; p < pm1; p++)
            {
                for (int i = 0; i < MR; i++)
                {
                    int r = p * MR + i;
                    bias[(p - pm0) * MR + i] = r < _mg ? _bias0[_grp * _mg + r] : 0f;
                }
            }
        }

        long kg = _kg;
        for (int kb = 0; kb < kg; kb += Gemm.KC)
        {
            int kc = (int)Math.Min(Gemm.KC, kg - kb);
            bool acc = kb > 0;
            for (int t = t0; t < t1; t++)
            {
                long n0 = (long)t * NR;
                int cols = (int)Math.Min(NR, _hw - n0);
                float* b;
                long ldb;
                if (_direct && cols == NR)
                {
                    b = _xg + (long)kb * _hw + n0;
                    ldb = _hw;
                }
                else
                {
                    b = _bpack + (_direct ? 0 : (long)t * kg * NR) + (long)kb * NR;
                    ldb = NR;
                }
                for (int p = pm0; p < pm1; p++)
                {
                    float* a = _apack + ((long)(_grp * _mp + p) * kg + kb) * MR;
                    float* c = _yg + (long)p * MR * _hw + n0;
                    int rows = Math.Min(MR, _mg - p * MR);
                    float* bi = _bias0 != null ? bias + (p - pm0) * MR : null;
                    if (rows == MR && cols == NR)
                    {
                        Gemm.Tile(a, b, ldb, c, _hw, kc, bi, acc);
                    }
                    else
                    {
                        Gemm.TileEdge(a, b, ldb, c, _hw, kc, bi, acc, rows, cols);
                    }

                    if (_epilogue is { } epilogue && kb + kc >= kg)
                    {
                        long offset = c - (float*)_y.P;
                        for (int i = 0; i < rows; i++)
                        {
                            epilogue.Apply(c + (long)i * _hw, offset + (long)i * _hw, cols);
                        }
                    }
                }
            }
        }
    }

    /// <summary>Interior outputs per row from which the depthwise rows use the vector loops instead of one AXPY per tap.</summary>
    private const int DwVectorMinWidth = 16;

    private void DwBody(int chunk)
    {
        long s = chunk * _perChunk;
        long e = Math.Min(_planes, s + _perChunk);
        int mult = _o / _c;
        int taps = _kh * _kw;
        Vector512<float>* w512 = stackalloc Vector512<float>[CpuIsa.Use512 ? taps : 0];
        Vector256<float>* w256 = stackalloc Vector256<float>[CpuIsa.Use256 && !CpuIsa.Use512 ? taps : 0];
        // Interior outputs: every tap of the row is inside the input, so the vector loop needs no bounds checks.
        int owLo = Math.Min(_wo, (_pl + _sw - 1) / _sw);
        int lastTap = (_kw - 1) * _dw - _pl;
        int owHi = _w - 1 - lastTap < 0 ? owLo : Math.Clamp((_w - 1 - lastTap) / _sw + 1, owLo, _wo);
        for (long pl = s; pl < e; pl++)
        {
            long n = pl / _o;
            int o = (int)(pl % _o);
            int c = o / mult;
            float* xp = (float*)_x.P + (n * _c + c) * _h * _w;
            float* yp = (float*)_y.P + pl * _hw;
            float* wt = _wdw + (long)o * taps;
            float bv = _bias0 == null ? 0f : _bias0[o];
            for (int t = 0; t < taps; t++)
            {
                if (CpuIsa.Use512)
                {
                    w512[t] = Vector512.Create(wt[t]);
                }
                else if (CpuIsa.Use256)
                {
                    w256[t] = Vector256.Create(wt[t]);
                }
            }

            for (int oh = 0; oh < _ho; oh++)
            {
                float* yr = yp + (long)oh * _wo;
                int fh0 = 0;
                int fh1 = _kh;
                while (fh0 < _kh && (uint)(oh * _sh - _pt + fh0 * _dh) >= (uint)_h)
                {
                    fh0++;
                }

                while (fh1 > fh0 && (uint)(oh * _sh - _pt + (fh1 - 1) * _dh) >= (uint)_h)
                {
                    fh1--;
                }

                float* row0 = xp + (long)(oh * _sh - _pt) * _w;
                if (owHi - owLo < DwVectorMinWidth || _sw > 2 || !CpuIsa.Use256)
                {
                    DwAxpyRow(yr, row0, wt, bv, fh0, fh1);
                }
                else
                {
                    int ow = owLo;
                    if (CpuIsa.Use512)
                    {
                        ow = DwRow512(yr, row0, w512, bv, fh0, fh1, owLo, owHi);
                    }

                    if (_sw == 1)
                    {
                        ow = CpuIsa.Use512 ? DwRow256(yr, row0, (Vector256<float>*)null, wt, bv, fh0, fh1, ow, owHi) : DwRow256(yr, row0, w256, wt, bv, fh0, fh1, ow, owHi);
                    }

                    DwScalar(yr, row0, wt, bv, fh0, fh1, 0, owLo);
                    DwScalar(yr, row0, wt, bv, fh0, fh1, ow, _wo);
                }

                _epilogue?.Apply(yr, yr - (float*)_y.P, _wo);
            }
        }
    }

    /// <summary>Interior outputs [owLo, owHi) in steps of 16 (stride 1, or 2 by an even-lane permute); returns the first one left.</summary>
    private int DwRow512(float* yr, float* row0, Vector512<float>* w, float bias, int fh0, int fh1, int owLo, int owHi)
    {
        var even = Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        int ow = owLo;
        // Stride 2 loads 32 inputs per tap; the last one is past the final output's tap and must still be inside the row.
        for (; ow + 16 <= owHi && (_sw == 1 || (ow + 15) * 2 - _pl + (_kw - 1) * _dw + 1 < _w); ow += 16)
        {
            var acc = Vector512.Create(bias);
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w + ow * _sw - _pl;
                for (int fw = 0; fw < _kw; fw++)
                {
                    float* p = xr + fw * _dw;
                    var v = _sw == 1 ? Vector512.Load(p) : Avx512F.PermuteVar16x32x2(Vector512.Load(p), even, Vector512.Load(p + 16));
                    acc = Vector512.FusedMultiplyAdd(v, w[fh * _kw + fw], acc);
                }
            }

            acc.Store(yr + ow);
        }

        return ow;
    }

    /// <summary>Interior outputs [owLo, owHi) in steps of 8, stride 1 (weights splatted in <paramref name="w"/>, or from <paramref name="wt"/> when null); returns the first one left.</summary>
    private int DwRow256(float* yr, float* row0, Vector256<float>* w, float* wt, float bias, int fh0, int fh1, int owLo, int owHi)
    {
        int ow = owLo;
        for (; ow + 8 <= owHi; ow += 8)
        {
            var acc = Vector256.Create(bias);
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w + ow - _pl;
                for (int fw = 0; fw < _kw; fw++)
                {
                    var wv = w != null ? w[fh * _kw + fw] : Vector256.Create(wt[fh * _kw + fw]);
                    acc = Vector256.FusedMultiplyAdd(Vector256.Load(xr + fw * _dw), wv, acc);
                }
            }

            acc.Store(yr + ow);
        }

        return ow;
    }

    /// <summary>
    /// One output row as one AXPY per tap over the valid output range (rows too narrow for the interior vector loop, strides
    /// above 2, or no 256-bit vectors).
    /// </summary>
    private void DwAxpyRow(float* yr, float* row0, float* wt, float bias, int fh0, int fh1)
    {
        new Span<float>(yr, _wo).Fill(bias);
        for (int fh = fh0; fh < fh1; fh++)
        {
            float* xr = row0 + (long)fh * _dh * _w;
            for (int fw = 0; fw < _kw; fw++)
            {
                float wv = wt[fh * _kw + fw];
                int off = fw * _dw - _pl;
                int lo = off >= 0 ? 0 : (-off + _sw - 1) / _sw;
                int hi = _w - 1 - off < 0 ? 0 : Math.Min(_wo, (_w - 1 - off) / _sw + 1);
                if (hi <= lo)
                {
                    continue;
                }

                if (_sw == 1)
                {
                    Simd.Axpy(yr + lo, xr + off + lo, wv, hi - lo);
                }
                else
                {
                    for (int ow = lo; ow < hi; ow++)
                    {
                        yr[ow] += wv * xr[ow * _sw + off];
                    }
                }
            }
        }
    }

    /// <summary>Outputs [ow0, ow1) of one row with bounds checks (borders and remainders of the vector loops).</summary>
    private void DwScalar(float* yr, float* row0, float* wt, float bias, int fh0, int fh1, int ow0, int ow1)
    {
        for (int ow = ow0; ow < ow1; ow++)
        {
            float acc = bias;
            for (int fh = fh0; fh < fh1; fh++)
            {
                float* xr = row0 + (long)fh * _dh * _w;
                for (int fw = 0; fw < _kw; fw++)
                {
                    int iw = ow * _sw - _pl + fw * _dw;
                    if ((uint)iw < (uint)_w)
                    {
                        acc = MathF.FusedMultiplyAdd(xr[iw], wt[fh * _kw + fw], acc);
                    }
                }
            }

            yr[ow] = acc;
        }
    }
}
