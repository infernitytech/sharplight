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
    readonly TRef _x, _y; readonly TRef? _bias;
    readonly int _c, _h, _w, _o, _kh, _kw, _sh, _sw, _pt, _pl, _dh, _dw, _g, _ho, _wo;
    readonly int _cg, _mg, _kg, _hw, _mp, _nt;
    readonly bool _depthwise, _direct;
    readonly float* _apack;     // packed weights [g][panel][k][MR]
    readonly float* _wdw;       // depthwise: dense f32 weights [O, KH*KW]
    readonly float* _bpack;     // packed im2col [tile][k][NR]
    readonly Action<int> _packBody, _gemmBody, _dwBody;

    // per-run state
    long _n; int _img, _grp, _kSplit, _chunkC; int _mpt, _ntt, _nBlocks, _mBlocks;
    float* _xg, _yg, _bias0;
    long _planes, _perChunk;

    public override string Name => _depthwise ? "Conv.depthwise" : _direct ? "Conv.1x1" : "Conv";

    ConvOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int sh, int sw, int pt, int pl, int dh, int dw)
    {
        _x = x; _y = y; _bias = bias;
        _c = (int)x.Dims[1]; _h = (int)x.Dims[2]; _w = (int)x.Dims[3];
        _o = (int)y.Dims[1]; _ho = (int)y.Dims[2]; _wo = (int)y.Dims[3];
        _kh = kh; _kw = kw; _g = groups; _sh = sh; _sw = sw; _pt = pt; _pl = pl; _dh = dh; _dw = dw;
        _cg = _c / _g; _mg = _o / _g; _kg = _cg * kh * kw; _hw = _ho * _wo;
        _packBody = PackBody; _gemmBody = GemmBody; _dwBody = DwBody;
        _depthwise = _g == _c && _cg == 1;
        if (_depthwise)
        {
            long n = (long)_o * kh * kw;
            _wdw = (float*)Mem.Alloc(n * 4, false);
            new ReadOnlySpan<float>(wDense, (int)n).CopyTo(new Span<float>(_wdw, (int)n));
            return;
        }
        int MR = Gemm.MR, NR = Gemm.NR;
        _mp = (_mg + MR - 1) / MR; _nt = (_hw + NR - 1) / NR;
        _direct = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && _hw == _h * _w && ((long)_hw * 4) % 4096 != 0;
        long aBytes = (long)_g * _mp * _kg * MR * 4;
        _apack = (float*)Mem.Alloc(aBytes, true);
        for (int g = 0; g < _g; g++)
            for (int p = 0; p < _mp; p++)
            {
                int r0 = g * _mg + p * MR, rows = Math.Min(MR, _mg - p * MR);
                Gemm.PackA(_apack + ((long)(g * _mp + p) * _kg) * MR, wDense + (long)r0 * _kg, _kg, 1, rows, _kg, 1f);
            }
        long tiles = _direct ? 1 : _nt;
        _bpack = (float*)Mem.Alloc(tiles * _kg * NR * 4, false);
    }

    public static ConvOp Create(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int[] strides, int[] pads, int[] dil)
        => new(x, wDense, kh, kw, bias, y, groups, strides[0], strides[1], pads[0], pads[1], dil[0], dil[1]);

    public override void Dispose()
    {
        Mem.Free((byte*)_apack); Mem.Free((byte*)_wdw); Mem.Free((byte*)_bpack);
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        if (_n == 0 || _hw == 0) return;
        _bias0 = _bias is null ? null : (float*)_bias.P;
        if (_depthwise)
        {
            _planes = _n * _o;
            long work = (long)_hw * _kh * _kw;
            int chunks = (int)Math.Min(_planes, Par.Chunks(_planes * work, 16384));
            _perChunk = (_planes + chunks - 1) / chunks;
            Par.For(chunks, _dwBody);
            return;
        }
        int MR = Gemm.MR, NR = Gemm.NR;
        for (int img = 0; img < _n; img++)
            for (int g = 0; g < _g; g++)
            {
                _img = img; _grp = g;
                _xg = (float*)_x.P + ((long)img * _c + (long)g * _cg) * _h * _w;
                _yg = (float*)_y.P + ((long)img * _o + (long)g * _mg) * _hw;
                // phase 1: im2col into packed panels (only the partial tail tile when the input is read in place)
                bool tail = _hw % NR != 0;
                if (!_direct || tail)
                {
                    int tiles = _direct ? 1 : _nt;
                    int ks = Math.Clamp((2 * Par.Threads + tiles - 1) / tiles, 1, _cg);
                    _kSplit = ks; _chunkC = (_cg + ks - 1) / ks;
                    Par.For(tiles * ks, _packBody);
                }
                // phase 2: blocked GEMM over (M block, N block) tasks
                int mpt = Math.Min(_mp, 16), ntt = _nt;
                long target = 4L * Par.Threads;
                while (((_mp + mpt - 1) / mpt) * (long)((_nt + ntt - 1) / ntt) < target && (ntt > 1 || mpt > 1))
                {
                    if (ntt > 1) ntt = (ntt + 1) / 2; else mpt = (mpt + 1) / 2;
                }
                _mpt = mpt; _ntt = ntt;
                _mBlocks = (_mp + mpt - 1) / mpt; _nBlocks = (_nt + ntt - 1) / ntt;
                Par.For(_mBlocks * _nBlocks, _gemmBody);
            }
    }

    /// <summary>Task = (output tile, channel range): fills the B panel rows of those channels.</summary>
    void PackBody(int task)
    {
        int ks = _kSplit;
        int tileSel = task / ks, part = task % ks;
        int tile = _direct ? _nt - 1 : tileSel;
        int c0 = part * _chunkC, c1 = Math.Min(_cg, c0 + _chunkC);
        if (c0 >= c1) return;
        int NR = Gemm.NR;
        long p0 = (long)tile * NR;
        int ncols = (int)Math.Min(NR, _hw - p0);
        float* panel = _bpack + (_direct ? 0 : (long)tile * _kg * NR);
        int* segOh = stackalloc int[NR]; int* segOw = stackalloc int[NR]; int* segJ = stackalloc int[NR]; int* segLen = stackalloc int[NR];
        int ns = 0;
        for (int j = 0; j < ncols;)
        {
            long pp = p0 + j; int oh = (int)(pp / _wo), ow = (int)(pp % _wo);
            int len = Math.Min(_wo - ow, ncols - j);
            segOh[ns] = oh; segOw[ns] = ow; segJ[ns] = j; segLen[ns] = len; ns++; j += len;
        }
        int W = _w, H = _h, kh = _kh, kw = _kw;
        for (int c = c0; c < c1; c++)
        {
            float* xc = _xg + (long)c * H * W;
            for (int fh = 0; fh < kh; fh++)
                for (int fw = 0; fw < kw; fw++)
                {
                    float* row = panel + (long)((c * kh + fh) * kw + fw) * NR;
                    for (int s = 0; s < ns; s++)
                    {
                        float* d = row + segJ[s]; int len = segLen[s];
                        int ih = segOh[s] * _sh - _pt + fh * _dh;
                        if ((uint)ih >= (uint)H) { new Span<float>(d, len).Clear(); continue; }
                        float* xr = xc + (long)ih * W;
                        int off = segOw[s] * _sw - _pl + fw * _dw;
                        if (_sw == 1)
                        {
                            int lo = Math.Clamp(-off, 0, len), hi = Math.Clamp(W - off, lo, len);
                            if (lo > 0) new Span<float>(d, lo).Clear();
                            if (hi > lo) new ReadOnlySpan<float>(xr + off + lo, hi - lo).CopyTo(new Span<float>(d + lo, hi - lo));
                            if (hi < len) new Span<float>(d + hi, len - hi).Clear();
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
                    if (ncols < NR) new Span<float>(row + ncols, NR - ncols).Clear();
                }
        }
    }

    void GemmBody(int task)
    {
        int MR = Gemm.MR, NR = Gemm.NR;
        int mb = task / _nBlocks, nb = task % _nBlocks;
        int pm0 = mb * _mpt, pm1 = Math.Min(_mp, pm0 + _mpt);
        int t0 = nb * _ntt, t1 = Math.Min(_nt, t0 + _ntt);
        float* bias = stackalloc float[MR * _mpt];
        if (_bias0 != null)
            for (int p = pm0; p < pm1; p++)
                for (int i = 0; i < MR; i++)
                {
                    int r = p * MR + i;
                    bias[(p - pm0) * MR + i] = r < _mg ? _bias0[_grp * _mg + r] : 0f;
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
                float* b; long ldb;
                if (_direct && cols == NR) { b = _xg + (long)kb * _hw + n0; ldb = _hw; }
                else { b = _bpack + (_direct ? 0 : (long)t * kg * NR) + (long)kb * NR; ldb = NR; }
                for (int p = pm0; p < pm1; p++)
                {
                    float* a = _apack + ((long)(_grp * _mp + p) * kg + kb) * MR;
                    float* c = _yg + (long)p * MR * _hw + n0;
                    int rows = Math.Min(MR, _mg - p * MR);
                    float* bi = _bias0 != null ? bias + (p - pm0) * MR : null;
                    if (rows == MR && cols == NR) Gemm.Tile(a, b, ldb, c, _hw, kc, bi, acc);
                    else Gemm.TileEdge(a, b, ldb, c, _hw, kc, bi, acc, rows, cols);
                }
            }
        }
    }

    void DwBody(int chunk)
    {
        long s = chunk * _perChunk, e = Math.Min(_planes, s + _perChunk);
        int mult = _o / _c;
        for (long pl = s; pl < e; pl++)
        {
            long n = pl / _o; int o = (int)(pl % _o); int c = o / mult;
            float* xp = (float*)_x.P + (n * _c + c) * _h * _w;
            float* yp = (float*)_y.P + pl * _hw;
            float* wt = _wdw + (long)o * _kh * _kw;
            float bv = _bias0 == null ? 0f : _bias0[o];
            for (int oh = 0; oh < _ho; oh++)
            {
                float* yr = yp + (long)oh * _wo;
                new Span<float>(yr, _wo).Fill(bv);
                for (int fh = 0; fh < _kh; fh++)
                {
                    int ih = oh * _sh - _pt + fh * _dh;
                    if ((uint)ih >= (uint)_h) continue;
                    float* xr = xp + (long)ih * _w;
                    for (int fw = 0; fw < _kw; fw++)
                    {
                        float wv = wt[fh * _kw + fw];
                        int off = fw * _dw - _pl;
                        int lo = off >= 0 ? 0 : (-off + _sw - 1) / _sw;
                        int hi = _w - 1 - off < 0 ? 0 : Math.Min(_wo, (_w - 1 - off) / _sw + 1);
                        if (hi <= lo) continue;
                        if (_sw == 1) Simd.Axpy(yr + lo, xr + off + lo, wv, hi - lo);
                        else
                            for (int ow = lo; ow < hi; ow++) yr[ow] += wv * xr[ow * _sw + off];
                    }
                }
            }
        }
    }
}
