using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Dense 2-D convolution (groups == 1) on channels-last activations (<see cref="ChannelsLast"/>; the input may be a
/// channels-last view, such as one part of a channel split). The output pixels
/// (over N·HO·WO) form tiles of <see cref="_mr"/> pixels and the output channels panels of <see cref="_nb"/> channels; a
/// micro-kernel keeps one tile x panel block in registers over all of K and broadcasts each input value straight from the
/// activations: in this layout every kernel tap of a pixel is a contiguous run of C input channels, so the implicit GEMM
/// needs neither im2col nor packing of the activations. Taps that fall into the padding read a zero row. Convs with few
/// input channels (the stem) instead copy their input once into a zero-padded channels-last buffer, where a whole kernel
/// row (KW·C values) is one contiguous run; that copy also reads NCHW input. The weights are packed once into
/// [panel][tap][channel][NB] panels. Bias, a residual addend in the output's layout and Relu / Clip are applied in
/// registers, other activations by the <see cref="OutputEpilogue"/> on each stored tile (or pixel). With AVX-512 the
/// tiles are 8 pixels x 32 channels, or 12 x 16 for convs with at most 16 output channels; with AVX2, 6 x 16.
/// </summary>
internal sealed unsafe partial class ChannelsLastConvOp : CpuOp
{
    /// <summary>Input channels below which the conv runs on the padded copy in kernel-row runs.</summary>
    public const int RowRunMaxChannels = 16;

    /// <summary>K values per chunk: a chunk of a weight panel (values x NB floats) stays in L1 while the task's tiles run over it.</summary>
    private const int ChunkValues = 256;

    /// <summary>Planning estimate: f32 multiply-add rate of one thread (flop/s).</summary>
    private const double ThreadFlops = 1.3e11;

    /// <summary>Planning estimate: bandwidth of the shared caches and memory for operand fetches (bytes/s).</summary>
    private const double SharedBytesPerSecond = 3e11;

    /// <summary>Planning estimate: fixed cost of one task (s).</summary>
    private const double TaskSeconds = 2e-7;

    /// <summary>Operand blocks up to this size are assumed to stay in a core's L2 between that core's tasks.</summary>
    private const long ResidentBlockBytes = 512 * 1024;

    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly OutputEpilogue? _epilogue;
    private readonly bool _xChannelsLast;
    private readonly long _xps;
    private readonly long _xrs;
    private readonly long _xis;
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
    private readonly int _ho;
    private readonly int _wo;
    private readonly int _mr;
    private readonly int _nb;
    private readonly bool _narrow;
    private readonly int _panels;
    private readonly int _k;
    private readonly bool _rowRuns;
    private readonly int _segs;
    private readonly int _len;
    private readonly int _hp;
    private readonly int _wp;
    private readonly float* _wpack;     // [panel][K][NB]
    private readonly float* _biasPad;   // [panels * NB] or null
    private readonly float* _zero;      // a zero row of _len floats
    private readonly float _lo;
    private readonly float _hi;
    private readonly bool _inlineActivation;
    private readonly (int S0, int S1, int K0, int K1)[] _chunks;
    private readonly Action<int> _body;
    private readonly Action<int> _padBody;

    // per-run state
    private long _n;
    private long _m;
    private int _tiles;
    private int _ntt;
    private int _npt;
    private int _tBlocks;
    private int _pBlocks;
    private float* _xpad;
    private long _xpadFloats;
    private long _plannedM = -1;

    private ChannelsLastConvOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int sh, int sw, int pt, int pl, int dh, int dw, OutputEpilogue? epilogue)
    {
        _x = x;
        _y = y;
        _bias = bias;
        _epilogue = epilogue;
        _xChannelsLast = ChannelsLast.IsView(x);
        _xps = _xChannelsLast ? ChannelsLast.PixelStride(x) : 0;
        _xrs = _xChannelsLast ? ChannelsLast.RowStride(x) : 0;
        _xis = _xChannelsLast ? ChannelsLast.ImageStride(x) : 0;
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _o = (int)y.Dims[1];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _kh = kh;
        _kw = kw;
        _sh = sh;
        _sw = sw;
        _pt = pt;
        _pl = pl;
        _dh = dh;
        _dw = dw;
        _body = Body;
        _padBody = PadBody;
        _narrow = CpuIsa.Use512 && _o <= 16;
        _mr = CpuIsa.Use512 ? (_narrow ? 12 : 8) : 6;
        _nb = CpuIsa.Use512 && !_narrow ? 32 : 16;
        _panels = (_o + _nb - 1) / _nb;
        _k = _c * kh * kw;
        _rowRuns = UsesRowRuns(_c, dw);
        _segs = _rowRuns ? kh : kh * kw;
        _len = _rowRuns ? kw * _c : _c;
        _chunks = DiagnosticSwitches.NoChannelsLastChunks ? [(0, _segs, 0, _len)] : Chunks(_segs, _len);
        if (_rowRuns)
        {
            // Padded extent: every row and column any output reads.
            _hp = Math.Max(_h + _pt, (_ho - 1) * sh + (kh - 1) * dh + 1);
            _wp = Math.Max(_w + _pl, (_wo - 1) * sw + (kw - 1) + 1);
        }

        _wpack = (float*)Mem.Alloc((long)_panels * _k * _nb * 4, true);
        for (int o = 0; o < _o; o++)
        {
            float* panel = _wpack + (long)(o / _nb) * _k * _nb + o % _nb;
            for (int fh = 0; fh < kh; fh++)
            {
                for (int fw = 0; fw < kw; fw++)
                {
                    for (int c = 0; c < _c; c++)
                    {
                        long k = ((long)fh * kw + fw) * _c + c;
                        panel[k * _nb] = wDense[(((long)o * _c + c) * kh + fh) * kw + fw];
                    }
                }
            }
        }

        _biasPad = bias is null ? null : (float*)Mem.Alloc((long)_panels * _nb * 4, true);
        _zero = (float*)Mem.Alloc(Math.Max(_len, 16) * 4L, true);
        _lo = float.NegativeInfinity;
        _hi = float.PositiveInfinity;
        _inlineActivation = epilogue?.Activation is null;
        if (epilogue?.Activation == UnaryKind.Relu)
        {
            _lo = 0;
            _inlineActivation = true;
        }
        else if (epilogue?.Activation == UnaryKind.Clip)
        {
            _lo = epilogue.Scale;
            _hi = epilogue.Bias;
            _inlineActivation = true;
        }
    }

    /// <inheritdoc />
    public override string Name => "Conv.cl" + (_kh * _kw == 1 ? ".1x1" : "") + _epilogue?.Label;

    /// <inheritdoc />
    public override string Detail =>
        $"{_n}x{_c}x{_h}x{_w}>{_o} k{_kh}s{_sh} {2.0 * _m * _o * _k / 1e6:F0}MF{(_rowRuns ? " rows" : "")} tasks {_pBlocks}x{_tBlocks}";

    /// <summary>True when a conv with <paramref name="channels"/> inputs runs on the padded copy in kernel-row runs (which may also read NCHW input).</summary>
    public static bool UsesRowRuns(int channels, int dilationW) => dilationW == 1 && channels < RowRunMaxChannels;

    /// <summary>
    /// Creates the op. <paramref name="x"/> is a channels-last view, or NCHW dense when the conv takes kernel-row runs
    /// (<see cref="UsesRowRuns"/>); <paramref name="y"/> and the epilogue's residual are channels-last.
    /// </summary>
    public static ChannelsLastConvOp Create(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int[] strides, int[] pads, int[] dil, OutputEpilogue? epilogue)
        => new(x, wDense, kh, kw, bias, y, strides[0], strides[1], pads[0], pads[1], dil[0], dil[1], epilogue);

    /// <inheritdoc />
    public override void Dispose()
    {
        Mem.Free((byte*)_wpack);
        Mem.Free((byte*)_biasPad);
        Mem.Free((byte*)_zero);
        Mem.Free((byte*)_xpad);
    }

    /// <inheritdoc />
    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _m = _n * _ho * _wo;
        if (_m == 0)
        {
            return;
        }

        if (_biasPad != null)
        {
            new ReadOnlySpan<float>((float*)_bias!.P, _o).CopyTo(new Span<float>(_biasPad, _o));
        }

        if (_rowRuns)
        {
            long floats = _n * _hp * _wp * _c;
            if (floats > _xpadFloats)
            {
                Mem.Free((byte*)_xpad);
                _xpad = (float*)Mem.Alloc(floats * 4, true);
                _xpadFloats = floats;
            }

            CpuParallel.For((int)(_n * _hp), _padBody);
        }

        if (_m != _plannedM)
        {
            Plan();
            _plannedM = _m;
        }

        CpuParallel.For(_pBlocks * _tBlocks, _body);
    }

    /// <summary>
    /// Chooses panels and tiles per task by an estimate of the run time: the multiply-adds spread over the threads, plus the operand bytes moved through the shared caches (each weight block once
    /// per tile block and each input block once per panel block, blocks that stay in L2 once per thread), plus a cost per
    /// task; with few tasks the threads wait for the last one. Late layers with few pixels and large weights then take few tasks that each read their weights once.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Plan()
    {
        int threads = CpuParallel.Threads;
        _tiles = (int)((_m + _mr - 1) / _mr);
        long aBytes = (long)_panels * _k * _nb * 4;
        long xBytes = _m * _c * 4L * _sh * _sw;
        double flops = 2.0 * _tiles * _mr * _panels * _nb * _k;
        double best = double.MaxValue;
        int bestNpt = _panels;
        int bestNtt = _tiles;
        for (int npt = 1; npt <= _panels; npt++)
        {
            int pBlocks = (_panels + npt - 1) / npt;
            if (npt > 1 && (_panels + npt - 2) / (npt - 1) == pBlocks)
            {
                continue;   // same block count as a smaller npt
            }

            for (int ntt = 1; ; ntt = Math.Max(ntt + 1, ntt * 5 / 4))
            {
                int tBlocks = (_tiles + ntt - 1) / ntt;
                long tasks = (long)pBlocks * tBlocks;
                double aBlock = (double)aBytes / pBlocks;
                double xBlock = (double)xBytes / tBlocks;
                double aReads = aBlock <= ResidentBlockBytes ? Math.Min(tBlocks, threads) : tBlocks;
                double xReads = xBlock <= ResidentBlockBytes ? Math.Min(pBlocks, threads) : pBlocks;
                // Inside a task the panels run outermost: an input block larger than L2 is fetched again for every panel.
                double inner = xBlock > ResidentBlockBytes ? xBlock * (npt - 1) * tasks : 0;
                double traffic = aBytes * aReads + xBytes * xReads + inner;
                // Dynamic claiming balances whole tasks: the last task still running costs up to one task's time on top.
                double compute = flops / ThreadFlops * (1.0 / threads + 1.0 / tasks);
                double time = compute + traffic / SharedBytesPerSecond + tasks * TaskSeconds;
                if (time < best)
                {
                    best = time;
                    bestNpt = npt;
                    bestNtt = ntt;
                }

                if (ntt >= _tiles)
                {
                    break;
                }
            }
        }

        _npt = bestNpt;
        _ntt = bestNtt;
        _pBlocks = (_panels + _npt - 1) / _npt;
        _tBlocks = (_tiles + _ntt - 1) / _ntt;
    }

    /// <summary>Copies padded input row <paramref name="task"/> (image * HP + row) into the zero-padded channels-last buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PadBody(int task)
    {
        long n = task / _hp;
        int ihp = (int)(task - n * _hp);
        float* dst = _xpad + (n * _hp + ihp) * _wp * _c;
        int ih = ihp - _pt;
        if ((uint)ih >= (uint)_h)
        {
            new Span<float>(dst, _wp * _c).Clear();
            return;
        }

        new Span<float>(dst, _pl * _c).Clear();
        int tail = _wp - _pl - _w;
        if (tail > 0)
        {
            new Span<float>(dst + (long)(_pl + _w) * _c, tail * _c).Clear();
        }

        float* d = dst + (long)_pl * _c;
        if (_xChannelsLast)
        {
            float* s = (float*)_x.P + n * _xis + ih * _xrs;
            if (_xps == _c)
            {
                new ReadOnlySpan<float>(s, _w * _c).CopyTo(new Span<float>(d, _w * _c));
                return;
            }

            for (int iw = 0; iw < _w; iw++)
            {
                new ReadOnlySpan<float>(s + iw * _xps, _c).CopyTo(new Span<float>(d + (long)iw * _c, _c));
            }

            return;
        }

        long plane = (long)_h * _w;
        float* src = (float*)_x.P + n * _c * plane + (long)ih * _w;
        for (int c = 0; c < _c; c++)
        {
            float* s = src + c * plane;
            for (int iw = 0; iw < _w; iw++)
            {
                d[(long)iw * _c + c] = s[iw];
            }
        }
    }

    /// <summary>
    /// One task: panels [p0, p1) over tiles [t0, t1). Per panel, K runs in chunks (<see cref="_chunks"/>) outermost, so that a
    /// chunk of the panel's weights stays in L1 while every tile of the task runs over it; partial sums are kept in the output
    /// (full tiles) or in per-tile scratch (edge tiles), and the epilogue is applied with the last chunk.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        int pb = task / _tBlocks;
        int tb = task - pb * _tBlocks;
        int p0 = pb * _npt;
        int p1 = Math.Min(_panels, p0 + _npt);
        int t0 = tb * _ntt;
        int t1 = Math.Min(_tiles, t0 + _ntt);
        if (p0 >= p1 || t0 >= t1)
        {
            return;
        }

        int perTile = _segs * _mr;
        int nt = t1 - t0;
        long tileFloats = (long)_mr * _nb;
        float** ptrs = (float**)ThreadScratch.Get(ThreadScratch.Worker, (long)nt * perTile * sizeof(nint) + nt * tileFloats * 4 + 64);
        float* tmpBase = (float*)(((nint)(ptrs + (long)nt * perTile) + 63) & ~(nint)63);
        for (int t = t0; t < t1; t++)
        {
            TilePointers(t, ptrs + (long)(t - t0) * perTile);
        }

        float* y = (float*)_y.P;
        float* residual = _epilogue?.Residual is { } r ? (float*)r.P : null;
        int chunks = _chunks.Length;
        for (int p = p0; p < p1; p++)
        {
            float* b = _wpack + (long)p * _k * _nb;
            float* bias = _biasPad == null ? null : _biasPad + (long)p * _nb;
            int o0 = p * _nb;
            int cols = Math.Min(_nb, _o - o0);
            for (int ch = 0; ch < chunks; ch++)
            {
                var (s0, s1, k0, k1) = _chunks[ch];
                bool acc = ch > 0;
                bool last = ch == chunks - 1;
                for (int t = t0; t < t1; t++)
                {
                    long m0 = (long)t * _mr;
                    int rows = (int)Math.Min(_mr, _m - m0);
                    float** tp = ptrs + (long)(t - t0) * perTile;
                    long off = m0 * _o + o0;
                    if (rows == _mr && cols == _nb)
                    {
                        Kernel(tp, s0, s1, k0, k1, b, bias, acc, y + off, _o, last && residual != null ? residual + off : null, last ? _lo : float.NegativeInfinity, last ? _hi : float.PositiveInfinity);
                        if (last && !_inlineActivation && cols == _o)
                        {
                            // The tile's rows are contiguous: one call over all of them.
                            ScalarMath.Unary(_epilogue!.Activation!.Value, y + off, y + off, rows * _o, _epilogue.Scale, _epilogue.Bias);
                        }
                        else if (last && !_inlineActivation)
                        {
                            for (int i = 0; i < rows; i++)
                            {
                                ScalarMath.Unary(_epilogue!.Activation!.Value, y + off + (long)i * _o, y + off + (long)i * _o, cols, _epilogue.Scale, _epilogue.Bias);
                            }
                        }

                        continue;
                    }

                    float* tmp = tmpBase + (t - t0) * tileFloats;
                    Kernel(tp, s0, s1, k0, k1, b, bias, acc, tmp, _nb, null, float.NegativeInfinity, float.PositiveInfinity);
                    if (!last)
                    {
                        continue;
                    }

                    for (int i = 0; i < rows; i++)
                    {
                        float* dst = y + off + (long)i * _o;
                        if (_epilogue is null)
                        {
                            new ReadOnlySpan<float>(tmp + i * _nb, cols).CopyTo(new Span<float>(dst, cols));
                        }
                        else
                        {
                            _epilogue.ApplyFrom(tmp + i * _nb, dst, off + (long)i * _o, cols);
                        }
                    }
                }
            }
        }
    }

    /// <summary>The input row pointers of tile <paramref name="t"/>: [segment][pixel], the zero row for padding and past the last pixel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void TilePointers(int t, float** dst)
    {
        long hw = (long)_ho * _wo;
        float* x = _rowRuns ? _xpad : (float*)_x.P;
        long m = (long)t * _mr;
        long n = m / hw;
        int r = (int)(m - n * hw);
        int oh = r / _wo;
        int ow = r - oh * _wo;
        for (int i = 0; i < _mr; i++, m++)
        {
            if (m >= _m)
            {
                for (int s = 0; s < _segs; s++)
                {
                    dst[s * _mr + i] = _zero;
                }

                continue;
            }

            if (_rowRuns)
            {
                float* row = x + ((n * _hp + oh * _sh) * _wp + (long)ow * _sw) * _c;
                long rowStep = (long)_dh * _wp * _c;
                for (int fh = 0; fh < _kh; fh++)
                {
                    dst[fh * _mr + i] = row + fh * rowStep;
                }
            }
            else
            {
                int ih0 = oh * _sh - _pt;
                int iw0 = ow * _sw - _pl;
                float* image = x + n * _xis;
                for (int fh = 0; fh < _kh; fh++)
                {
                    int ih = ih0 + fh * _dh;
                    for (int fw = 0; fw < _kw; fw++)
                    {
                        int iw = iw0 + fw * _dw;
                        dst[(fh * _kw + fw) * _mr + i] = (uint)ih < (uint)_h && (uint)iw < (uint)_w ? image + ih * _xrs + iw * _xps : _zero;
                    }
                }
            }

            if (++ow == _wo)
            {
                ow = 0;
                if (++oh == _ho)
                {
                    oh = 0;
                    n++;
                }
            }
        }
    }

    private void Kernel(float** ptrs, int s0, int s1, int k0, int k1, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        if (_narrow)
        {
            Kernel12x16(ptrs, s0, s1, k0, k1, _len, b, bias, acc, c, ldc, residual, lo, hi);
        }
        else if (CpuIsa.Use512)
        {
            Kernel8x32(ptrs, s0, s1, k0, k1, _len, b, bias, acc, c, ldc, residual, lo, hi);
        }
        else
        {
            Kernel6x16(ptrs, s0, s1, k0, k1, _len, b, bias, acc, c, ldc, residual, lo, hi);
        }
    }

    /// <summary>
    /// K chunks of about <see cref="ChunkValues"/> values: runs of whole segments, or slices of one segment's values when a
    /// segment alone is longer.
    /// </summary>
    private static (int S0, int S1, int K0, int K1)[] Chunks(int segs, int len)
    {
        var chunks = new List<(int, int, int, int)>();
        if (len <= ChunkValues)
        {
            int per = Math.Max(1, ChunkValues / len);
            int count = (segs + per - 1) / per;
            per = (segs + count - 1) / count;
            for (int s = 0; s < segs; s += per)
            {
                chunks.Add((s, Math.Min(segs, s + per), 0, len));
            }

            return [.. chunks];
        }

        int slices = (len + ChunkValues - 1) / ChunkValues;
        int width = (len + slices - 1) / slices;
        for (int s = 0; s < segs; s++)
        {
            for (int k = 0; k < len; k += width)
            {
                chunks.Add((s, s + 1, k, Math.Min(len, k + width)));
            }
        }

        return [.. chunks];
    }
}
