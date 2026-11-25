using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 2-D convolution on NCHW tensors. Dense and grouped convs run as one implicit GEMM per group over all images:
/// C[o, col] = sum_k W[o, k] * im2col(x)[k, col], col over N*HO*WO, so tiles of <see cref="Gemm.NR"/> output pixels may span
/// images. The weights are packed once into <see cref="Gemm.MR"/>-row panels (A). One parallel loop per run hands out tasks
/// of (weight panels, output tiles); each task walks K in L2-sized chunks, builds the im2col micro-panel of one tile (B, kept in
/// L1) right before every weight panel of the task runs over it, accumulates in a task-local C block and finally writes the
/// valid rows and columns to y with the fused epilogue (bias, residual, activation). When the output channels must be split
/// across tasks for parallelism, B is packed once for all tasks first; interior tiles of 1x1 stride-1 convs read the input in
/// place; strided 1x1 convs read a subsampled copy of the input the same way. Stride-1 convs with larger kernels need no
/// im2col at all (the shifted form, ConvOp.Shifted.cs). Depthwise convs (groups == C) use a direct kernel
/// (ConvOp.Depthwise.cs). The task split weighs the operand traffic of splitting the weights against splitting the tiles.
/// </summary>
internal sealed unsafe partial class ConvOp : CpuOp
{
    /// <summary>Bytes of the A block (task panels times the K chunk) aimed to stay in L2 while the task's tiles run over it.</summary>
    private const long ABlockBytes = 512 * 1024;

    /// <summary>Most (weight panel, output tile) pairs of one task's C block (one pair is MR x NR floats).</summary>
    private const int MaxPairsPerTask = 256;

    /// <summary>Output pixels per image, in tiles, from which strided 1x1 convs run on the subsampled input.</summary>
    private const int MinSubsampleTiles = 4;

    /// <summary>Tasks per thread aimed for when the output tiles are split (dynamic claiming then balances the threads).</summary>
    private const int TasksPerThread = 6;

    /// <summary>Operand blocks up to this size are assumed to stay in a core's L2 between that core's tasks.</summary>
    private const long ResidentBlockBytes = 512 * 1024;

    /// <summary>Most output tiles per task.</summary>
    private const int MaxTilesPerTask = 8;

    /// <summary>Largest packed-B buffer for the pack-once form; larger ones pack per task instead.</summary>
    private const long MaxPrepackBytes = 64L * 1024 * 1024;

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
    private readonly bool _depthwise;
    private readonly bool _direct;
    private readonly bool _subsample;
    private readonly bool _tapMajor;
    private readonly float* _apack;      // packed weights [g][panel][k][MR]
    private readonly float* _biasPanels; // bias per panel row [g][panel][MR], zero padded (null without bias)
    private readonly float* _wdw;        // depthwise: dense f32 weights [O, KH*KW]
    private readonly Action<int> _packBody;
    private readonly Action<int> _gemmBody;
    private readonly Action<int> _dwBody;
    private readonly Action<int> _padBody;
    private readonly Action<int> _subsampleBody;

    // per-run state
    private long _n;
    private long _nc;
    private int _tiles;
    private int _mpt;
    private int _ntt;
    private int _mBlocks;
    private int _cBlocks;
    private int _kc;
    private int _kChunks;
    private bool _prepacked;
    private bool _shifted;
    private float* _bpack;               // pack-once B tiles [g][tile][k][NR]
    private long _bpackBytes;
    private float* _bias0;
    private float* _xsub;                // strided 1x1 convs: the subsampled input [N][C][HO*WO]
    private long _xsubFloats;
    private long _planes;
    private long _perChunk;

    public override string Name => (_depthwise ? "Conv.depthwise" : _direct ? "Conv.1x1" : _subsample ? "Conv.1x1s" : "Conv") + _epilogue?.Label;

    /// <inheritdoc />
    public override string Detail =>
        $"{_n}x{_c}x{_h}x{_w}>{_o} k{_kh}s{_sh}g{_g} {2.0 * _n * _o * _kg * _hw / 1e6:F0}MF{(_shifted ? " shifted" : _prepacked ? " prepacked" : "")} tasks {_mBlocks}x{_cBlocks}";

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
        _padBody = PadBody;
        _subsampleBody = SubsampleBody;
        _depthwise = _g == _c && _cg == 1;
        if (_depthwise)
        {
            long n = (long)_o * kh * kw;
            _wdw = (float*)Mem.Alloc(n * 4, false);
            new ReadOnlySpan<float>(wDense, (int)n).CopyTo(new Span<float>(_wdw, (int)n));
            return;
        }

        int MR = Gemm.MR;
        _mp = (_mg + MR - 1) / MR;
        _direct = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && _hw == _h * _w;
        // Strided 1x1 convs without leading padding read the input subsampled once per run, then run like the direct form.
        // Small outputs keep the pack-once form: most of their tiles would be edge tiles, packed again by every task.
        _subsample = kh == 1 && kw == 1 && !_direct && pt == 0 && pl == 0 && (_ho - 1) * sh < _h && (_wo - 1) * sw < _w
            && _hw >= MinSubsampleTiles * Gemm.NR;
        // Stride-1 convs order K (tap, input channel), for both im2col packing and the shifted form, so one weight packing
        // serves either choice at run time.
        _tapMajor = !_direct && sh == 1 && sw == 1;
        float* weights = wDense;
        if (_tapMajor)
        {
            InitShifted();
            weights = TapMajorWeights(wDense);
        }

        _apack = (float*)Mem.Alloc((long)_g * _mp * _kg * MR * 4, true);
        for (int g = 0; g < _g; g++)
        {
            for (int p = 0; p < _mp; p++)
            {
                int r0 = g * _mg + p * MR;
                int rows = Math.Min(MR, _mg - p * MR);
                Gemm.PackA(_apack + ((long)(g * _mp + p) * _kg) * MR, weights + (long)r0 * _kg, _kg, 1, rows, _kg, 1f);
            }
        }

        if (weights != wDense)
        {
            Mem.Free((byte*)weights);
        }

        _biasPanels = bias is null ? null : (float*)Mem.Alloc((long)_g * _mp * MR * 4, true);
    }

    public static ConvOp Create(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int[] strides, int[] pads, int[] dil, OutputEpilogue? epilogue)
        => new(x, wDense, kh, kw, bias, y, groups, strides[0], strides[1], pads[0], pads[1], dil[0], dil[1], epilogue);

    public override void Dispose()
    {
        Mem.Free((byte*)_apack);
        Mem.Free((byte*)_biasPanels);
        Mem.Free((byte*)_wdw);
        Mem.Free((byte*)_bpack);
        Mem.Free((byte*)_xpad);
        Mem.Free((byte*)_tapOffsets);
        Mem.Free((byte*)_kOffsets);
        Mem.Free((byte*)_xsub);
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

        FillBiasPanels();
        Plan();
        if (_shifted)
        {
            PreparePadded();
            CpuParallel.For((int)(_n * _c), _padBody);
            CpuParallel.For(_g * _mBlocks * _cBlocks, _gemmBody);
            return;
        }

        if (_prepacked)
        {
            CpuParallel.For(_g * _tiles * _kChunks, _packBody);
        }

        if (_subsample)
        {
            long floats = _n * _c * _hw;
            if (floats > _xsubFloats)
            {
                Mem.Free((byte*)_xsub);
                _xsub = (float*)Mem.Alloc(floats * 4, false);
                _xsubFloats = floats;
            }

            CpuParallel.For((int)(_n * _c), _subsampleBody);
        }

        CpuParallel.For(_g * _mBlocks * _cBlocks, _gemmBody);
    }

    /// <summary>Copies the bias into the zero-padded per-panel layout the micro-kernel initializes its accumulators from.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void FillBiasPanels()
    {
        if (_bias0 == null)
        {
            return;
        }

        int MR = Gemm.MR;
        for (int g = 0; g < _g; g++)
        {
            for (int r = 0; r < _mp * MR; r++)
            {
                _biasPanels[(long)g * _mp * MR + r] = r < _mg ? _bias0[g * _mg + r] : 0f;
            }
        }
    }

    /// <summary>
    /// Splits the run into tasks: all weight panels per task while the output tiles alone give about two tasks per thread
    /// (each im2col element is then packed once, inside its task), else fewer panels per task with B packed once up front,
    /// or, for stride-1 convs, the shifted form, which needs no packing at all. Tiles per task follow from about three tasks
    /// per thread and the C block bound; the K chunk keeps the task's A block in L2.
    /// </summary>
    private void Plan()
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        _nc = _n * _hw;
        _shifted = false;
        int tiles = (int)((_nc + NR - 1) / NR);
        PlanTasks(tiles, _direct || _subsample ? _n * _c * _hw * 4 : (long)_g * tiles * _kg * NR * 4);
        if (_mBlocks > 1 && _tapMajor)
        {
            _shifted = true;
            PlanTasks((int)(_n * _tilesPerImage), _n * _c * _plane * 4);
        }

        int kcMax = (int)Math.Clamp(ABlockBytes / ((long)_mpt * MR * 4), 64, 512);
        _kChunks = (_kg + kcMax - 1) / kcMax;
        _kc = (_kg + _kChunks - 1) / _kChunks;

        long prepackBytes = (long)_g * _tiles * _kg * NR * 4;
        _prepacked = _mBlocks > 1 && !_direct && !_subsample && !_shifted && prepackBytes <= MaxPrepackBytes;
        if (_prepacked && prepackBytes > _bpackBytes)
        {
            Mem.Free((byte*)_bpack);
            _bpack = (float*)Mem.Alloc(prepackBytes, false);
            _bpackBytes = prepackBytes;
        }
    }

    /// <summary>
    /// Weight panels and output tiles per task for <paramref name="tiles"/> output tiles whose B operand (the input, or its
    /// im2col view) is <paramref name="bBytes"/> bytes. Two splits are considered: few weight-panel blocks (B read about
    /// once, the weights once per tile block) and few tile blocks (the weights read about once, B once per panel block).
    /// The one with less operand traffic wins; small batches of late layers, where the weights are much larger than the
    /// activations, take the second.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PlanTasks(int tiles, long bBytes)
    {
        int threads = CpuParallel.Threads;
        _tiles = tiles;
        long aBytes = (long)_g * _mp * _kg * Gemm.MR * 4;

        // Few panel blocks: split the weights only until the tiles give about two tasks per thread.
        int mBlocks = 1;
        int mpt = _mp;
        while ((long)_g * mBlocks * _tiles < 2L * threads && mpt > 1)
        {
            mBlocks = Math.Min(_mp, mBlocks * 2);
            mpt = (_mp + mBlocks - 1) / mBlocks;
            mBlocks = (_mp + mpt - 1) / mpt;
        }

        int maxTiles = Math.Clamp(MaxPairsPerTask / mpt, 1, MaxTilesPerTask);
        int ntt = (int)Math.Clamp((long)_g * mBlocks * _tiles / ((long)TasksPerThread * threads), 1, maxTiles);
        int cBlocks = (_tiles + ntt - 1) / ntt;
        ntt = (_tiles + cBlocks - 1) / cBlocks;
        double traffic = Traffic(aBytes, bBytes, mBlocks, cBlocks, threads);

        // Few tile blocks: all tiles per task as far as the C block allows, panels split for two tasks per thread.
        int ntt2 = Math.Min(_tiles, MaxPairsPerTask);
        int cBlocks2 = (_tiles + ntt2 - 1) / ntt2;
        ntt2 = (_tiles + cBlocks2 - 1) / cBlocks2;
        int mpt2 = Math.Max(1, (int)Math.Min(_mp, (long)_g * _mp * cBlocks2 / (2L * threads)));
        mpt2 = Math.Min(mpt2, Math.Max(1, MaxPairsPerTask / ntt2));
        int mBlocks2 = (_mp + mpt2 - 1) / mpt2;
        mpt2 = (_mp + mBlocks2 - 1) / mBlocks2;
        while ((long)_g * mBlocks2 * cBlocks2 < threads && ntt2 > 1)
        {
            ntt2--;
            cBlocks2 = (_tiles + ntt2 - 1) / ntt2;
        }

        double traffic2 = Traffic(aBytes, bBytes, mBlocks2, cBlocks2, threads);
        if (traffic2 < 0.75 * traffic)
        {
            mBlocks = mBlocks2;
            mpt = mpt2;
            cBlocks = cBlocks2;
            ntt = ntt2;
        }

        _cBlocks = cBlocks;
        _ntt = ntt;
        _mpt = mpt;
        _mBlocks = mBlocks;
    }

    /// <summary>Strided 1x1 convs: copies the sampled pixels of input plane <paramref name="task"/> (image * C + channel).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void SubsampleBody(int task)
    {
        float* src = (float*)_x.P + (long)task * _h * _w;
        float* dst = _xsub + (long)task * _hw;
        for (int oh = 0; oh < _ho; oh++)
        {
            float* s = src + (long)oh * _sh * _w;
            float* d = dst + (long)oh * _wo;
            if (_sw == 2)
            {
                Simd.CompressEven(s, d, _wo);
                continue;
            }

            for (int ow = 0; ow < _wo; ow++)
            {
                d[ow] = s[(long)ow * _sw];
            }
        }
    }

    /// <summary>
    /// Operand bytes moved through the shared caches for a split into <paramref name="mBlocks"/> x <paramref name="cBlocks"/>
    /// tasks: each weight block is read by every tile block and each B block by every panel block, except that a block small
    /// enough to stay in a core's L2 is fetched about once per thread rather than once per task.
    /// </summary>
    private static double Traffic(long aBytes, long bBytes, int mBlocks, int cBlocks, int threads)
    {
        double aReads = aBytes / mBlocks <= ResidentBlockBytes ? Math.Min(cBlocks, threads) : cBlocks;
        double bReads = bBytes / cBlocks <= ResidentBlockBytes ? Math.Min(mBlocks, threads) : mBlocks;
        return aBytes * aReads + bBytes * bReads;
    }

    /// <summary>Pack-once task: K chunk of one output tile of one group into the shared B buffer.</summary>
    private void PackBody(int task)
    {
        int chunk = task % _kChunks;
        int tile = task / _kChunks % _tiles;
        int g = task / (_kChunks * _tiles);
        int k0 = chunk * _kc;
        int k1 = Math.Min(_kg, k0 + _kc);
        PackColumns(g, (long)tile * Gemm.NR, k0, k1, _bpack + ((long)(g * _tiles + tile) * _kg + k0) * Gemm.NR);
    }

    /// <summary>
    /// GEMM task: weight panels [p0, p1) of one group over output tiles [t0, t1), all of K, into the task's C block; then the
    /// block's valid part goes to y through the epilogue.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void GemmBody(int task)
    {
        int MR = Gemm.MR;
        int NR = Gemm.NR;
        int cb = task % _cBlocks;
        int mb = task / _cBlocks % _mBlocks;
        int g = task / (_cBlocks * _mBlocks);
        int p0 = mb * _mpt;
        int p1 = Math.Min(_mp, p0 + _mpt);
        int t0 = cb * _ntt;
        int t1 = Math.Min(_tiles, t0 + _ntt);
        if (p0 >= p1 || t0 >= t1)
        {
            return;
        }

        long ldc = (long)(t1 - t0) * NR;
        long cFloats = (p1 - p0) * MR * ldc;
        float* cbuf = (float*)ThreadScratch.Get(ThreadScratch.Worker, (cFloats + (long)_kc * NR) * 4);
        if (_shifted)
        {
            ShiftedGemm(g, p0, p1, t0, t1, cbuf, ldc);
            ShiftedCopyOut(g, p0, p1, t0, t1, cbuf, ldc);
            return;
        }

        float* bbuf = cbuf + cFloats;
        long kg = _kg;
        float* aGroup = _apack + (long)g * _mp * kg * MR;
        float* biasGroup = _biasPanels == null ? null : _biasPanels + (long)g * _mp * MR;
        for (int kb = 0; kb < _kg; kb += _kc)
        {
            int kc = Math.Min(_kc, _kg - kb);
            bool acc = kb > 0;
            for (int t = t0; t < t1; t++)
            {
                float* b = TileB(g, t, kb, kc, bbuf, out long ldb);
                float* ct = cbuf + (long)(t - t0) * NR;
                for (int p = p0; p < p1; p++)
                {
                    float* bias = acc || biasGroup == null ? null : biasGroup + (long)p * MR;
                    Gemm.Tile(aGroup + ((long)p * kg + kb) * MR, b, ldb, ct + (long)(p - p0) * MR * ldc, ldc, kc, bias, acc);
                }
            }
        }

        CopyOut(g, p0, p1, t0, t1, cbuf, ldc);
    }

    /// <summary>
    /// The [kc][NR] B micro-panel of tile <paramref name="t"/> at K offset <paramref name="kb"/>: from the pack-once buffer, in
    /// place for interior tiles of 1x1 stride-1 convs (row stride HW), else packed into <paramref name="scratch"/>.
    /// </summary>
    private float* TileB(int g, int t, int kb, int kc, float* scratch, out long ldb)
    {
        int NR = Gemm.NR;
        long col0 = (long)t * NR;
        if (_prepacked)
        {
            ldb = NR;
            return _bpack + ((long)(g * _tiles + t) * _kg + kb) * NR;
        }

        if (_direct || _subsample)
        {
            long img = col0 / _hw;
            long pix = col0 - img * _hw;
            if (pix + NR <= _hw)
            {
                ldb = _hw;
                float* src = _subsample ? _xsub : (float*)_x.P;
                return src + ((img * _c + (long)g * _cg) * _hw) + (long)kb * _hw + pix;
            }
        }

        PackColumns(g, col0, kb, kb + kc, scratch);
        ldb = NR;
        return scratch;
    }

    /// <summary>
    /// Fills rows [k0, k1) of the im2col micro-panel of output columns [col0, col0 + NR) of group <paramref name="g"/> into
    /// <paramref name="panel"/> ([k1 - k0][NR]); columns past the last output are zero.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void PackColumns(int g, long col0, int k0, int k1, float* panel)
    {
        int NR = Gemm.NR;
        int ncols = (int)Math.Min(NR, _nc - col0);
        // Segments: runs of output columns inside one image and one output row.
        float** segX = stackalloc float*[NR];
        int* segOh = stackalloc int[NR];
        int* segOw = stackalloc int[NR];
        int* segJ = stackalloc int[NR];
        int* segLen = stackalloc int[NR];
        int ns = 0;
        long plane = (long)_h * _w;
        for (int j = 0; j < ncols;)
        {
            long col = col0 + j;
            long img = col / _hw;
            int pix = (int)(col - img * _hw);
            int oh = pix / _wo;
            int ow = pix - oh * _wo;
            int len = Math.Min(_wo - ow, ncols - j);
            segX[ns] = (float*)_x.P + (img * _c + (long)g * _cg) * plane;
            segOh[ns] = oh;
            segOw[ns] = ow;
            segJ[ns] = j;
            segLen[ns] = len;
            ns++;
            j += len;
        }

        int W = _w;
        int H = _h;
        int khw = _kh * _kw;
        // K order: (c, fh, fw), or (fh, fw, c) for tap-major weights.
        int c = _tapMajor ? k0 % _cg : k0 / khw;
        int r = _tapMajor ? k0 / _cg : k0 - c * khw;
        int fh = r / _kw;
        int fw = r - fh * _kw;
        for (int k = k0; k < k1; k++)
        {
            float* row = panel + (long)(k - k0) * NR;
            long cOff = (long)c * plane;
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

                float* xr = segX[s] + cOff + (long)ih * W;
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
                else if (_sw == 2)
                {
                    // In range: 0 <= off + 2t < W.
                    int lo = Math.Clamp((1 - off) / 2, 0, len);
                    if (off + 2 * lo < 0)
                    {
                        lo = Math.Min(len, lo + 1);
                    }

                    int hi = Math.Clamp(W - off <= 0 ? 0 : (W - off + 1) / 2, lo, len);
                    new Span<float>(d, lo).Clear();
                    Simd.CompressEven(xr + off + 2 * lo, d + lo, hi - lo);
                    new Span<float>(d + hi, len - hi).Clear();
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

            if (_tapMajor)
            {
                if (++c == _cg)
                {
                    c = 0;
                    if (++fw == _kw)
                    {
                        fw = 0;
                        fh++;
                    }
                }
            }
            else if (++fw == _kw)
            {
                fw = 0;
                if (++fh == _kh)
                {
                    fh = 0;
                    c++;
                }
            }
        }
    }

    /// <summary>
    /// Writes the valid rows (output channels) and columns (pixels of each image) of a task's C block to y, applying the
    /// epilogue to each contiguous output row segment.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void CopyOut(int g, int p0, int p1, int t0, int t1, float* cbuf, long ldc)
    {
        int MR = Gemm.MR;
        long colStart = (long)t0 * Gemm.NR;
        long colEnd = Math.Min(_nc, (long)t1 * Gemm.NR);
        int rowStart = p0 * MR;
        int rowEnd = Math.Min(_mg, p1 * MR);
        float* y = (float*)_y.P;
        for (long col = colStart; col < colEnd;)
        {
            long img = col / _hw;
            long pix = col - img * _hw;
            int len = (int)Math.Min(_hw - pix, colEnd - col);
            float* src = cbuf + (col - colStart);
            float* dst = y + (img * _o + (long)g * _mg + rowStart) * _hw + pix;
            for (int r = rowStart; r < rowEnd; r++)
            {
                if (_epilogue is null)
                {
                    new ReadOnlySpan<float>(src, len).CopyTo(new Span<float>(dst, len));
                }
                else
                {
                    _epilogue.ApplyFrom(src, dst, dst - y, len);
                }
                src += ldc;
                dst += _hw;
            }

            col += len;
        }
    }
}
