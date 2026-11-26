using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// 3x3 stride-1 convolution (groups 1, no dilation) on channels-last activations by Winograd F(4x4, 3x3): every 6x6 input
/// tile (overlapping by two pixels) gives a 4x4 output tile with 36 multiplies per channel pair instead of 144. The input
/// transform (Bᵀ d B) of a task's tiles is written as 36 row-major [tile][C] matrices, each multiplied by its transformed
/// weights (G g Gᵀ, packed once into [position][panel][C][NB] panels) with the channels-last conv's register micro-kernels,
/// and the output transform (Aᵀ m A) adds the bias, a residual addend and Relu / Clip in registers (other activations per
/// stored vector). A task is a block of tiles x a block of output-channel panels and runs over the input channels in chunks
/// of <see cref="ChannelChunk"/> (transform, then accumulate the products), so everything it touches stays in its core's
/// caches; the input transform is repeated per panel block, which is cheap against the products. The transforms use the points 0, ±1, ±2: the f32 error is a few
/// times the direct conv's (measured in <c>WinogradConvTests</c>), see <see cref="Applies"/> for the shapes it takes.
/// </summary>
internal sealed unsafe class WinogradConvOp : CpuOp
{
    /// <summary>Fewest input and output channels for which the transforms pay off.</summary>
    public const int MinChannels = 32;

    /// <summary>Fewest 4x4 output tiles per image (see <see cref="Applies"/>).</summary>
    public const int MinTilesPerImage = 16;

    /// <summary>Largest transformed weights (bytes, see <see cref="Applies"/>).</summary>
    public const long MaxWeightBytes = 16L << 20;

    private const int Positions = 36;

    /// <summary>K values per chunk of a weight panel, so that the chunk stays in L1 while the task's tile groups run over it.</summary>
    private const int ChunkValues = 256;

    /// <summary>
    /// Input channels transformed at a time inside a task: the products accumulate over the chunks, so a task's scratch holds
    /// the transformed input of one chunk and the products, and a task can take more tiles per read of its weights.
    /// </summary>
    private const int ChannelChunk = 64;

    /// <summary>Upper bound of a task's transformed input and product scratch (bytes), about half of a core's L2.</summary>
    private const long ScratchBytes = 640 * 1024;

    /// <summary>Planning estimate: f32 multiply-add rate of one thread (flop/s).</summary>
    private const double ThreadFlops = 1.3e11;

    /// <summary>
    /// Planning estimate: bandwidth of the shared caches for operand fetches (bytes/s). Lower than the direct conv's: on a
    /// two-CCD Ryzen the tasks' input and weight reads cross between the CCDs' L3s (8 → 16 threads gained nothing before
    /// input re-reads were counted).
    /// </summary>
    private const double SharedBytesPerSecond = 1.2e11;

    /// <summary>Planning estimate: fixed cost of one task (s).</summary>
    private const double TaskSeconds = 2e-7;

    /// <summary>Planning estimate: operations of the input transform per tile and input channel (36 values).</summary>
    private const double InputTransformOps = 400;

    /// <summary>Batch from which <c>auto</c> runs Winograd (on tiers without AVX-512).</summary>
    public const int AutoMinBatch = 4;

    /// <summary>
    /// When convs run on Winograd: <c>auto</c> (default) on tiers without AVX-512 from a batch of <see cref="AutoMinBatch"/>,
    /// <c>n</c> &gt; 0 from a batch of n on every tier (<c>1</c>: always), <c>0</c> never; batches below the minimum take the
    /// direct conv at run time. Read from <c>SHARPLIGHT_CPU_WINOGRAD</c>; tests and benchmarks set it to compare with the
    /// direct channels-last conv. Measured on a Ryzen 9950X (docs/benchmarks.md §117): with AVX-512 the direct conv's 8x32
    /// kernels beat Winograd inside models at batch 1 and 8; with AVX2 Winograd wins about 10 % of ResNet-50 at batch 8 and
    /// loses at batch 1.
    /// </summary>
    internal static string Policy { get; set; } = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_WINOGRAD") is { Length: > 0 } v ? v.Trim() : "auto";

    /// <summary>The smallest batch that runs on Winograd under <see cref="Policy"/>, or <see cref="int.MaxValue"/> for none.</summary>
    internal static int MinBatch => int.TryParse(Policy, out int n)
        ? n > 0 ? n : int.MaxValue
        : CpuIsa.Use512 ? int.MaxValue : AutoMinBatch;

    private readonly CpuOp? _direct;  // runs batches below _minBatch
    private readonly int _minBatch;
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly OutputEpilogue? _epilogue;
    private readonly long _xps;
    private readonly long _xrs;
    private readonly long _xis;
    private readonly int _c;
    private readonly int _h;
    private readonly int _w;
    private readonly int _o;
    private readonly int _pt;
    private readonly int _pl;
    private readonly int _ho;
    private readonly int _wo;
    private readonly int _th;
    private readonly int _tw;
    private readonly int _mr;
    private readonly int _nb;
    private readonly int _panels;
    private readonly float* _upack;   // [position][panel][C][NB]
    private readonly float* _zero;    // a zero row of C floats
    private readonly float _lo;
    private readonly float _hi;
    private readonly bool _inlineActivation;
    private readonly Action<int> _body;

    // per-run state
    private long _n;
    private long _tiles;
    private int _tpt;
    private int _npt;
    private int _tBlocks;
    private int _pBlocks;
    private long _plannedTiles = -1;
    private bool _ranDirect;

    private WinogradConvOp(TRef x, float* wDense, TRef? bias, TRef y, int pt, int pl, OutputEpilogue? epilogue, CpuOp? direct, int minBatch)
    {
        _direct = direct;
        _minBatch = minBatch;
        _x = x;
        _y = y;
        _bias = bias;
        _epilogue = epilogue;
        _xps = ChannelsLast.PixelStride(x);
        _xrs = ChannelsLast.RowStride(x);
        _xis = ChannelsLast.ImageStride(x);
        _c = (int)x.Dims[1];
        _h = (int)x.Dims[2];
        _w = (int)x.Dims[3];
        _o = (int)y.Dims[1];
        _ho = (int)y.Dims[2];
        _wo = (int)y.Dims[3];
        _pt = pt;
        _pl = pl;
        _th = (_ho + 3) / 4;
        _tw = (_wo + 3) / 4;
        _mr = CpuIsa.Use512 ? 8 : 6;
        _nb = CpuIsa.Use512 ? 32 : 16;
        _panels = (_o + _nb - 1) / _nb;
        _body = Body;
        _upack = (float*)Mem.Alloc((long)Positions * _panels * _c * _nb * 4, true);
        TransformWeights(wDense);
        _zero = (float*)Mem.Alloc(_c * 4L, true);
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
    public override string Name => _ranDirect ? _direct!.Name : "Conv.wino" + _epilogue?.Label;

    /// <inheritdoc />
    public override string Detail => _ranDirect ? _direct!.Detail : $"{_n}x{_c}x{_h}x{_w}>{_o} k3s1 {2.0 * _n * _ho * _wo * _o * _c * 9 / 1e6:F0}MF tiles {_tiles} tasks {_pBlocks}x{_tBlocks} ({_tpt} tiles)";

    /// <summary>
    /// True when a conv with these parameters runs on Winograd: a 3x3 kernel at stride 1 without dilation, at least
    /// <see cref="MinChannels"/> input and output channels (multiples of 16), 256-bit vectors, no column-bias epilogue, and
    /// enough tiles per image to amortize the transformed weights, which are four times the size of the direct conv's: at
    /// least <see cref="MinTilesPerImage"/> 4x4 output tiles, and transformed weights of at most <see cref="MaxWeightBytes"/>.
    /// Late 7x7 layers with 512 channels (4 tiles, 38 MB of transformed weights) ran at 0.55-0.65x of the direct conv.
    /// </summary>
    public static bool Applies(int channels, int outputs, int outH, int outW, int kh, int kw, int[] strides, int[] dil, OutputEpilogue? epilogue)
        => MinBatch < int.MaxValue && CpuIsa.Use256 && kh == 3 && kw == 3 && strides is [1, 1] && dil is [1, 1]
            && channels >= MinChannels && outputs >= MinChannels && channels % 16 == 0 && outputs % 16 == 0
            && epilogue?.ColumnBias is null && (long)((outH + 3) / 4) * ((outW + 3) / 4) >= MinTilesPerImage
            && (long)Positions * channels * outputs * 4 <= MaxWeightBytes;

    /// <summary>
    /// The Winograd op for a conv that <see cref="Applies"/>, or the op of <paramref name="direct"/> when the batch is fixed
    /// below <see cref="MinBatch"/>; with a symbolic batch that may fall below it, the Winograd op keeps the direct op and runs
    /// it for small batches. <paramref name="x"/> is a channels-last view, <paramref name="y"/> and the epilogue's residual
    /// are channels-last.
    /// </summary>
    public static CpuOp Create(TRef x, float* wDense, TRef? bias, TRef y, int[] pads, OutputEpilogue? epilogue, Func<CpuOp> direct)
    {
        int minBatch = MinBatch;
        if (x.Sym is null)
        {
            return x.Dims[0] >= minBatch ? new WinogradConvOp(x, wDense, bias, y, pads[0], pads[1], epilogue, null, minBatch) : direct();
        }

        return new WinogradConvOp(x, wDense, bias, y, pads[0], pads[1], epilogue, minBatch > 1 ? direct() : null, minBatch);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Mem.Free((byte*)_upack);
        Mem.Free((byte*)_zero);
        _direct?.Dispose();
    }

    /// <inheritdoc />
    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _ranDirect = _n < _minBatch && _direct != null;
        if (_ranDirect)
        {
            _direct!.Run(p);
            return;
        }

        _tiles = _n * _th * _tw;
        if (_tiles == 0)
        {
            return;
        }

        if (_tiles != _plannedTiles)
        {
            Plan();
            _plannedTiles = _tiles;
        }

        CpuParallel.For(_pBlocks * _tBlocks, _body);
    }

    /// <summary>U = G g Gᵀ per output and input channel, in f64, packed into [position][panel][C][NB].</summary>
    private void TransformWeights(float* wDense)
    {
        ReadOnlySpan<double> g =
        [
            1.0 / 4, 0, 0,
            -1.0 / 6, -1.0 / 6, -1.0 / 6,
            -1.0 / 6, 1.0 / 6, -1.0 / 6,
            1.0 / 24, 1.0 / 12, 1.0 / 6,
            1.0 / 24, -1.0 / 12, 1.0 / 6,
            0, 0, 1,
        ];
        Span<double> t = stackalloc double[18];
        for (int o = 0; o < _o; o++)
        {
            int panel = o / _nb;
            int lane = o % _nb;
            for (int c = 0; c < _c; c++)
            {
                float* k = wDense + ((long)o * _c + c) * 9;
                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        t[i * 3 + j] = g[i * 3] * k[j] + g[i * 3 + 1] * k[3 + j] + g[i * 3 + 2] * k[6 + j];
                    }
                }

                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        double u = t[i * 3] * g[j * 3] + t[i * 3 + 1] * g[j * 3 + 1] + t[i * 3 + 2] * g[j * 3 + 2];
                        _upack[(((long)(i * 6 + j) * _panels + panel) * _c + c) * _nb + lane] = (float)u;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Chooses tiles and panels per task by an estimate of the run time: products and input transforms (repeated per panel
    /// block) spread over the threads with the last task's tail, each task's block of the transformed weights read once
    /// through the shared caches, and a cost per task; a task's scratch stays within <see cref="ScratchBytes"/>. Every task
    /// keeps its transforms in its own core's caches: a schedule of separate transform and product passes over shared
    /// buffers was 1.3-2.5x slower on the two-CCD Ryzen 9950X, whose cross-CCD traffic is slow.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Plan()
    {
        int threads = CpuParallel.Threads;
        long groups = (_tiles + _mr - 1) / _mr;
        double inputBytes = (double)_n * _h * _w * _c * 4;
        double best = double.MaxValue;
        _tpt = _mr;
        _npt = 1;
        for (int npt = 1; npt <= _panels; npt++)
        {
            int pBlocks = (_panels + npt - 1) / npt;
            if (npt > 1 && (_panels + npt - 2) / (npt - 1) == pBlocks)
            {
                continue;
            }

            for (long gpt = 1; gpt <= groups; gpt = Math.Max(gpt + 1, gpt * 5 / 4))
            {
                int tpt = (int)(gpt * _mr);
                long scratch = (long)Positions * tpt * (Math.Min(_c, ChannelChunk) + npt * _nb) * 4;
                if (scratch > ScratchBytes && gpt > 1)
                {
                    break;
                }

                long tBlocks = (groups + gpt - 1) / gpt;
                long tasks = tBlocks * pBlocks;
                double products = 2.0 * Positions * tBlocks * tpt * _c * (double)pBlocks * npt * _nb;
                double transforms = InputTransformOps * tBlocks * tpt * _c * pBlocks;
                double compute = (products + transforms) / ThreadFlops * (1.0 / threads + 1.0 / tasks);
                double weights = (double)Positions * _c * npt * _nb * 4 * tasks;
                // Every panel block reads the input again (the 6x6 patches overlap by 2.25x, mostly within a task's tiles).
                double input = inputBytes * pBlocks * (1 + 2.0 / Math.Max(1, tpt / _tw));
                double time = compute + (weights + input) / SharedBytesPerSecond + tasks * TaskSeconds;
                if (time < best)
                {
                    best = time;
                    _tpt = tpt;
                    _npt = npt;
                }
            }
        }

        _pBlocks = (_panels + _npt - 1) / _npt;
        _tBlocks = (int)((_tiles + _tpt - 1) / _tpt);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int task)
    {
        if (CpuIsa.Use512)
        {
            RunTask<FloatLanes512>(task);
        }
        else
        {
            RunTask<FloatLanes256>(task);
        }
    }

    /// <summary>One fused task: the input transform of tiles [t0, t0 + T), the 36 products for panels [p0, p1), the output transform.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void RunTask<TL>(int task) where TL : struct, IFloatLanes<TL>
    {
        int pb = task / _tBlocks;
        int tb = task - pb * _tBlocks;
        int p0 = pb * _npt;
        int p1 = Math.Min(_panels, p0 + _npt);
        long t0 = (long)tb * _tpt;
        if (p0 >= p1 || t0 >= _tiles)
        {
            return;
        }

        int count = (int)Math.Min(_tpt, _tiles - t0);
        int tpt = (count + _mr - 1) / _mr * _mr;
        int ldm = (p1 - p0) * _nb;
        int cc = Math.Min(_c, ChannelChunk);
        long vFloats = (long)Positions * tpt * cc;
        long mFloats = (long)Positions * tpt * ldm;
        float* v = (float*)ThreadScratch.Get(ThreadScratch.Worker, (vFloats + mFloats) * 4 + 64);
        v = (float*)(((nint)v + 63) & ~(nint)63);
        float* m = v + vFloats;
        for (int c0 = 0; c0 < _c; c0 += cc)
        {
            int c1 = Math.Min(_c, c0 + cc);
            int len = c1 - c0;
            for (int i = 0; i < tpt; i++)
            {
                InputTransform<TL>(t0 + i, i < count, c0, c1, v + (long)i * len, (long)tpt * len);
            }

            for (int xi = 0; xi < Positions; xi++)
            {
                Products(xi, v + (long)xi * tpt * len, c0, c1, m + (long)xi * tpt * ldm, ldm, 0, tpt, p0, p1);
            }
        }

        int cols = Math.Min(ldm, _o - p0 * _nb);
        for (int i = 0; i < count; i++)
        {
            OutputTransform<TL>(t0 + i, m + (long)i * ldm, (long)tpt * ldm, p0 * _nb, cols);
        }
    }

    /// <summary>
    /// Adds the products of position <paramref name="xi"/> over input channels [c0, c1) for tile rows [r0, r1) (multiples of the
    /// tile height) and panels [p0, p1) (the first chunk stores them): transformed input row r holds channels [c0, c1) at
    /// <paramref name="v"/> + r * (c1 - c0), the results go to <paramref name="m"/> + row * <paramref name="ldm"/> + (panel - p0) * NB.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Products(int xi, float* v, int c0, int c1, float* m, long ldm, int r0, int r1, int p0, int p1)
    {
        float** ptrs = stackalloc float*[8];
        int len = c1 - c0;
        for (int p = p0; p < p1; p++)
        {
            float* u = _upack + ((long)xi * _panels + p) * _c * _nb;
            for (int k0 = c0; k0 < c1; k0 += ChunkValues)
            {
                int k1 = Math.Min(c1, k0 + ChunkValues);
                bool acc = k0 > 0;
                for (int g = r0; g < r1; g += _mr)
                {
                    // The kernel reads value k of row i at ptrs[i][k]: point the rows c0 values before their chunk.
                    for (int i = 0; i < _mr; i++)
                    {
                        ptrs[i] = v + (long)(g + i) * len - c0;
                    }

                    float* c = m + g * ldm + (p - p0) * _nb;
                    if (CpuIsa.Use512)
                    {
                        ChannelsLastConvOp.Kernel8x32(ptrs, 0, 1, k0, k1, _c, u, null, acc, c, ldm, null, float.NegativeInfinity, float.PositiveInfinity);
                    }
                    else
                    {
                        ChannelsLastConvOp.Kernel6x16(ptrs, 0, 1, k0, k1, _c, u, null, acc, c, ldm, null, float.NegativeInfinity, float.PositiveInfinity);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Bᵀ d B of tile <paramref name="tile"/> for channels [c0, c1): position xi goes to <paramref name="v"/> + xi *
    /// <paramref name="stride"/> (zeros for a padding tile past the last one).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void InputTransform<TL>(long tile, bool real, int c0, int c1, float* v, long stride) where TL : struct, IFloatLanes<TL>
    {
        float** rows = stackalloc float*[Positions];
        if (real)
        {
            long per = (long)_th * _tw;
            long n = tile / per;
            int r = (int)(tile - n * per);
            int th = r / _tw;
            int tw = r - th * _tw;
            float* image = (float*)_x.P + n * _xis;
            for (int i = 0; i < 6; i++)
            {
                int ih = th * 4 - _pt + i;
                for (int j = 0; j < 6; j++)
                {
                    int iw = tw * 4 - _pl + j;
                    rows[i * 6 + j] = (uint)ih < (uint)_h && (uint)iw < (uint)_w ? image + ih * _xrs + iw * _xps : _zero;
                }
            }
        }
        else
        {
            for (int i = 0; i < Positions; i++)
            {
                rows[i] = _zero;
            }
        }

        int lanes = TL.Count;
        float* tmp = stackalloc float[Positions * 16];
        for (int c = c0; c < c1; c += lanes)
        {
            for (int j = 0; j < 6; j++)
            {
                Forward(
                    TL.Load(rows[j] + c), TL.Load(rows[6 + j] + c), TL.Load(rows[12 + j] + c),
                    TL.Load(rows[18 + j] + c), TL.Load(rows[24 + j] + c), TL.Load(rows[30 + j] + c),
                    tmp + j * lanes, 6L * lanes);
            }

            for (int i = 0; i < 6; i++)
            {
                float* t = tmp + i * 6 * lanes;
                Forward(
                    TL.Load(t), TL.Load(t + lanes), TL.Load(t + 2 * lanes), TL.Load(t + 3 * lanes), TL.Load(t + 4 * lanes),
                    TL.Load(t + 5 * lanes), v + i * 6 * stride + (c - c0), stride);
            }
        }
    }

    /// <summary>
    /// Aᵀ m A of tile <paramref name="tile"/> for output channels [o0, o0 + cols): product position xi is at <paramref name="m"/> +
    /// xi * <paramref name="stride"/>; adds the bias and the residual, clamps and stores the valid pixels.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void OutputTransform<TL>(long tile, float* m, long stride, int o0, int cols) where TL : struct, IFloatLanes<TL>
    {
        long per = (long)_th * _tw;
        long n = tile / per;
        int r = (int)(tile - n * per);
        int th = r / _tw;
        int tw = r - th * _tw;
        int rows = Math.Min(4, _ho - th * 4);
        int wcols = Math.Min(4, _wo - tw * 4);
        float* y = (float*)_y.P;
        float* residual = _epilogue?.Residual is { } res ? (float*)res.P : null;
        float* bias = _bias is null ? null : (float*)_bias.P;
        int lanes = TL.Count;
        float* tmp = stackalloc float[24 * 16];
        var lo = TL.Broadcast(_lo);
        var hi = TL.Broadcast(_hi);
        float* outRow = stackalloc float[4 * 16];
        for (int c = 0; c < cols; c += lanes)
        {
            for (int j = 0; j < 6; j++)
            {
                float* mj = m + j * stride + c;
                Inverse(
                    TL.Load(mj), TL.Load(mj + 6 * stride), TL.Load(mj + 12 * stride), TL.Load(mj + 18 * stride),
                    TL.Load(mj + 24 * stride), TL.Load(mj + 30 * stride), tmp + j * lanes, 6L * lanes);
            }

            var b = bias == null ? TL.Zero : TL.Load(bias + o0 + c);
            for (int i = 0; i < rows; i++)
            {
                float* t = tmp + i * 6 * lanes;
                Inverse(
                    TL.Load(t), TL.Load(t + lanes), TL.Load(t + 2 * lanes), TL.Load(t + 3 * lanes), TL.Load(t + 4 * lanes),
                    TL.Load(t + 5 * lanes), outRow, lanes);
                long pixel = (n * _ho + th * 4 + i) * _wo + tw * 4;
                for (int j = 0; j < wcols; j++)
                {
                    long off = (pixel + j) * _o + o0 + c;
                    var value = TL.Add(TL.Load(outRow + j * lanes), b);
                    if (residual != null)
                    {
                        value = TL.Add(value, TL.Load(residual + off));
                    }

                    TL.Store(TL.Max(TL.Min(value, hi), lo), y + off);
                    if (!_inlineActivation)
                    {
                        ScalarMath.Unary(_epilogue!.Activation!.Value, y + off, y + off, lanes, _epilogue.Scale, _epilogue.Bias);
                    }
                }
            }
        }
    }

    /// <summary>The six values of Bᵀ d for one column d0..d5, stored at <paramref name="dst"/> + i * <paramref name="stride"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Forward<TL>(TL d0, TL d1, TL d2, TL d3, TL d4, TL d5, float* dst, long stride) where TL : struct, IFloatLanes<TL>
    {
        var m4 = TL.Broadcast(-4f);
        var m5 = TL.Broadcast(-5f);
        var p4 = TL.Broadcast(4f);
        var a = TL.Fma(m4, d2, d4);
        var b = TL.Fma(m4, d1, d3);
        var c = TL.Subtract(d4, d2);
        var e = TL.Subtract(d3, d1);
        e = TL.Add(e, e);
        TL.Store(TL.Fma(p4, d0, TL.Fma(m5, d2, d4)), dst);
        TL.Store(TL.Add(a, b), dst + stride);
        TL.Store(TL.Subtract(a, b), dst + 2 * stride);
        TL.Store(TL.Add(c, e), dst + 3 * stride);
        TL.Store(TL.Subtract(c, e), dst + 4 * stride);
        TL.Store(TL.Fma(p4, d1, TL.Fma(m5, d3, d5)), dst + 5 * stride);
    }

    /// <summary>The four values of Aᵀ m for one column m0..m5, stored at <paramref name="dst"/> + i * <paramref name="stride"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Inverse<TL>(TL m0, TL m1, TL m2, TL m3, TL m4, TL m5, float* dst, long stride) where TL : struct, IFloatLanes<TL>
    {
        var p = TL.Add(m1, m2);
        var q = TL.Subtract(m1, m2);
        var r = TL.Add(m3, m4);
        var s = TL.Subtract(m3, m4);
        TL.Store(TL.Add(TL.Add(m0, p), r), dst);
        TL.Store(TL.Fma(TL.Broadcast(2f), s, q), dst + stride);
        TL.Store(TL.Fma(TL.Broadcast(4f), r, p), dst + 2 * stride);
        TL.Store(TL.Add(TL.Fma(TL.Broadcast(8f), s, q), m5), dst + 3 * stride);
    }
}
