using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// y = x · Wᵀ for f32 activation rows x and a weight W [N, K] in any <see cref="WeightFormat"/>. Three paths:
/// <list type="bullet">
/// <item>GEMM (at least <see cref="GemmMinRows"/> rows, <see cref="IntDotGemmMinRows"/> with integer dots): the weight is dequantized panel by panel into the packed B operand
/// of the register-blocked <see cref="Gemm"/> micro-kernel; the activations are packed once as the A operand.</item>
/// <item>Integer dot (fewer rows, <see cref="DeviceFeatures.IntDot"/> enabled and the format supported): activations are
/// quantized to 8 bits and multiplied with the decoded weight codes in int32 (<see cref="IntDot"/>).</item>
/// <item>Float rows: on-the-fly dequantization with f32 dot products (fused in registers for Q8_0 / Q4_0 at one row).</item>
/// </list>
/// </summary>
internal sealed unsafe class MatMulOp : CpuOp
{
    private static readonly int s_gemmMinRowsOverride =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_GEMM_MIN_ROWS"), out var v) && v > 0 ? v : 0;

    /// <summary>Rows from which the GEMM path is used when the float row path is the alternative.</summary>
    public static readonly int GemmMinRows = s_gemmMinRowsOverride > 0 ? s_gemmMinRowsOverride : 8;

    /// <summary>
    /// Rows from which the GEMM path is used when integer dots are the alternative: they decode each weight row once for
    /// all activation rows and stay ahead of the GEMM up to about 32 rows (9950X, 17408x5120: 2x at 16 rows, even at 32).
    /// </summary>
    public static readonly int IntDotGemmMinRows = s_gemmMinRowsOverride > 0 ? s_gemmMinRowsOverride : 32;

    /// <summary>Largest static f32 weight (bytes) kept a second time in the GEMM's packed B layout.</summary>
    public const long PrepackMaxBytes = 64L << 20;

    /// <summary><c>SHARPLIGHT_CPU_NO_PREPACK=1</c> packs the B panels of static f32 weights on every run instead of once.</summary>
    private static readonly bool s_noPrepack = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_PREPACK") == "1";

    private const int KB = 1024;
    private const int GemmMinTilesPerBlock = 4;
    private const int IntDotMinRowsPerChunk = 16;
    private const int PrefetchRows = 2;
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _y;
    private readonly WeightFormat _fmt;
    private readonly long _n;
    private readonly long _k;
    private readonly long _rowBytes;
    private readonly bool _fused;
    private readonly bool _intDot;
    private readonly bool _smallCodes;
    private readonly float** _xrows;
    private readonly Action<int> _body;
    private readonly Action<int> _intDotBody;
    private readonly Action<int> _packBody;
    private readonly Action<int> _gemmBody;
    private readonly Action<int> _prepackBody;
    private readonly bool _prepack;
    private readonly OutputEpilogue? _epilogue;
    private readonly Action<int> _epilogueBody;
    private float* _bprepacked;
    [ThreadStatic] private static float[]? s_acc;
    private long _m;
    private long _groups;
    private long _perChunk;
    private byte* _activations;
    private float* _apack;
    private int _mTiles;
    private int _panels;
    private int _tilesPerBlock;
    private int _mBlocks;

    public override string Name => $"MatMul.{_fmt} {_n}x{_k}{_epilogue?.Label}";

    /// <param name="x">Activation rows [M, K].</param>
    /// <param name="w">Weight [N, K].</param>
    /// <param name="y">Output rows [M, N].</param>
    /// <param name="fmt">Storage format of <paramref name="w"/>.</param>
    /// <param name="intDot">Whether 8-bit activation dots may be used.</param>
    /// <param name="staticWeight">
    /// Whether <paramref name="w"/> never changes between runs: an f32 weight up to <see cref="PrepackMaxBytes"/> is then packed
    /// into the GEMM's B layout once, on the first run that takes the GEMM path.
    /// </param>
    /// <param name="epilogue">Elementwise ops applied to the output (rows of <paramref name="y"/>), or null.</param>
    public MatMulOp(TRef x, TRef w, TRef y, WeightFormat fmt, bool intDot, bool staticWeight = false, OutputEpilogue? epilogue = null)
    {
        _x = x;
        _w = w;
        _y = y;
        _fmt = fmt;
        _n = w.Dims[0];
        _k = w.Dims[1];
        _rowBytes = _k / Quant.BlockSize(fmt) * Quant.BlockBytes(fmt);
        if (_k % Quant.BlockSize(fmt) != 0)
        {
            throw new ArgumentException($"K={_k} is not a multiple of the {fmt} block size.");
        }

        long maxRows = Math.Max(1, x.MaxElements / Math.Max(1, _k));
        _fused = Quant.HasFusedKernel(fmt);
        _intDot = intDot && IntDot.Supports(fmt) && _k % IntDot.BlockSize == 0 && _k <= int.MaxValue;
        _smallCodes = IntDot.HasSmallCodes(fmt);
        _xrows = (float**)Mem.Alloc(maxRows * sizeof(float*), false);
        _body = Body;
        _intDotBody = IntDotBody;
        _packBody = PackBody;
        _gemmBody = GemmBody;
        _prepackBody = PrepackPanel;
        _epilogue = epilogue;
        _epilogueBody = EpilogueBody;
        _prepack = staticWeight && !s_noPrepack && fmt == WeightFormat.F32 && _n * _k * sizeof(float) <= PrepackMaxBytes;
    }

    public override void Dispose()
    {
        Mem.Free((byte*)_xrows);
        Mem.Free((byte*)_bprepacked);
        _bprepacked = null;
    }

    public override void Run(RunParameters p)
    {
        _m = _x.Rows(p);
        if (_m == 0)
        {
            return;
        }

        for (long r = 0; r < _m; r++)
        {
            _xrows[r] = _x.F(r);
        }

        if (_m >= (_intDot ? IntDotGemmMinRows : GemmMinRows))
        {
            RunGemm();   // applies the epilogue per tile
            return;
        }

        if (_intDot)
        {
            RunIntDot();
        }
        else
        {
            _groups = (_n + 3) / 4;
            int chunks = (int)Math.Min(_groups, CpuParallel.Threads * 8L);
            _perChunk = (_groups + chunks - 1) / chunks;
            CpuParallel.For(chunks, _body);
        }

        if (_epilogue is not null)
        {
            CpuParallel.For((int)Math.Min(_m, CpuParallel.Threads), _epilogueBody);
        }
    }

    /// <summary>Applies the epilogue to a contiguous share of the output rows (paths without per-tile epilogues).</summary>
    private void EpilogueBody(int chunk)
    {
        int chunks = (int)Math.Min(_m, CpuParallel.Threads);
        long r0 = _m * chunk / chunks;
        long r1 = _m * (chunk + 1) / chunks;
        _epilogue!.ApplyRows((float*)_y.P, r0, r1);
    }

    // ---- integer dot (few rows) ------------------------------------------------------------------------------------

    private void RunIntDot()
    {
        long rowBytes = IntDot.RowBytes(_k);
        _activations = ThreadScratch.Get(ThreadScratch.Caller, _m * rowBytes);
        for (long r = 0; r < _m; r++)
        {
            IntDot.Quantize(_xrows[r], _k, IntDot.RowAt(_activations + r * rowBytes, _k));
        }

        int chunks = (int)Math.Clamp(_n / IntDotMinRowsPerChunk, 1, CpuParallel.Threads * 4L);
        _perChunk = (_n + chunks - 1) / chunks;
        CpuParallel.For(chunks, _intDotBody);
    }

    private void IntDotBody(int c)
    {
        long n0 = c * _perChunk;
        long n1 = Math.Min(_n, n0 + _perChunk);
        int k = (int)_k;
        long rowBytes = IntDot.RowBytes(_k);
        var level = CpuIsa.Level;
        byte* u = ThreadScratch.Get(ThreadScratch.Worker, IntDot.DecodedBytes(_k));
        float* a = (float*)(u + ((_k + 63) & ~63L));
        float* b = a + ((_k / IntDot.GroupSize + 15) & ~15L);
        float* y = (float*)_y.P;
        for (long n = n0; n < n1; n++)
        {
            if (n + PrefetchRows < n1)
            {
                Simd.Prefetch(_w.P + (n + PrefetchRows) * _rowBytes, _rowBytes);
            }

            IntDot.Decode(_fmt, _w.P + n * _rowBytes, _k, u, a, b);
            for (long t = 0; t < _m; t++)
            {
                y[t * _n + n] = IntDot.Dot(level, u, a, b, _smallCodes, IntDot.RowAt(_activations + t * rowBytes, _k), k);
            }
        }
    }

    // ---- GEMM (many rows) ------------------------------------------------------------------------------------------

    private void RunGemm()
    {
        int mr = Gemm.MR;
        _mTiles = (int)((_m + mr - 1) / mr);
        _apack = (float*)ThreadScratch.Get(ThreadScratch.Caller, (long)_mTiles * _k * mr * sizeof(float));
        CpuParallel.For(_mTiles, _packBody);

        // Split the rows into blocks only as far as needed to give every thread work: each block dequantizes the weights again.
        _panels = (int)((_n + Gemm.NR - 1) / Gemm.NR);
        int threads = CpuParallel.Threads;
        _mBlocks = 1;
        while ((long)_panels * _mBlocks < 2L * threads && _mTiles / (_mBlocks * 2) >= GemmMinTilesPerBlock)
        {
            _mBlocks *= 2;
        }

        _tilesPerBlock = (_mTiles + _mBlocks - 1) / _mBlocks;
        if (_prepack && _bprepacked == null)
        {
            _bprepacked = (float*)Mem.Alloc((long)_panels * _k * Gemm.NR * sizeof(float), false);
            CpuParallel.For(_panels, _prepackBody);
        }

        CpuParallel.For(_panels * _mBlocks, _gemmBody);
    }

    /// <summary>Packs weight rows [panel·NR, panel·NR + NR) as [K][NR] (zero columns past N); panel p starts at p·K·NR.</summary>
    private void PrepackPanel(int panel)
    {
        int nr = Gemm.NR;
        long n0 = (long)panel * nr;
        int cols = (int)Math.Min(nr, _n - n0);
        float* dst = _bprepacked + (long)panel * _k * nr;
        for (int j = 0; j < nr; j++)
        {
            float* src = j < cols ? (float*)(_w.P + (n0 + j) * _rowBytes) : null;
            for (long k = 0; k < _k; k++)
            {
                dst[k * nr + j] = src == null ? 0f : src[k];
            }
        }
    }

    /// <summary>Packs MR activation rows as [K][MR] (zero rows past the end).</summary>
    private void PackBody(int tile)
    {
        int mr = Gemm.MR;
        long r0 = (long)tile * mr;
        int rows = (int)Math.Min(mr, _m - r0);
        float* dst = _apack + (long)tile * _k * mr;
        float** src = stackalloc float*[mr];
        for (int i = 0; i < mr; i++)
        {
            src[i] = i < rows ? _xrows[r0 + i] : null;
        }

        for (long k = 0; k < _k; k++)
        {
            float* d = dst + k * mr;
            for (int i = 0; i < mr; i++)
            {
                d[i] = i < rows ? src[i][k] : 0f;
            }
        }
    }

    private void GemmBody(int task)
    {
        int mr = Gemm.MR;
        int nr = Gemm.NR;
        int kcMax = Gemm.KC;
        int panel = task % _panels;
        int block = task / _panels;
        int t0 = block * _tilesPerBlock;
        int t1 = Math.Min(_mTiles, t0 + _tilesPerBlock);
        if (t0 >= t1)
        {
            return;
        }

        long n0 = (long)panel * nr;
        int cols = (int)Math.Min(nr, _n - n0);
        float* bpack = (float*)ThreadScratch.Get(ThreadScratch.Worker, ((long)kcMax * nr + kcMax) * sizeof(float));
        float* row = bpack + (long)kcMax * nr;
        float* y = (float*)_y.P;
        int bs = Quant.BlockSize(_fmt);
        int bb = Quant.BlockBytes(_fmt);
        for (long kb = 0; kb < _k; kb += kcMax)
        {
            int kc = (int)Math.Min(kcMax, _k - kb);
            if (_bprepacked != null)
            {
                bpack = _bprepacked + ((long)panel * _k + kb) * nr;
            }

            for (int j = 0; j < nr && _bprepacked == null; j++)
            {
                if (j < cols)
                {
                    float* src = row;
                    if (_fmt == WeightFormat.F32)
                    {
                        src = (float*)(_w.P + (n0 + j) * _rowBytes) + kb;
                    }
                    else
                    {
                        Quant.Dequantize(_fmt, _w.P + (n0 + j) * _rowBytes + kb / bs * bb, kc, row);
                    }

                    for (int k = 0; k < kc; k++)
                    {
                        bpack[k * nr + j] = src[k];
                    }
                }
                else
                {
                    for (int k = 0; k < kc; k++)
                    {
                        bpack[k * nr + j] = 0f;
                    }
                }
            }

            bool accumulate = kb > 0;
            for (int t = t0; t < t1; t++)
            {
                float* a = _apack + ((long)t * _k + kb) * mr;
                float* c = y + (long)t * mr * _n + n0;
                int rows = (int)Math.Min(mr, _m - (long)t * mr);
                if (rows == mr && cols == nr)
                {
                    Gemm.Tile(a, bpack, nr, c, _n, kc, null, accumulate);
                }
                else
                {
                    Gemm.TileEdge(a, bpack, nr, c, _n, kc, null, accumulate, rows, cols);
                }

                if (_epilogue is not null && kb + kc >= _k)
                {
                    long r0 = (long)t * mr;
                    for (int i = 0; i < rows; i++)
                    {
                        _epilogue.Apply(c + (long)i * _n, (r0 + i) * _n + n0, cols);
                    }
                }
            }
        }
    }

    // ---- float rows (few rows without integer dot) -----------------------------------------------------------------

    private void Body(int c)
    {
        long g0 = c * _perChunk;
        long g1 = Math.Min(_groups, g0 + _perChunk);
        long m = _m;
        var accArr = s_acc;
        if (accArr is null || accArr.Length < m * 4)
        {
            s_acc = accArr = new float[Math.Max(m * 4, 4096)];
        }

        float* wbuf = stackalloc float[4 * KB];
        float* res = stackalloc float[8];
        float** wp = stackalloc float*[4];
        byte** rows = stackalloc byte*[4];
        int bs = Quant.BlockSize(_fmt);
        int bb = Quant.BlockBytes(_fmt);
        fixed (float* acc = accArr)
        {
            for (long g = g0; g < g1; g++)
            {
                long n0 = g * 4;
                int nr = (int)Math.Min(4, _n - n0);
                for (int r = 0; r < 4; r++)
                {
                    rows[r] = _w.P + Math.Min(n0 + r, _n - 1) * _rowBytes;
                }

                if (m == 1 && _fused)
                {
                    Quant.DotRows4(_fmt, _xrows[0], rows, _k / bs, res);
                    float* y1 = (float*)_y.P + n0;
                    for (int r = 0; r < nr; r++)
                    {
                        y1[r] = res[r];
                    }

                    continue;
                }
                new Span<float>(acc, (int)(m * 4)).Clear();
                for (long kb = 0; kb < _k; kb += KB)
                {
                    long len = Math.Min(KB, _k - kb);
                    for (int r = 0; r < 4; r++)
                    {
                        if (_fmt == WeightFormat.F32)
                        {
                            wp[r] = (float*)rows[r] + kb;
                        }
                        else
                        {
                            Quant.Dequantize(_fmt, rows[r] + kb / bs * bb, len, wbuf + r * KB);
                            wp[r] = wbuf + r * KB;
                        }
                    }
                    long t = 0;
                    for (; t + 2 <= m; t += 2)
                    {
                        Simd.Dot4x2(_xrows[t] + kb, _xrows[t + 1] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        float* a1 = a0 + 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                        a1[0] += res[4];
                        a1[1] += res[5];
                        a1[2] += res[6];
                        a1[3] += res[7];
                    }
                    if (t < m)
                    {
                        Simd.Dot4(_xrows[t] + kb, wp[0], wp[1], wp[2], wp[3], len, res);
                        float* a0 = acc + t * 4;
                        a0[0] += res[0];
                        a0[1] += res[1];
                        a0[2] += res[2];
                        a0[3] += res[3];
                    }
                }
                for (long t = 0; t < m; t++)
                {
                    float* y = (float*)_y.P + t * _n + n0;
                    for (int r = 0; r < nr; r++)
                    {
                        y[r] = acc[t * 4 + r];
                    }
                }
            }
        }
    }
}
