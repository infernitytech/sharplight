using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Threads per block of <c>gemm_fp8_native.cu</c> (2 x 4 warps).</summary>
    private const int NativeFp8Threads = 256;

    /// <summary>Token rows and weight rows of a <c>gemm_fp8_native.cu</c> tile.</summary>
    private const int NativeFp8Tile = 128;

    /// <summary>Dynamic shared memory of <c>gemm_fp8_native.cu</c>: three stages of 128 + 128 rows of 128 codes and one scale each.</summary>
    private const int NativeFp8SharedBytes = 3 * (256 * 128 + 256 * 4);

    /// <summary>
    /// Fewest activation rows a matmul runs on a hand-written provider GEMM with (<c>SHARPLIGHT_NATIVE_GEMM_MIN_ROWS</c>, default
    /// 256): below that the generated GEMMs' 64-row tiles and K splits fill the GPU better.
    /// </summary>
    private static long NativeGemmMinRows => DiagnosticSwitches.NativeGemmMinRows ?? 65;

    /// <summary>
    /// True when matmuls of block-scaled E4M3 weights [<paramref name="n"/>, <paramref name="k"/>] run on the hand-written FP8 GEMM
    /// (<see cref="DeviceFeatures.NativeGemmFp8"/>, <c>gemm_fp8_native.cu</c>) from <see cref="NativeGemmMinRows"/> rows: weights in
    /// the 128-row groups of <see cref="WeightRepack.Fp8B128Grouped"/> (see <see cref="NativeFp8GroupedWeight"/>), on a target that
    /// takes CUDA C++.
    /// </summary>
    private bool UseNativeFp8Gemm(DataType type, long n, long k)
        => NativeGemmFp8 && _opt.MaxSharedMemory >= NativeFp8SharedBytes && type == ShaderTypes.F8E4M3_B128G && k % 512 == 0 && n % WeightRepacker.Fp8GroupRows == 0;

    /// <summary>True when the device runs the hand-written FP8 GEMM (feature and a CUDA C++ target).</summary>
    private bool NativeGemmFp8 => _opt.Has(DeviceFeatures.NativeGemmFp8) && _compiler.CompilesCuda(_opt.Target);

    /// <summary>
    /// True when a block-scaled E4M3 weight is stored in the 128-row groups of <see cref="WeightRepack.Fp8B128Grouped"/> for the
    /// hand-written FP8 GEMM: a [N, K] matrix (N % 128 == 0, K % 512 == 0) that only matmuls read as their weight. Every generated
    /// kernel reading FP8 weights decodes that layout too (<c>FP8_GROUPED</c>, <c>F8_GROUPED</c>), so the smaller batches and
    /// decode keep their kernels. <c>SHARPLIGHT_REPACK_FP8=0</c> keeps the stored block layout.
    /// </summary>
    private bool NativeFp8GroupedWeight(Value weight)
    {
        var w = weight.Weight!;
        if (!(DiagnosticSwitches.Fp8Grouped ?? NativeGemmFp8) || DiagnosticSwitches.NoWeightRepack || !DiagnosticSwitches.Fp8Repack || w.Type != DataType.F8E4M3_B128
            || w.Shape.Rank != 2 || w.Shape.Dims.Any(d => d.IsSymbolic) || w.Shape[0].Fixed % WeightRepacker.Fp8GroupRows != 0
            || w.Shape[1].Fixed % 512 != 0)
        {
            return false;
        }

        bool any = false;
        foreach (var node in _seg.Nodes)
        {
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                if (node.Inputs[i] != weight)
                {
                    continue;
                }

                if (node.Op.Name != "MatMul" || i != 1)
                {
                    return false;
                }

                any = true;
            }
        }

        return any;
    }

    /// <summary>
    /// Row capacity of the dense copy of the E4M3 activation scales (<c>[K / 128][xcap]</c> f32 after <c>xcap</c> rows, see
    /// <c>gemm_fp8_native.cu</c>) that the writers of activation rows for <paramref name="rows"/> rows add when the device runs the
    /// hand-written FP8 GEMM, or 0 (no copy).
    /// </summary>
    private long DenseScaleCapacity(long rows) => NativeGemmFp8 ? (rows + 127) / 128 * 128 : 0;

    /// <summary>
    /// Rows of activation-row tensors of <paramref name="rows"/> rows of [K codes][K / 128 scales] (<paramref name="rowBytes"/>
    /// bytes) with room for the dense scale copy after <see cref="DenseScaleCapacity"/> rows.
    /// </summary>
    private long Fp8ActivationRows(long rows, long rowBytes)
    {
        long cap = DenseScaleCapacity(rows);
        // The copy: cap * K / 32 bytes, less than cap / 32 rows of K + K / 32 bytes.
        return cap == 0 ? rows : cap + (cap * (rowBytes / 33) + rowBytes - 1) / rowBytes;
    }

    /// <summary>Row capacities of the dense scale copies of E4M3 activation rows, by buffer and offset (see <see cref="Fp8ActivationRows"/>).</summary>
    private readonly Dictionary<(BufferDecl Buffer, long Offset), long> _denseScaleCaps = [];

    /// <summary>Records that E4M3 activation rows <paramref name="quantized"/> for <paramref name="rows"/> rows carry the dense scale copy.</summary>
    private void NoteDenseScales(TensorRef quantized, long rows)
    {
        if (DenseScaleCapacity(rows) is > 0 and var cap)
        {
            _denseScaleCaps[(quantized.Buffer, quantized.ByteOffset)] = cap;
        }
    }

    /// <summary>The dense scale copy's row capacity of E4M3 activation rows <paramref name="quantized"/>, or 0 when they have none.</summary>
    private long DenseScaleCapacityOf(TensorRef quantized) => _denseScaleCaps.GetValueOrDefault((quantized.Buffer, quantized.ByteOffset));

    /// <summary>
    /// Relative time of a block of the hand-written FP8 GEMM with 128 token rows against one with 64, fitted to the tile choice that
    /// was faster for the Qwen3-8B shapes at 80-384 rows on the RTX PRO 4500 Blackwell (64-row tiles faster at 160, 192 and 320
    /// rows, 128-row tiles at 256 and 384: a ratio between 1.6 and 1.875).
    /// </summary>
    private const double NativeFp8TallBlockCost = 1.75;

    /// <summary>
    /// <see cref="NativeFp8TallBlockCost"/> when the 128-row tiles run on <c>gemm_fp8_tma.cu</c> (64-row tiles keep
    /// <c>gemm_fp8_native.cu</c>): fitted to Qwen3-8B-FP8 prefill (8192 rows in 1024-row passes: 9,971 tok/s at 1.75, 10,515 at 1.3,
    /// 10,486 at 1.0; 512 and 2048 rows within 1 %).
    /// </summary>
    private const double NativeFp8TmaTallBlockCost = 1.3;

    /// <summary>
    /// Token rows of the hand-written FP8 GEMM's tile for a batch (64 or 128): the one with the fewest estimated block waves times
    /// block time on the GPU's SMs (one block per SM: both use the 99 KB of shared memory).
    /// </summary>
    private int NativeFp8TileRows(long rows, long n, bool dual)
    {
        long sms = Math.Max(1, _opt.ComputeUnits);
        long columns = Grid.Div(n, dual ? NativeFp8Tile / 2 : NativeFp8Tile);
        double Cost(int tileRows, double blockCost) => (Grid.Div(rows, tileRows) * columns + sms - 1) / sms * blockCost;
        return Cost(64, 1.0) <= Cost(128, DiagnosticSwitches.NativeFp8TallCost ?? (NativeFp8TensorCopies ? NativeFp8TmaTallBlockCost : NativeFp8TallBlockCost)) ? 64 : 128;
    }

    /// <summary>
    /// Launch variants of the hand-written FP8 GEMM for [<paramref name="n"/>, <paramref name="k"/>] weights (single, or dual with
    /// <paramref name="dualEpilogue"/>) reading activation rows whose dense scale copy has <paramref name="xcap"/> rows, for batches of
    /// <paramref name="minRows"/> to <paramref name="maxRows"/> rows: one per token tile height (<see cref="NativeFp8TileRows"/>).
    /// </summary>
    private KernelVariant[] NativeFp8Variants(long n, long k, string? dualEpilogue, long xcap, long minRows, long maxRows, Func<Bindings, long> rows)
    {
        bool dual = dualEpilogue is not null;
        var used = new SortedSet<int>();
        for (long r = minRows; r <= maxRows; r += 64)
        {
            used.Add(NativeFp8TileRows(r, n, dual));
        }

        used.Add(NativeFp8TileRows(maxRows, n, dual));
        return [.. used.Select(tileRows =>
        {
            if (tileRows == NativeFp8Tile && NativeFp8TensorCopies)
            {
                return NativeFp8TmaVariant(n, k, dualEpilogue, xcap, b => rows(b) >= minRows && NativeFp8TileRows(rows(b), n, dual) == tileRows, rows);
            }

            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("XCAP", xcap), ("F8N_BM", tileRows), ("DUAL", dual ? 1 : 0));
            var kernel = _compiler.CompileCuda("gemm_fp8_native.cu", dualEpilogue ?? "", "gemm_fp8_native", NativeFp8Threads, NativeFp8SharedBytes,
                _opt.Target, macros, _opt.Cache);
            return new KernelVariant(b => rows(b) >= minRows && NativeFp8TileRows(rows(b), n, dual) == tileRows, kernel,
                b => new Grid(Grid.Div(rows(b), tileRows), Grid.Div(n, dual ? NativeFp8Tile / 2 : NativeFp8Tile)));
        })];
    }

    /// <summary>Threads per block of <c>gemm_fp8_tma.cu</c>: 2 x 4 computing warps and the copy warp.</summary>
    private const int NativeFp8TmaThreads = 288;

    /// <summary>Dynamic shared memory of <c>gemm_fp8_tma.cu</c>: three stages of 256 rows of 128 codes, two slots of 256 scales, eight barriers.</summary>
    private const int NativeFp8TmaSharedBytes = 3 * 256 * 128 + 2 * 256 * 4 + 64;

    /// <summary>
    /// True when the hand-written FP8 GEMM's 128-row tiles run on its warp-specialized form with tensor copies (<c>gemm_fp8_tma.cu</c>,
    /// <see cref="DeviceFeatures.TensorCopies"/>): persistent blocks, a copy warp and full / empty barriers instead of per-thread
    /// asynchronous copies and a block barrier per K block (Qwen3-8B-FP8 prefill +8-9 % at 512-8192 rows on the RTX PRO 4500
    /// Blackwell, benchmarks.md section 57). Disabling <c>tensor_copies</c> or <c>SHARPLIGHT_FP8_TMA=0</c> keeps
    /// <c>gemm_fp8_native.cu</c>.
    /// </summary>
    private bool NativeFp8TensorCopies => DiagnosticSwitches.NativeFp8Tma && _opt.Has(DeviceFeatures.TensorCopies) && _opt.MaxSharedMemory >= NativeFp8TmaSharedBytes;

    /// <summary>
    /// The launch variant of <c>gemm_fp8_tma.cu</c> for the batches <paramref name="serves"/> selects: one block per SM at most, each
    /// walking its share of the 128-row tiles, with the tensor maps over x (its <paramref name="xcap"/> rows) and the 128-row weight groups.
    /// </summary>
    private KernelVariant NativeFp8TmaVariant(long n, long k, string? dualEpilogue, long xcap, Func<Bindings, bool> serves, Func<Bindings, long> rows)
    {
        bool dual = dualEpilogue is not null;
        long rowBytes = k + k / 32;
        var inv = CultureInfo.InvariantCulture;
        string WeightMap(int pointer) => string.Create(inv, $"{pointer} 3 {k},128,{n / 128} {k},{128 * rowBytes} 128,{(dual ? 64 : 128)},1");
        var maps = new List<string> { string.Create(inv, $"0 2 {k},{xcap} {rowBytes} 128,128"), WeightMap(1) };
        if (dual)
        {
            maps.Add(WeightMap(2));
        }

        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("XCAP", xcap), ("DUAL", dual ? 1 : 0));
        var kernel = _compiler.CompileCuda("gemm_fp8_tma.cu", dualEpilogue ?? "", "gemm_fp8_tma", NativeFp8TmaThreads, NativeFp8TmaSharedBytes,
            _opt.Target, macros, _opt.Cache, string.Join(';', maps));
        long sms = Math.Max(1, _opt.ComputeUnits);
        long columns = Grid.Div(n, dual ? NativeFp8Tile / 2 : NativeFp8Tile);
        return new KernelVariant(serves, kernel, b => new Grid((uint)Math.Min(sms, (long)Grid.Div(rows(b), NativeFp8Tile) * columns)));
    }

    /// <summary>Threads per block of <c>gemm_f16_native.cu</c> per tile configuration (0: 128 x 256, 1: 128 x 128).</summary>
    private static readonly int[] s_nativeF16Threads = [256, 128];

    /// <summary>Weight rows per tile of <c>gemm_f16_native.cu</c> per tile configuration.</summary>
    private static readonly int[] s_nativeF16TileColumns = [256, 128];

    /// <summary>Dynamic shared memory of <c>gemm_f16_native.cu</c> per tile configuration: two stages of 384 rows, three of 256 rows.</summary>
    private static readonly int[] s_nativeF16Shared = [2 * 384 * 128, 3 * 256 * 128];

    /// <summary>
    /// Relative speed of the f16 tile configurations at full waves (RTX PRO 4500 Blackwell, Qwen3-8B shapes at 2048 rows: 128 x 256
    /// 155-161 TFLOPS, 128 x 128 140-150).
    /// </summary>
    private static readonly double[] s_nativeF16Speed = [1.0, 0.94];

    /// <summary>
    /// True when the f16 products of the library GEMM path (bf16 weights stored as f16, the f16 copies of x) run on the hand-written
    /// f16 GEMM of the provider (<see cref="DeviceFeatures.NativeGemmF16"/>, <c>gemm_f16_native.cu</c>), preferred over the vendor
    /// library when both are there; disabling <c>native_gemm_f16</c> leaves them to the library (or the generated GEMMs).
    /// </summary>
    private bool NativeGemmF16 => _opt.Has(DeviceFeatures.NativeGemmF16) && _compiler.CompilesCuda(_opt.Target);

    /// <summary>The tile configuration of the hand-written f16 GEMM with the fewest estimated waves of work for a batch.</summary>
    private int NativeF16Config(long rows, long n, bool dual)
    {
        int sms = Math.Max(1, _opt.ComputeUnits);
        double best = double.MaxValue;
        int config = 0;
        for (int c = 0; c < s_nativeF16TileColumns.Length; c++)
        {
            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            long tiles = Grid.Div(rows, 128) * ((n + columns - 1) / columns);
            double cost = (double)((tiles + sms - 1) / sms) * s_nativeF16TileColumns[c] / s_nativeF16Speed[c];
            if (cost < best)
            {
                best = cost;
                config = c;
            }
        }

        return config;
    }

    /// <summary>
    /// Launch variants of the hand-written f16 GEMM (single, or dual with <paramref name="dualEpilogue"/>) for batches of
    /// <paramref name="minRows"/> rows and more: one per tile configuration, each serving the row counts it is estimated fastest for.
    /// </summary>
    private KernelVariant[] NativeF16Variants(long n, long k, long maxRows, string? dualEpilogue, Func<Bindings, long> rows, long minRows)
    {
        bool dual = dualEpilogue is not null;
        var used = new SortedSet<int>();
        for (long r = minRows; r <= Math.Max(minRows, maxRows); r += 128)
        {
            used.Add(NativeF16Config(r, n, dual));
        }

        used.Add(NativeF16Config(Math.Max(minRows, maxRows), n, dual));
        return [.. used.Select(c =>
        {
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("DUAL", dual ? 1 : 0), ("H_CFG", c));
            var kernel = _compiler.CompileCuda("gemm_f16_native.cu", dualEpilogue?.Replace("F8N_EPI2", "H_EPI2", StringComparison.Ordinal) ?? "",
                "gemm_f16_native", s_nativeF16Threads[c], s_nativeF16Shared[c], _opt.Target, macros, _opt.Cache);
            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            return new KernelVariant(b => rows(b) >= minRows && NativeF16Config(rows(b), n, dual) == c, kernel,
                b => new Grid(Grid.Div(rows(b), 128), Grid.Div(n, columns)));
        })];
    }

    /// <summary>Token rows of the routing tiles of <c>gemm_moe_native.cu</c> (one warp row of 8 row fragments).</summary>
    private const int NativeMoeTileRows = 128;

    /// <summary>Warps of a <c>gemm_moe_native.cu</c> block (all side by side along the weight rows).</summary>
    private const int NativeMoeWarps = 8;

    /// <summary>Stages of the <c>cp.async</c> ring of <c>gemm_moe_native.cu</c> where they fit (else 3).</summary>
    private const int NativeMoeStages = 4;

    /// <summary>Static shared memory of <c>gemm_moe_native.cu</c> (its E8M0 factor table) next to the dynamic stages.</summary>
    private const int NativeMoeStaticShared = 1024;

    /// <summary>
    /// True when the expert products of <paramref name="w"/> ([E, <paramref name="n"/>, <paramref name="k"/>]) can run on the hand-written
    /// grouped MXFP4 GEMM (<see cref="DeviceFeatures.NativeGemmMoe"/>, <c>gemm_moe_native.cu</c>): MXFP4 experts in the 16-row groups of
    /// <see cref="WeightRepack.Mxfp4Split"/>, K a multiple of 64, on a target that takes CUDA C++ with room for three stages.
    /// </summary>
    private bool UseNativeMoeGemm(TensorRef w, long k, long n)
        => _opt.Has(DeviceFeatures.NativeGemmMoe) && _compiler.CompilesCuda(_opt.Target) && w.Type == ShaderTypes.MXFP4R && k % 64 == 0 && n % 32 == 0
           && DiagnosticSwitches.MoeSortedHalfActivations && _opt.MaxSharedMemory >= NativeMoeStageBytes(256, true) * 3 + NativeMoeStaticShared;

    /// <summary>
    /// Routed items (token, slot) from which the hand-written grouped MXFP4 GEMM serves a step of <paramref name="experts"/> experts:
    /// 12 rows per expert on average (where the expert-grouped skinny GEMM hands over), or <c>SHARPLIGHT_NATIVE_MOE_MIN_ROWS</c> per
    /// expert. RTX PRO 4500, gpt-oss-20b mixed serving steps (64 decode rows plus a prompt, GPU ms): 171 rows 33.7 → 25.8, 256 rows
    /// 40.2 → 29.6, 512 rows 58.2 → 45.6, 2048 rows 172.2 → 147.0 against gemm_cm2.
    /// </summary>
    private static long NativeMoeMinItems(long experts) => (DiagnosticSwitches.NativeMoeMinRows ?? 12) * experts;

    /// <summary>Bytes of one stage of <c>gemm_moe_native.cu</c>: activation rows, codes and scales of its weight rows.</summary>
    private static int NativeMoeStageBytes(int columns, bool dual)
        => NativeMoeTileRows * 128 + 2 * columns * 16 + 2 * (dual ? 2 : 1) * (dual ? columns / 2 : columns) / 16 * 16;

    /// <summary>Dynamic shared memory of <c>gemm_moe_native.cu</c> for weight tiles of <paramref name="columns"/> rows.</summary>
    private int NativeMoeSharedBytes(int columns, bool dual)
        => NativeMoeStageBytes(columns, dual) * NativeMoeStagesFor(columns, dual);

    /// <summary>The stages of <c>gemm_moe_native.cu</c> that fit the device's shared memory (<see cref="NativeMoeStages"/> or 3).</summary>
    private int NativeMoeStagesFor(int columns, bool dual)
        => _opt.MaxSharedMemory >= NativeMoeStageBytes(columns, dual) * NativeMoeStages + NativeMoeStaticShared ? NativeMoeStages : 3;

    /// <summary>
    /// Weight rows per tile of <c>gemm_moe_native.cu</c> for <paramref name="n"/> outputs: the width (256, 192 or 128 rows; dual pairs
    /// 256 or 128, their 8 warps each holding whole gate / up fragment pairs) with the most useful columns per estimated time.
    /// RTX PRO 4500, gpt-oss-20b shapes at 2048 tokens: 192 and 256 rows equally fast per column, 128 rows ~10 % slower (more
    /// activation traffic per product).
    /// </summary>
    private static int NativeMoeTileColumns(long n, bool dual)
    {
        int[] widths = dual ? [256, 128] : [256, 192, 128];
        int best = widths[0];
        double bestScore = 0;
        foreach (int width in widths)
        {
            long outputs = dual ? width / 2 : width;
            double score = (double)n / (Grid.Div(n, outputs) * outputs) * (width == 128 ? 0.9 : 1.0);
            if (score > bestScore)
            {
                bestScore = score;
                best = width;
            }
        }

        return best;
    }

    /// <summary>
    /// The CUDA C++ form of the dual epilogue <c>epi2</c> (see <see cref="DualEpilogueSource"/>) for the hand-written dual GEMM, as the
    /// <c>F8N_EPI2</c> macro over a device function, or null when the group has an operation without a CUDA form here (the generated
    /// dual GEMM then serves the pair). The expressions are those of the Slang epilogue with single-precision literals and <c>__expf</c>.
    /// </summary>
    private static string? DualEpilogueCudaSource(Node a, Node bNode, List<Node> group, Value output)
    {
        string[] supported = ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Relu", "Square", "Recip",
            "Add", "Sub", "Mul", "Max", "Min", "Scale", "Cast", "Clip"];
        if (group.Any(node => !supported.Contains(node.Op.Name)))
        {
            return null;
        }

        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        var src = new StringBuilder();
        src.AppendLine("static __device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }");
        src.AppendLine("static __device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }");
        src.AppendLine("static __device__ __forceinline__ float sl_epi2(float m0, float m1)");
        src.AppendLine("{");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            if (!name.ContainsKey(node.Inputs[0]) || node.Inputs.Skip(1).Any(i => !name.ContainsKey(i)))
            {
                // An input from outside the pair (a bias, a mask): not expressible in this epilogue.
                return null;
            }

            src.AppendLine(CultureInfo.InvariantCulture, $"    float {local} = {SinglePrecisionLiterals(OpExpr(node, [.. node.Inputs.Select(i => name[i])]))};");
            name[node.Output] = local;
        }

        src.AppendLine(CultureInfo.InvariantCulture, $"    return {name[output]};");
        src.AppendLine("}");
        src.AppendLine("#define F8N_EPI2(a, b) sl_epi2(a, b)");
        return src.ToString();
    }

    /// <summary>Floating-point literals of a generated expression as <c>float</c> literals (C++ reads <c>1.0</c> as a double).</summary>
    /// <remarks><c>exp</c> becomes <c>__expf</c>, as in the CUDA form of the Slang kernels (<c>common.slang</c>), so the fused epilogue
    /// rounds like the elementwise launch it replaces.</remarks>
    private static string SinglePrecisionLiterals(string expression)
        => CudaExpression(expression);

    /// <summary>A generated elementwise expression in its CUDA C++ form: <c>float</c> literals and <c>__expf</c> (see <see cref="SinglePrecisionLiterals"/>).</summary>
    internal static string CudaExpression(string expression)
        => ExpCall().Replace(FloatLiteral().Replace(expression, m => m.Value + "f"), "__expf(");

    [GeneratedRegex(@"(?<![\w])exp\(")]
    private static partial Regex ExpCall();

    [GeneratedRegex(@"(?<![\w.])(\d+\.\d*([eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")]
    private static partial Regex FloatLiteral();
}
