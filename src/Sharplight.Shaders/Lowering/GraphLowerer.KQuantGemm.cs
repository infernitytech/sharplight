using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Threads per block of <c>gemm_kq_native.cu</c> (2 x 4 warps).</summary>
    private const int NativeKQuantThreads = 256;

    /// <summary>Weight rows per block of <c>gemm_kq_native.cu</c>.</summary>
    private const int NativeKQuantColumns = 64;

    /// <summary>Largest batch the hand-written K-quant decode GEMM serves (one token tile of 64 rows).</summary>
    private const long NativeKQuantMaxRows = 64;

    /// <summary>
    /// Fewest rows the hand-written K-quant decode GEMM serves (<c>SHARPLIGHT_NATIVE_KQ_MIN_ROWS</c>): from the first batch past the matvec.
    /// </summary>
    private static long NativeKQuantMinRows => DiagnosticSwitches.NativeKQuantMinRows ?? MatvecMaxRows + 1;

    /// <summary>Bytes of one block of 256 values of a K-quant format the hand-written decode GEMM reads (0: not one of them).</summary>
    private static int NativeKQuantBlockBytes(DataType type) => type == DataType.Q4_K ? 144 : type == DataType.Q5_K ? 176 : type == DataType.Q6_K ? 210 : 0;

    /// <summary>The <c>KQ_FMT</c> id of a K-quant format of <c>gemm_kq_native.cu</c>.</summary>
    private static int NativeKQuantFormat(DataType type) => type == DataType.Q4_K ? 4 : type == DataType.Q5_K ? 5 : 6;

    /// <summary>
    /// Dynamic shared memory of one stage of <c>gemm_kq_native.cu</c>: <paramref name="tileRows"/> activation rows of 256 f16 values and
    /// the 16-byte windows of 64 weight rows' blocks (the block itself when every block starts 16-byte aligned, else 14 bytes more).
    /// </summary>
    private static int NativeKQuantStageBytes(int tileRows, int blockBytes, long k)
    {
        bool aligned = blockBytes % 16 == 0 && k / 256 * blockBytes % 16 == 0;
        int chunks = aligned ? blockBytes / 16 : (blockBytes + 14 + 15) / 16;
        return tileRows * 512 + NativeKQuantColumns * chunks * 16;
    }

    /// <summary>
    /// Stages of <c>gemm_kq_native.cu</c>'s copy ring: as many as fit the device's shared memory (2 to 4), at most 96 KiB so that the
    /// L1 cache keeps some room; 0 or 1 when two do not fit.
    /// </summary>
    private int NativeKQuantStages(int tileRows, int blockBytes, long k)
    {
        int stage = NativeKQuantStageBytes(tileRows, blockBytes, k);
        long limit = Math.Min(_opt.MaxSharedMemory, 96L * 1024);
        return (int)Math.Clamp(limit / stage, 0, 4);
    }

    /// <summary>
    /// Token rows of the hand-written K-quant decode GEMM's tile for <paramref name="rows"/> rows (<c>SHARPLIGHT_NATIVE_KQ_BM</c>): 16 up to 16
    /// rows (four stages), 32 up to 32 (three), else 64 (two). bench/s77_kq, RTX PRO 4500, sustained µs (Q4_K [4096 → 12288] / Q6_K
    /// [12288 → 4096]): 16 rows 43.8 / 58.8 on 16-row tiles against 58.2 / 65.6 on 32-row tiles; 32 rows 52.4 / 60.4 on 32-row tiles
    /// against 67.0 / 93.1 on 16-row tiles and 96.3 / 106.6 on 64-row tiles; 64 rows 82.1 / 89.7 on 64-row tiles against 89.5 / 115.1
    /// on two 32-row tiles (each reads the weights).
    /// </summary>
    private static int NativeKQuantTileRows(long rows) => DiagnosticSwitches.NativeKQuantTileRows ?? (rows <= 16 ? 16 : rows <= 32 ? 32 : 64);

    /// <summary>
    /// Warps along the token rows of the hand-written K-quant decode GEMM's tile of <paramref name="tileRows"/> rows
    /// (<c>SHARPLIGHT_NATIVE_KQ_WM</c>, <c>KQ_WM</c>): 2 for 64-row tiles, where eight warps reading every token row's activations from
    /// shared memory cost more than dequantizing each weight value twice; 1 otherwise.
    /// </summary>
    private static int NativeKQuantWarpRows(int tileRows) => tileRows < 32 ? 1 : DiagnosticSwitches.NativeKQuantWarpRows ?? (tileRows >= 64 ? 2 : 1);

    /// <summary>
    /// True when the hand-written K-quant decode GEMM can take decode batches of <paramref name="type"/> weights on this device (the format
    /// and the feature; the shape is checked per matmul, <see cref="UseNativeKQuantDecode"/>), so the skinny GEMM leaves those batches to it.
    /// </summary>
    private bool UsesNativeKQuantDecode(DataType type)
        => NativeKQuantBlockBytes(type) > 0 && _opt.Has(DeviceFeatures.NativeGemmKQuant) && _compiler.CompilesCuda(_opt.Target) && NativeKQuantMinRows <= MatvecMaxRows + 1;

    /// <summary>
    /// The generated GEMM's K split for a plan whose decode batches the hand-written K-quant GEMM takes: at least 2 for those batches, so
    /// the unsplit gemm_cm2 launch leaves them to the split-K emission, where the K-quant GEMM serves them (it does not use the split);
    /// <paramref name="split"/> otherwise. Qwen3.8 27B Q4_K_M at 16 rows: gate / up [5120 → 17408] 165 µs each on unsplit gemm_cm2.
    /// </summary>
    private int NativeKQuantPlanSplit(DataType type, long n, long k, long rows, bool splitHalf, int split)
        => !splitHalf && split < 2 && UsesNativeKQuantDecode(type) && k % 256 == 0 && n >= NativeKQuantColumns && rows >= NativeKQuantMinRows
           && rows <= NativeKQuantMaxRows ? 2 : split;

    /// <summary>
    /// True when K-quant weights of <paramref name="type"/> [<paramref name="n"/>, <paramref name="k"/>] read by the f16 copy of x
    /// (<paramref name="macros"/>) run their decode batches on the hand-written K-quant GEMM (<see cref="DeviceFeatures.NativeGemmKQuant"/>,
    /// <c>gemm_kq_native.cu</c>): Q4_K, Q5_K or Q6_K, whole blocks per row, on a target that takes CUDA C++ with room for two stages.
    /// </summary>
    private bool UseNativeKQuantDecode(DataType type, long n, long k, IReadOnlyDictionary<string, string> macros)
        => _opt.Has(DeviceFeatures.NativeGemmKQuant) && _compiler.CompilesCuda(_opt.Target) && NativeKQuantBlockBytes(type) > 0 && k % 256 == 0
           && n >= NativeKQuantColumns && macros.GetValueOrDefault("X_F16") == "1" && macros.GetValueOrDefault("X_STRIDE") == k.ToString(CultureInfo.InvariantCulture)
           && NativeKQuantStages(NativeKQuantTileRows(NativeKQuantMaxRows), NativeKQuantBlockBytes(type), k) >= 2;

    /// <summary>
    /// K split of the hand-written K-quant decode GEMM for <paramref name="tiles"/> blocks of <paramref name="steps"/> superblocks
    /// (<c>SHARPLIGHT_NATIVE_KQ_SPLIT</c>), one block per SM (a block takes most of an SM's shared memory): grids of less than half the SMs
    /// split to fill them (at most 4 ways), grids whose last wave would hold less than a quarter of the SMs split in two; otherwise none
    /// (the blocks stream their weights, so a partly filled last wave runs faster than a full one). bench/s77_kq, sustained µs at 16 / 32
    /// rows: Q6_K [21504 → 5376] (84 blocks on 82 SMs) 220.4 / 217.3 → 182.5 / 179.7 split in two; Q4_K [4096 → 12288] (192 blocks)
    /// 43.8 / 52.4 unsplit against 44.9 / 55.4; [4096 → 4096] (64 blocks) 14.1 / 17.8 against 17.5 / 20.8.
    /// </summary>
    private int NativeKQuantSplit(long tiles, long steps)
    {
        if (DiagnosticSwitches.NativeKQuantSplit is { } forced)
        {
            return forced;
        }

        long sms = Math.Max(1, _opt.ComputeUnits);
        int split = tiles * 2 <= sms ? (int)Math.Min(4, sms / tiles) : tiles >= sms && tiles % sms != 0 && tiles % sms * 4 < sms ? 2 : 1;
        return (int)Math.Max(1, Math.Min(split, steps / 2));
    }

    /// <summary>
    /// The hand-written K-quant decode GEMM for the batches of <see cref="NativeKQuantMinRows"/> to <see cref="NativeKQuantMaxRows"/> rows that
    /// <paramref name="serves"/> (the split-K batches its multi-output group leaves to it), when the matmul reads the f16 copy of x with
    /// K-quant weights (<see cref="UseNativeKQuantDecode"/>) into dense rows and its epilogue is none or a lone bias row, which the store
    /// adds (and undoes a row-ranged copy's row scales). One launch per batch: its K split reduces in the last block of each tile.
    /// Returns the batches it serves, or null.
    /// </summary>
    private Func<Bindings, bool>? EmitNativeKQuantDecodeGemm(MatMulPlan p, TensorRef activations, MatMulEpilogueCode? epilogue, Func<Bindings, bool> serves)
    {
        var bias = EpilogueBias(epilogue);
        if (!MultiGemmCandidate(p) || p.RowsPerLead != 1 || p.WeightParts is not null || (epilogue is not null && bias is null)
            || !UseNativeKQuantDecode(p.W.Type, p.N, p.K, p.Macros) || p.W.ByteOffset % 16 != 0 || activations.ByteOffset % 16 != 0)
        {
            return null;
        }

        var y = epilogue?.Output ?? p.Y;
        if (!DenseRowsOf(y, p.N))
        {
            return null;
        }

        long rowsMax = Math.Min(NativeKQuantMaxRows, p.X.Shape[0]);
        if (rowsMax < NativeKQuantMinRows)
        {
            return null;
        }

        long? rangeScales = _rangeScales.TryGetValue(activations, out long xrs) ? xrs : null;
        Func<Bindings, bool> decode = b => serves(b) && p.RowCount(b) >= NativeKQuantMinRows && p.RowCount(b) <= rowsMax;
        var (variants, scratch, arrivals) = NativeKQuantVariants(p.W.Type, p.N, p.K, rowsMax, bias is not null, rangeScales, null, decode, p.RowCount);
        KernelArg[] biasArgs = bias is null ? [] : [Pointer(bias)];
        TensorRef[] biasReads = bias is null ? [] : [bias];
        Emit($"{MatMulLabel(p)} native kquant decode{epilogue?.Label}", [p.Node, .. epilogue?.Group ?? []], variants,
            [Pointer(activations), Pointer(p.W), Pointer(y), .. biasArgs, Pointer(scratch), Pointer(arrivals), TokenArg(p.X, p.RowsPerLead)],
            [activations, p.W, .. biasReads], [y, scratch, arrivals]);
        return decode;
    }

    /// <summary>
    /// The multi-output form of the hand-written K-quant decode GEMM over the group's concatenated weight rows (q / k / v of one format:
    /// each block's weight tile in one member's matrix, storing to that member's output with its bias) for the decode batches the
    /// multi-output GEMM serves (<paramref name="serves"/>), when every member takes it and every member boundary is a multiple of its
    /// 64-row weight tiles. Returns the batches it serves, or null.
    /// </summary>
    private Func<Bindings, bool>? EmitNativeKQuantMultiGemm(MultiGemmGroup group, List<MultiGemmPart> parts, long[] widths, long k, Func<Bindings, bool> serves)
    {
        var first = parts[0];
        long n = widths.Sum();
        if (parts.Count > 3 || parts.Any(pt => pt.Weight.Type != first.Weight.Type || !UseNativeKQuantDecode(pt.Weight.Type, n, k, pt.Macros) || pt.Weight.ByteOffset % 16 != 0)
            || first.Activations.ByteOffset % 16 != 0 || widths.SkipLast(1).Any(w => w % NativeKQuantColumns != 0))
        {
            return null;
        }

        long rowsMax = Math.Min(NativeKQuantMaxRows, first.RowsMax);
        if (rowsMax < NativeKQuantMinRows)
        {
            return null;
        }

        bool bias = first.Bias is not null;
        var rowsOf = first.Rows;
        Func<Bindings, bool> decode = b => serves(b) && rowsOf(b) >= NativeKQuantMinRows && rowsOf(b) <= rowsMax;
        var multi = new Dictionary<string, string>
        {
            ["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture),
            ["MULTI_O1"] = widths[0].ToString(CultureInfo.InvariantCulture),
            ["MULTI_O2"] = (parts.Count > 2 ? widths[0] + widths[1] : n).ToString(CultureInfo.InvariantCulture),
        };
        long? rangeScales = _rangeScales.TryGetValue(first.Activations, out long xrs) ? xrs : null;
        var (variants, scratch, arrivals) = NativeKQuantVariants(first.Weight.Type, n, k, rowsMax, bias, rangeScales, multi, decode, rowsOf);
        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        KernelArg[] firstBias = bias ? [Pointer(first.Bias!)] : [];
        KernelArg[] biases = bias ? [.. parts.Skip(1).Select(pt => Pointer(pt.Bias!))] : [];
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} native kquant decode", group.Members, variants,
            [Pointer(first.Activations), Pointer(first.Weight), Pointer(first.Output), .. firstBias, .. weights, .. outputs, .. biases, Pointer(scratch),
                Pointer(arrivals), first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight), .. bias ? parts.Select(pt => pt.Bias!) : []], [.. parts.Select(pt => pt.Output), scratch, arrivals]);
        return decode;
    }

    /// <summary>
    /// The dual gate / up form of the hand-written K-quant decode GEMM for two matmuls sharing x whose products feed one elementwise
    /// group (<paramref name="epilogue"/>, the group's CUDA form from <see cref="DualEpilogueCudaSource"/>): one launch over the
    /// interleaved rows of both matrices, each block 32 outputs of both, applying the group in its store, for the decode batches of
    /// <see cref="NativeKQuantMinRows"/> to <see cref="NativeKQuantMaxRows"/> rows above <paramref name="fusedBelow"/> (instead of two
    /// K-quant GEMM launches and the elementwise launch). Returns the row counts it serves, or null when it does not apply.
    /// </summary>
    private Func<long, bool>? EmitNativeKQuantDualGemm(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead,
        TensorRef weightA, TensorRef weightB, TensorRef y, long n, long k, string? epilogue, long fusedBelow)
    {
        if (epilogue is null || !DiagnosticSwitches.NativeKQuantDual || weightA.Type != weightB.Type || !UsesNativeKQuantDecode(weightA.Type) || k % 256 != 0
            || n < NativeKQuantColumns || rowsPerLead != 1 || weightA.ByteOffset % 16 != 0 || weightB.ByteOffset % 16 != 0 || !y.IsDense
            || !DiagnosticSwitches.Gemm2 || !DiagnosticSwitches.GemmHalfActivations
            || NativeKQuantStages(NativeKQuantTileRows(NativeKQuantMaxRows), NativeKQuantBlockBytes(weightA.Type), k) < 2)
        {
            return null;
        }

        long minRows = Math.Max(NativeKQuantMinRows, fusedBelow + 1);
        long rowsMax = Math.Min(NativeKQuantMaxRows, x.Shape[0]);
        if (rowsMax < minRows)
        {
            return null;
        }

        var half = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.F16);
        if (half.ByteOffset % 16 != 0)
        {
            return null;
        }

        var tx = x;
        Func<Bindings, long> rowsOf = b => Rows(b, tx);
        Func<Bindings, bool> decode = b => rowsOf(b) >= minRows && rowsOf(b) <= rowsMax;
        long? rangeScales = _rangeScales.TryGetValue(half, out long xrs) ? xrs : null;
        var (variants, scratch, arrivals) = NativeKQuantVariants(weightA.Type, n, k, rowsMax, false, rangeScales, new Dictionary<string, string> { ["KQ_DUAL"] = "1" },
            decode, rowsOf, epilogue.Replace("F8N_EPI2", "KQ_EPI2", StringComparison.Ordinal), dual: true);
        Emit($"{label} native kquant decode", nodes, variants,
            [Pointer(half), Pointer(weightA), Pointer(y), Pointer(weightB), Pointer(scratch), Pointer(arrivals), TokenArg(x, rowsPerLead)],
            [half, weightA, weightB], [y, scratch, arrivals]);
        return r => r >= minRows && r <= rowsMax;
    }

    /// <summary>
    /// The launch variants of <c>gemm_kq_native.cu</c> for [<paramref name="n"/>, <paramref name="k"/>] weights of <paramref name="type"/> at
    /// the batches <paramref name="decode"/> of up to <paramref name="rowsMax"/> rows: one per (token tile rows, K split), with the bias
    /// store, the row scales of a row-ranged copy of x at byte <paramref name="rangeScales"/> and the <paramref name="extra"/> macros (the
    /// multi-output form's); the split-K partial slices and the arrival counters they use. The <paramref name="dual"/> form's tiles cover
    /// the interleaved rows of two matrices (2 <paramref name="n"/> rows), its source starting with <paramref name="prefix"/> (the epilogue).
    /// </summary>
    private (KernelVariant[] Variants, TensorRef Scratch, TensorRef Arrivals) NativeKQuantVariants(DataType type, long n, long k, long rowsMax, bool bias,
        long? rangeScales, IReadOnlyDictionary<string, string>? extra, Func<Bindings, bool> decode, Func<Bindings, long> rowsOf, string prefix = "",
        bool dual = false)
    {
        int blockBytes = NativeKQuantBlockBytes(type);
        long tileRowsTotal = dual ? 2 * n : n;
        long columns = Grid.Div(tileRowsTotal, NativeKQuantColumns);
        long steps = k / 256;
        int Split(long rows) => NativeKQuantSplit(columns * Grid.Div(rows, NativeKQuantTileRows(rows)), steps);
        // One variant per (token tile rows, split) over the served batches; the split depends on rows only through the token tiles.
        var configs = new SortedSet<(int TileRows, int Split)>();
        for (long rows = NativeKQuantMinRows; rows <= rowsMax; rows++)
        {
            configs.Add((NativeKQuantTileRows(rows), Split(rows)));
        }

        long scratchBytes = Math.Max(16, configs.Max(c => c.Split > 1 ? c.Split * rowsMax * tileRowsTotal * 4 : 0));
        var buffer = SplitScratch(scratchBytes);
        var scratch = new TensorRef(buffer, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var arrivals = SplitArrivals(columns * Grid.Div(rowsMax, configs.Min(c => c.TileRows)));
        var variants = configs.Select(c =>
        {
            int stages = NativeKQuantStages(c.TileRows, blockBytes, k);
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("KQ_FMT", NativeKQuantFormat(type)), ("KQ_BM", c.TileRows), ("KQ_STAGES", stages),
                ("KQ_SPLIT", c.Split), ("KQ_WM", NativeKQuantWarpRows(c.TileRows)));
            if (bias)
            {
                macros["H_BIAS"] = "1";
            }

            if (rangeScales is { } scales)
            {
                macros["H_XRS"] = scales.ToString(CultureInfo.InvariantCulture) + "ull";
            }

            foreach (var (name, value) in extra ?? new Dictionary<string, string>())
            {
                macros[name] = value;
            }

            var kernel = _compiler.CompileCuda("gemm_kq_native.cu", prefix, "gemm_kq_native", NativeKQuantThreads, stages * NativeKQuantStageBytes(c.TileRows, blockBytes, k),
                _opt.Target, macros, _opt.Cache);
            Func<Bindings, bool> pick = b => decode(b) && NativeKQuantTileRows(rowsOf(b)) == c.TileRows && Split(rowsOf(b)) == c.Split;
            return new KernelVariant(pick, kernel, b => new Grid(Grid.Div(rowsOf(b), c.TileRows), (uint)columns, (uint)c.Split));
        }).ToArray();
        return (variants, scratch, arrivals);
    }
}
