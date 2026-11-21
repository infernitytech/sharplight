using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Threads per block of <c>gemm_kq_native.cu</c> (2 x 4 warps).</summary>
    private const int NativeKQuantThreads = 256;

    /// <summary>Weight rows per block of <c>gemm_kq_native.cu</c>.</summary>
    private const int NativeKQuantColumns = 64;

    /// <summary>Largest batch the hand-written K-quant decode GEMM serves (two token tiles of 32 rows, or one of 64).</summary>
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
    /// L1 cache keeps some room.
    /// </summary>
    private int NativeKQuantStages(int tileRows, int blockBytes, long k)
    {
        int stage = NativeKQuantStageBytes(tileRows, blockBytes, k);
        long limit = Math.Min(_opt.MaxSharedMemory, 96L * 1024);
        return (int)Math.Clamp(limit / stage, 0, 4);
    }

    /// <summary>
    /// Token rows of the hand-written K-quant decode GEMM's tile for <paramref name="rows"/> rows (<c>SHARPLIGHT_NATIVE_KQ_BM</c>): 32 up to
    /// 32 rows (three stages), else 64 (two stages).
    /// </summary>
    private static int NativeKQuantTileRows(long rows) => DiagnosticSwitches.NativeKQuantTileRows ?? (rows <= 32 ? 32 : 64);

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
    /// K split of the hand-written K-quant decode GEMM for <paramref name="tiles"/> blocks of <paramref name="steps"/> superblocks: the
    /// fewest estimated superblocks per block times waves of blocks (<see cref="WaveFillSplit"/>, one block per SM: a block takes most of
    /// an SM's shared memory), at most 8.
    /// </summary>
    private int NativeKQuantSplit(long tiles, long steps)
        => DiagnosticSwitches.NativeKQuantSplit ?? Math.Min(8, WaveFillSplit(tiles, steps, Math.Max(1, _opt.ComputeUnits)));

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
        long minRows = NativeKQuantMinRows;
        if (rowsMax < minRows)
        {
            return null;
        }

        int blockBytes = NativeKQuantBlockBytes(p.W.Type);
        long columns = Grid.Div(p.N, NativeKQuantColumns);
        long steps = p.K / 256;
        // One variant per (token tile rows, split) over the served batches; the split depends on rows only through the token tiles.
        var configs = new SortedSet<(int TileRows, int Split)>();
        for (long rows = minRows; rows <= rowsMax; rows++)
        {
            int tileRows = NativeKQuantTileRows(rows);
            configs.Add((tileRows, NativeKQuantSplit(columns * Grid.Div(rows, tileRows), steps)));
        }

        long scratchBytes = configs.Max(c => c.Split > 1 ? c.Split * rowsMax * p.N * 4 : 0);
        var scratch = SplitScratch(Math.Max(16, scratchBytes));
        var partial = new TensorRef(scratch, 0, DataType.F32, [Math.Max(16, scratchBytes) / 4], [1], null);
        var arrivals = SplitArrivals(columns * Grid.Div(rowsMax, configs.Min(c => c.TileRows)));
        long? rangeScales = _rangeScales.TryGetValue(activations, out long xrs) ? xrs : null;
        Func<Bindings, bool> decode = b => serves(b) && p.RowCount(b) >= minRows && p.RowCount(b) <= rowsMax;
        var variants = configs.Select(c =>
        {
            var macros = Macros(("K", p.K), ("N", p.N), ("Y_STRIDE", p.N), ("KQ_FMT", NativeKQuantFormat(p.W.Type)), ("KQ_BM", c.TileRows),
                ("KQ_STAGES", NativeKQuantStages(c.TileRows, blockBytes, p.K)), ("KQ_SPLIT", c.Split));
            if (bias is not null)
            {
                macros["H_BIAS"] = "1";
            }

            if (rangeScales is { } scales)
            {
                macros["H_XRS"] = scales.ToString(CultureInfo.InvariantCulture) + "ull";
            }

            int shared = NativeKQuantStages(c.TileRows, blockBytes, p.K) * NativeKQuantStageBytes(c.TileRows, blockBytes, p.K);
            var kernel = _compiler.CompileCuda("gemm_kq_native.cu", "", "gemm_kq_native", NativeKQuantThreads, shared, _opt.Target, macros, _opt.Cache);
            Func<Bindings, bool> pick = b => decode(b) && NativeKQuantTileRows(p.RowCount(b)) == c.TileRows
                && NativeKQuantSplit(columns * Grid.Div(p.RowCount(b), c.TileRows), steps) == c.Split;
            return new KernelVariant(pick, kernel, b => new Grid(Grid.Div(p.RowCount(b), c.TileRows), (uint)columns, (uint)c.Split));
        }).ToList();
        KernelArg[] biasArgs = bias is null ? [] : [Pointer(bias)];
        TensorRef[] biasReads = bias is null ? [] : [bias];
        Emit($"{MatMulLabel(p)} native kquant decode{epilogue?.Label}", [p.Node, .. epilogue?.Group ?? []], [.. variants],
            [Pointer(activations), Pointer(p.W), Pointer(y), .. biasArgs, new PointerArg(scratch, 0), Pointer(arrivals), TokenArg(p.X, p.RowsPerLead)],
            [activations, p.W, .. biasReads], [y, partial, arrivals]);
        return decode;
    }
}
