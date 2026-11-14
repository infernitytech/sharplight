using System.Globalization;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Weight matmuls without tensor cores (f32 weights under strict precision, devices without matrix features): the tiled
/// SIMT <c>gemm</c> with 16-token tiles for few rows and 64-token tiles otherwise, and the 128 x 128 <c>gemm_wide</c> for large
/// grids, split over K when the grid would leave most of the GPU idle.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Largest row count that uses the 16-token tile.</summary>
    private const long SimtSmallTileMaxRows = 32;

    /// <summary>Output rows per tile of the SIMT gemm and its K step.</summary>
    private const long SimtTileColumns = 64;
    private const long SimtTileK = 32;

    /// <summary>Workgroups the split-K SIMT gemm aims for (a couple per multiprocessor of a large GPU).</summary>
    private const long SimtTargetWorkgroups = 1024;

    /// <summary>Largest K split of the SIMT gemm.</summary>
    private const int SimtMaxSplit = 16;

    /// <summary>Smallest row count and output width served by the 128 x 128 <c>gemm_wide</c> tile.</summary>
    private const long SimtWideMinRows = 512;
    private const long SimtWideMinColumns = 256;

    /// <summary>Tile edge (tokens and weight rows) and K step of <c>gemm_wide</c>.</summary>
    private const long SimtWideTile = 128;
    private const long SimtWideTileK = 16;

    /// <summary>True when <paramref name="rows"/> rows of an <paramref name="n"/>-wide output use <c>gemm_wide</c>.</summary>
    private static bool SimtWide(long rows, long n) => rows >= SimtWideMinRows && n >= SimtWideMinColumns && DiagnosticSwitches.WideSimtGemm;

    /// <summary>Token rows per SIMT gemm tile for <paramref name="rows"/> activation rows of an <paramref name="n"/>-wide output.</summary>
    private static long SimtTileRows(long rows, long n) => SimtWide(rows, n) ? SimtWideTile : rows <= SimtSmallTileMaxRows ? 16 : 64;

    private Dictionary<string, string> SimtMacros(MatMulPlan p, long tileRows, int split) => new(p.Macros)
    {
        ["GEMM_BM"] = tileRows.ToString(CultureInfo.InvariantCulture),
        ["GEMM_SPLITK"] = split.ToString(CultureInfo.InvariantCulture),
    };

    /// <summary>The SIMT kernel for tiles of <paramref name="tileRows"/> tokens: <c>gemm_wide</c> for 128, else <c>gemm</c>.</summary>
    private CompiledKernel SimtKernel(MatMulPlan p, long tileRows, int split)
        => tileRows == SimtWideTile ? Lib("matmul.slang", "gemm_wide", SimtMacros(p, tileRows, split)) : Lib("matmul.slang", "gemm", SimtMacros(p, tileRows, split));

    private static Grid SimtGrid(MatMulPlan p, Bindings b, uint split)
    {
        long rows = p.RowCount(b);
        long tileRows = SimtTileRows(rows, p.N);
        long tileColumns = tileRows == SimtWideTile ? SimtWideTile : SimtTileColumns;
        return new Grid(Grid.Div(p.N, tileColumns), Grid.Div(rows, tileRows), split);
    }

    /// <summary>The unsplit SIMT gemm for every tile height (rows beyond the matvecs).</summary>
    private List<KernelVariant> SimtGemmVariants(MatMulPlan p) =>
    [
        .. new long[] { SimtWideTile, 16, 64 }.Select(tileRows => new KernelVariant(
            b => SimtTileRows(p.RowCount(b), p.N) == tileRows && p.Split(b) == 1, SimtKernel(p, tileRows, 1), b => SimtGrid(p, b, 1))),
    ];

    /// <summary>
    /// K split of the SIMT gemm for <paramref name="rows"/> activation rows: doubles while the grid is below the target
    /// and every split keeps at least two K steps; 1 for rows the matvecs handle.
    /// </summary>
    private static int ChooseSimtSplit(long N, long K, long rows)
    {
        if (rows <= MatvecMaxRows)
        {
            return 1;
        }

        long tileRows = SimtTileRows(rows, N);
        bool wide = tileRows == SimtWideTile;
        long tileColumns = wide ? SimtWideTile : SimtTileColumns;
        long tiles = (N + tileColumns - 1) / tileColumns * ((rows + tileRows - 1) / tileRows);
        long steps = (K + (wide ? SimtWideTileK : SimtTileK) - 1) / (wide ? SimtWideTileK : SimtTileK);
        // A wide workgroup does four times the work of a 64 x 64 one: a quarter of the workgroups fills the GPU, and each
        // split keeps at least four 16-value K steps.
        long target = wide ? SimtTargetWorkgroups / 4 : SimtTargetWorkgroups;
        long minSteps = wide ? 4 : 2;
        int split = 1;
        while (split < SimtMaxSplit && tiles * split < target && steps / (split * 2) >= minSteps)
        {
            split *= 2;
        }

        // No empty splits: the last split must still own a K step.
        while (split > 1 && (split - 1) * ((steps + split - 1) / split) >= steps)
        {
            split /= 2;
        }

        return split;
    }

    /// <summary>Split-K SIMT gemm: partial sums into a split-K scratch buffer, then <c>splitk_reduce</c>.</summary>
    private void EmitSimtSplitKMatMul(MatMulPlan p)
    {
        long N = p.N;
        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long scratchBytes = 0;
        var configs = new SortedSet<(long TileRows, int Split)>();
        // The split depends on rows only through the tile height and count: probing every 16 rows covers every
        // configuration and the largest row count of each.
        for (long rows = 16; rows < rowsMax + 16; rows += 16)
        {
            long probe = Math.Min(rows, rowsMax);
            int split = p.SplitRows(probe);
            if (probe > MatvecMaxRows && split > 1)
            {
                configs.Add((SimtTileRows(probe, N), split));
                scratchBytes = Math.Max(scratchBytes, split * probe * N * 4);
            }
        }

        if (configs.Count == 0)
        {
            return;
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        bool Selected(Bindings b, long tileRows, int split)
            => p.RowCount(b) > MatvecMaxRows && SimtTileRows(p.RowCount(b), p.N) == tileRows && p.Split(b) == split;

        Emit($"{MatMulLabel(p)} split-k", [p.Node],
            [.. configs.Select(c => new KernelVariant(b => Selected(b, c.TileRows, c.Split),
                SimtKernel(p, c.TileRows, c.Split), b => SimtGrid(p, b, (uint)c.Split)))],
            [Pointer(p.X), Pointer(p.W), new PointerArg(scratch, 0), TokenArg(p.X, p.RowsPerLead)],
            [p.X, p.W], [partial]);

        Emit($"matmul {p.Node.Output.Name} split-k reduce", [p.Node],
            [.. configs.Select(c => c.Split).Distinct().Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
                Lib("matmul.slang", "splitk_reduce", new Dictionary<string, string>(p.Macros) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) }),
                b => new Grid(Grid.Div(p.RowCount(b) * N, 256))))],
            [new PointerArg(scratch, 0), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [partial], [p.Y]);
    }
}
