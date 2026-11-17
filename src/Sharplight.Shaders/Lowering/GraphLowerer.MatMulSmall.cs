using System.Globalization;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Weight matmuls on 8x8 subgroup matrices (<see cref="Runtime.DeviceFeatures.MatrixF16Small"/>, Metal simdgroup_matrix): the
/// <c>gemm_sg</c> tile of 64 tokens x 128 rows, split over K when the grid would leave most of the GPU idle (partials into the split-K
/// scratch, then <c>splitk_reduce</c>).
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Token rows, output columns and K step of a <c>gemm_sg</c> tile.</summary>
    private const long SmallMatrixTileRows = 64;
    private const long SmallMatrixTileColumns = 128;
    private const long SmallMatrixTileK = 32;

    /// <summary>Largest K split of <c>gemm_sg</c>, and the K steps each split keeps at least.</summary>
    private const int SmallMatrixMaxSplit = 8;
    private const long SmallMatrixMinSteps = 4;

    private CompiledKernel SmallMatrixKernel(MatMulPlan p, int split)
        => Lib("matmul.slang", "gemm_sg", new Dictionary<string, string>(p.Macros) { ["GEMM_SPLITK"] = split.ToString(CultureInfo.InvariantCulture) });

    private static Grid SmallMatrixGrid(MatMulPlan p, Bindings b, uint split)
        => new(Grid.Div(p.N, SmallMatrixTileColumns), Grid.Div(p.RowCount(b), SmallMatrixTileRows), split);

    /// <summary>
    /// K split of <c>gemm_sg</c> for <paramref name="rows"/> activation rows: doubles while the grid is below the device's concurrent
    /// workgroups and every split keeps <see cref="SmallMatrixMinSteps"/> K steps; 1 for rows the matvecs handle.
    /// </summary>
    private int ChooseSmallMatrixSplit(long n, long k, long rows)
    {
        if (rows <= MatvecMaxRows)
        {
            return 1;
        }

        long tiles = (n + SmallMatrixTileColumns - 1) / SmallMatrixTileColumns * ((rows + SmallMatrixTileRows - 1) / SmallMatrixTileRows);
        long steps = (k + SmallMatrixTileK - 1) / SmallMatrixTileK;
        int split = 1;
        while (split < SmallMatrixMaxSplit && tiles * split < SimtWorkgroupTarget && steps / (split * 2) >= SmallMatrixMinSteps)
        {
            split *= 2;
        }

        return split;
    }

    /// <summary>Split-K <c>gemm_sg</c>: partial sums into the split-K scratch, then <c>splitk_reduce</c>.</summary>
    private void EmitSmallMatrixSplitKMatMul(MatMulPlan p)
    {
        long n = p.N;
        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long scratchBytes = 0;
        var splits = new SortedSet<int>();
        // The split depends on rows only through the tile count: probing every 64 rows covers every split and its largest row count.
        for (long rows = SmallMatrixTileRows; rows < rowsMax + SmallMatrixTileRows; rows += SmallMatrixTileRows)
        {
            long probe = Math.Min(rows, rowsMax);
            int split = p.SplitRows(probe);
            if (probe > MatvecMaxRows && split > 1)
            {
                splits.Add(split);
                scratchBytes = Math.Max(scratchBytes, split * probe * n * 4);
            }
        }

        if (splits.Count == 0)
        {
            return;
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        Emit($"{MatMulLabel(p)} split-k", [p.Node],
            [.. splits.Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
                SmallMatrixKernel(p, split), b => SmallMatrixGrid(p, b, (uint)split)))],
            [Pointer(p.X), Pointer(p.W), new PointerArg(scratch, 0), TokenArg(p.X, p.RowsPerLead)],
            [p.X, p.W], [partial]);

        Emit($"matmul {p.Node.Output.Name} split-k reduce", [p.Node],
            [.. splits.Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
                Lib("matmul.slang", "splitk_reduce", new Dictionary<string, string>(p.Macros) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) }),
                b => new Grid(Grid.Div(p.RowCount(b) * n, 256))))],
            [new PointerArg(scratch, 0), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [partial], [p.Y]);
    }
}
