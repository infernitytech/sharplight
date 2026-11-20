using System.Globalization;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Largest batch the multi-output GEMM serves (decode and small mixed steps; it bounds its split-K scratch).</summary>
    private const long MultiGemmMaxRows = 256;

    /// <summary>Most weight matrices one multi-output GEMM concatenates (splitk.slang MULTI).</summary>
    private const int MultiGemmMaxMembers = 3;

    /// <summary>The multi-output group of each member matmul (see <see cref="FindMultiGemmGroups"/>).</summary>
    private readonly Dictionary<Node, MultiGemmGroup> _multiGemmOf = [];

    /// <summary>
    /// Groups single-matmul units that read the same activations with weights of one format and K, all with a bias row or none
    /// (<see cref="LoweringUnit.BiasAdd"/>): the q / k / v projections. At most one member's width is not a multiple of 128 (the
    /// column tile of the multi-output GEMM); it goes last. Each member is still lowered on its own; see <see cref="EmitMultiGemm"/>.
    /// </summary>
    private void FindMultiGemmGroups(List<LoweringUnit> units)
    {
        if (!DiagnosticSwitches.MultiGemm || !DiagnosticSwitches.SplitKFused)
        {
            return;
        }

        var byInput = new Dictionary<Value, List<(Node MatMul, bool Bias)>>();
        foreach (var unit in units)
        {
            if (unit.Node is not { Op.Name: "MatMul" } matmul || unit.Fused is not null || unit.Absorbed is not null || unit.Halves is not null
                || unit.Gathers is not null)
            {
                continue;
            }

            var w = matmul.Inputs[1];
            if (w.Kind != ValueKind.Weight || w.Shape.Rank != 2 || w.Shape[0].IsSymbolic || w.Shape[1].IsSymbolic || WeightFormat(w.Type) is null)
            {
                continue;
            }

            if (!byInput.TryGetValue(matmul.Inputs[0], out var list))
            {
                byInput[matmul.Inputs[0]] = list = [];
            }

            list.Add((matmul, unit.BiasAdd is not null));
        }

        foreach (var candidates in byInput.Values)
        {
            foreach (var kind in candidates.GroupBy(c => (c.MatMul.Inputs[1].Type, c.MatMul.Inputs[1].Shape[1].Fixed, c.Bias)))
            {
                var aligned = kind.Select(c => c.MatMul).Where(m => m.Inputs[1].Shape[0].Fixed % 128 == 0).ToList();
                var unaligned = kind.Select(c => c.MatMul).Where(m => m.Inputs[1].Shape[0].Fixed % 128 != 0).Take(1);
                var members = aligned.Take(MultiGemmMaxMembers - (unaligned.Any() && aligned.Count >= MultiGemmMaxMembers ? 1 : 0))
                    .Concat(unaligned).Take(MultiGemmMaxMembers).ToList();
                if (members.Count < 2)
                {
                    continue;
                }

                var group = new MultiGemmGroup { Members = members };
                foreach (var member in members)
                {
                    _multiGemmOf[member] = group;
                }
            }
        }
    }

    /// <summary>
    /// Records a member's fused split-K launch for its multi-output group (none: no group) and returns the condition its own
    /// launch adds: the bindings the multi-output GEMM does not serve.
    /// </summary>
    private Func<Bindings, bool> NoteMultiGemmPart(Node matmul, MultiGemmPart part)
    {
        if (!_multiGemmOf.TryGetValue(matmul, out var group))
        {
            return _ => true;
        }

        if (!group.Parts.TryAdd(matmul, part))
        {
            group.Usable = false;
        }

        return b => !(group.Serves?.Invoke(b) ?? false);
    }

    /// <summary>The bias row of a split-K reduction epilogue that is a lone Add of a row vector, else null.</summary>
    private static TensorRef? EpilogueBias(MatMulEpilogueCode? epilogue)
        => epilogue is { Group: [{ Op.Name: "Add" }], Args: [var bias], Kinds: [1] } ? bias : null;

    /// <summary>Marks matmul <paramref name="matmul"/> as lowered; after a group's last member, emits its multi-output GEMM.</summary>
    private void NoteMatMulLowered(Node matmul)
    {
        if (_multiGemmOf.TryGetValue(matmul, out var group) && ++group.Lowered == group.Members.Count)
        {
            EmitMultiGemm(group);
        }
    }

    /// <summary>
    /// The multi-output GEMM of a group (splitk.slang MULTI): one gemm_cm2 or fp8 GEMM over the concatenated weight rows (each
    /// matrix its own buffer), K split, reduced in the last workgroup of each tile into every member's own output (with its
    /// bias). It serves the batches where every member runs its fused split-K launch and the concatenation splits K too, up to
    /// <see cref="MultiGemmMaxRows"/> rows; the members' split launches skip those (<see cref="MultiGemmGroup.Serves"/>). One
    /// launch instead of three, and the short k / v matrices share the grid of q (each split launch of a 512-row matrix runs a
    /// few column tiles). Each output element sums its K slices in the same order as its member's launch would if their splits
    /// agree; where the splits differ, its partial sums round differently (as between any two split counts).
    /// </summary>
    private void EmitMultiGemm(MultiGemmGroup group)
    {
        if (!group.Usable || group.Parts.Count != group.Members.Count)
        {
            return;
        }

        var parts = group.Members.Select(m => group.Parts[m]).ToList();
        var first = parts[0];
        bool fp8 = first.Fp8;
        bool bias = first.Bias is not null;
        if (parts.Any(pt => pt.Fp8 != fp8 || (pt.Bias is not null) != bias || pt.Activations.Buffer != first.Activations.Buffer
                            || pt.Activations.ByteOffset != first.Activations.ByteOffset || pt.Weight.Type != first.Weight.Type
                            || pt.Weight.Shape[1] != first.Weight.Shape[1] || !DenseRowsOf(pt.Output, pt.Weight.Shape[0]))
            || (fp8 && bias))
        {
            return;
        }

        foreach (var key in new[] { "X_F16", "X_STRIDE", "X_RS", "WFMT", "ROW_BYTES" })
        {
            if (parts.Any(pt => pt.Macros.GetValueOrDefault(key) != first.Macros.GetValueOrDefault(key)))
            {
                return;
            }
        }

        long[] widths = [.. parts.Select(pt => pt.Weight.Shape[0])];
        long n = widths.Sum();
        long k = first.Weight.Shape[1];
        long o1 = widths[0];
        long o2 = parts.Count > 2 ? widths[0] + widths[1] : n;
        long rowsMax = Math.Min(first.RowsMax, MultiGemmMaxRows);
        int blockK = DiagnosticSwitches.GemmBlockK == 64 && k % 64 == 0 ? 64 : 32;
        int TileRows(long rows) => fp8 ? MmqTileRowsFor(rows, fp8: true) : Gemm2TileRowsFor(rows);
        int Columns(int tileRows) => fp8 ? MmqColumns(tileRows) : Gemm2Columns(tileRows);
        int Split(long rows) => DiagnosticSwitches.MultiGemmSplit ?? (fp8 ? QuantizedGemmSplit(n, k, rows, 128, fp8: true)
            : ChooseSplit(n, k, rows, TileRows(rows), Columns(TileRows(rows)), blockK, GemmWaveTiles));

        var tiledSplits = new SortedSet<(int TileRows, int Split)>();
        long scratchBytes = 0;
        long tiles = 0;
        for (long rows = MatvecMaxRows + 1; rows <= rowsMax; rows++)
        {
            int split = Split(rows);
            int tileRows = TileRows(rows);
            if (split <= 1 || o1 % Columns(tileRows) != 0 || o2 % Columns(tileRows) != 0)
            {
                continue;
            }

            tiledSplits.Add((tileRows, split));
            scratchBytes = Math.Max(scratchBytes, split * rows * n * 4);
            tiles = Math.Max(tiles, Grid.Div(n, Columns(tileRows)) * Grid.Div(rows, tileRows));
        }

        if (tiledSplits.Count == 0)
        {
            return;
        }

        var rowsOf = first.Rows;
        var fits = tiledSplits.ToHashSet();
        group.Serves = b =>
        {
            long rows = rowsOf(b);
            return rows > MatvecMaxRows && rows <= rowsMax && fits.Contains((TileRows(rows), Split(rows))) && parts.All(pt => pt.Serves(b));
        };

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var arrivals = SplitArrivals(tiles);
        var serves = group.Serves;
        var variants = tiledSplits.Select(ts =>
        {
            var macros = fp8
                ? new Dictionary<string, string>(Fp8GemmMacros(first.Weight.Type, n, k, ts.TileRows, ts.Split))
                : new Dictionary<string, string>(Gemm2Macros(new Dictionary<string, string>(first.Macros)
                {
                    ["N"] = n.ToString(CultureInfo.InvariantCulture),
                    ["Y_STRIDE"] = n.ToString(CultureInfo.InvariantCulture),
                }, ts.TileRows)) { ["SPLITK"] = ts.Split.ToString(CultureInfo.InvariantCulture) };
            macros["SPLITK_FUSED"] = "1";
            macros["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_O1"] = o1.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_O2"] = o2.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_BIAS"] = bias ? "1" : "0";
            var kernel = fp8 ? Lib("gemm_fp8.slang", "gemm_fp8", macros) : Lib("matmul.slang", "gemm_cm2", macros);
            int columns = Columns(ts.TileRows);
            return new KernelVariant(b => serves(b) && TileRows(rowsOf(b)) == ts.TileRows && Split(rowsOf(b)) == ts.Split, kernel,
                fp8 ? b => new Grid(Grid.Div(rowsOf(b), ts.TileRows), Grid.Div(n, columns), (uint)ts.Split)
                    : b => new Grid(Grid.Div(n, columns), Grid.Div(rowsOf(b), ts.TileRows), (uint)ts.Split));
        });

        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        KernelArg[] biases = bias ? [.. parts.Select(pt => Pointer(pt.Bias!))] : [];
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} split-k", group.Members, [.. variants],
            [Pointer(first.Activations), Pointer(first.Weight), new PointerArg(scratch, 0), Pointer(first.Output), Pointer(arrivals),
             .. weights, .. outputs, .. biases, first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight), .. bias ? parts.Select(pt => pt.Bias!) : []],
            [partial, .. parts.Select(pt => pt.Output), arrivals]);
    }

    /// <summary>True when <paramref name="r"/> holds dense rows of <paramref name="width"/> values.</summary>
    private static bool DenseRowsOf(TensorRef r, long width)
        => r.Strides.Length >= 1 && r.Strides[^1] == 1 && r.Shape[^1] == width
           && Enumerable.Range(0, r.Shape.Length - 1).All(i => r.Strides[i] == r.Strides[i + 1] * r.Shape[i + 1]);
}
