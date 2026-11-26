using System.Globalization;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Largest batch the multi-output GEMM serves (<c>SHARPLIGHT_MULTI_GEMM_MAX_ROWS</c>; it bounds its split-K scratch). It serves only
    /// the batches whose members all run their own split-K launches, so above 256 rows only projections that stay on gemm_cm2 there take
    /// it: f16 projections with a bias row below <see cref="NativeBiasGemmMinRows"/>, at the row counts that still split K (mixed decode +
    /// prompt steps of 257 to ~512 rows). RTX PRO 4500, 16 decode rows + one or two prompts of 250-500 rows, GPU ms against 256: Qwen2.5-1.5B
    /// -0.5 to -1.6 %, gpt-oss-20b -0.5 % at 266 rows (benchmarks.md section 100.2).
    /// </summary>
    private static long MultiGemmMaxRows => DiagnosticSwitches.MultiGemmMaxRows ?? 1024;

    /// <summary>Most weight matrices one multi-output GEMM concatenates (splitk.slang MULTI).</summary>
    private const int MultiGemmMaxMembers = 3;

    /// <summary>The multi-output group of each member matmul (see <see cref="FindMultiGemmGroups"/>).</summary>
    private readonly Dictionary<Node, MultiGemmGroup> _multiGemmOf = [];

    /// <summary>
    /// Groups single-matmul units that read the same activations with weights of one format and K, all with a bias row or none
    /// (<see cref="LoweringUnit.BiasAdd"/>): the q / k / v projections. At most one member's width is not a multiple of 128 (the
    /// column tile of the multi-output GEMM); it goes last. Each member is still lowered on its own; see <see cref="EmitMultiGemm(MultiGemmGroup)"/>.
    /// </summary>
    private void FindMultiGemmGroups(List<LoweringUnit> units)
    {
        if (!DiagnosticSwitches.MultiGemm)
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

        if (!group.Parts.TryAdd((matmul, part.Fp8), part))
        {
            group.Usable = false;
        }

        return b => !(group.Serves.TryGetValue(part.Fp8, out var serves) && serves(b));
    }

    /// <summary>The bias row of a split-K reduction epilogue that is a lone Add of a row vector, else null.</summary>
    private static TensorRef? EpilogueBias(MatMulEpilogueCode? epilogue)
        => epilogue is { Group: [{ Op.Name: "Add" }], Args: [var bias], Kinds: [1] } ? bias : null;

    /// <summary>Marks matmul <paramref name="matmul"/> as lowered; after a group's last member, emits its multi-output GEMM.</summary>
    private void NoteMatMulLowered(Node matmul)
    {
        if (_multiGemmOf.TryGetValue(matmul, out var group) && ++group.Lowered == group.Members.Count)
        {
            EmitNativeF16TallMultiGemm(group);
            EmitNativeFp8TallMultiGemm(group);
            EmitNativeFp4MultiGemm(group);
            EmitMultiGemm(group);
            foreach (var emit in group.AfterEmit)
            {
                emit();
            }

            group.AfterEmit.Clear();
        }
    }

    /// <summary>
    /// Lowers the elementwise group <paramref name="epilogue"/> that reads matmul <paramref name="matmul"/>'s product as its own launch
    /// under <paramref name="condition"/>. For a member of a multi-output group, the bindings the group's GEMM serves are lowered
    /// after that GEMM (<see cref="MultiGemmGroup.AfterEmit"/>): it is emitted after the group's last member, so a launch emitted
    /// with this member would read the product before it is written (gpt-oss-20b's q / k / v biases on CUDA when the split-K
    /// reduction did not add them, 572eeb80; the biases of FP8 and int-dot projections). The other bindings keep their launch in place.
    /// </summary>
    private void LowerSeparateEpilogue(Node matmul, MatMulEpilogue epilogue, Func<Bindings, bool>? condition)
    {
        var outer = _launchCondition;
        if (!_multiGemmOf.TryGetValue(matmul, out var group) || group.Lowered >= group.Members.Count)
        {
            _launchCondition = condition;
            LowerElementwise(epilogue.Group, [epilogue.Output]);
            _launchCondition = outer;
            return;
        }

        Func<Bindings, bool> served = b => group.Serves.Values.Any(serves => serves(b));
        // Batches whose multi-output GEMM stored the bias (int-dot weights' q / k / v biases) need no launch after it.
        Func<Bindings, bool> servedWithoutBias = b => group.Serves.Any(kv => !group.StoresBias.Contains(kv.Key) && kv.Value(b));
        _launchCondition = b => (condition?.Invoke(b) ?? true) && !served(b);
        LowerElementwise(epilogue.Group, [epilogue.Output]);
        _launchCondition = outer;
        group.AfterEmit.Add(() =>
        {
            if (group.Serves.Count == 0)
            {
                return;
            }

            var before = _launchCondition;
            _launchCondition = b => (condition?.Invoke(b) ?? true) && servedWithoutBias(b);
            LowerElementwise(epilogue.Group, [epilogue.Output]);
            _launchCondition = before;
        });
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
        foreach (bool family in new[] { false, true })
        {
            // gemm_cm2's multi-output form only where the target compiles CUDA C++: on the NVIDIA Vulkan driver gpt-oss-20b's
            // 64-sequence decode step lost 2.5 % with it (16.85 → 17.28 ms; benchmarks.md section 61), the fp8 form gained 1 %.
            if (!family && !_compiler.CompilesCuda(_opt.Target))
            {
                continue;
            }

            if (group.Members.All(m => group.Parts.ContainsKey((m, family))))
            {
                EmitMultiGemm(group, family);
            }
        }
    }

    /// <summary>The multi-output GEMM of <paramref name="group"/> on one kernel family (see <see cref="EmitMultiGemm(MultiGemmGroup)"/>).</summary>
    private void EmitMultiGemm(MultiGemmGroup group, bool family)
    {
        void Skip(string why)
        {
            if (DiagnosticSwitches.MultiGemmDebug)
            {
                Console.Error.WriteLine($"multi-output GEMM of {string.Join(", ", group.Members.Select(m => m.Output.Name))} not emitted: {why}");
            }
        }

        if (!group.Usable)
        {
            Skip("a member's split launch does not fit");
            return;
        }

        var parts = group.Members.Select(m => group.Parts[(m, family)]).ToList();
        var first = parts[0];
        bool fp8 = first.Fp8;
        bool bias = first.Bias is not null;
        if (parts.Any(pt => pt.Fp8 != fp8 || (pt.Bias is not null) != bias || pt.Activations.Buffer != first.Activations.Buffer
                            || pt.Activations.ByteOffset != first.Activations.ByteOffset || pt.Weight.Type != first.Weight.Type
                            || pt.Weight.Shape[1] != first.Weight.Shape[1] || !DenseRowsOf(pt.Output, pt.Weight.Shape[0]))
            || (fp8 && bias))
        {
            Skip("members differ (kernel, bias, activations, weights or output layout)");
            return;
        }

        foreach (var key in new[] { "X_F16", "X_STRIDE", "X_RS", "WFMT", "ROW_BYTES" })
        {
            if (parts.Any(pt => pt.Macros.GetValueOrDefault(key) != first.Macros.GetValueOrDefault(key)))
            {
                Skip("macro " + key + " differs");
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
            Skip("no batch splits K");
            return;
        }

        var rowsOf = first.Rows;
        var fits = tiledSplits.ToHashSet();
        group.Serves[family] = b =>
        {
            long rows = rowsOf(b);
            return rows > MatvecMaxRows && rows <= rowsMax && fits.Contains((TileRows(rows), Split(rows))) && parts.All(pt => pt.Serves(b));
        };

        if (DiagnosticSwitches.MultiGemmDebug)
        {
            Console.Error.WriteLine($"multi-output GEMM of {string.Join(", ", group.Members.Select(m => m.Output.Name))} ({(fp8 ? "fp8" : "gemm_cm2")}, "
                                    + $"bias {bias}): splits {string.Join(" ", tiledSplits)}");
        }

        if (bias)
        {
            group.StoresBias.Add(family);
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        bool fused = DiagnosticSwitches.MultiGemmFused && (!fp8 || _compiler.CompilesCuda(_opt.Target));
        var arrivals = fused ? SplitArrivals(tiles) : null;
        var serves = group.Serves[family];
        var native = fp8 ? EmitNativeFp8MultiGemm(group, parts, widths, k, serves)
            : EmitNativeF16MultiGemm(group, parts, widths, k, serves) ?? EmitNativeKQuantMultiGemm(group, parts, widths, k, serves);
        Func<Bindings, bool> splitServes = native is null ? serves : b => serves(b) && !native(b);
        var variants = tiledSplits.Select(ts =>
        {
            var macros = fp8
                ? new Dictionary<string, string>(Fp8GemmMacros(first.Weight.Type, n, k, ts.TileRows, ts.Split))
                : new Dictionary<string, string>(Gemm2Macros(new Dictionary<string, string>(first.Macros)
                {
                    ["N"] = n.ToString(CultureInfo.InvariantCulture),
                    ["Y_STRIDE"] = n.ToString(CultureInfo.InvariantCulture),
                }, ts.TileRows)) { ["SPLITK"] = ts.Split.ToString(CultureInfo.InvariantCulture) };
            if (fused)
            {
                macros["SPLITK_FUSED"] = "1";
            }

            macros["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_O1"] = o1.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_O2"] = o2.ToString(CultureInfo.InvariantCulture);
            macros["MULTI_BIAS"] = bias ? "1" : "0";
            var kernel = fp8 ? Lib("gemm_fp8.slang", "gemm_fp8", macros) : Lib("matmul.slang", "gemm_cm2", macros);
            int columns = Columns(ts.TileRows);
            return new KernelVariant(b => splitServes(b) && TileRows(rowsOf(b)) == ts.TileRows && Split(rowsOf(b)) == ts.Split, kernel,
                fp8 ? b => new Grid(Grid.Div(rowsOf(b), ts.TileRows), Grid.Div(n, columns), (uint)ts.Split)
                    : b => new Grid(Grid.Div(n, columns), Grid.Div(rowsOf(b), ts.TileRows), (uint)ts.Split));
        });

        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        KernelArg[] biases = bias ? [.. parts.Select(pt => Pointer(pt.Bias!))] : [];
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} split-k", group.Members, [.. variants],
            [Pointer(first.Activations), Pointer(first.Weight), new PointerArg(scratch, 0), Pointer(first.Output), .. arrivals is null ? (KernelArg[])[] : [Pointer(arrivals)],
             .. weights, .. outputs, .. biases, first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight), .. bias ? parts.Select(pt => pt.Bias!) : []],
            fused ? [partial, .. parts.Select(pt => pt.Output), arrivals!] : [partial]);
        if (fused)
        {
            return;
        }

        // The reduction of the concatenated partials, each column stored to its member's output (with its bias); w is the row-ranged
        // f16 copy of the activations where the GEMM read one (its row scales are undone here), else the first weight.
        bool ranged = first.Macros.ContainsKey("X_RS");
        var reduceVariants = tiledSplits.Select(ts => ts.Split).Distinct().Select(splitCount =>
        {
            var macros = new Dictionary<string, string>(WithoutSplit(first.Macros))
            {
                ["N"] = n.ToString(CultureInfo.InvariantCulture),
                ["Y_STRIDE"] = n.ToString(CultureInfo.InvariantCulture),
                ["SPLITK"] = splitCount.ToString(CultureInfo.InvariantCulture),
                ["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture),
                ["MULTI_O1"] = o1.ToString(CultureInfo.InvariantCulture),
                ["MULTI_O2"] = o2.ToString(CultureInfo.InvariantCulture),
                ["MULTI_BIAS"] = bias ? "1" : "0",
            };
            if (fp8)
            {
                macros.Remove("X_RS");
            }

            return new KernelVariant(b => splitServes(b) && Split(rowsOf(b)) == splitCount, Lib("matmul.slang", "splitk_reduce", macros),
                b => new Grid(Grid.Div(rowsOf(b) * n, 256)));
        });
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] split-k reduce", group.Members, [.. reduceVariants],
            [new PointerArg(scratch, 0), ranged && !fp8 ? Pointer(first.Activations) : Pointer(first.Weight), Pointer(first.Output), Pointer(first.Output),
             .. weights, .. outputs, .. biases, first.Tokens],
            [partial, .. ranged && !fp8 ? [first.Activations] : (TensorRef[])[], .. bias ? parts.Select(pt => pt.Bias!) : []],
            [.. parts.Select(pt => pt.Output)]);
    }

    /// <summary>
    /// The decode form of the hand-written FP8 GEMM over the group's concatenated weight rows (<c>gemm_fp8_native.cu</c> MULTI: 64-row
    /// token and weight tiles, 32-row token tiles up to <see cref="NativeFp8ShortTileMaxRows"/> rows, each block in one member's matrix,
    /// storing to that member's output) for the decode batches of at most <see cref="NativeFp8ShortTileMaxRows"/> and of
    /// <see cref="NativeFp8MultiMinRows"/> to 64 rows the multi-output GEMM serves (<paramref name="serves"/>), when every member is a
    /// matrix the hand-written GEMM takes. Returns the batches it serves, or null. One launch instead of the K-split GEMM and its
    /// reduction (k and v alone on the decode form ran 16 blocks each, ~300 GB/s). Each output sums its K blocks in the order of the
    /// member's own decode-form launch.
    /// </summary>
    /// <summary>
    /// Rows from which the multi-output decode form serves on 64-row token tiles (<see cref="EmitNativeFp8MultiGemm"/>). RTX PRO 4500,
    /// Qwen3-8B-FP8 decode steps (BatchBench <c>--text</c>, GPU ms, two alternated pairs): 48 sequences 15.58-15.59 → 15.45-15.48, 64: 16.65
    /// → 16.48, but 32: 14.66-14.67 → 14.92-14.98 (its 96 blocks of q / k / v run two waves at one block per SM, 42 µs against 40.6 + 2.8
    /// for the K split). Up to <see cref="NativeFp8ShortTileMaxRows"/> rows the 32-row token tiles serve instead (two blocks per SM).
    /// </summary>
    private const long NativeFp8MultiMinRows = 48;

    private Func<Bindings, bool>? EmitNativeFp8MultiGemm(MultiGemmGroup group, List<MultiGemmPart> parts, long[] widths, long k, Func<Bindings, bool> serves)
    {
        var first = parts[0];
        int columns = NativeFp8DecodeColumns;
        var xcap = DenseScaleCapacityOf(first.Activations);
        if (columns == NativeFp8Tile || DiagnosticSwitches.NativeGemmMinRows is not null || parts.Count > 3 || xcap <= 0
            || parts.Any(pt => !UseNativeFp8Gemm(pt.Weight.Type, pt.Weight.Shape[0], k)))
        {
            return null;
        }

        long n = widths.Sum();
        var rowsOf = first.Rows;
        Func<Bindings, bool> tall = b => serves(b) && rowsOf(b) >= NativeFp8MultiMinRows && rowsOf(b) <= 64;
        Func<Bindings, bool> shortRows = b => serves(b) && rowsOf(b) <= NativeFp8ShortTileMaxRows;
        Func<Bindings, bool> decode = b => tall(b) || shortRows(b);
        var multi = new Dictionary<string, string>
        {
            ["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture),
            ["MULTI_O1"] = widths[0].ToString(CultureInfo.InvariantCulture),
            ["MULTI_O2"] = (parts.Count > 2 ? widths[0] + widths[1] : n).ToString(CultureInfo.InvariantCulture),
        };
        KernelVariant[] variants = [NativeFp8Variant(n, k, null, xcap, 64, columns, tall, rowsOf, multi), NativeFp8Variant(n, k, null, xcap, 32, columns, shortRows, rowsOf, multi)];
        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} native", group.Members, variants,
            [Pointer(first.Activations), Pointer(first.Weight), Pointer(first.Output), .. weights, .. outputs, first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight)], [.. parts.Select(pt => pt.Output)]);
        return decode;
    }

    /// <summary>
    /// The decode form of the hand-written f16 GEMM over the group's concatenated weight rows (<c>gemm_f16_native.cu</c> H_CFG 2 with
    /// MULTI: each block's weight tile in one member's matrix, storing to that member's output with its bias) for the decode batches of
    /// <see cref="NativeF16DecodeMinRows"/> to 64 rows the multi-output GEMM serves (<paramref name="serves"/>), when every member takes
    /// the decode form and every member boundary is a multiple of its weight tile. Returns the batches it serves, or null. One launch
    /// instead of the K-split GEMM and its reduction.
    /// </summary>
    private Func<Bindings, bool>? EmitNativeF16MultiGemm(MultiGemmGroup group, List<MultiGemmPart> parts, long[] widths, long k, Func<Bindings, bool> serves)
    {
        var first = parts[0];
        long n = widths.Sum();
        int columns = NativeF16DecodeColumns(n);
        if (parts.Count > 3 || parts.Any(pt => !UseNativeF16Decode(pt.Weight.Type, n, k, pt.Macros)) || widths.SkipLast(1).Any(w => w % columns != 0))
        {
            return null;
        }

        bool bias = first.Bias is not null;
        var rowsOf = first.Rows;
        Func<Bindings, bool> decode = b => serves(b) && rowsOf(b) >= NativeF16DecodeMinRows && rowsOf(b) <= NativeF16DecodeTileRows;
        var multi = new Dictionary<string, string>
        {
            ["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture),
            ["MULTI_O1"] = widths[0].ToString(CultureInfo.InvariantCulture),
            ["MULTI_O2"] = (parts.Count > 2 ? widths[0] + widths[1] : n).ToString(CultureInfo.InvariantCulture),
        };
        long? rangeScales = _rangeScales.TryGetValue(first.Activations, out long xrs) ? xrs : null;
        var variant = NativeF16DecodeVariant(n, k, decode, rowsOf, bias, rangeScales, multi);
        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        KernelArg[] firstBias = bias ? [Pointer(first.Bias!)] : [];
        KernelArg[] biases = bias ? [.. parts.Skip(1).Select(pt => Pointer(pt.Bias!))] : [];
        Emit($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} native decode", group.Members, [variant],
            [Pointer(first.Activations), Pointer(first.Weight), Pointer(first.Output), .. firstBias, .. weights, .. outputs, .. biases, first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight), .. bias ? parts.Select(pt => pt.Bias!) : []], [.. parts.Select(pt => pt.Output)]);
        return decode;
    }

    /// <summary>
    /// Records a member's launch on the hand-written f16 GEMM's prompt-sized tiles for its multi-output group (none: no group) and
    /// returns the condition its own launch adds: the bindings the group's multi-output form does not serve.
    /// </summary>
    private Func<Bindings, bool> NoteMultiGemmTallPart(Node matmul, MultiGemmTallPart part)
    {
        if (!DiagnosticSwitches.MultiGemmTall || !_multiGemmOf.TryGetValue(matmul, out var group) || !group.TallParts.TryAdd(matmul, part))
        {
            return _ => true;
        }

        return b => !(group.TallServes?.Invoke(b) ?? false);
    }

    /// <summary>The bindings of at least <paramref name="minRows"/> rows under the current launch condition.</summary>
    private Func<Bindings, bool> TallCondition(long minRows, Func<Bindings, long> rows)
    {
        var condition = _launchCondition;
        return b => (condition?.Invoke(b) ?? true) && rows(b) >= minRows;
    }

    /// <summary>The variants, each also requiring <paramref name="condition"/> (evaluated once per recorded shape, see <see cref="Bindings.Holds"/>).</summary>
    private static KernelVariant[] Unless(KernelVariant[] variants, Func<Bindings, bool> condition)
        => [.. variants.Select(v => v with { When = b => b.Holds(condition) && v.When(b) })];

    /// <summary>
    /// The multi-output form of the hand-written f16 GEMM's prompt-sized tiles (<c>gemm_f16_native.cu</c> H_CFG 0 / 1 with MULTI)
    /// over the concatenated weight rows of a group whose members all run on those tiles (<see cref="MultiGemmGroup.TallParts"/>):
    /// one launch instead of one per member, each block's weight tile in one member's matrix, storing to its output with its bias.
    /// The short k / v matrices alone fill less than a wave each (gpt-oss-20b at 2,008 rows: 2 x 16 blocks of 128 x 256 on 82 SMs,
    /// 51 µs each) and q leaves its fourth wave nearly empty; together they fill it (RTX PRO 4500, 2,048 rows: 378 + 2 x 51.4 →
    /// 406 µs per layer). Each output is the product its member's launch computes (the same tile and K order).
    /// <c>SHARPLIGHT_MULTI_GEMM_TALL=0</c> keeps the members' launches.
    /// </summary>
    private void EmitNativeF16TallMultiGemm(MultiGemmGroup group)
    {
        if (group.Members.Count > 3 || !group.Members.All(group.TallParts.ContainsKey))
        {
            return;
        }

        var parts = group.Members.Select(m => group.TallParts[m]).ToList();
        if (parts.Any(pt => pt.Fp8))
        {
            return;
        }

        var first = parts[0];
        bool bias = first.Bias is not null;
        if (parts.Any(pt => pt.Activations.Buffer != first.Activations.Buffer || pt.Activations.ByteOffset != first.Activations.ByteOffset
                            || (pt.Bias is not null) != bias || pt.RangeScales != first.RangeScales || pt.MinRows != first.MinRows
                            || pt.MaxRows != first.MaxRows || pt.Weight.Type != first.Weight.Type || pt.Weight.Shape[1] != first.Weight.Shape[1]
                            || !DenseRowsOf(pt.Output, pt.Weight.Shape[0]) || pt.Weight.Shape[0] % 2 != 0))
        {
            return;
        }

        long[] widths = [.. parts.Select(pt => pt.Weight.Shape[0])];
        long n = widths.Sum();
        long k = first.Weight.Shape[1];
        bool Allowed(int config) => widths.SkipLast(1).All(w => w % s_nativeF16TileColumns[config] == 0);
        if (!s_nativeF16Configs.Any(Allowed))
        {
            return;
        }

        var multi = new Dictionary<string, string>
        {
            ["MULTI"] = parts.Count.ToString(CultureInfo.InvariantCulture),
            ["MULTI_O1"] = widths[0].ToString(CultureInfo.InvariantCulture),
            ["MULTI_O2"] = (parts.Count > 2 ? widths[0] + widths[1] : n).ToString(CultureInfo.InvariantCulture),
        };
        Func<Bindings, bool> serves = b => parts.All(pt => pt.Serves(b));
        var rowsOf = first.Rows;
        KernelVariant[] Variants(bool sk) => Unless(NativeF16Variants(n, k, first.MaxRows, null, rowsOf, first.MinRows, bias, first.RangeScales, Allowed, multi, sk), serves);
        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        KernelArg[] firstBias = bias ? [Pointer(first.Bias!)] : [];
        KernelArg[] biases = bias ? [.. parts.Skip(1).Select(pt => Pointer(pt.Bias!))] : [];
        var outer = _launchCondition;
        _launchCondition = null;
        EmitNativeF16($"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} native{(bias ? " fused[Add]" : "")}", group.Members, Variants,
            [Pointer(first.Activations), Pointer(first.Weight), Pointer(first.Output), .. firstBias, .. weights, .. outputs, .. biases, first.Tokens],
            [first.Activations, .. parts.Select(pt => pt.Weight), .. bias ? parts.Select(pt => pt.Bias!) : []], [.. parts.Select(pt => pt.Output)]);
        _launchCondition = outer;
        group.TallServes = serves;
    }

    /// <summary>
    /// The multi-output form of the hand-written FP8 GEMM's 128-row tiles (<c>gemm_fp8_tma.cu</c> MULTI) over the concatenated weight
    /// rows of a group whose members all run there on the same E4M3 activation rows (<see cref="MultiGemmGroup.TallParts"/> with
    /// <see cref="MultiGemmTallPart.Fp8"/>): one launch instead of one per member, each tile in one member's matrix, storing to its
    /// output, for the batches where the concatenation runs on 128-row tiles. The short k / v matrices alone fill 1.6 waves each
    /// (Qwen3-8B-FP8 at 2,048 rows: 128 tiles on 82 SMs, 72-79 µs). Each output is the product its member's launch computes (the same
    /// tile and K order). <c>SHARPLIGHT_MULTI_GEMM_TALL=0</c> keeps the members' launches.
    /// </summary>
    private void EmitNativeFp8TallMultiGemm(MultiGemmGroup group)
    {
        if (group.Members.Count > 3 || !group.Members.All(group.TallParts.ContainsKey))
        {
            return;
        }

        var parts = group.Members.Select(m => group.TallParts[m]).ToList();
        var first = parts[0];
        if (parts.Any(pt => !pt.Fp8 || pt.Bias is not null || pt.Activations.Buffer != first.Activations.Buffer
                            || pt.Activations.ByteOffset != first.Activations.ByteOffset || pt.MinRows != first.MinRows || pt.MaxRows != first.MaxRows
                            || pt.Weight.Type != first.Weight.Type || pt.Weight.Shape[1] != first.Weight.Shape[1]
                            || !DenseRowsOf(pt.Output, pt.Weight.Shape[0]) || pt.Weight.Shape[0] % NativeFp8Tile != 0)
            || DenseScaleCapacityOf(first.Activations) <= 0 || !NativeFp8TensorCopies)
        {
            return;
        }

        long xcap = DenseScaleCapacityOf(first.Activations);
        long[] widths = [.. parts.Select(pt => pt.Weight.Shape[0])];
        long n = widths.Sum();
        long k = first.Weight.Shape[1];
        var rowsOf = first.Rows;
        Func<Bindings, bool> serves = b => parts.All(pt => pt.Serves(b)) && NativeFp8TileRows(rowsOf(b), n, k, false) == NativeFp8Tile;
        bool uniform = parts.All(pt => pt.UniformScales);
        var variants = NativeFp8TmaVariants(n, k, null, xcap, serves, rowsOf, uniformScales: uniform, multiWidths: widths, streamK: false);
        var streamKVariants = NativeFp8StreamKUsed(n, k, first.MinRows, first.MaxRows)
            ? NativeFp8TmaVariants(n, k, null, xcap, serves, rowsOf, uniformScales: uniform, multiWidths: widths, streamK: true) : [];
        KernelArg[] weights = [.. parts.Skip(1).Select(pt => Pointer(pt.Weight))];
        KernelArg[] outputs = [.. parts.Skip(1).Select(pt => Pointer(pt.Output))];
        var outer = _launchCondition;
        _launchCondition = null;
        string label = $"matmul multi [{k}→{string.Join("+", widths)}] {first.Weight.Type} fp8 native";
        KernelArg[] args = [Pointer(first.Activations), Pointer(first.Weight), Pointer(first.Output), .. weights, .. outputs, first.Tokens];
        TensorRef[] reads = [first.Activations, .. parts.Select(pt => pt.Weight)];
        TensorRef[] writes = [.. parts.Select(pt => pt.Output)];
        Emit(label, group.Members, variants, args, reads, writes);
        EmitStreamK(label, group.Members, streamKVariants, args, reads, writes, NativeFp8StreamKSlotBytes);
        _launchCondition = outer;
        group.TallServes = serves;
    }

    /// <summary>True when <paramref name="r"/> holds dense rows of <paramref name="width"/> values.</summary>
    private static bool DenseRowsOf(TensorRef r, long width)
        => r.Strides.Length >= 1 && r.Strides[^1] == 1 && r.Shape[^1] == width
           && Enumerable.Range(0, r.Shape.Length - 1).All(i => r.Strides[i] == r.Strides[i + 1] * r.Shape[i + 1]);
}
