using System.Globalization;
using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowering of the expert-indexed matrix product (MatMulId, see kernels/moe.slang).</summary>
public sealed partial class GraphLowerer
{
    private const int MoeMaxExperts = 1024;
    private const int MoeMinTile = 16;

    /// <summary>Rows of the expert tiles of the expert-grouped skinny GEMM (one 16-row token fragment).</summary>
    private const int MoeSkinnyTile = 16;

    /// <summary>
    /// Average (token, slot) items per expert up to which the expert-grouped skinny GEMM serves a step (<see cref="MoeSkinnyTile"/>-row
    /// tiles); above, gemm_cm2's 32-128-row tiles. <c>SHARPLIGHT_MOE_SKINNY=n</c> overrides it, 0 disables the skinny form.
    /// </summary>
    private static int MoeSkinnyItemsPerExpert => DiagnosticSwitches.MoeSkinnyItemsPerExpert ?? 12;

    private static OpSupport SupportsMoe(Node node)
    {
        var x = node.Inputs[0];
        var w = node.Inputs[1];
        var ids = node.Inputs[2];
        if (x.Type != DataType.F32)
        {
            return OpSupport.No($"MatMulId activations must be f32, got {x.Type}");
        }

        if (ids.Type != DataType.I32)
        {
            return OpSupport.No("MatMulId expert indices must be i32");
        }

        if (w.Kind != ValueKind.Weight)
        {
            return OpSupport.No("MatMulId weight must be a model parameter");
        }

        if (WeightFormat(w.Type) is null)
        {
            return OpSupport.No($"MatMulId weight type {w.Type} not supported");
        }

        if (w.Shape.Rank != 3 || w.Shape.Dims.Any(d => d.IsSymbolic))
        {
            return OpSupport.No("MatMulId weight shape must be static [E, N, K]");
        }

        if (w.Shape[0].Fixed > MoeMaxExperts)
        {
            return OpSupport.No($"MatMulId with more than {MoeMaxExperts} experts");
        }

        if (w.Shape[2].Fixed % 8 != 0)
        {
            return OpSupport.No($"MatMulId K={w.Shape[2].Fixed} is not a multiple of 8");
        }

        if (ids.Shape[1].IsSymbolic)
        {
            return OpSupport.No("MatMulId slot count must be static");
        }

        return OpSupport.Yes;
    }

    // Routing (expert tiles + sorted items) is shared by the MatMulIds of one expert block (same ids, same E, same tiling: the
    // tensor-core path depends on K, so the projections of one block may tile differently) that also run their grouped GEMMs for
    // the same batches: the same matvec bound (an int-dot gate / up serves up to 128 items per step, a down projection in a format
    // without int-dot only 16, and reusing the gate's routing left the down GEMM without one for 17-128 items) under the same
    // launch condition.
    private BufferDecl? _moeScratch;
    private (Value Ids, long E, long MaxTiles, bool Coop, long MatvecMaxItems, object? Condition)? _moeRouted;

    // Expert-sorted f16 copy of the activations of the grouped GEMMs, shared by the MatMulIds that read the same x with the same
    // routing (gate and up of an expert block).
    private BufferDecl? _moeHalfActivations;
    private (Value X, Value Ids, long MaxTiles, long MatvecMaxItems, object? Condition)? _moeGathered;

    /// <summary>
    /// Emits (once per activations and routing) the gather of the routed rows of x into expert-sorted f16 rows and returns them
    /// ([items, K], row o = sorted position o of the routing). The rows are row-ranged (common.slang, f16_range_scale) unless
    /// <c>SHARPLIGHT_F16_RANGE=0</c>: x need not be bounded (the down projection reads a gated FFN's product), so a row reaching
    /// 2^15 is stored scaled, with its scale word at byte <paramref name="rangeScales"/> + 4 o (null for plain rows), which the
    /// grouped gemm_cm2 undoes in its store. Rows below 2^15 are stored exactly as by the plain gather.
    /// </summary>
    private TensorRef EmitSortedHalfActivations(Node n, TensorRef x, long xStride, bool perSlot, long S, long maxItems, long maxTiles, long K, int[] tileSizes,
        long matvecMaxItems, Func<Bindings, long> items, Func<Bindings, int> tile, Func<Bindings, uint> tiles, TensorRef scratch,
        PointerArg scratchArg, TensorRef ids, out long? rangeScales)
    {
        bool ranged = DiagnosticSwitches.F16RangeGuard;
        long rowBytes = maxItems * K * 2;
        rangeScales = ranged ? rowBytes : null;
        _moeHalfActivations ??= NewBuffer(BufferKind.Activations, "moe-x16", 0);
        _moeHalfActivations.Bytes = Math.Max(_moeHalfActivations.Bytes, rowBytes + (ranged ? maxItems * 4 : 0));
        var half = new TensorRef(_moeHalfActivations, 0, DataType.F16, [maxItems, K], [K, 1], null);
        if (_moeGathered is { } g && g.X == n.Inputs[0] && g.Ids == n.Inputs[2] && g.MaxTiles == maxTiles && g.MatvecMaxItems == matvecMaxItems
            && ReferenceEquals(g.Condition, _launchCondition))
        {
            return half;
        }

        Emit($"moe-gather {n.Inputs[0].Name} x→{(ranged ? "ranged " : "")}f16", [n],
            [.. tileSizes.Select(t => new KernelVariant(b => items(b) > matvecMaxItems && tile(b) == t,
                Lib("moe.slang", "moe_gather_x16", ranged
                    ? Macros(("K", K), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xStride), ("MAXTILES", maxTiles), ("TILE_M", t), ("Y_RS", rowBytes + "ull"))
                    : Macros(("K", K), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xStride), ("MAXTILES", maxTiles), ("TILE_M", t))),
                ranged ? b => new Grid((uint)t, tiles(b)) : b => new Grid(Grid.Div(t * K / 8, 256), tiles(b))))],
            [Pointer(x), scratchArg, Pointer(half), TokenArg(ids)],
            [x, scratch], [half]);
        _moeGathered = (n.Inputs[0], n.Inputs[2], maxTiles, matvecMaxItems, _launchCondition);
        return half;
    }

    /// <summary>
    /// Items up to which the expert matvec (one item per token and slot, each computed alone) serves a step: at least the items of the
    /// row-invariant rows (<see cref="InvariantRows"/> tokens of <paramref name="slots"/> experts each), so a speculative verification
    /// step computes its rows exactly as single-row steps do instead of switching to the grouped GEMM.
    /// </summary>
    private long InvariantMatvecItems(long items, long slots) => Math.Max(items, InvariantRows * slots);

    /// <summary>
    /// Items up to which the per-item int-dot expert matvec serves a step: <see cref="DiagnosticSwitches.MoeIntDotMaxItems"/> when
    /// set, else half the expert count (16-128) when the tensor-core grouped GEMM takes the larger steps. The per-item matvec reads
    /// an expert's weights once per item routed to it, the grouped GEMM once per expert tile, so the GEMM wins from about one item
    /// per two experts (RTX PRO 4500: Qwen3-30B-A3B, 128 experts, 16 sequences 21.0 → 15.5 ms per step; gpt-oss-20b, 32 experts,
    /// 16 sequences 20.6 → 10.7 ms); without tensor cores the scalar GEMM only wins much later.
    /// </summary>
    private long IntDotPerItemMaxItems(long experts, long slots, bool tensorCoreGemm)
        => InvariantMatvecItems(DiagnosticSwitches.MoeIntDotMaxItems > 0 ? DiagnosticSwitches.MoeIntDotMaxItems
            : tensorCoreGemm ? Math.Clamp(experts / 2, 16, 128) : 128, slots);

    /// <summary>Whether the expert products of <paramref name="w"/> (inner size <paramref name="k"/>) run the tensor-core grouped GEMM.</summary>
    private bool MoeTensorCoreGemm(TensorRef w, long k)
        => !DiagnosticSwitches.MoeNoCoop && (_opt.Has(DeviceFeatures.MatrixF16) || SmallFragments) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32
           && k % 32 == 0 && (w.Type != DataType.F32 || _opt.Compile.AllowReducedPrecision);

    /// <summary>
    /// Two expert products of the same activations and routing (gate / up) whose outputs only feed one elementwise group, which
    /// may also read per-expert rows of f32 tables through the same routing (<paramref name="gathers"/>, e.g. the experts' biases):
    /// batches on the int-dot expert matvec (decode) run one launch that computes both products and applies the group in its
    /// store; other batches run the two products and the group.
    /// </summary>
    private void LowerDualMatMulId(Node a, Node bNode, List<Node> gathers, List<Node> group, Value output)
    {
        var w = Ref(a.Inputs[1]);
        long N = w.Shape[1];
        long K = w.Shape[2];
        var outer = _launchCondition;
        if (UseIntDotMatvec(a, w.Type, K))
        {
            var x = PrepareMatMulInput(Ref(a.Inputs[0]), "matmul-id", out long xs, out long rowsPerLead);
            var ids = Ref(a.Inputs[2]);
            if (!ids.IsDense || ids.Type != DataType.I32)
            {
                ids = DenseTemp(ids, "matmul-id ids");
            }

            long S = ids.Shape[1];
            var y = Produce(output);
            var m = Macros(("WFMT", WeightFormat(w.Type)!.Value), ("K", K), ("N", N), ("E", w.Shape[0]), ("S", S), ("PER_SLOT", x.Rank == 3),
                ("X_STRIDE", xs), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"), ("EXPERT_BYTES", w.Type.ByteSize(K) * N + "ull"));
            string source = DualExpertEpilogueSource(a, bNode, gathers, group, output, N);
            long mv = IntDotPerItemMaxItems(w.Shape[0], S, MoeTensorCoreGemm(w, K));
            var tids = ids;
            EmitIntDotExpertMatvec($"matmul-id2 {output.Name} [{w.Shape[0]}x{K}→{N}]x2 {w.Type} fused[{string.Join(",", group.Select(g => g.Op.Name))}]",
                [.. gathers, a, bNode, .. group], a, x, xs, rowsPerLead, [w, Ref(bNode.Inputs[1])], ids, y, m, mv, S,
                [.. gathers.Select(g => Ref(g.Inputs[0]))],
                macros => _compiler.Compile("mmvq_moe_dual.slang", source, "moe_mmvq", _opt.Target, macros, _opt.Cache));
            _launchCondition = b => (outer?.Invoke(b) ?? true) && Rows(b, tids) * S > mv;
        }

        LowerMatMulId(a);
        LowerMatMulId(bNode);
        LowerElementwise(group, [output]);
        _launchCondition = outer;
    }

    /// <summary>
    /// The routed-expert sum of a MoE block (<see cref="LoweringUnit.ExpertSum"/>): batches on the int-dot expert matvec run one
    /// launch per token that computes the product of every slot, applies the group (which may read the gathered per-expert rows and
    /// per-slot scalars) and sums the slots in order, as the narrow ReduceSum does; other batches run the product, the group and the
    /// sum.
    /// </summary>
    private void LowerExpertSum(Node product, List<Node> gathers, List<Node> group, Node permute, Node reduce)
    {
        var w = Ref(product.Inputs[1]);
        long N = w.Shape[1];
        long K = w.Shape[2];
        var summed = permute.Inputs[0];
        var outer = _launchCondition;
        // One subgroup per slot: the workgroup holds S subgroups.
        if (UseIntDotMatvec(product, w.Type, K) && product.Inputs[2].Shape[1] is { IsSymbolic: false } slotDim
            && slotDim.Fixed * _opt.SubgroupSize <= 1024)
        {
            var x = PrepareMatMulInput(Ref(product.Inputs[0]), "matmul-id", out long xs, out long rowsPerLead);
            var ids = Ref(product.Inputs[2]);
            if (!ids.IsDense || ids.Type != DataType.I32)
            {
                ids = DenseTemp(ids, "matmul-id ids");
            }

            long S = ids.Shape[1];
            var inGroup = group.Select(g => g.Output).ToHashSet();
            var gathered = gathers.Select(g => g.Output).ToHashSet();
            var scalars = group.SelectMany(g => g.Inputs).Where(v => v != product.Output && !inGroup.Contains(v) && !gathered.Contains(v)).Distinct()
                .Select(Ref).ToList();
            var y = Produce(reduce.Output);
            var m = Macros(("WFMT", WeightFormat(w.Type)!.Value), ("K", K), ("N", N), ("E", w.Shape[0]), ("S", S), ("PER_SLOT", x.Rank == 3),
                ("X_STRIDE", xs), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"), ("EXPERT_BYTES", w.Type.ByteSize(K) * N + "ull"));
            string source = ExpertSumEpilogueSource(product, gathers, [.. group.SelectMany(g => g.Inputs).Where(v => v != product.Output && !inGroup.Contains(v)
                && !gathered.Contains(v)).Distinct()], scalars, group, summed, N);
            long mv = IntDotPerItemMaxItems(w.Shape[0], S, MoeTensorCoreGemm(w, K));
            var tids = ids;
            EmitIntDotExpertMatvec($"matmul-id-sum {reduce.Output.Name} [{w.Shape[0]}x{K}→{N}] {w.Type} fused[{string.Join(",", group.Select(g => g.Op.Name))}]+slot-sum",
                [.. gathers, product, .. group, permute, reduce], product, x, xs, rowsPerLead, [w], ids, y, m, mv, S,
                [.. gathers.Select(g => Ref(g.Inputs[0])), .. scalars],
                macros => _compiler.Compile("mmvq_moe_sum.slang", source, "moe_mmvq", _opt.Target, macros, _opt.Cache), slotSum: true);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && Rows(b, tids) * S > mv;
        }

        LowerMatMulId(product);
        LowerElementwise(group, [summed]);
        LowerNodeOp(permute);
        LowerNodeOp(reduce);
        _launchCondition = outer;
    }

    /// <summary>
    /// Source of the <c>MOE_EPI1</c> epilogue of the slot-sum expert matvec: the group over the product, the gathered rows and the
    /// per-slot scalars (element (t, s) at t * stride0 + s * stride1).
    /// </summary>
    private static string ExpertSumEpilogueSource(Node product, List<Node> gathers, List<Value> scalarValues, List<TensorRef> scalars, List<Node> group,
        Value output, long n)
    {
        var name = new Dictionary<Value, string> { [product.Output] = "m0" };
        for (int i = 0; i < gathers.Count; i++)
        {
            name[gathers[i].Output] = "g" + i;
        }

        for (int i = 0; i < scalarValues.Count; i++)
        {
            name[scalarValues[i]] = "c" + i;
        }

        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine($"#define MOE_EPI_FIELDS {string.Concat(gathers.Select((_, i) => $"float* gt{i}; "))}{string.Concat(scalars.Select((_, i) => $"float* sc{i}; "))}");
        src.AppendLine($"float epi1(float m0{string.Concat(gathers.Select((_, i) => $", float g{i}"))}{string.Concat(scalars.Select((_, i) => $", float c{i}"))}) {{");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            src.AppendLine($"  float {local} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = local;
        }

        src.AppendLine($"  return {name[output]};");
        src.AppendLine("}");
        string rows = string.Concat(gathers.Select((_, i) => $", p.gt{i}[uint64_t(e) * {n} + (n)]"));
        string perSlot = string.Concat(scalars.Select((r, i) =>
            $", p.sc{i}[uint64_t(t) * {r.Strides[0]} + uint64_t(s) * {r.Strides[1]}]"));
        src.AppendLine($"#define MOE_EPI1(m0, e, n, t, s) epi1(m0{rows}{perSlot})");
        src.AppendLine("#include \"mmvq.slang\"");
        return src.ToString();
    }

    /// <summary>Source of the <c>MOE_EPI</c> epilogue of the dual expert matvec: the group over both products and the gathered rows.</summary>
    private static string DualExpertEpilogueSource(Node a, Node bNode, List<Node> gathers, List<Node> group, Value output, long n)
    {
        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        for (int i = 0; i < gathers.Count; i++)
        {
            name[gathers[i].Output] = "g" + i;
        }

        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine($"#define MOE_EPI_FIELDS {string.Concat(gathers.Select((_, i) => $"float* gt{i}; "))}");
        src.AppendLine($"float epi2(float m0, float m1{string.Concat(gathers.Select((_, i) => $", float g{i}"))}) {{");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            src.AppendLine($"  float {local} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = local;
        }

        src.AppendLine($"  return {name[output]};");
        src.AppendLine("}");
        string rows = string.Concat(gathers.Select((_, i) => $", p.gt{i}[uint64_t(e) * {n} + (n)]"));
        src.AppendLine($"#define MOE_EPI(m0, m1, e, n) epi2(m0, m1{rows})");
        src.AppendLine("#include \"mmvq.slang\"");
        return src.ToString();
    }

    private void LowerMatMulId(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var ids = Ref(n.Inputs[2]);
        long E = w.Shape[0];
        long N = w.Shape[1];
        long K = w.Shape[2];
        long S = ids.Shape[1];
        bool perSlot = x.Rank == 3;
        if (!ids.IsDense || ids.Type != DataType.I32)
        {
            ids = DenseTemp(ids, "matmul-id ids");
        }

        x = PrepareMatMulInput(x, "matmul-id", out long xs, out long rowsPerLead);
        var y = Produce(n.Output);
        int fmt = WeightFormat(w.Type)!.Value;
        long rowBytes = w.Type.ByteSize(K);
        var m = Macros(("WFMT", fmt), ("K", K), ("N", N), ("E", E), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xs),
            ("ROW_BYTES", rowBytes + "ull"), ("EXPERT_BYTES", rowBytes * N + "ull"));
        var tids = ids;
        var tx = x;
        long maxItems = ids.Shape[0] * S;
        Func<Bindings, long> items = b => Rows(b, tids) * S;
        bool coop = MoeTensorCoreGemm(w, K);
        // Items up to which the per-item matvec is used (tensor-core GEMMs win from a few rows, the scalar tiled GEMM much
        // later, the int-dot matvec only from about 16 items per expert).
        bool intDot = UseIntDotMatvec(n, w.Type, K);
        long mv = intDot ? IntDotPerItemMaxItems(E, S, coop) : InvariantMatvecItems(coop ? DiagnosticSwitches.MoeMatvecMaxItems : Math.Max(DiagnosticSwitches.MoeMatvecMaxItems, DiagnosticSwitches.MoeScalarRows * E), S);
        string label = $"matmul-id {n.Output.Name} [{E}x{K}→{N}] {w.Type}";

        // Decode / few rows: one matvec per (token, slot), with int8 dot products when the format allows.
        if (intDot)
        {
            EmitIntDotExpertMatvec(label, n, x, xs, rowsPerLead, w, ids, y, m, mv, S);
        }

        Emit(label + " matvec", [n],
            [new KernelVariant(b => items(b) <= mv && !intDot, Lib("moe.slang", "moe_matvec", m), b => new Grid(Grid.Div(N, 4), (uint)Math.Max(1, items(b))))],
            [Pointer(x), Pointer(w), Pointer(ids),
             Pointer(y), TokenArg(ids)],
            [x, w, ids], [y]);

        // Many rows: group by expert, then a GEMM per expert tile.
        long maxTiles = Math.Min(maxItems, maxItems / MoeMinTile + E);
        long scratchWords = 1 + 3 * maxTiles + maxItems;
        _moeScratch ??= NewBuffer(BufferKind.Activations, "moe-routing", 0);
        _moeScratch.Bytes = Math.Max(_moeScratch.Bytes, Math.Max(16, scratchWords * 4));
        var scratch = new TensorRef(_moeScratch, 0, DataType.U32, [scratchWords], [1], null);

        // Tile rows per expert: small tiles when experts see few rows on average (less wasted MMA work). The grouped gemm_cm2
        // also has 128-row tiles for experts that see many rows.
        bool gemm2 = coop && DiagnosticSwitches.MoeGemm2 && !SmallFragments;
        // 8x8 fragments (Metal) keep to the 32-row tile: 16 accumulators per SIMD-group, as gemm_sg (the 64-row tile's 32 crashed
        // Metal's backend compiler on the CI GPU).
        int[] coopTiles = gemm2 ? [32, 64, 128] : SmallFragments ? [32] : [32, 64];
        // Few items per expert (batched decode): 16-row expert tiles on the expert-grouped skinny GEMM, which reads the weights
        // at the matvec's bandwidth instead of through gemm_cm2's 32-row tiles.
        bool skinnyMoe = gemm2 && DiagnosticSwitches.MoeSortedHalfActivations && DiagnosticSwitches.SkinnyGemm && K % 32 == 0
                         && MoeSkinnyItemsPerExpert > 0;
        long skinnyItems = MoeSkinnyItemsPerExpert * E;
        int[] routeTiles = skinnyMoe ? [MoeSkinnyTile, .. coopTiles] : coopTiles;
        Func<Bindings, int> tile = b => !coop ? 64
            : SmallFragments ? 32
            : DiagnosticSwitches.MoeTile != 0 ? DiagnosticSwitches.MoeTile
            : skinnyMoe && items(b) <= skinnyItems ? MoeSkinnyTile
            : items(b) <= 24 * E ? 32
            : !gemm2 || items(b) <= 160 * E ? 64 : 128;
        Func<Bindings, uint> tiles = b => (uint)Math.Max(1, Math.Min(items(b), items(b) / tile(b) + E));
        var scratchArg = new PointerArg(_moeScratch, 0);

        if (_moeRouted is not { } r || r.Ids != n.Inputs[2] || r.E != E || r.MaxTiles != maxTiles || r.Coop != coop || r.MatvecMaxItems != mv
            || !ReferenceEquals(r.Condition, _launchCondition))
        {
            int[] tiling = coop ? routeTiles : [64];
            Emit($"moe-route {n.Inputs[2].Name} [{E} experts]", [n],
                [.. tiling.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_route", new Dictionary<string, string>(Macros(("E", E), ("S", S), ("TILE_M", t), ("MAXTILES", maxTiles)))),
                    _ => new Grid(1)))],
                [Pointer(ids), scratchArg, TokenArg(ids)],
                [ids], [scratch]);
            _moeRouted = (n.Inputs[2], E, maxTiles, coop, mv, _launchCondition);
        }

        var gm = new Dictionary<string, string>(m) { ["MAXTILES"] = maxTiles.ToString(CultureInfo.InvariantCulture) };
        var gargs = new KernelArg[] { Pointer(x), Pointer(w), scratchArg, Pointer(y), TokenArg(ids) };
        if (gemm2)
        {
            // The dense tensor-core GEMM in its expert-grouped form: one routing tile per workgroup row.
            var groupedMacros = new Dictionary<string, string>(gm) { ["MOE"] = "1", ["Y_STRIDE"] = N.ToString(CultureInfo.InvariantCulture) };
            var gemmX = x;
            if (DiagnosticSwitches.MoeSortedHalfActivations)
            {
                gemmX = EmitSortedHalfActivations(n, x, xs, perSlot, S, maxItems, maxTiles, K, routeTiles, mv, items, tile, tiles, scratch, scratchArg, ids,
                    out long? rangeScales);
                gargs = [Pointer(gemmX), Pointer(w), scratchArg, Pointer(y), TokenArg(ids)];
                groupedMacros["X_F16"] = "1";
                groupedMacros["X_SORTED"] = "1";
                groupedMacros["X_STRIDE"] = K.ToString(CultureInfo.InvariantCulture);
                if (rangeScales is { } offset)
                {
                    groupedMacros["X_RS"] = offset.ToString(CultureInfo.InvariantCulture) + "ull";
                }
            }

            // Split tiles: the 64- and 128-row launches skip routing tiles of at most half their height (an expert's last tile
            // with few rows), which a second launch runs on half-height tiles. Each output row keeps its K order, so the result
            // does not depend on the tile height. The two launches write disjoint output rows, so they share a write group
            // and run without a barrier between them.
            bool split = DiagnosticSwitches.MoeSplitTiles;
            _writeGroup = split ? new object() : null;
            Dictionary<string, string> Rows(int t, string limit, int rows)
                => new(Gemm2Macros(groupedMacros, t)) { [limit] = rows.ToString(CultureInfo.InvariantCulture) };
            Emit(label + " gemm-cm2", [n],
                [.. coopTiles.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("matmul.slang", "gemm_cm2", split && t > 32 ? Rows(t, "MOE_MIN_ROWS", t / 2 + 1) : Gemm2Macros(groupedMacros, t)),
                    b => new Grid(Grid.Div(N, Gemm2Columns(t)), tiles(b))))],
                gargs, [gemmX, w, scratch], [y]);
            if (split)
            {
                Emit(label + " gemm-cm2 half tiles", [n],
                    [.. coopTiles.Where(t => t > 32).Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                        Lib("matmul.slang", "gemm_cm2", Rows(t / 2, "MOE_MAX_ROWS", t / 2)),
                        b => new Grid(Grid.Div(N, Gemm2Columns(t / 2)), tiles(b))))],
                    gargs, [gemmX, w, scratch], [y]);
            }

            _writeGroup = null;
            if (skinnyMoe)
            {
                // Two weight fragments and 4 subgroups per workgroup on the direct path: an expert tile's workgroups each read 32
                // weight rows, so the fixed costs (layout probe, the subgroups' final reduction) are spread over more bytes
                // (RTX PRO 4500, gpt-oss-20b MXFP4 experts, gate / up + down at 16 / 64 sequences: 1 fragment x 8 subgroups 557 /
                // 709 us, 2 x 4 495 / 562 us, 4 x 4 508 / 579 us, 2 x 8 512 / 601 us).
                bool direct = UseDirectSkinny(w.Type, half: true, 1, K);
                var (fn, sg, bk) = direct && DiagnosticSwitches.SkinnyGemmConfig is null ? (2, (int)Math.Clamp(K / 64, 1, 4), 64)
                    : SkinnyConfig(N, K, 1, 1, SkinnySharedBytes, direct);
                var skinnyMacros = new Dictionary<string, string>(groupedMacros)
                {
                    ["SK_FM"] = "1",
                    ["SK_FN"] = fn.ToString(CultureInfo.InvariantCulture),
                    ["SK_SG"] = sg.ToString(CultureInfo.InvariantCulture),
                    ["SK_BK"] = bk.ToString(CultureInfo.InvariantCulture),
                    ["SK_DIRECT"] = direct ? "1" : "0",
                };
                Emit(label + " skinny", [n],
                    [new KernelVariant(b => items(b) > mv && tile(b) == MoeSkinnyTile, Lib("matmul.slang", "gemm_sk", skinnyMacros),
                        b => new Grid(Grid.Div(N, fn * 16), tiles(b)))],
                    gargs, [gemmX, w, scratch], [y]);
            }
        }
        else if (coop)
        {
            Emit(label + " gemm-cm", [n],
                [.. coopTiles.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_gemm_cm", new Dictionary<string, string>(gm)
                    {
                        ["WARPS_M"] = "2", ["WARPS_N"] = "2", ["FRAG_M"] = (t / 32).ToString(CultureInfo.InvariantCulture), ["FRAG_N"] = "4",
                        ["MOE_FRAG8"] = SmallFragments ? "1" : "0",
                    }),
                    b => new Grid(Grid.Div(N, 128), tiles(b))))],
                gargs, [x, w, scratch], [y]);
        }
        else
        {
            Emit(label + " gemm", [n],
                [new KernelVariant(b => items(b) > mv, Lib("moe.slang", "moe_gemm", gm), b => new Grid(Grid.Div(N, 64), tiles(b)))],
                gargs, [x, w, scratch], [y]);
        }
    }
}
