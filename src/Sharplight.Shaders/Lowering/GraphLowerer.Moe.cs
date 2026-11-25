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
    /// Average routed rows per expert above which the grouped gemm_cm2 runs 128-row expert tiles instead of 64-row ones. RTX PRO
    /// 4500, gpt-oss-20b 2048-row prompt step (256 rows per expert): 64-row tiles 163.8 ms against 170.1 ms with 128-row
    /// tiles (per layer, mean of the profile: gate / up 2.70 → 2.61 ms, down 1.50 → 1.38 ms); the threshold was 160 until round 10.
    /// </summary>
    private const int MoeTallTileMinItems = 320;

    /// <summary>
    /// (token, slot) items up to which the expert-grouped skinny GEMM serves a step of <paramref name="w"/> (<see cref="MoeSkinnyTile"/>-row
    /// tiles; above, gemm_cm2's 32-128-row tiles), or 0 when it does not serve the matrix: 12 per expert where the skinny GEMM loads
    /// the operands straight into the fragments (<see cref="UseDirectSkinny"/>), 1 per expert on its staged path.
    /// Where the hand-written grouped MXFP4 GEMM serves the matrix it takes over earlier (<see cref="NativeMoeMinItems"/>).
    /// <c>SHARPLIGHT_MOE_SKINNY=n</c> sets the items per expert, 0 disables the skinny form. RTX PRO 4500, ms per decode step:
    /// gpt-oss-20b (MXFP4, direct, 32 experts) at 8 / 16 / 32 / 64 sequences 11.05 / 11.78 / 14.16 / 18.85 → 9.53 / 10.07 / 12.91 /
    /// 18.62, uniform routing at 96 sequences (12 per expert) 731 → 615 us per expert block, at 128 level; Qwen3-30B-A3B (Q4_K,
    /// staged, 128 experts) at 8 / 16 sequences 13.45 → 12.44 / 15.20 → 14.35 but 32 sequences (2 per expert) 18.98 → 19.87.
    /// </summary>
    private long MoeSkinnyItems(TensorRef w, long k)
    {
        bool served = MoeTensorCoreGemm(w, k) && DiagnosticSwitches.MoeGemm2 && !SmallFragments && DiagnosticSwitches.MoeSortedHalfActivations
                      && DiagnosticSwitches.SkinnyGemm && k % 32 == 0;
        long perExpert = DiagnosticSwitches.MoeSkinnyItemsPerExpert ?? (UseDirectSkinny(w.Type, half: true, 1, k) ? 12 : 1);
        return served ? perExpert * w.Shape[0] : 0;
    }

    /// <summary>
    /// Wide expert tiles of the direct skinny GEMM for <paramref name="w"/> (several 16-row token fragments per expert tile, each
    /// decoded weight serving all of them), or null when they do not serve the matrix.
    /// </summary>
    private (int Fm, int Fn, int Subgroups, int KSplit, int ItemsPerExpert)? MoeWideTiles(TensorRef w, long k)
    {
        if (!UseDirectSkinny(w.Type, half: true, 1, k) || k % 64 != 0)
        {
            return null;
        }

        if (DiagnosticSwitches.MoeWideConfig is { } c)
        {
            return c.Length == 5 && c[0] > 0 ? (c[0], c[1], c[2], c[3], c[4]) : null;
        }

        return null;
    }

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
    // launch condition, or a routing emitted for every batch (gpt-oss's gate / up, whose down product is lowered under the
    // condition its int-dot expert sum leaves: a second routing per block before). Tiling: the tile heights and the item bounds that
    // choose them.
    private BufferDecl? _moeScratch;
    private (Value Ids, long E, long MaxTiles, bool Coop, long MatvecMaxItems, object? Condition, bool Native, string Tiling)? _moeRouted;

    // Expert-sorted f16 copy of the activations of the grouped GEMMs, shared by the MatMulIds that read the same x with the same
    // routing (gate and up of an expert block).
    private BufferDecl? _moeHalfActivations;
    private (Value X, Value Ids, long MaxTiles, long MatvecMaxItems, object? Condition)? _moeGathered;

    // Routings emitted so far (a routing's sorted positions hold until the next one), the expert-sorted f16 rows a dual expert GEMM
    // stores for the next expert product, and its store (see MoeSortedStore).
    private int _moeRouteSerial;
    private BufferDecl? _moeSortedRows;
    private MoeSortedStore? _moeSortedStore;

    /// <summary>
    /// Emits (once per activations and routing) the gather of the routed rows of x into expert-sorted f16 rows and returns them
    /// ([items, K], row o = sorted position o of the routing). The rows are row-ranged (common.slang, f16_range_scale) unless
    /// <c>SHARPLIGHT_F16_RANGE=0</c>: x need not be bounded (the down projection reads a gated FFN's product), so a row reaching
    /// 2^15 is stored scaled, with its scale word at byte <paramref name="rangeScales"/> + 4 o (null for plain rows), which the
    /// grouped gemm_cm2 undoes in its store. Rows below 2^15 are stored exactly as by the plain gather.
    /// </summary>
    private TensorRef EmitSortedHalfActivations(Node n, TensorRef x, long xStride, bool perSlot, long S, long maxItems, long maxTiles, long K, int[] tileSizes,
        long matvecMaxItems, Func<Bindings, long> items, Func<Bindings, int> tile, Func<Bindings, uint> tiles, TensorRef scratch,
        PointerArg scratchArg, TensorRef ids, out long? rangeScales, MoeSortedStore? stored = null, Func<Bindings, bool>? skip = null)
    {
        bool ranged = DiagnosticSwitches.F16RangeGuard;
        long rowBytes = maxItems * K * 2;
        rangeScales = ranged ? rowBytes : null;
        TensorRef half;
        if (stored is not null)
        {
            // The rows the dual expert GEMM before stored (its sorted store); the gather fills them for the other batches.
            half = stored.Rows;
        }
        else
        {
            _moeHalfActivations ??= NewBuffer(BufferKind.Activations, "moe-x16", 0);
            _moeHalfActivations.Bytes = Math.Max(_moeHalfActivations.Bytes, rowBytes + (ranged ? maxItems * 4 : 0));
            half = new TensorRef(_moeHalfActivations, 0, DataType.F16, [maxItems, K], [K, 1], null);
            if (_moeGathered is { } g && g.X == n.Inputs[0] && g.Ids == n.Inputs[2] && g.MaxTiles == maxTiles && g.MatvecMaxItems == matvecMaxItems
                && ReferenceEquals(g.Condition, _launchCondition))
            {
                return half;
            }
        }

        Func<Bindings, bool> gathers = stored is null ? skip is null ? _ => true : b => !skip(b) : b => !stored.Serves(b);
        Emit($"moe-gather {n.Inputs[0].Name} x→{(ranged ? "ranged " : "")}f16", [n],
            [.. tileSizes.Select(t => new KernelVariant(b => items(b) > matvecMaxItems && tile(b) == t && gathers(b),
                Lib("moe.slang", "moe_gather_x16", ranged
                    ? Macros(("K", K), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xStride), ("MAXTILES", maxTiles), ("TILE_M", t), ("Y_RS", rowBytes + "ull"))
                    : Macros(("K", K), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xStride), ("MAXTILES", maxTiles), ("TILE_M", t))),
                ranged ? b => new Grid((uint)t, tiles(b)) : b => new Grid(Grid.Div(t * K / 8, 256), tiles(b))))],
            [Pointer(x), scratchArg, Pointer(half), TokenArg(ids)],
            [x, scratch], [half]);
        if (stored is null)
        {
            _moeGathered = (n.Inputs[0], n.Inputs[2], maxTiles, matvecMaxItems, _launchCondition);
        }

        return half;
    }

    /// <summary>
    /// The f16 copy of <paramref name="x"/> (rows of <paramref name="k"/> values, stride <paramref name="xStride"/>) another GEMM already
    /// reads, written for the batches of MinRows up to at least <paramref name="maxRows"/> rows where its launch condition holds
    /// (Written), with the byte offset of its scale words when it is row-ranged; null when there is none.
    /// </summary>
    private (TensorRef Copy, long MinRows, Func<Bindings, bool>? Written, long? RangeScales)? SharedHalfCopy(Value source, TensorRef x, long xStride,
        long k, long maxRows)
    {
        var root = RootOf(source);
        (TensorRef Copy, long MinRows, Func<Bindings, bool>? Written, long? RangeScales)? best = null;
        foreach (var (key, copy) in _quantizedActivations)
        {
            if (key.Root != root || key.ByteOffset != x.ByteOffset || key.XStride != xStride || key.K != k || key.Code != ActivationCode.F16
                || key.MaxRows < maxRows || (key.Condition is not null and not Func<Bindings, bool>) || copy.Strides[^1] != 1
                || copy.Shape[^1] != 2 * k || (best is { } b && b.MinRows <= key.MinRows))
            {
                continue;
            }

            best = (copy, key.MinRows, key.Condition as Func<Bindings, bool>, _rangeScales.TryGetValue(copy, out long scales) ? scales : null);
        }

        return best;
    }

    /// <summary>True when <paramref name="reader"/> is the only node of the segment reading <paramref name="value"/>, and it is not an output.</summary>
    private bool SoleReader(Value value, Node reader)
        => !_segmentOutputs.Contains(value) && _seg.Nodes.All(node => node == reader || !node.Inputs.Contains(value));

    /// <summary>
    /// Items up to which the expert matvec (one item per token and slot, each computed alone) serves a step: at least the items of the
    /// row-invariant rows (<see cref="InvariantRows"/> tokens of <paramref name="slots"/> experts each), so a speculative verification
    /// step computes its rows exactly as single-row steps do instead of switching to the grouped GEMM.
    /// </summary>
    private long InvariantMatvecItems(long items, long slots) => Math.Max(items, InvariantRows * slots);

    /// <summary>
    /// Items up to which the per-item int-dot expert matvec serves a step of <paramref name="w"/>:
    /// <see cref="DiagnosticSwitches.MoeIntDotMaxItems"/> when set, else a quarter of the expert count (8-128) when the expert-grouped
    /// skinny GEMM takes the next steps (<see cref="MoeSkinnyItems"/> reaches past it), half (16-128) when gemm_cm2 does. The per-item matvec reads
    /// an expert's weights once per item routed to it, the grouped GEMMs once per expert tile, so the gemm_cm2 wins from about one
    /// item per two experts (RTX PRO 4500: Qwen3-30B-A3B, 128 experts, 16 sequences 21.0 → 15.5 ms per step; gpt-oss-20b, 32
    /// experts, 16 sequences 20.6 → 10.7 ms) and the skinny GEMM from one per four (gpt-oss-20b at 3 / 4 sequences 8.04 / 9.19 →
    /// 7.48 / 7.67 ms; Qwen3-30B-A3B at 8 sequences 13.71 → 12.45 ms); without tensor cores the scalar GEMM only wins much later.
    /// </summary>
    private long IntDotPerItemMaxItems(TensorRef w, long k, long slots)
    {
        long experts = w.Shape[0];
        long quarter = Math.Clamp(experts / 4, 8, 128);
        return InvariantMatvecItems(DiagnosticSwitches.MoeIntDotMaxItems > 0 ? DiagnosticSwitches.MoeIntDotMaxItems
            : MoeSkinnyItems(w, k) > quarter ? quarter
            : MoeTensorCoreGemm(w, k) ? Math.Clamp(experts / 2, 16, 128) : 128, slots);
    }

    /// <summary>Whether the expert products of <paramref name="w"/> (inner size <paramref name="k"/>) run the tensor-core grouped GEMM.</summary>
    private bool MoeTensorCoreGemm(TensorRef w, long k)
        => !DiagnosticSwitches.MoeNoCoop && (_opt.Has(DeviceFeatures.MatrixF16) || SmallFragments) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32
           && k % 32 == 0 && (w.Type != DataType.F32 || HalfActivations);

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
        if (UseIntDotExpertMatvec(a, w.Type, K))
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
            string source = DualExpertEpilogueSource(a, bNode, gathers, group, output, N)!;
            long mv = IntDotPerItemMaxItems(w, K, S);
            var tids = ids;
            EmitIntDotExpertMatvec($"matmul-id2 {output.Name} [{w.Shape[0]}x{K}→{N}]x2 {w.Type} fused[{string.Join(",", group.Select(g => g.Op.Name))}]",
                [.. gathers, a, bNode, .. group], a, x, xs, rowsPerLead, [w, Ref(bNode.Inputs[1])], ids, y, m, mv, S,
                [.. gathers.Select(g => Ref(g.Inputs[0]))],
                macros => _compiler.Compile("mmvq_moe_dual.slang", source, "moe_mmvq", _opt.Target, macros, _opt.Cache));
            _launchCondition = b => (outer?.Invoke(b) ?? true) && Rows(b, tids) * S > mv;
        }

        // Batches on the grouped gemm_cm2: one dual launch multiplies both expert matrices (their 16-row fragments alternate in
        // the B tile, each activation tile staged once) and stores the group, instead of two GEMMs writing their f32 products and
        // the group's launch reading them back.
        Func<Bindings, bool>? fused = null;
        var dualY = Produce(output);
        // NVFP4 experts on the grouped FP4 GEMM: one dual launch with the group in its store when the group reads only the pair, else
        // the two products and the group as their own launches.
        bool fp4 = UseNativeFp4Moe(a, w.Type, N, K, Ref(a.Inputs[0]).Shape[^1]);
        string? fp4Epilogue = fp4 && gathers.Count == 0 && dualY.IsDense && Ref(bNode.Inputs[1]).Type == w.Type && UseIntDotExpertMatvec(a, w.Type, K)
            ? DualEpilogueCudaSource(a, bNode, group, output)
            : null;
        if (fp4Epilogue is not null)
        {
            fused = LowerMatMulId(a, (Ref(bNode.Inputs[1]), dualY, [], "", [a, bNode, .. group], false, fp4Epilogue));
            if (fused is not null)
            {
                // The integer-dot matvec above serves the other batches, so the partner and the group have nothing left to run.
                _launchCondition = outer;
                return;
            }
        }

        if (DiagnosticSwitches.MoeDual != 0 && dualY.IsDense && !fp4)
        {
            string source = DualExpertEpilogueSource(a, bNode, gathers, group, output, N, "matmul.slang")!;
            fused = LowerMatMulId(a, (Ref(bNode.Inputs[1]), dualY, [.. gathers.Select(g => Ref(g.Inputs[0]))], source, [.. gathers, a, bNode, .. group], false,
                DualExpertEpilogueSource(a, bNode, gathers, group, output, N, cuda: true)));
        }
        else
        {
            LowerMatMulId(a);
        }

        LowerMatMulId(bNode, skipGrouped: fused is not null);
        var beforeGroup = _launchCondition;
        if (fused is not null)
        {
            _launchCondition = b => (beforeGroup?.Invoke(b) ?? true) && !fused(b);
        }

        LowerElementwise(group, [output]);
        _launchCondition = outer;
    }

    /// <summary>
    /// An expert product whose only reader adds the routed experts' bias rows (<see cref="LoweringUnit.BiasAdd"/>, the folded
    /// gather <paramref name="gather"/>): the grouped GEMMs (gemm_cm2 and the expert-grouped skinny GEMM) add the bias row of
    /// each tile's expert in their store and write the sum; batches on the per-item matvecs run the product and the Add.
    /// </summary>
    private void LowerExpertBias(Node product, Node add, Node gather)
    {
        var outer = _launchCondition;
        var y = Produce(add.Output);
        var served = y.IsDense && DiagnosticSwitches.MoeBiasEpilogue
            ? LowerMatMulId(product, single: (y, [Ref(gather.Inputs[0])], null, [gather, product, add], BiasEpilogueCudaSource(Ref(product.Inputs[1]).Shape[1])))
            : LowerMatMulId(product);
        if (served is not null)
        {
            _launchCondition = b => (outer?.Invoke(b) ?? true) && !served(b);
        }

        LowerElementwise([add], [add.Output]);
        _launchCondition = outer;
    }

    /// <summary>
    /// An expert product with a fused gate / up matrix (<see cref="LoweringUnit.Halves"/>): its output, after the optional bias Add
    /// <paramref name="add"/> of the routed experts' rows (<paramref name="gather"/>), is read only as two halves by one elementwise
    /// group. The grouped GEMMs compute the halves as a dual pair (w2 = the upper half's rows of each expert) and store the group
    /// (bias rows added first), so the 2N-wide product is never stored; batches on the per-item matvecs run the product, the Add
    /// and the group.
    /// </summary>
    private void LowerExpertHalves(Node product, Node? add, Node? gather, (LoweringUnit Group, Node Low, Node High) halves)
    {
        var outer = _launchCondition;
        var group = halves.Group;
        var w = Ref(product.Inputs[1]);
        long n = w.Shape[1];
        long half = n / 2;
        var gy = Produce(group.FusedOutputs![0]);
        Func<Bindings, bool>? served = null;
        if (DiagnosticSwitches.MoeDual != 0 && gy.IsDense && (add is null) == (gather is null))
        {
            string source = HalvesEpilogueSource(product, add, gather, halves, n)!;
            var w2 = w with { ByteOffset = w.ByteOffset + half * w.Type.ByteSize(w.Shape[2]) };
            Node[] nodes = [.. gather is null ? Array.Empty<Node>() : [gather], product, .. add is null ? Array.Empty<Node>() : [add], halves.Low, halves.High,
                .. group.Fused!];
            served = LowerMatMulId(product, (w2, gy, gather is null ? [] : [Ref(gather.Inputs[0])], source, nodes, true,
                HalvesEpilogueSource(product, add, gather, halves, n, cuda: true)));
        }
        else if (add is not null && gather is not null)
        {
            LowerExpertBias(product, add, gather);
            return;
        }
        else
        {
            LowerMatMulId(product);
        }

        if (served is { } s)
        {
            _halvesServed[group] = s;
            _launchCondition = b => (outer?.Invoke(b) ?? true) && !s(b);
        }

        if (add is not null)
        {
            LowerElementwise([add], [add.Output]);
        }

        _launchCondition = outer;
    }

    /// <summary>
    /// Source of the <c>MOE_EPI</c> epilogue of the dual grouped GEMM over the halves of an expert product (n outputs): the bias
    /// Add of each half (rows e·n + c and e·n + n / 2 + c of the bias table), then the group over the two halves. With
    /// <paramref name="cuda"/>, the CUDA C++ form for <c>gemm_moe_native.cu</c> (<c>NM_EPI2</c>), or null when the group has an
    /// operation without one.
    /// </summary>
    private static string? HalvesEpilogueSource(Node product, Node? add, Node? gather, (LoweringUnit Group, Node Low, Node High) halves, long n,
        bool cuda = false)
    {
        if (cuda && !CudaEpilogueOps([.. add is null ? Array.Empty<Node>() : [add], .. halves.Group.Fused!]))
        {
            return null;
        }

        var src = new StringBuilder();
        var e = new MoeEpilogueWriter(src, cuda);
        e.Header();
        if (gather is not null)
        {
            e.Fields("float* gt0;");
        }

        e.Function("epi2", "float m0, float m1, float g0, float g1");
        var name = new Dictionary<Value, string>();
        if (add is not null)
        {
            e.Line("s0", OpExpr(add, [.. add.Inputs.Select(v => v == product.Output ? "m0" : "g0")]));
            e.Line("s1", OpExpr(add, [.. add.Inputs.Select(v => v == product.Output ? "m1" : "g1")]));
            name[halves.Low.Output] = "s0";
            name[halves.High.Output] = "s1";
        }
        else
        {
            name[halves.Low.Output] = "m0";
            name[halves.High.Output] = "m1";
        }

        foreach (var node in halves.Group.Fused!)
        {
            var local = "v" + name.Count;
            e.Line(local, OpExpr(node, [.. node.Inputs.Select(i => name[i])]));
            name[node.Output] = local;
        }

        e.Return(name[halves.Group.FusedOutputs![0]]);
        e.Dual(gather is null ? "epi2(m0, m1, 0.0, 0.0)"
            : $"epi2(m0, m1, p.gt0[{e.U64("e")} * {n} + (n)], p.gt0[{e.U64("e")} * {n} + {n / 2} + (n)])");
        e.Footer("matmul.slang");
        return src.ToString();
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
        if (UseIntDotExpertMatvec(product, w.Type, K) && product.Inputs[2].Shape[1] is { IsSymbolic: false } slotDim
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
                && !gathered.Contains(v)).Distinct()], scalars, group, summed, N)!;
            long mv = IntDotPerItemMaxItems(w, K, S);
            var tids = ids;
            EmitIntDotExpertMatvec($"matmul-id-sum {reduce.Output.Name} [{w.Shape[0]}x{K}→{N}] {w.Type} fused[{string.Join(",", group.Select(g => g.Op.Name))}]+slot-sum",
                [.. gathers, product, .. group, permute, reduce], product, x, xs, rowsPerLead, [w], ids, y, m, mv, S,
                [.. gathers.Select(g => Ref(g.Inputs[0])), .. scalars],
                macros => _compiler.Compile("mmvq_moe_sum.slang", source, "moe_mmvq", _opt.Target, macros, _opt.Cache), slotSum: true);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && Rows(b, tids) * S > mv;
        }

        // Batches on the grouped GEMMs apply the group (bias rows, router weights) in their store; the slot sum follows as usual.
        var ys = Produce(summed);
        Func<Bindings, bool>? served = null;
        if (DiagnosticSwitches.MoeBiasEpilogue && ys.IsDense)
        {
            var inGroupValues = group.Select(g => g.Output).ToHashSet();
            var gatheredValues = gathers.Select(g => g.Output).ToHashSet();
            var scalarValues = group.SelectMany(g => g.Inputs).Where(v => v != product.Output && !inGroupValues.Contains(v) && !gatheredValues.Contains(v))
                .Distinct().ToList();
            var scalarRefs = scalarValues.Select(Ref).ToList();
            if (scalarRefs.All(r => r.Rank == 3 && r.Shape[2] == 1))
            {
                string source = ExpertSumEpilogueSource(product, gathers, scalarValues, scalarRefs, group, summed, w.Shape[1], "matmul.slang")!;
                served = LowerMatMulId(product, single: (ys, [.. gathers.Select(g => Ref(g.Inputs[0])), .. scalarRefs], source, [.. gathers, product, .. group],
                    ExpertSumEpilogueSource(product, gathers, scalarValues, scalarRefs, group, summed, w.Shape[1], cuda: true)));
            }
            else
            {
                LowerMatMulId(product);
            }
        }
        else
        {
            LowerMatMulId(product);
        }

        var beforeGroup = _launchCondition;
        if (served is { } s)
        {
            _launchCondition = b => (beforeGroup?.Invoke(b) ?? true) && !s(b);
        }

        LowerElementwise(group, [summed]);
        _launchCondition = beforeGroup;
        LowerNodeOp(permute);
        LowerNodeOp(reduce);
        _launchCondition = outer;
    }

    /// <summary>
    /// Source of the <c>MOE_EPI1</c> epilogue of the slot-sum expert matvec: the group over the product, the gathered rows and the
    /// per-slot scalars (element (t, s) at t * stride0 + s * stride1). With <paramref name="cuda"/>, the CUDA C++ form for
    /// <c>gemm_moe_native.cu</c> (<c>NM_EPI1</c>), or null when the group has an operation without one.
    /// </summary>
    private static string? ExpertSumEpilogueSource(Node product, List<Node> gathers, List<Value> scalarValues, List<TensorRef> scalars, List<Node> group,
        Value output, long n, string library = "mmvq.slang", bool cuda = false)
    {
        if (cuda && !CudaEpilogueOps(group))
        {
            return null;
        }

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
        var e = new MoeEpilogueWriter(src, cuda);
        e.Header();
        e.Fields($"{string.Concat(gathers.Select((_, i) => $"float* gt{i}; "))}{string.Concat(scalars.Select((_, i) => $"float* sc{i}; "))}");
        e.Function("epi1", $"float m0{string.Concat(gathers.Select((_, i) => $", float g{i}"))}{string.Concat(scalars.Select((_, i) => $", float c{i}"))}");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            e.Line(local, OpExpr(node, [.. node.Inputs.Select(i => name[i])]));
            name[node.Output] = local;
        }

        e.Return(name[output]);
        string rows = string.Concat(gathers.Select((_, i) => $", p.gt{i}[{e.U64("e")} * {n} + (n)]"));
        string perSlot = string.Concat(scalars.Select((r, i) =>
            $", p.sc{i}[{e.U64("t")} * {r.Strides[0]} + {e.U64("s")} * {r.Strides[1]}]"));
        e.Single($"epi1(m0{rows}{perSlot})");
        e.Footer(library);
        return src.ToString();
    }

    /// <summary>
    /// Source of the <c>MOE_EPI</c> epilogue of the dual expert matvec (<paramref name="library"/> mmvq.slang) or of the dual grouped
    /// gemm_cm2 (matmul.slang): the group over both products and the gathered rows. With <paramref name="cuda"/>, the CUDA C++ form
    /// for <c>gemm_moe_native.cu</c> (<c>NM_EPI2</c>), or null when the group has an operation without one.
    /// </summary>
    private static string? DualExpertEpilogueSource(Node a, Node bNode, List<Node> gathers, List<Node> group, Value output, long n, string library = "mmvq.slang",
        bool cuda = false)
    {
        if (cuda && !CudaEpilogueOps(group))
        {
            return null;
        }

        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        for (int i = 0; i < gathers.Count; i++)
        {
            name[gathers[i].Output] = "g" + i;
        }

        var src = new StringBuilder();
        var e = new MoeEpilogueWriter(src, cuda);
        e.Header();
        e.Fields(string.Concat(gathers.Select((_, i) => $"float* gt{i}; ")));
        e.Function("epi2", $"float m0, float m1{string.Concat(gathers.Select((_, i) => $", float g{i}"))}");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            e.Line(local, OpExpr(node, [.. node.Inputs.Select(i => name[i])]));
            name[node.Output] = local;
        }

        e.Return(name[output]);
        string rows = string.Concat(gathers.Select((_, i) => $", p.gt{i}[{e.U64("e")} * {n} + (n)]"));
        e.Dual($"epi2(m0, m1{rows})");
        e.Footer(library);
        return src.ToString();
    }

    /// <summary>
    /// The CUDA C++ epilogue of <c>gemm_moe_native.cu</c> that adds the bias row of each tile's expert (the table [E, <paramref name="n"/>]
    /// after y), as <c>MOE_BIAS</c> of matmul.slang.
    /// </summary>
    private static string BiasEpilogueCudaSource(long n)
        => $"#define NM_EPI_FIELDS float* moe_bias;\n#define NM_EPI1(m0, e, n, t, s) ((m0) + p.moe_bias[(unsigned long long)(e) * {n} + (n)])\n";

    /// <summary>The elementwise operations the CUDA C++ epilogues of the hand-written GEMMs express (see <see cref="DualEpilogueCudaSource"/>).</summary>
    private static bool CudaEpilogueOps(IEnumerable<Node> group) => group.All(node => s_cudaEpilogueOps.Contains(node.Op.Name));

    private static readonly HashSet<string> s_cudaEpilogueOps = ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Relu", "Square",
        "Recip", "Add", "Sub", "Mul", "Max", "Min", "Scale", "Cast", "Clip"];

    /// <summary>
    /// Lowers the expert product <paramref name="n"/>. With <paramref name="dual"/>, the grouped gemm_cm2 launches compute it together
    /// with a second expert matrix of the same shape (same activations and routing) and store the dual epilogue <c>MOE_EPI</c> of
    /// the generated source into the group's output instead of the product. <paramref name="skipGrouped"/> leaves the grouped
    /// gemm_cm2 launches out (a dual launch of the partner serves those batches). With <paramref name="single"/>, the grouped
    /// launches apply an epilogue in their store and write <c>Y</c> instead of the product: without a source, the bias row of each
    /// tile's expert (table [E, N]); with one, its <c>MOE_EPI1</c> over the tables.
    /// Returns the batches the dual or biased grouped launches serve (null when there are none).
    /// </summary>
    private Func<Bindings, bool>? LowerMatMulId(Node n, (TensorRef W2, TensorRef Y, TensorRef[] Tables, string Source, Node[] Nodes, bool Halves, string? Cuda)? dual = null,
        bool skipGrouped = false, (TensorRef Y, TensorRef[] Tables, string? Source, Node[] Nodes, string? Cuda)? single = null)
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
        bool intDot = UseIntDotExpertMatvec(n, w.Type, K);
        long mv = intDot ? IntDotPerItemMaxItems(w, K, S) : InvariantMatvecItems(coop ? DiagnosticSwitches.MoeMatvecMaxItems : Math.Max(DiagnosticSwitches.MoeMatvecMaxItems, DiagnosticSwitches.MoeScalarRows * E), S);
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
        // The hand-written grouped MXFP4 GEMM (CUDA, many rows per expert) takes over from the skinny GEMM at nativeMinItems.
        bool nativeMoe = gemm2 && UseNativeMoeGemm(w, K, N);
        long nativeMinItems = NativeMoeMinItems(E);
        long skinnyItems = nativeMoe ? Math.Min(MoeSkinnyItems(w, K), nativeMinItems - 1) : MoeSkinnyItems(w, K);
        bool skinnyMoe = gemm2 && skinnyItems > 0;
        var wide = skinnyMoe ? MoeWideTiles(w, K) : null;
        long wideItems = wide is { } wc ? wc.ItemsPerExpert * E : 0;
        int wideTile = wide is { } wt ? 16 * wt.Fm : 0;
        Func<Bindings, bool> isWide = b => wide is not null && items(b) > skinnyItems && items(b) <= wideItems;
        // Many rows per expert on the hand-written grouped MXFP4 GEMM (CUDA): its batches route NativeMoeTileRows-row tiles (which the
        // gemm_cm2 tiles include, so a product whose epilogue has no CUDA form runs gemm_cm2 on the same routing).
        bool nativeOn = false;
        Func<Bindings, bool> nativeRows = b => nativeMoe && items(b) > mv && items(b) >= nativeMinItems && !isWide(b) && (!skinnyMoe || items(b) > skinnyItems);
        Func<Bindings, bool> isNative = b => nativeOn && nativeRows(b);
        // NVFP4 experts on the block-scaled FP4 MMA (CUDA 12.x): every batch above the matvec range on the grouped FP4 GEMM.
        // A dual pair takes it with the group in its store when the group reads no per-expert tables (its CUDA form as for dense pairs).
        var fp4Dual = dual is { Tables.Length: 0, Halves: false, Cuda: not null, Y.IsDense: true } dd4 && dd4.W2.Type == w.Type ? dd4 : default;
        bool fp4Moe = (dual is null || fp4Dual.Cuda is not null) && single is null && !skipGrouped && y.IsDense && UseNativeFp4Moe(n, w.Type, N, K, xs);
        int[] routeTiles = skinnyMoe ? [.. new[] { MoeSkinnyTile, wideTile }.Where(t => t > 0), .. coopTiles] : coopTiles;
        routeTiles = fp4Moe ? [.. routeTiles, .. NativeFp4MoeTiles] : routeTiles;
        routeTiles = [.. routeTiles.Distinct()];
        Func<Bindings, int> tile = b => fp4Moe ? NativeFp4MoeTileRows(items(b), E)
            : !coop ? 64
            : SmallFragments ? 32
            : DiagnosticSwitches.MoeTile != 0 ? DiagnosticSwitches.MoeTile
            : skinnyMoe && items(b) <= skinnyItems ? MoeSkinnyTile
            : isWide(b) ? wideTile
            : nativeRows(b) ? NativeMoeTileRowsFor(items(b), E)
            : items(b) <= 24 * E ? 32
            : !gemm2 || items(b) <= MoeTallTileMinItems * E ? 64 : 128;
        Func<Bindings, uint> tiles = b => (uint)Math.Max(1, Math.Min(items(b), items(b) / tile(b) + E));
        var scratchArg = new PointerArg(_moeScratch, 0);

        string routeKey = string.Create(CultureInfo.InvariantCulture, $"{string.Join(",", routeTiles)} {skinnyItems} {wideItems} {wideTile} {nativeMinItems} {gemm2}");
        if (_moeRouted is not { } r || r.Ids != n.Inputs[2] || r.E != E || r.MaxTiles != maxTiles || r.Coop != coop || r.MatvecMaxItems != mv
            || !(r.Condition is null || ReferenceEquals(r.Condition, _launchCondition)) || r.Native != nativeMoe || r.Tiling != routeKey)
        {
            int[] tiling = coop ? routeTiles : [64];
            Emit($"moe-route {n.Inputs[2].Name} [{E} experts]", [n],
                [.. tiling.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_route", new Dictionary<string, string>(Macros(("E", E), ("S", S), ("TILE_M", t), ("MAXTILES", maxTiles)))),
                    _ => new Grid(1)))],
                [Pointer(ids), scratchArg, TokenArg(ids)],
                [ids], [scratch]);
            _moeRouted = (n.Inputs[2], E, maxTiles, coop, mv, _launchCondition, nativeMoe, routeKey);
            _moeRouteSerial++;
        }

        if (fp4Moe)
        {
            EmitNativeFp4Moe(label, fp4Dual.Nodes ?? [n], n, x, xs, rowsPerLead, perSlot, w, ids, fp4Dual.Cuda is null ? y : fp4Dual.Y, maxTiles, mv, items, tile,
                tiles, scratch, scratchArg, fp4Dual.Cuda is null ? null : (fp4Dual.W2, fp4Dual.Cuda));
            return fp4Dual.Cuda is null ? null : b => items(b) > mv;
        }

        var gm = new Dictionary<string, string>(m) { ["MAXTILES"] = maxTiles.ToString(CultureInfo.InvariantCulture) };
        var gargs = new KernelArg[] { Pointer(x), Pointer(w), scratchArg, Pointer(y), TokenArg(ids) };
        if (gemm2)
        {
            // The dense tensor-core GEMM in its expert-grouped form: one routing tile per workgroup row.
            var groupedMacros = new Dictionary<string, string>(gm) { ["MOE"] = "1", ["Y_STRIDE"] = N.ToString(CultureInfo.InvariantCulture) };
            bool cm2TileMajor = (DiagnosticSwitches.MoeTileMajor & 2) != 0;
            bool skTileMajor = (DiagnosticSwitches.MoeTileMajor & 1) != 0;
            Grid TileGrid(bool major, uint columnTiles, uint expertTiles) => major ? new Grid(expertTiles, columnTiles) : new Grid(columnTiles, expertTiles);
            var gemmX = x;
            TensorRef? tokenCopy = null;
            long? tokenScales = null;
            Func<Bindings, bool> tokenRead = _ => false;
            if (DiagnosticSwitches.MoeSortedHalfActivations)
            {
                // The expert-sorted f16 rows a dual expert GEMM stored for this product on the same routing (its only reader).
                var stored = _moeSortedStore is { } st && st.Output == n.Inputs[0] && st.RouteSerial == _moeRouteSerial && st.Width == K
                             && st.MaxItems == maxItems && perSlot && x.Buffer == st.Spill.Buffer && x.ByteOffset == st.Spill.ByteOffset
                             && st.RangeScales.HasValue == DiagnosticSwitches.F16RangeGuard && SoleReader(n.Inputs[0], n)
                    ? st
                    : null;
                if (stored is not null)
                {
                    var readCondition = _launchCondition;
                    stored.Consumer = b => (readCondition?.Invoke(b) ?? true) && items(b) > mv;
                }

                _moeSortedStore = null;
                // The f16 copy of x per token that another GEMM reads (the router's): the hand-written grouped GEMM reads its items'
                // token rows from it, and the gather skips those batches.
                if (stored is null && !perSlot && nativeMoe && DiagnosticSwitches.MoeTokenRows && SharedHalfCopy(n.Inputs[0], x, xs, K, ids.Shape[0]) is { } copy)
                {
                    tokenCopy = copy.Copy;
                    tokenScales = copy.RangeScales;
                    long copyMin = copy.MinRows;
                    var written = copy.Written;
                    tokenRead = b => items(b) / S >= copyMin && (written?.Invoke(b) ?? true);
                }

                var readsTokens = tokenRead;
                gemmX = EmitSortedHalfActivations(n, x, xs, perSlot, S, maxItems, maxTiles, K, routeTiles, mv, items, tile, tiles, scratch, scratchArg, ids,
                    out long? rangeScales, stored, tokenCopy is null ? null : b => isNative(b) && readsTokens(b));
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
            Dictionary<string, string> Rows(int t, string limit, int rows)
                => new(Gemm2Macros(groupedMacros, t)) { [limit] = rows.ToString(CultureInfo.InvariantCulture) };
            // The dual form (see the dual parameter) alternates 16-row fragments of the two matrices in the B tile: each tile layout
            // needs an even fragment count along N, and the expert-sorted f16 rows.
            // (skipGrouped: this is the partner of such a pair, whose launches its dual partner made.)
            // Halves: the pair is the two halves of n's own output rows (a fused gate / up matrix): every grouped launch is dual, with
            // N / 2 outputs and w2 at the upper half's first row.
            bool halves = dual is { Halves: true };
            bool dualPair = (dual is not null || skipGrouped) && DiagnosticSwitches.MoeSortedHalfActivations
                            && coopTiles.All(t => int.Parse(Gemm2Macros(groupedMacros, t)["FRAG_N"], CultureInfo.InvariantCulture) % 2 == 0)
                            && (!halves || N % 32 == 0);
            var d = dualPair && dual is { } dd ? dd : default;
            bool dualCm2 = dualPair && (halves || (DiagnosticSwitches.MoeDual & 1) != 0);
            bool dualSkinny = dualPair && (halves || (DiagnosticSwitches.MoeDual & 2) != 0) && skinnyMoe;
            long dualN = halves ? N / 2 : N;
            Dictionary<string, string> DualMacros(Dictionary<string, string> macros, string kernel)
            {
                var r = new Dictionary<string, string>(macros) { ["DUAL"] = "1", ["KERNEL_" + kernel] = "1" };
                if (halves)
                {
                    r["N"] = dualN.ToString(CultureInfo.InvariantCulture);
                    r["Y_STRIDE"] = dualN.ToString(CultureInfo.InvariantCulture);
                }

                return r;
            }

            bool isDual = dualCm2 && d.Source is not null;
            // Batches the dual launches serve (the partner's own launches and the group are skipped for them).
            Func<Bindings, bool> grouped = b => items(b) > mv && (((dualCm2 || single is not null) && coopTiles.Contains(tile(b)) && !isWide(b))
                                                                  || ((dualSkinny || single is not null) && skinnyMoe && tile(b) == MoeSkinnyTile));
            KernelArg[] dualArgs = d.Source is null ? gargs
                : [Pointer(gemmX), Pointer(w), Pointer(d.W2), scratchArg, Pointer(d.Y), .. d.Tables.Select(Pointer), TokenArg(ids)];
            TensorRef[] dualReads = d.Source is null ? [gemmX, w, scratch] : [gemmX, w, d.W2, scratch, .. d.Tables];
            // The single-product epilogue: the bias rows (no source: MOE_BIAS, the table follows y) or a generated MOE_EPI1.
            var bb = !isDual && single is { } bv ? bv : default;
            bool biased = bb.Y is not null;
            Dictionary<string, string> SingleMacros(Dictionary<string, string> macros, string kernel)
                => bb.Source is null ? new(macros) { ["MOE_BIAS"] = "1" } : new(macros) { ["KERNEL_" + kernel] = "1" };
            CompiledKernel Single(Dictionary<string, string> macros, string kernel)
                => bb.Source is null ? Lib("matmul.slang", kernel.ToLowerInvariant(), SingleMacros(macros, kernel))
                    : _compiler.Compile("matmul_moe_epi.slang", bb.Source, kernel.ToLowerInvariant(), _opt.Target, SingleMacros(macros, kernel), _opt.Cache);
            CompiledKernel Grouped(Dictionary<string, string> macros)
            {
                if (cm2TileMajor)
                {
                    macros = new Dictionary<string, string>(macros) { ["MOE_TILE_MAJOR"] = "1" };
                }

                if (biased)
                {
                    return Single(macros, "GEMM_CM2");
                }

                return isDual ? _compiler.Compile("matmul_moe_dual.slang", d.Source!, "gemm_cm2", _opt.Target, DualMacros(macros, "GEMM_CM2"), _opt.Cache)
                    : Lib("matmul.slang", "gemm_cm2", macros);
            }

            int Columns(int t) => isDual ? Gemm2Columns(t) / 2 : Gemm2Columns(t);
            long gridN = isDual ? dualN : N;
            string groupedLabel = isDual ? $"{label}x2 fused gemm-cm2" : label + " gemm-cm2";
            Node[] groupedNodes = isDual ? d.Nodes : [n];
            // The bias epilogue (single products only): the bias table follows y.
            KernelArg[] biasArgs = biased ? [Pointer(gemmX), Pointer(w), scratchArg, Pointer(bb.Y), .. bb.Tables.Select(Pointer), TokenArg(ids)] : gargs;
            TensorRef[] biasReads = biased ? [gemmX, w, scratch, .. bb.Tables] : [gemmX, w, scratch];
            KernelArg[] groupedArgs = isDual ? dualArgs : biased ? biasArgs : gargs;
            TensorRef[] groupedReads = isDual ? dualReads : biased ? biasReads : [gemmX, w, scratch];
            TensorRef groupedOut = isDual ? d.Y : biased ? bb.Y : y;
            if (biased)
            {
                groupedLabel += bb.Source is null ? " +bias" : " +epilogue";
                groupedNodes = bb.Nodes;
            }
            if (!skipGrouped || !dualCm2)
            {
                _writeGroup = split ? new object() : null;
                Emit(groupedLabel, groupedNodes,
                    [.. coopTiles.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t && !isWide(b) && !isNative(b),
                        Grouped(split && t > 32 ? Rows(t, "MOE_MIN_ROWS", t / 2 + 1) : Gemm2Macros(groupedMacros, t)),
                        b => TileGrid(cm2TileMajor, Grid.Div(gridN, Columns(t)), tiles(b))))],
                    groupedArgs, groupedReads, [groupedOut]);
                if (split)
                {
                    Emit(groupedLabel + " half tiles", groupedNodes,
                        [.. coopTiles.Where(t => t > 32).Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t && !isWide(b) && !isNative(b),
                            Grouped(Rows(t / 2, "MOE_MAX_ROWS", t / 2)),
                            b => TileGrid(cm2TileMajor, Grid.Div(gridN, Columns(t / 2)), tiles(b))))],
                        groupedArgs, groupedReads, [groupedOut]);
                }

                _writeGroup = null;
                // The hand-written grouped MXFP4 GEMM with the same epilogue (its CUDA form), when there is one.
                string? nativeEpilogue = isDual ? d.Cuda : biased ? bb.Cuda : "";
                if (nativeMoe && nativeEpilogue is not null)
                {
                    nativeOn = true;
                    long nativeN = isDual ? dualN : N;
                    // The dual product's sorted f16 store for the down projection (switched on by it, see MoeSortedStore).
                    MoeSortedStore? store = null;
                    if (isDual && DiagnosticSwitches.MoeSortedStore && DiagnosticSwitches.MoeSortedHalfActivations && d.Y.IsDense
                        && dualN % 2 == 0 && d.Nodes.Length > 0 && d.Nodes[^1].Output is { } storedOutput && _refs.TryGetValue(storedOutput, out var spill)
                        && spill.Buffer == d.Y.Buffer && spill.ByteOffset == d.Y.ByteOffset)
                    {
                        bool rangedRows = DiagnosticSwitches.F16RangeGuard;
                        long sortedBytes = maxItems * dualN * 2;
                        _moeSortedRows ??= NewBuffer(BufferKind.Activations, "moe-y16", 0);
                        _moeSortedRows.Bytes = Math.Max(_moeSortedRows.Bytes, sortedBytes + (rangedRows ? maxItems * 4 : 0));
                        var nativeServes = isNative;
                        store = new MoeSortedStore
                        {
                            Output = storedOutput,
                            Spill = d.Y,
                            Rows = new TensorRef(_moeSortedRows, 0, DataType.F16, [maxItems, dualN], [dualN, 1], null),
                            RangeScales = rangedRows ? sortedBytes : null,
                            RouteSerial = _moeRouteSerial,
                            Width = dualN,
                            MaxItems = maxItems,
                            Native = nativeServes,
                        };
                    }

                    foreach (int nativeTile in NativeMoeTiles)
                    {
                        var (bn, stages, minBlocks, table) = NativeMoeConfig(nativeN, isDual, nativeTile);
                        var nativeMacros = new Dictionary<string, string>
                        {
                            ["K"] = K.ToString(CultureInfo.InvariantCulture),
                            ["N"] = nativeN.ToString(CultureInfo.InvariantCulture),
                            ["EXPERT_BYTES"] = gm["EXPERT_BYTES"],
                            ["ROW_BYTES"] = gm["ROW_BYTES"],
                            ["S"] = S.ToString(CultureInfo.InvariantCulture),
                            ["MAXTILES"] = gm["MAXTILES"],
                            ["NM_BM"] = nativeTile.ToString(CultureInfo.InvariantCulture),
                            ["NM_BN"] = bn.ToString(CultureInfo.InvariantCulture),
                            ["NM_WM"] = "1",
                            ["NM_WN"] = NativeMoeWarps.ToString(CultureInfo.InvariantCulture),
                            ["NM_STAGES"] = stages.ToString(CultureInfo.InvariantCulture),
                            ["NM_MINB"] = minBlocks.ToString(CultureInfo.InvariantCulture),
                            ["NM_TABLE"] = table ? "1" : "0",
                            ["DUAL"] = isDual ? "1" : "0",
                        };
                        if (groupedMacros.TryGetValue("X_RS", out var xrs))
                        {
                            nativeMacros["X_RS"] = xrs;
                        }

                        long columns = isDual ? bn / 2 : bn;
                        int t0 = nativeTile;
                        // One launch per input form (expert-sorted rows, or token rows of the shared copy) and store (f32 rows per item,
                        // or the sorted f16 store), each for the batches it serves.
                        foreach (bool token in tokenCopy is null ? [false] : new[] { false, true })
                        {
                            foreach (bool sorted in store is null ? [false] : new[] { false, true })
                            {
                                var macros = new Dictionary<string, string>(nativeMacros);
                                if (token)
                                {
                                    macros.Remove("X_RS");
                                    macros["NM_X_TOKEN"] = "1";
                                    if (tokenScales is { } ts)
                                    {
                                        macros["X_RS"] = ts.ToString(CultureInfo.InvariantCulture) + "ull";
                                    }
                                }

                                if (sorted && store!.RangeScales is { } words)
                                {
                                    macros["NM_Y16_RS"] = words.ToString(CultureInfo.InvariantCulture) + "ull";
                                }

                                if (sorted)
                                {
                                    macros["NM_Y16"] = "1";
                                }

                                var kernel = _compiler.CompileCuda("gemm_moe_native.cu", nativeEpilogue, "gemm_moe_native", NativeMoeWarps * 32,
                                    NativeMoeStageBytes(bn, isDual, nativeTile) * stages, _opt.Target, macros, _opt.Cache);
                                bool tokenForm = token, sortedForm = sorted;
                                var input = token ? tokenCopy! : gemmX;
                                KernelArg[] args = sorted
                                    ? [Pointer(input), Pointer(w), Pointer(d.W2), scratchArg, Pointer(d.Y), Pointer(store!.Rows), .. d.Tables.Select(Pointer), TokenArg(ids)]
                                    : [Pointer(input), .. groupedArgs.Skip(1)];
                                TensorRef[] reads = [input, .. groupedReads.Skip(1)];
                                Emit(groupedLabel.Replace(" gemm-cm2", " native", StringComparison.Ordinal) + (token ? " token rows" : "") + (sorted ? " sorted f16" : ""),
                                    groupedNodes,
                                    [new KernelVariant(b => isNative(b) && tile(b) == t0 && tokenRead(b) == tokenForm && (store?.Serves(b) ?? false) == sortedForm,
                                        kernel, b => new Grid(Grid.Div(nativeN, columns), tiles(b)))],
                                    args, reads, sorted ? [d.Y, store!.Rows] : [groupedOut]);
                            }
                        }
                    }

                    if (store is not null)
                    {
                        if (store.RangeScales is { } words)
                        {
                            Emit($"moe-range-fixup {store.Output.Name}", groupedNodes,
                                [new KernelVariant(store.Serves, Lib("moe.slang", "moe_range_fixup",
                                    Macros(("K", dualN), ("S", S), ("Y_RS", words + "ull"))), b => new Grid(Grid.Div(items(b), 256)))],
                                [Pointer(store.Rows), Pointer(d.Y), TokenArg(ids)],
                                [store.Rows, d.Y], [store.Rows]);
                        }

                        _moeSortedStore = store;
                    }
                }
            }
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
                    ["MOE_TILE_MAJOR"] = skTileMajor ? "1" : "0",
                };
                foreach (var (name, value) in DiagnosticSwitches.GemmExtraMacros)
                {
                    skinnyMacros[name] = value;
                }

                if (!skipGrouped || !dualSkinny)
                {
                    var skinnyKernel = dualSkinny && d.Source is not null
                        ? _compiler.Compile("matmul_moe_dual.slang", d.Source!, "gemm_sk", _opt.Target, DualMacros(skinnyMacros, "GEMM_SK"), _opt.Cache)
                        : Lib("matmul.slang", "gemm_sk", skinnyMacros);
                    bool fusedSkinny = dualSkinny && d.Source is not null;
                    if (biased && !fusedSkinny)
                    {
                        skinnyKernel = Single(skinnyMacros, "GEMM_SK");
                    }

                    Emit(fusedSkinny ? $"{label}x2 fused skinny" : biased ? label + (bb.Source is null ? " skinny +bias" : " skinny +epilogue") : label + " skinny",
                        fusedSkinny ? d.Nodes : biased ? bb.Nodes : [n],
                        [new KernelVariant(b => items(b) > mv && tile(b) == MoeSkinnyTile, skinnyKernel,
                            b => TileGrid(skTileMajor, Grid.Div(fusedSkinny ? dualN : N, fn * 16), tiles(b)))],
                        fusedSkinny ? dualArgs : biased ? biasArgs : gargs,
                        fusedSkinny ? dualReads : biased ? biasReads : [gemmX, w, scratch], [fusedSkinny ? d.Y : biased ? bb.Y : y]);
                }
            }

            if (wide is { } wcfg)
            {
                var wideMacros = new Dictionary<string, string>(groupedMacros)
                {
                    ["SK_FM"] = wcfg.Fm.ToString(CultureInfo.InvariantCulture),
                    ["SK_FN"] = wcfg.Fn.ToString(CultureInfo.InvariantCulture),
                    ["SK_SG"] = wcfg.Subgroups.ToString(CultureInfo.InvariantCulture),
                    ["SK_KS"] = wcfg.KSplit.ToString(CultureInfo.InvariantCulture),
                    ["SK_BK"] = "64",
                    ["SK_DIRECT"] = "1",
                };
                foreach (var (name, value) in DiagnosticSwitches.GemmExtraMacros)
                {
                    wideMacros[name] = value;
                }

                int cols = wcfg.Subgroups / wcfg.KSplit * wcfg.Fn * 16;
                Emit(label + " skinny wide", [n],
                    [new KernelVariant(b => items(b) > mv && isWide(b), Lib("matmul.slang", "gemm_sk", wideMacros),
                        b => new Grid(Grid.Div(N, cols), tiles(b)))],
                    gargs, [gemmX, w, scratch], [y]);
            }

            return (dual is not null && (dualCm2 || dualSkinny)) || (single is not null && !isDual) ? grouped : null;
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

        return null;
    }
}
