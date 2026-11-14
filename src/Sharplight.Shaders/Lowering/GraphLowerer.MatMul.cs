using System.Globalization;
using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Largest row count served by the matrix-vector kernels.</summary>
    private const long MatvecMaxRows = 8;

    // Tensor-core GEMM (gemm_cm2) tile for prefill: both large warp layouts produce 128x128 tiles. Small batches use
    // 32- or 64-token tiles (see Gemm2TileRowsFor) so the MMA work stays proportional to the rows.
    private const int Gemm2TileRows = 128;
    private const int Gemm2TileColumns = 128;

    /// <summary>Grid size (workgroups) from which the 8-warp gemm_cm2 layout beats the 4-warp one.</summary>
    private const long LargeGridWorkgroups = 384;

    private static readonly int NvTileTokens = DiagnosticSwitches.WorkgroupMatrixTile[0];
    private static readonly int NvTileColumns = DiagnosticSwitches.WorkgroupMatrixTile[1];
    private static readonly int NvTileK = DiagnosticSwitches.WorkgroupMatrixTile[2];

    /// <summary>Split-K partial-sum buffers, used in rotation (see <see cref="SplitScratch"/>).</summary>
    private readonly BufferDecl?[] _splitScratch = new BufferDecl?[3];

    private int _splitScratchNext;

    /// <summary>
    /// Use the workgroup-scope matrix GEMM (<see cref="DeviceFeatures.MatrixWorkgroup"/>) for prefill matmuls when the
    /// device has it. Off by default: on the drivers measured so far (NVIDIA Blackwell, 2026) the kernel-staged
    /// gemm_cm2 is as fast for f16 weights and ~1.5x faster for q8_0, whose decode-on-load is per element.
    /// Defaults to true only when <c>SHARPLIGHT_GEMM_NV2=1</c> is set. Process-wide; set before lowering.
    /// </summary>
    public static bool PreferWorkgroupMatrixGemm { get; set; } = DiagnosticSwitches.WorkgroupMatrixGemm;

    /// <summary>Matrix-vector variants for up to 8 rows of x (memory bound: decode, small batches).</summary>
    private static List<KernelVariant> MatvecVariants(Func<string, Dictionary<string, string>, CompiledKernel> lib, Dictionary<string, string> m,
        long N, long K, Func<Bindings, long> rows)
    {
        // Decode (one token): split-K matvec, one workgroup per output row, K spread over enough threads that
        // each walks only a few 8-weight groups (keeps many loads in flight per row).
        int skSg = N <= 2048 || K >= 12288 ? 8 : 4;
        var singleRow = new Dictionary<string, string>(m) { ["MV_COLS"] = "1", ["MV_SK_SG"] = skSg.ToString(CultureInfo.InvariantCulture) };
        var decode = K >= 1024
            ? new KernelVariant(b => rows(b) == 1, lib("matvec_sk", singleRow), b => new Grid((uint)N))
            : new KernelVariant(b => rows(b) == 1, lib("matvec", singleRow), b => new Grid(Grid.Div(N, 4)));
        // Small N: one split-K workgroup per row, otherwise one subgroup per row.
        var smallBatch = N <= 2048 && K >= 1024
            ? new KernelVariant(b => rows(b) <= MatvecMaxRows, lib("matvec_sk", m), b => new Grid((uint)N))
            : new KernelVariant(b => rows(b) <= MatvecMaxRows, lib("matvec", m), b => new Grid(Grid.Div(N, 4)));
        return [decode, smallBatch];
    }

    /// <summary>
    /// Makes x readable by the weight kernels: rows of 16-byte aligned f32 with a stride divisible by four, copying to a
    /// dense temporary otherwise. Reports the row stride and the rows per leading index.
    /// </summary>
    private TensorRef PrepareMatMulInput(TensorRef x, string why, out long rowStride, out long rowsPerLead)
    {
        if (!TryRows(x, out rowStride, out rowsPerLead) || rowStride % 4 != 0 || x.ByteOffset % 16 != 0 || x.Type != DataType.F32)
        {
            x = DenseTemp(x, why);
            TryRows(x, out rowStride, out rowsPerLead);
        }

        return x;
    }

    /// <summary>
    /// Two matmuls sharing x whose outputs only feed one elementwise group (e.g. SwiGLU gate/up + silu·mul):
    /// small batches run one dual matvec / int-dot matvec / skinny GEMM that computes both and applies the group per element,
    /// fp8 weights from 17 rows the dual fp8 GEMM; other batches run the regular launches (matmul, matmul, elementwise).
    /// </summary>
    private void LowerDualMatMul(Node a, Node bNode, List<Node> group, Value output)
    {
        var x = Ref(a.Inputs[0]);
        var weightA = Ref(a.Inputs[1]);
        var weightB = Ref(bNode.Inputs[1]);
        long N = weightA.Shape[0];
        long K = weightA.Shape[1];
        x = PrepareMatMulInput(x, "matmul", out long xStride, out long rowsPerLead);
        var y = Produce(output);
        int format = WeightFormat(weightA.Type) ?? throw new NotSupportedException($"MatMul weight type {weightA.Type}");
        var macros = Macros(("WFMT", format), ("K", K), ("N", N), ("X_STRIDE", xStride), ("Y_STRIDE", N), ("ROW_BYTES", weightA.Type.ByteSize(K) + "ull"), ("DUAL", 1));

        var tx = x;
        Func<Bindings, long> rows = b => Rows(b, tx) * rowsPerLead;
        var groupNames = string.Join(",", group.Select(g => g.Op.Name));
        var outer = _launchCondition;
        long fusedMaxRows = MatvecMaxRows;
        if (UseIntDotMatvec(a, weightA.Type, K))
        {
            fusedMaxRows = Math.Max(MatvecMaxRows, IntDotMaxRows(weightA.Type, K));
            string intDotSource = DualEpilogueSource(a, bNode, group, output, "mmvq.slang");
            EmitIntDotMatvec($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x, xStride, rowsPerLead,
                [weightA, weightB], y, N, K,
                intDotMacros => _compiler.Compile("mmvq_dual.slang", intDotSource, "mmvq", _opt.Target, intDotMacros, _opt.Cache));
            var outsideIntDot = OutsideIntDotRange(x, rowsPerLead, weightA.Type, K);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && outsideIntDot(b);
        }

        string source = DualEpilogueSource(a, bNode, group, output, "matmul.slang");
        long dualSkinnyMin = UseIntDotMatvec(a, weightA.Type, K) ? IntDotSkinnyMinRows(weightA.Type, K, N) : SkinnyMinRows;
        if (DiagnosticSwitches.SkinnyGemm && CoopUsable(a, weightA.Type, K) && SkinnyMaxRowsFor(N) >= dualSkinnyMin)
        {
            // Up to 16 rows (above the int-dot range for formats with one): both matrices on the skinny GEMM with the group as
            // its epilogue.
            var beforeSkinny = _launchCondition;
            EmitSkinnyGemm(dualSkinnyMin, $"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x, xStride, rowsPerLead,
                [weightA, weightB], y, N, K, macros,
                skinnyMacros => _compiler.Compile("matmul_dual.slang", source, "gemm_sk", _opt.Target,
                    new Dictionary<string, string>(skinnyMacros) { ["KERNEL_GEMM_SK"] = "1" }, _opt.Cache));
            _launchCondition = b => (beforeSkinny?.Invoke(b) ?? true) && !InSkinnyRange(rows(b), N, dualSkinnyMin);
            fusedMaxRows = Math.Max(fusedMaxRows, SkinnyMaxRowsFor(N));
        }

        Emit($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group],
            MatvecVariants((entry, entryMacros) => _compiler.Compile("matmul_dual.slang", source, entry, _opt.Target,
                new Dictionary<string, string>(entryMacros) { ["KERNEL_" + entry.ToUpperInvariant()] = "1" }, _opt.Cache), macros, N, K, rows),
            [Pointer(x), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [x, weightA, weightB], [y]);

        // Larger batches: fp8 weights run one dual GEMM with the group as its epilogue from 17 rows (above 64 rows only where the
        // separate GEMMs would not split K); otherwise the regular matmuls and the elementwise group, skipped while a fused
        // launch serves the rows.
        long fusedBelow = fusedMaxRows;
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > fusedBelow;
        bool dualFp8 = EmitDualFp8Gemm($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x,
            xStride, rowsPerLead, weightA, weightB, y, N, K, DualEpilogueSource(a, bNode, group, output, "gemm_fp8.slang"));
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > fusedBelow && !(dualFp8 && DualFp8Serves(rows(b), N, K));
        LowerMatMul(a);
        LowerMatMul(bNode);
        LowerElementwise(group, [output]);
        _launchCondition = outer;
    }

    /// <summary>
    /// A matmul whose output only feeds the Add of a bias row vector (<see cref="LoweringUnit.BiasAdd"/>): single activation rows
    /// on the f32 split-K matvec add the bias in its store (one launch and one dependency level less per projection, e.g. the
    /// biased q / k / v / o projections of a decode step); other row counts, and formats whose single rows take the integer-dot
    /// matvec, run the matmul and the Add as before. The fused store computes the same f32 sum as the Add kernel.
    /// </summary>
    private void LowerMatMulBias(Node matmul, Node add)
    {
        var w = Ref(matmul.Inputs[1]);
        long N = w.Shape[0];
        long K = w.Shape[1];
        var outer = _launchCondition;
        bool singleRowMatvec = K % 8 == 0 && K >= 1024 && (!UseIntDotMatvec(matmul, w.Type, K) || IntDotMinRows(w.Type) > 1);
        if (singleRowMatvec)
        {
            var x = PrepareMatMulInput(Ref(matmul.Inputs[0]), "matmul", out long xStride, out long rowsPerLead);
            var bias = Ref(add.Inputs[matmul.Output == add.Inputs[0] ? 1 : 0]);
            var y = Produce(add.Output);
            int format = WeightFormat(w.Type) ?? throw new NotSupportedException($"MatMul weight type {w.Type}");
            int skSg = N <= 2048 || K >= 12288 ? 8 : 4;
            var macros = Macros(("WFMT", format), ("K", K), ("N", N), ("X_STRIDE", xStride), ("Y_STRIDE", N), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"),
                ("MV_COLS", 1), ("MV_SK_SG", skSg), ("BIAS", 1));
            var tx = x;
            Func<Bindings, long> rows = b => Rows(b, tx) * rowsPerLead;
            Emit($"matmul {matmul.Output.Name} [{K}→{N}] {w.Type} +bias", [matmul, add],
                [new KernelVariant(b => rows(b) == 1, Lib("matmul.slang", "matvec_sk", macros), _ => new Grid((uint)N))],
                [Pointer(x), Pointer(w), Pointer(y), Pointer(bias), TokenArg(x, rowsPerLead)],
                [x, w, bias], [y]);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) != 1;
        }

        LowerMatMul(matmul);
        LowerElementwise([add], [add.Output]);
        _launchCondition = outer;
    }

    /// <summary>Source of the <c>epi2</c> epilogue that applies the fused elementwise group to both matmul results.</summary>
    private static string DualEpilogueSource(Node a, Node bNode, List<Node> group, Value output, string kernelFile)
    {
        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine("float epi2(float m0, float m1) {");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            src.AppendLine($"  float {local} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = local;
        }

        src.AppendLine($"  return {name[output]};");
        src.AppendLine("}");
        src.AppendLine($"#include \"{kernelFile}\"");
        return src.ToString();
    }

    /// <summary>
    /// Token rows of the gemm_cm2 tile for a batch of <paramref name="rows"/>: 32 or 64 for small batches (decode of many
    /// sequences), where a 128-token tile would mostly multiply padding, else 128.
    /// </summary>
    private static int Gemm2TileRowsFor(long rows) => !DiagnosticSwitches.Gemm2Skinny ? Gemm2TileRows : rows <= 32 ? 32 : rows <= 64 ? 64 : Gemm2TileRows;

    /// <summary>
    /// gemm_cm2 layout (warps M, warps N, fragments M, fragments N) of a tile with <paramref name="tileRows"/> tokens and
    /// a K block of 64 (<paramref name="wideK"/>) or 32. Every thread stages at least one 8-value activation group per K
    /// block, so the 32-token tile with a 32-value K block has only 4 warps.
    /// </summary>
    private static int[] Gemm2Layout(int tileRows, bool large, bool wideK = true) => tileRows switch
    {
        32 => DiagnosticSwitches.Gemm2Layout32 ?? (wideK ? [2, 4, 1, 2] : [1, 4, 2, 2]),
        64 => DiagnosticSwitches.Gemm2Layout64 ?? [2, 4, 2, 2],
        _ => DiagnosticSwitches.Gemm2Layout ?? (large ? [2, 4, 4, 2] : [2, 2, 4, 4]),
    };

    /// <summary>Weight rows (output columns) of the gemm_cm2 tile with <paramref name="tileRows"/> tokens.</summary>
    private static int Gemm2Columns(int tileRows)
    {
        var layout = Gemm2Layout(tileRows, large: false);
        return layout[1] * layout[3] * 16;
    }

    /// <summary>Macros of gemm_cm2 for a tile of <paramref name="tileRows"/> tokens; 128-token tiles use the 4-warp (<paramref name="large"/> false) or 8-warp layout.</summary>
    private static Dictionary<string, string> Gemm2Macros(Dictionary<string, string> m, int tileRows, bool large)
    {
        bool wideK = DiagnosticSwitches.GemmBlockK == 64 && long.Parse(m["K"], CultureInfo.InvariantCulture) % 64 == 0;
        int[] layout = Gemm2Layout(tileRows, large, wideK);
        int threads = layout[0] * layout[1] * 32;
        int columns = layout[1] * layout[3] * 16;
        int blockK = wideK ? 64 : 32;
        if (layout[0] * layout[2] * 16 != tileRows || (tileRows == Gemm2TileRows && columns != Gemm2TileColumns)
            || columns != Gemm2Columns(tileRows) || tileRows * blockK / 8 % threads != 0 || columns * blockK / 16 % threads != 0)
        {
            throw new NotSupportedException($"GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles staged evenly by its threads.");
        }

        return new(m)
        {
            ["WARPS_M"] = layout[0].ToString(CultureInfo.InvariantCulture),
            ["WARPS_N"] = layout[1].ToString(CultureInfo.InvariantCulture),
            ["FRAG_M"] = layout[2].ToString(CultureInfo.InvariantCulture),
            ["FRAG_N"] = layout[3].ToString(CultureInfo.InvariantCulture),
            ["GBK"] = blockK.ToString(CultureInfo.InvariantCulture),
        };
    }

    /// <summary>
    /// The gemm_cm2 variants (no split-K) for activations with macros <paramref name="m"/>, each also gated by
    /// <paramref name="when"/>: the 8-warp 128-token layout for large grids, the skinny tiles, then the 4-warp layout.
    /// </summary>
    private List<KernelVariant> Gemm2Variants(MatMulPlan p, Dictionary<string, string> m, Func<Bindings, bool> when)
    {
        List<KernelVariant> variants =
        [
            new KernelVariant(b => when(b) && p.Split(b) == 1 && LargeGemm2Grid(p, b),
                Lib("matmul.slang", "gemm_cm2", Gemm2Macros(m, Gemm2TileRows, large: true)), b => Gemm2Grid(p, b)),
        ];
        foreach (int tileRows in new[] { 32, 64 })
        {
            variants.Add(new KernelVariant(b => when(b) && p.Split(b) == 1 && Gemm2TileRowsFor(p.RowCount(b)) == tileRows,
                Lib("matmul.slang", "gemm_cm2", Gemm2Macros(m, tileRows, large: false)), b => Gemm2Grid(p, b)));
        }

        variants.Add(new KernelVariant(b => when(b) && p.Split(b) == 1,
            Lib("matmul.slang", "gemm_cm2", Gemm2Macros(m, Gemm2TileRows, large: false)), b => Gemm2Grid(p, b)));
        return variants;
    }

    private void LowerMatMul(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        if (w.Shape[1] % 8 != 0 && !w.Type.IsQuantized)
        {
            // Weight kernels walk K in groups of 8; odd K goes through the general strided matmul (x · Wᵀ).
            LowerBmm(n, Ref(n.Inputs[0]), w, ta: false, tb: true, alpha: 1f);
            return;
        }

        var plan = PlanMatMul(n, x, w);
        var outer = _launchCondition;
        bool intDot = UseIntDotMatvec(n, plan.W.Type, plan.K);
        bool mmq = QuantizedActivationsAllowed(n) && UseMmq(plan.W.Type, plan.N, plan.K);
        bool fp8Gemm = !mmq && UseFp8Gemm(plan.W.Type, plan.K);
        bool skinny = UseSkinnyGemm(plan);
        long skinnyMin = intDot ? IntDotSkinnyMinRows(plan.W.Type, plan.K, plan.N) : SkinnyMinRows;
        bool skinnyHere = skinny && SkinnyMaxRowsFor(plan.N) >= skinnyMin;
        // One f16 copy of x for every f16 tensor-core launch of this matmul (skinny GEMM, gemm_cm2), emitted under the outer
        // condition so that the matmuls reading the same rows (Q/K/V, gate/up, ...) share it.
        var half = SharedHalfActivations(plan, intDot, skinny && !intDot, mmq ? MmqMinRows : fp8Gemm ? Fp8GemmMinRows : null);
        if (intDot)
        {
            EmitIntDotMatvec(MatMulLabel(plan), [n], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], plan.Y, plan.N, plan.K,
                macros => Lib("mmvq.slang", "mmvq", macros));
            var outsideIntDot = OutsideIntDotRange(plan.X, plan.RowsPerLead, plan.W.Type, plan.K);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && outsideIntDot(b);
        }

        if (mmq)
        {
            // Emitted under the outer condition only, so that matmuls reading the same rows share the q8 activations.
            var beforeMmq = _launchCondition;
            _launchCondition = outer;
            EmitMmq(plan);
            _launchCondition = b => (beforeMmq?.Invoke(b) ?? true) && plan.RowCount(b) < MmqMinRows;
        }
        else if (fp8Gemm)
        {
            var beforeFp8 = _launchCondition;
            _launchCondition = outer;
            EmitFp8Gemm(plan);
            _launchCondition = b => (beforeFp8?.Invoke(b) ?? true) && plan.RowCount(b) < Fp8GemmMinRows;
        }

        if (skinnyHere)
        {
            // Up to 16 rows (above the int-dot range for formats with one): weights read once by the skinny GEMM; the launches below
            // keep single rows (matvec) and larger batches (gemm_cm2).
            var beforeSkinny = _launchCondition;
            EmitSkinnyGemm(skinnyMin, MatMulLabel(plan), [n], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], plan.Y, plan.N, plan.K, plan.Macros,
                macros => Lib("matmul.slang", "gemm_sk", macros), half);
            _launchCondition = b => (beforeSkinny?.Invoke(b) ?? true) && !InSkinnyRange(plan.RowCount(b), plan.N, skinnyMin);
        }

        EmitMatMulLaunches(plan, half);
        _launchCondition = outer;
    }

    /// <summary>
    /// The shared f16 copy of a matmul's activations for the row counts its f16 tensor-core launches serve: from 2 rows with the
    /// skinny GEMM, else above the int-dot or matvec range, up to the first row count of the int8 / fp8 GEMM
    /// (<paramref name="quantizedGemmMinRows"/>) when one serves larger batches. Null when no tensor-core launch reads f16 activations.
    /// </summary>
    private TensorRef? SharedHalfActivations(MatMulPlan p, bool intDot, bool skinny, long? quantizedGemmMinRows)
    {
        if (!p.Coop || !DiagnosticSwitches.Gemm2 || !DiagnosticSwitches.GemmHalfActivations)
        {
            return null;
        }

        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long minRows = intDot ? (UseSkinnyGemm(p) ? IntDotSkinnyMinRows(p.W.Type, p.K, p.N) : Math.Max(MatvecMaxRows, IntDotMaxRows(p.W.Type, p.K)) + 1)
            : skinny ? Math.Min(Math.Max(SkinnyMinRows, SkinnyHalfMinRows), Math.Max(MatvecMaxRows, SkinnyMaxRows) + 1)
            : MatvecMaxRows + 1;
        long maxRows = quantizedGemmMinRows is { } gemmMin ? Math.Min(rowsMax, gemmMin - 1) : rowsMax;
        if (minRows > maxRows)
        {
            return null;
        }

        return QuantizedActivations(MatMulLabel(p), [p.Node], p.Node.Inputs[0], p.X, p.XStride, p.RowsPerLead, p.K, minRows, maxRows, ActivationCode.F16);
    }

    /// <summary>
    /// The f32-activation launches of a planned matmul: matvec variants, GEMM (optionally on the f16 activations
    /// <paramref name="half"/>), split-K.
    /// </summary>
    private void EmitMatMulLaunches(MatMulPlan plan, TensorRef? half = null)
    {
        var activations = plan.X;
        if (plan.Coop && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations)
        {
            // Tensor-core GEMMs (more than 8 rows) read an f16 copy of x made by a small conversion kernel: half the
            // bytes per tile load (x is re-read by every column tile). Matvecs keep reading f32 x. Separate launches,
            // since a launch's variants share one argument list; the ones not applicable to a binding are skipped.
            var (halfActivations, halfMacros) = EmitHalfActivationMatMul(plan, half);
            activations = halfActivations;
            plan = new MatMulPlan
            {
                Node = plan.Node,
                X = plan.X,
                W = plan.W,
                Y = plan.Y,
                N = plan.N,
                K = plan.K,
                XStride = plan.XStride,
                RowsPerLead = plan.RowsPerLead,
                Macros = halfMacros,
                Coop = plan.Coop,
                WorkgroupMatrix = plan.WorkgroupMatrix,
                SplitRows = plan.SplitRows,
            };
        }
        else
        {
            EmitDirectMatMul(plan);
        }

        EmitSplitKMatMul(plan, activations);
    }

    /// <summary>Resolves operands, kernel format macros and which GEMM family applies to a MatMul node.</summary>
    private MatMulPlan PlanMatMul(Node n, TensorRef x, TensorRef w)
    {
        long N = w.Shape[0];
        long K = w.Shape[1];
        x = PrepareMatMulInput(x, "matmul", out long xStride, out long rowsPerLead);
        var y = Produce(n.Output);
        int format = WeightFormat(w.Type) ?? throw new NotSupportedException($"MatMul weight type {w.Type}");
        var macros = Macros(("WFMT", format), ("K", K), ("N", N), ("X_STRIDE", xStride), ("Y_STRIDE", N), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"));

        bool coop = CoopUsable(n, w.Type, K);
        // Workgroup-scope matrix GEMM (device-staged tiles, decode on load) for the formats it decodes.
        bool workgroupMatrix = coop && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations && PreferWorkgroupMatrixGemm
                               && _opt.Has(DeviceFeatures.MatrixWorkgroup) && K % NvTileK == 0 && format is 1 or 8 or 108;
        Func<long, int> splitRows = workgroupMatrix
            ? rows => ChooseSplit(N, K, rows, NvTileTokens, NvTileColumns, NvTileK)
            : coop
                ? rows => ChooseSplit(N, K, rows, Gemm2TileRowsFor(rows), Gemm2Columns(Gemm2TileRowsFor(rows)),
                    DiagnosticSwitches.GemmBlockK == 64 && K % 64 == 0 ? 64 : 32)
                : rows => ChooseSimtSplit(N, K, rows);

        return new MatMulPlan
        {
            Node = n,
            X = x,
            W = w,
            Y = y,
            N = N,
            K = K,
            XStride = xStride,
            RowsPerLead = rowsPerLead,
            Macros = macros,
            Coop = coop,
            WorkgroupMatrix = workgroupMatrix,
            SplitRows = splitRows,
        };
    }

    /// <summary>
    /// True when the f16 tensor-core GEMMs serve weights of <paramref name="type"/> with reduction length <paramref name="k"/>:
    /// f16 matrix MMA (+ f16 arithmetic), the 32-wide subgroups gemm_cm is written for, K in 32-value blocks, and reduced
    /// precision allowed for f32 weights.
    /// </summary>
    /// <summary>
    /// <see cref="CoopUsable(DataType, long)"/> for matmul node <paramref name="n"/>: a node marked
    /// <see cref="StandardOps.F32ActivationsAttribute"/> (block-quantized ONNX weights, whose ORT reference keeps f32
    /// activations) uses the f16 tensor-core GEMMs only with reduced precision allowed.
    /// </summary>
    private bool CoopUsable(Node n, DataType type, long k) => CoopUsable(type, k) && QuantizedActivationsAllowed(n);

    private bool CoopUsable(DataType type, long k)
        => _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && k % 32 == 0
           && (type != DataType.F32 || _opt.Compile.AllowReducedPrecision);

    private static string MatMulLabel(MatMulPlan p) => $"matmul {p.Node.Output.Name} [{p.K}→{p.N}] {p.W.Type}";

    private static Grid Gemm2Grid(MatMulPlan p, Bindings b, uint splits = 1)
    {
        int tileRows = Gemm2TileRowsFor(p.RowCount(b));
        return new(Grid.Div(p.N, Gemm2Columns(tileRows)), Grid.Div(p.RowCount(b), tileRows), splits);
    }

    private static bool LargeGemm2Grid(MatMulPlan p, Bindings b)
        => Gemm2TileRowsFor(p.RowCount(b)) == Gemm2TileRows
           && Grid.Div(p.N, Gemm2TileColumns) * Grid.Div(p.RowCount(b), Gemm2TileRows) >= LargeGridWorkgroups;

    /// <summary>One launch with matvec variants and GEMM variants that read f32 activations.</summary>
    private void EmitDirectMatMul(MatMulPlan p)
    {
        var m = p.Macros;
        if (!p.Coop)
        {
            Emit(MatMulLabel(p), [p.Node],
                [.. MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), m, p.N, p.K, p.RowCount), .. SimtGemmVariants(p)],
                [Pointer(p.X), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
                [p.X, p.W], [p.Y]);
            return;
        }

        List<KernelVariant> variants = MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), m, p.N, p.K, p.RowCount);
        if (DiagnosticSwitches.Gemm2)
        {
            variants.AddRange(Gemm2Variants(p, m, _ => true));
        }
        else
        {
            variants.Add(new KernelVariant(_ => true, Lib("matmul.slang", "gemm_cm", m),
                b => new Grid(Grid.Div(p.N, 128), Grid.Div(p.RowCount(b), 64))));
        }

        Emit(MatMulLabel(p), [p.Node], variants,
            [Pointer(p.X), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [p.X, p.W], [p.Y]);
    }

    /// <summary>
    /// Emits the matvec launch, the f32→f16 activation conversion and the GEMM launch reading the f16 copy.
    /// Returns the f16 activations and the macros the GEMM variants were built from.
    /// </summary>
    private (TensorRef Activations, Dictionary<string, string> Macros) EmitHalfActivationMatMul(MatMulPlan p, TensorRef? shared)
    {
        var x = p.X;
        long K = p.K;
        long rowsPerLead = p.RowsPerLead;
        Emit(MatMulLabel(p), [p.Node],
            MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), p.Macros, p.N, K, p.RowCount),
            [Pointer(x), Pointer(p.W), Pointer(p.Y), TokenArg(x, rowsPerLead)],
            [x, p.W], [p.Y]);

        var half = shared;
        if (half is null)
        {
            long rowsMax = x.Shape[0] * rowsPerLead;
            half = Temp(DataType.F16, [rowsMax, K], x.Symbol);
            Emit($"matmul {p.Node.Output.Name} x→f16", [p.Node],
                [new KernelVariant(b => p.RowCount(b) > MatvecMaxRows, Lib("data.slang", "to_f16", Macros(("K", K), ("X_STRIDE", p.XStride))),
                    b => new Grid(Grid.Div(p.RowCount(b) * (K / 8), 256)))],
                [Pointer(x), Pointer(half), TokenArg(x, rowsPerLead)],
                [x], [half]);
        }

        var halfMacros = new Dictionary<string, string>(p.Macros) { ["X_STRIDE"] = K.ToString(CultureInfo.InvariantCulture), ["X_F16"] = "1" };
        if (p.WorkgroupMatrix)
        {
            halfMacros = new Dictionary<string, string>(halfMacros.Concat(new Dictionary<string, string>
            {
                ["NV_BT"] = $"{NvTileTokens}",
                ["NV_BN"] = $"{NvTileColumns}",
                ["NV_BK"] = $"{NvTileK}",
            }));
        }

        List<KernelVariant> gemmVariants;
        if (p.WorkgroupMatrix)
        {
            gemmVariants =
            [
                new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == 1, Lib("matmul.slang", "gemm_nv2", halfMacros),
                    b => new Grid(Grid.Div(p.N, NvTileColumns), Grid.Div(p.RowCount(b), NvTileTokens))),
            ];
        }
        else
        {
            gemmVariants = Gemm2Variants(p, halfMacros, b => p.RowCount(b) > MatvecMaxRows);
        }

        Emit(MatMulLabel(p), [p.Node], gemmVariants,
            [Pointer(half), Pointer(p.W), Pointer(p.Y), TokenArg(x, rowsPerLead)],
            [half, p.W], [p.Y]);
        return (half, halfMacros);
    }

    /// <summary>
    /// Split-K for grids too small to fill the GPU (small N or few token tiles): partial sums into a shared
    /// scratch buffer, then a reduction. Both launches only run when the chosen split is greater than one.
    /// </summary>
    private void EmitSplitKMatMul(MatMulPlan p, TensorRef activations)
    {
        if (!p.Coop)
        {
            EmitSimtSplitKMatMul(p);
            return;
        }

        if (!DiagnosticSwitches.Gemm2)
        {
            return;
        }

        long N = p.N;
        long rowsPerLead = p.RowsPerLead;
        long rowsMax = p.X.Shape[0] * rowsPerLead;
        long scratchBytes = 0;
        var splits = new SortedSet<int>();
        var tiledSplits = new SortedSet<(int TileRows, int Split)>();
        // The split depends on rows only through the token tile size and count; the largest row count per tile count
        // bounds scratch.
        for (long rows = Math.Min(rowsMax, MatvecMaxRows + 1); rows <= rowsMax; rows = NextSplitProbe(rows, rowsMax))
        {
            int split = p.SplitRows(rows);
            if (split > 1)
            {
                splits.Add(split);
                tiledSplits.Add((p.WorkgroupMatrix ? NvTileTokens : Gemm2TileRowsFor(rows), split));
                scratchBytes = Math.Max(scratchBytes, split * rows * N * 4);
            }
        }

        if (splits.Count == 0)
        {
            return;
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var splitMacros = (int split) => new Dictionary<string, string>(p.Macros) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) };

        var partialVariants = tiledSplits.Select(ts => p.WorkgroupMatrix
            ? new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == ts.Split,
                Lib("matmul.slang", "gemm_nv2", splitMacros(ts.Split)),
                b => new Grid(Grid.Div(N, NvTileColumns), Grid.Div(p.RowCount(b), NvTileTokens), (uint)ts.Split))
            : new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == ts.Split && Gemm2TileRowsFor(p.RowCount(b)) == ts.TileRows,
                Lib("matmul.slang", "gemm_cm2", new Dictionary<string, string>(Gemm2Macros(p.Macros, ts.TileRows, large: false))
                {
                    ["SPLITK"] = ts.Split.ToString(CultureInfo.InvariantCulture),
                }),
                b => Gemm2Grid(p, b, (uint)ts.Split)));
        Emit($"{MatMulLabel(p)} split-k", [p.Node],
            [.. partialVariants],
            [Pointer(activations), Pointer(p.W), new PointerArg(scratch, 0), TokenArg(activations, rowsPerLead)],
            [activations, p.W], [partial]);

        var reduceVariants = splits.Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
            Lib("matmul.slang", "splitk_reduce", splitMacros(split)),
            b => new Grid(Grid.Div(p.RowCount(b) * N, 256))));
        Emit($"matmul {p.Node.Output.Name} split-k reduce", [p.Node],
            [.. reduceVariants],
            [new PointerArg(scratch, 0), Pointer(p.W), Pointer(p.Y), TokenArg(activations, rowsPerLead)],
            [partial], [p.Y]);
    }

    /// <summary>
    /// The split-K partial-sum buffer for the next matmul, grown to <paramref name="bytes"/>. Consecutive matmuls get different
    /// buffers (three in rotation), so sibling matmuls of one dependency level (Q/K/V, gate/up) do not order each other through
    /// the scratch and can share barriers.
    /// </summary>
    private BufferDecl SplitScratch(long bytes)
    {
        int slot = _splitScratchNext++ % _splitScratch.Length;
        var buffer = _splitScratch[slot] ??= NewBuffer(BufferKind.Activations, "splitk" + slot, 0);
        buffer.Bytes = Math.Max(buffer.Bytes, bytes);
        return buffer;
    }

    /// <summary>
    /// Next row count to probe when searching the split factors: the largest row count of each token tile size (32, 64,
    /// 128) first, then 128-row steps, finishing at <paramref name="rowsMax"/>.
    /// </summary>
    private static long NextSplitProbe(long rows, long rowsMax)
    {
        foreach (long tileEnd in new long[] { 32, 64, 128 })
        {
            if (rows < tileEnd && rowsMax > tileEnd)
            {
                return tileEnd;
            }
        }

        return rows == rowsMax ? rowsMax + 1 : Math.Min(rowsMax, rows + 128);
    }

    /// <summary>
    /// K split for the tensor-core GEMM tiles: aim for ~256 workgroups, keep at least 4 K blocks per split
    /// and no empty splits.
    /// </summary>
    private static int ChooseSplit(long N, long K, long rows, long blockRows = Gemm2TileRows, long blockColumns = Gemm2TileColumns, long blockK = 32)
    {
        if (rows <= MatvecMaxRows)
        {
            return 1;
        }

        long tiles = (N + blockColumns - 1) / blockColumns * ((rows + blockRows - 1) / blockRows);
        long steps = K / blockK;
        if (tiles >= 192)
        {
            return 1;
        }

        int split = 1;
        while (split < 32 && tiles * split < 256 && steps / (split * 2) >= 4)
        {
            split *= 2;
        }

        while (split > 1 && (split - 1) * ((steps + split - 1) / split) >= steps)
        {
            split /= 2;
        }

        return split;
    }
}
