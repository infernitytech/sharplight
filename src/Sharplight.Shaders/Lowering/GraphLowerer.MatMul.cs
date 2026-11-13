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

    // Tensor-core GEMM (gemm_cm2) tile: both supported warp layouts produce 128x128 tiles.
    private const int Gemm2TileRows = 128;
    private const int Gemm2TileColumns = 128;

    /// <summary>Grid size (workgroups) from which the 8-warp gemm_cm2 layout beats the 4-warp one.</summary>
    private const long LargeGridWorkgroups = 384;

    private static readonly int NvTileTokens = DiagnosticSwitches.WorkgroupMatrixTile[0];
    private static readonly int NvTileColumns = DiagnosticSwitches.WorkgroupMatrixTile[1];
    private static readonly int NvTileK = DiagnosticSwitches.WorkgroupMatrixTile[2];

    private BufferDecl? _splitScratch;

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
    /// for up to 8 rows one dual matvec computes both and applies the group per element. Larger batches run
    /// the regular launches (matmul, matmul, elementwise), which are skipped for small batches.
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
        if (UseIntDotMatvec(weightA.Type, K))
        {
            fusedMaxRows = Math.Max(MatvecMaxRows, IntDotMaxRows);
            string intDotSource = DualEpilogueSource(a, bNode, group, output, "mmvq.slang");
            EmitIntDotMatvec($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x, xStride, rowsPerLead,
                [weightA, weightB], y, N, K,
                intDotMacros => _compiler.Compile("mmvq_dual.slang", intDotSource, "mmvq", _opt.Target, intDotMacros, _opt.Cache));
            var outsideIntDot = OutsideIntDotRange(x, rowsPerLead, weightA.Type);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && outsideIntDot(b);
        }

        string source = DualEpilogueSource(a, bNode, group, output, "matmul.slang");
        Emit($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group],
            MatvecVariants((entry, entryMacros) => _compiler.Compile("matmul_dual.slang", source, entry, _opt.Target,
                new Dictionary<string, string>(entryMacros) { ["KERNEL_" + entry.ToUpperInvariant()] = "1" }, _opt.Cache), macros, N, K, rows),
            [Pointer(x), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [x, weightA, weightB], [y]);

        // Larger batches: the regular matmuls and the elementwise group, skipped while a fused launch serves the rows.
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > fusedMaxRows;
        LowerMatMul(a);
        LowerMatMul(bNode);
        LowerElementwise(group, [output]);
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

    /// <summary>Macros of gemm_cm2 for the 4-warp (<paramref name="large"/> false) or 8-warp layout.</summary>
    private static Dictionary<string, string> Gemm2Macros(Dictionary<string, string> m, bool large)
    {
        // warps M, warps N, fragments M, fragments N
        int[] layout = DiagnosticSwitches.Gemm2Layout ?? (large ? [2, 4, 4, 2] : [2, 2, 4, 4]);
        if (layout[0] * layout[2] * 16 != Gemm2TileRows || layout[1] * layout[3] * 16 != Gemm2TileColumns)
        {
            throw new NotSupportedException("GEMM layout must give 128x128 tiles.");
        }

        bool wideK = DiagnosticSwitches.GemmBlockK == 64 && long.Parse(m["K"], CultureInfo.InvariantCulture) % 64 == 0;
        return new(m)
        {
            ["WARPS_M"] = layout[0].ToString(CultureInfo.InvariantCulture),
            ["WARPS_N"] = layout[1].ToString(CultureInfo.InvariantCulture),
            ["FRAG_M"] = layout[2].ToString(CultureInfo.InvariantCulture),
            ["FRAG_N"] = layout[3].ToString(CultureInfo.InvariantCulture),
            ["GBK"] = wideK ? "64" : "32",
        };
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
        if (UseIntDotMatvec(plan.W.Type, plan.K))
        {
            EmitIntDotMatvec(MatMulLabel(plan), [n], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], plan.Y, plan.N, plan.K,
                macros => Lib("mmvq.slang", "mmvq", macros));
            var outsideIntDot = OutsideIntDotRange(plan.X, plan.RowsPerLead, plan.W.Type);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && outsideIntDot(b);
        }

        EmitMatMulLaunches(plan);
        _launchCondition = outer;
    }

    /// <summary>The f32-activation launches of a planned matmul: matvec variants, GEMM (optionally on f16 activations), split-K.</summary>
    private void EmitMatMulLaunches(MatMulPlan plan)
    {
        var activations = plan.X;
        if (plan.Coop && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations)
        {
            // Tensor-core GEMMs (more than 8 rows) read an f16 copy of x made by a small conversion kernel: half the
            // bytes per tile load (x is re-read by every column tile). Matvecs keep reading f32 x. Separate launches,
            // since a launch's variants share one argument list; the ones not applicable to a binding are skipped.
            var (halfActivations, halfMacros) = EmitHalfActivationMatMul(plan);
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

        // Tensor-core path needs f16 matrix MMA (+ f16 arithmetic) and the 32-wide subgroups gemm_cm is written for.
        bool coop = _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && K % 32 == 0
                    && (w.Type != DataType.F32 || _opt.Compile.AllowReducedPrecision);
        // Workgroup-scope matrix GEMM (device-staged tiles, decode on load) for the formats it decodes.
        bool workgroupMatrix = coop && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations && PreferWorkgroupMatrixGemm
                               && _opt.Has(DeviceFeatures.MatrixWorkgroup) && K % NvTileK == 0 && format is 1 or 8 or 108;
        Func<long, int> splitRows = workgroupMatrix
            ? rows => ChooseSplit(N, K, rows, NvTileTokens, NvTileColumns, NvTileK)
            : rows => ChooseSplit(N, K, rows, blockK: DiagnosticSwitches.GemmBlockK == 64 && K % 64 == 0 ? 64 : 32);

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

    private static string MatMulLabel(MatMulPlan p) => $"matmul {p.Node.Output.Name} [{p.K}→{p.N}] {p.W.Type}";

    private static Grid Gemm2Grid(MatMulPlan p, Bindings b, uint splits = 1)
        => new(Grid.Div(p.N, Gemm2TileColumns), Grid.Div(p.RowCount(b), Gemm2TileRows), splits);

    private static bool LargeGemm2Grid(MatMulPlan p, Bindings b)
        => Grid.Div(p.N, Gemm2TileColumns) * Grid.Div(p.RowCount(b), Gemm2TileRows) >= LargeGridWorkgroups;

    /// <summary>One launch with matvec variants and GEMM variants that read f32 activations.</summary>
    private void EmitDirectMatMul(MatMulPlan p)
    {
        var m = p.Macros;
        KernelVariant big;
        if (p.Coop && DiagnosticSwitches.Gemm2)
        {
            big = new KernelVariant(b => p.Split(b) == 1 && LargeGemm2Grid(p, b),
                Lib("matmul.slang", "gemm_cm2", Gemm2Macros(m, large: true)), b => Gemm2Grid(p, b));
        }
        else if (p.Coop)
        {
            big = new KernelVariant(_ => true, Lib("matmul.slang", "gemm_cm", m),
                b => new Grid(Grid.Div(p.N, 128), Grid.Div(p.RowCount(b), 64)));
        }
        else
        {
            big = new KernelVariant(_ => true, Lib("matmul.slang", "gemm", m),
                b => new Grid(Grid.Div(p.N, 64), Grid.Div(p.RowCount(b), 64)));
        }

        List<KernelVariant> variants = [.. MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), m, p.N, p.K, p.RowCount), big];
        if (p.Coop && DiagnosticSwitches.Gemm2)
        {
            variants.Add(new KernelVariant(b => p.Split(b) == 1, Lib("matmul.slang", "gemm_cm2", Gemm2Macros(m, large: false)), b => Gemm2Grid(p, b)));
        }

        Emit(MatMulLabel(p), [p.Node], variants,
            [Pointer(p.X), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [p.X, p.W], [p.Y]);
    }

    /// <summary>
    /// Emits the matvec launch, the f32→f16 activation conversion and the GEMM launch reading the f16 copy.
    /// Returns the f16 activations and the macros the GEMM variants were built from.
    /// </summary>
    private (TensorRef Activations, Dictionary<string, string> Macros) EmitHalfActivationMatMul(MatMulPlan p)
    {
        var x = p.X;
        long K = p.K;
        long rowsPerLead = p.RowsPerLead;
        Emit(MatMulLabel(p), [p.Node],
            MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), p.Macros, p.N, K, p.RowCount),
            [Pointer(x), Pointer(p.W), Pointer(p.Y), TokenArg(x, rowsPerLead)],
            [x, p.W], [p.Y]);

        long rowsMax = x.Shape[0] * rowsPerLead;
        var half = Temp(DataType.F16, [rowsMax, K], x.Symbol);
        Emit($"matmul {p.Node.Output.Name} x→f16", [p.Node],
            [new KernelVariant(b => p.RowCount(b) > MatvecMaxRows, Lib("data.slang", "to_f16", Macros(("K", K), ("X_STRIDE", p.XStride))),
                b => new Grid(Grid.Div(p.RowCount(b) * (K / 8), 256)))],
            [Pointer(x), Pointer(half), TokenArg(x, rowsPerLead)],
            [x], [half]);

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
            gemmVariants =
            [
                new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == 1 && LargeGemm2Grid(p, b),
                    Lib("matmul.slang", "gemm_cm2", Gemm2Macros(halfMacros, large: true)), b => Gemm2Grid(p, b)),
                new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == 1,
                    Lib("matmul.slang", "gemm_cm2", Gemm2Macros(halfMacros, large: false)), b => Gemm2Grid(p, b)),
            ];
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
        if (!p.Coop || !DiagnosticSwitches.Gemm2)
        {
            return;
        }

        long N = p.N;
        long rowsPerLead = p.RowsPerLead;
        long rowsMax = p.X.Shape[0] * rowsPerLead;
        long scratchBytes = 0;
        var splits = new SortedSet<int>();
        // The split depends on rows only through the token tile count; the largest row count per tile count bounds scratch.
        for (long rows = Math.Min(rowsMax, MatvecMaxRows + 1); rows <= rowsMax; rows = NextSplitProbe(rows, rowsMax))
        {
            int split = p.SplitRows(rows);
            if (split > 1)
            {
                splits.Add(split);
                scratchBytes = Math.Max(scratchBytes, split * rows * N * 4);
            }
        }

        if (splits.Count == 0)
        {
            return;
        }

        _splitScratch ??= NewBuffer(BufferKind.Activations, "splitk", 0);
        _splitScratch.Bytes = Math.Max(_splitScratch.Bytes, scratchBytes);
        var partial = new TensorRef(_splitScratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var splitMacros = (int split) => new Dictionary<string, string>(p.Macros) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) };

        var partialVariants = splits.Select(split => p.WorkgroupMatrix
            ? new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
                Lib("matmul.slang", "gemm_nv2", splitMacros(split)),
                b => new Grid(Grid.Div(N, NvTileColumns), Grid.Div(p.RowCount(b), NvTileTokens), (uint)split))
            : new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
                Lib("matmul.slang", "gemm_cm2", new Dictionary<string, string>(Gemm2Macros(p.Macros, large: false)) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) }),
                b => Gemm2Grid(p, b, (uint)split)));
        Emit($"{MatMulLabel(p)} split-k", [p.Node],
            [.. partialVariants],
            [Pointer(activations), Pointer(p.W), new PointerArg(_splitScratch, 0), TokenArg(activations, rowsPerLead)],
            [activations, p.W], [partial]);

        var reduceVariants = splits.Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split,
            Lib("matmul.slang", "splitk_reduce", splitMacros(split)),
            b => new Grid(Grid.Div(p.RowCount(b) * N, 256))));
        Emit($"matmul {p.Node.Output.Name} split-k reduce", [p.Node],
            [.. reduceVariants],
            [new PointerArg(_splitScratch, 0), Pointer(p.W), Pointer(p.Y), TokenArg(activations, rowsPerLead)],
            [partial], [p.Y]);
    }

    /// <summary>
    /// Next row count to probe when searching the split factors: 128 rows first (when the batch is larger), then
    /// 128-row steps, finishing at <paramref name="rowsMax"/>.
    /// </summary>
    private static long NextSplitProbe(long rows, long rowsMax)
    {
        if (rows < 128 && rowsMax > 128)
        {
            return 128;
        }

        return rows == rowsMax ? rowsMax + 1 : Math.Min(rowsMax, rows + 128);
    }

    /// <summary>
    /// K split for the 128x128 tensor-core GEMM: aim for ~256 workgroups, keep at least 4 K blocks per split
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
