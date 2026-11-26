using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Largest row count served by the matrix-vector kernels.</summary>
    private const long MatvecMaxRows = 8;

    // Tensor-core GEMM (gemm_cm2) tile for prefill: both 128-token warp layouts produce 128x128 tiles. Small batches use
    // 32- or 64-token tiles (see Gemm2TileRowsFor) so the MMA work stays proportional to the rows.
    private const int Gemm2TileRows = 128;
    private const int Gemm2TileColumns = 128;

    /// <summary>
    /// gemm_cm2 grids (workgroups, K splits included) below this run f32-weight matmuls on the SIMT GEMM instead, even under
    /// reduced precision (see <see cref="SmallCoopGrid"/>).
    /// </summary>
    private const long SmallCoopGridWorkgroups = 96;

    private static readonly int NvTileTokens = DiagnosticSwitches.WorkgroupMatrixTile[0];
    private static readonly int NvTileColumns = DiagnosticSwitches.WorkgroupMatrixTile[1];
    private static readonly int NvTileK = DiagnosticSwitches.WorkgroupMatrixTile[2];

    /// <summary>Split-K partial-sum buffers, used in rotation (see <see cref="SplitScratch"/>).</summary>
    private readonly BufferDecl?[] _splitScratch = new BufferDecl?[3];

    /// <summary>Arrival counters of the fused split-K reduction, one per split-K scratch slot (see <see cref="SplitArrivals"/>).</summary>
    private readonly BufferDecl?[] _splitArrivals = new BufferDecl?[3];

    private int _splitScratchNext;

    /// <summary>Split-f16 parts and row scales per constant weight (see <see cref="SplitWeights"/>).</summary>
    private readonly Dictionary<(Value Weight, long Taps), (TensorRef Parts, TensorRef Scales)> _splitWeights = [];

    /// <summary>
    /// Use the workgroup-scope matrix GEMM (<see cref="DeviceFeatures.MatrixWorkgroup"/>) for prefill matmuls when the
    /// device has it. Off by default: on the drivers measured so far (NVIDIA Blackwell, 2026) the kernel-staged
    /// gemm_cm2 is as fast for f16 weights and ~1.5x faster for q8_0, whose decode-on-load is per element.
    /// Defaults to true only when <c>SHARPLIGHT_GEMM_NV2=1</c> is set. Process-wide; set before lowering.
    /// </summary>
    public static bool PreferWorkgroupMatrixGemm { get; set; } = DiagnosticSwitches.WorkgroupMatrixGemm;

    /// <summary>
    /// Matrix-vector variants for up to 8 rows of x (memory bound: decode, small batches). Rows up to <see cref="InvariantRows"/>
    /// run the single-row kernel and K split with more columns, and every variant sums each 8-weight group as one explicit
    /// fma chain (<c>MV_FMA_CHAIN</c>), so each row's sum is computed exactly as for one row.
    /// </summary>
    private List<KernelVariant> MatvecVariants(Func<string, Dictionary<string, string>, CompiledKernel> lib, Dictionary<string, string> m,
        long N, long K, Func<Bindings, long> rows)
    {
        // Decode (one token): split-K matvec, one workgroup per output row, K spread over enough threads that
        // each walks only a few 8-weight groups (keeps many loads in flight per row).
        int skSg = SplitKMatvecSubgroups(N, K);
        long skRows = SplitKMatvecRows(N, dual: m.TryGetValue("DUAL", out var dualMacro) && dualMacro == "1");
        if (InvariantRows > 0)
        {
            m = new Dictionary<string, string>(m) { ["MV_FMA_CHAIN"] = "1" };
        }

        KernelVariant SingleRowKernel(Func<Bindings, bool> when, long columns)
        {
            var macros = new Dictionary<string, string>(m)
            {
                ["MV_COLS"] = columns.ToString(CultureInfo.InvariantCulture), ["MV_SK_SG"] = skSg.ToString(CultureInfo.InvariantCulture),
                ["MV_SK_ROWS"] = skRows.ToString(CultureInfo.InvariantCulture),
            };
            if (skRows > 1)
            {
                macros["MV_FMA_CHAIN"] = "1";
            }
            return K >= 1024
                ? new KernelVariant(when, lib("matvec_sk", macros), b => new Grid(Grid.Div(N, skRows)))
                : new KernelVariant(when, lib("matvec", macros), b => new Grid(Grid.Div(N, 4)));
        }

        List<KernelVariant> variants = [SingleRowKernel(b => rows(b) == 1, 1)];
        for (long columns = 2; columns <= MatvecMaxRows; columns *= 2)
        {
            if (InvariantBucket(columns))
            {
                long bucket = columns;
                variants.Add(SingleRowKernel(b => rows(b) <= bucket, bucket));
            }
        }

        // Small N: one split-K workgroup per row, otherwise one subgroup per row.
        variants.Add(N <= 2048 && K >= 1024
            ? new KernelVariant(b => rows(b) <= MatvecMaxRows, lib("matvec_sk", m), b => new Grid((uint)N))
            : new KernelVariant(b => rows(b) <= MatvecMaxRows, lib("matvec", m), b => new Grid(Grid.Div(N, 4))));
        return variants;
    }

    /// <summary>Subgroups per output row of the single-row split-K matvec (<c>MV_SK_SG</c> in <c>matvec_sk</c>).</summary>
    private static int SplitKMatvecSubgroups(long n, long k) => n <= 2048 || k >= 12288 ? 8 : 4;

    /// <summary>
    /// Output rows per workgroup of the single-row split-K matvec (<c>MV_SK_ROWS</c> in <c>matvec_sk</c>): 2 for single matrices
    /// with at least 1024 rows, else 1. Two rows per workgroup read each f32 activation from cache once for both rows (in f32,
    /// x is four times the bytes of FP8 weights) and keep two weight loads in flight per thread. Blackwell, FP8 at one row (µs):
    /// [17408→5120] 67 → 59, [5120→17408] 67 → 57, [5120→12288] 47 → 41, [5120→6144] 24 → 22 (27B FP8 decode step 22.5-23.1
    /// → 20.0-20.5 ms). The dual gate / up matvec (already two loads per thread) and narrow matrices (fewer workgroups:
    /// [5120→48] 3.6 → 4.4) keep one row.
    /// A row's products are added in the same order at either row count, but left to itself the driver contracts the plain sum
    /// of an 8-weight group differently in the two-row kernel (last-bit differences in about 60 % of the outputs). The two-row
    /// kernel therefore writes the sum as the explicit fma chain (<c>MV_FMA_CHAIN</c>), which is what the driver makes of the
    /// one-row kernel's plain sum, so the outputs equal those of one row per workgroup (greedy output unchanged on
    /// Llama-3.2-1B bf16 / Q8_0, Qwen3-0.6B bf16, Qwen3-8B Q4_K_M and gpt-oss-20b MXFP4; RTX PRO 4500 decode step: Llama-1B
    /// Q8_0 2.10 → 2.04 ms, gpt-oss-20b 4.37 → 4.16 ms, bf16 and Q4_K unchanged). <c>SHARPLIGHT_MV_ROWS=n</c> forces n (1 restores one row per workgroup).
    /// </summary>
    private static long SplitKMatvecRows(long n, bool dual) => DiagnosticSwitches.SplitKMatvecRows ?? (dual || n < 1024 ? 1 : 2);

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
        long dualSkinnyMin = UseIntDotMatvec(a, weightA.Type, K) ? IntDotSkinnyMinRows(weightA, K, N) : SkinnyMinRows;
        if (DiagnosticSwitches.SkinnyGemm && CoopUsable(a, weightA.Type, K) && SkinnyMaxRowsFor(N, weightA) >= dualSkinnyMin)
        {
            // Up to 16 rows (above the int-dot range for formats with one): both matrices on the skinny GEMM with the group as
            // its epilogue.
            var beforeSkinny = _launchCondition;
            EmitSkinnyGemm(dualSkinnyMin, $"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x, xStride, rowsPerLead,
                [weightA, weightB], y, N, K, macros,
                skinnyMacros => _compiler.Compile("matmul_dual.slang", source, "gemm_sk", _opt.Target,
                    new Dictionary<string, string>(skinnyMacros) { ["KERNEL_GEMM_SK"] = "1" }, _opt.Cache));
            _launchCondition = b => (beforeSkinny?.Invoke(b) ?? true) && !InSkinnyRange(rows(b), N, weightA, dualSkinnyMin);
            fusedMaxRows = Math.Max(fusedMaxRows, SkinnyMaxRowsFor(N, weightA));
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
        // 16-bit weights with the vendor GEMM library: from LibraryGemmMinRows rows the two matmuls run on the library (LowerMatMul
        // below) and the group as its own launch, instead of the dual gemm_cm2.
        bool library = LibraryGemmUsable(weightA.Type, N, K, rowsPerLead, x.Shape[0]) && weightB.Type == weightA.Type;
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > fusedBelow;
        bool dualFp8 = EmitDualFp8Gemm($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group], a.Inputs[0], x,
            xStride, rowsPerLead, weightA, weightB, y, N, K, DualEpilogueSource(a, bNode, group, output, "gemm_fp8.slang"),
            UseNativeFp8Gemm(weightA.Type, N, K) ? DualEpilogueCudaSource(a, bNode, group, output) : null, out var dualFp8Serves, output,
            UniformFp8ScaleGroups(a.Inputs[1], weightA) && UniformFp8ScaleGroups(bNode.Inputs[1], weightB));
        // The hand-written f16 GEMM serves the pair from LibraryGemmMinRows rows as one dual launch with the group as its epilogue
        // (instead of two GEMMs and the elementwise launch), unless x's f16 copy is row-ranged (its scaling must be undone before
        // a nonlinear epilogue).
        // NVFP4 pairs on the block-scaled FP4 MMA: one dual FP4 GEMM with the group in its store.
        var fp4Dual = dualFp8 ? _ => false : EmitDualFp4Gemm($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group],
            a.Inputs[0], x, xStride, rowsPerLead, weightA, weightB, y, N, K, DualEpilogueCudaSource(a, bNode, group, output));
        Func<long, bool> nativeDual = _ => false;
        if (!dualFp8 && library && NativeGemmF16 && DualEpilogueCudaSource(a, bNode, group, output) is { } f16Epilogue)
        {
            long rowsMax = x.Shape[0] * rowsPerLead;
            string label = $"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]";
            var half = QuantizedActivations(label, [a, bNode, .. group], a.Inputs[0], x, xStride, rowsPerLead, K, LibraryGemmMinRows, rowsMax, ActivationCode.F16);
            if (!_rangeScales.ContainsKey(half))
            {
                EmitNativeF16($"{label} native", [a, bNode, .. group], sk => NativeF16Variants(N, K, rowsMax, f16Epilogue, rows, LibraryGemmMinRows, streamK: sk),
                    [Pointer(half), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
                    [half, weightA, weightB], [y]);
                nativeDual = r => r >= LibraryGemmMinRows;
            }
        }

        // K-quant pairs at 9-64 rows: the hand-written K-quant GEMM's dual form, the group in its store.
        var kquantDual = dualFp8 ? null : EmitNativeKQuantDualGemm($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group],
            a.Inputs[0], x, xStride, rowsPerLead, weightA, weightB, y, N, K, DualEpilogueCudaSource(a, bNode, group, output),
            DualEpilogueSource(a, bNode, group, output, "gemm_kq.slang"), fusedBelow);
        var dualCondition = _launchCondition;
        _launchCondition = b => (dualCondition?.Invoke(b) ?? true) && !fp4Dual(rows(b));
        var dualGemm2 = dualFp8 ? null : EmitDualGemm2($"matmul2 {output.Name} [{K}→{N}]x2 {weightA.Type} fused[{groupNames}]", [a, bNode, .. group],
            a.Inputs[0], x, xStride, rowsPerLead, weightA, weightB, y, N, K, DualEpilogueSource(a, bNode, group, output, "matmul.slang"), output,
            library ? LibraryGemmMinRows : long.MaxValue);
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > fusedBelow && !(dualFp8 && dualFp8Serves(rows(b))) && !nativeDual(rows(b))
                                && !(dualGemm2?.Invoke(rows(b)) ?? false) && !(kquantDual?.Invoke(rows(b)) ?? false)
                                && !fp4Dual(rows(b));
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
            int skSg = SplitKMatvecSubgroups(N, K);
            long skRows = SplitKMatvecRows(N, dual: false);
            var macros = Macros(("WFMT", format), ("K", K), ("N", N), ("X_STRIDE", xStride), ("Y_STRIDE", N), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"),
                ("MV_COLS", 1), ("MV_SK_SG", skSg), ("MV_SK_ROWS", skRows), ("BIAS", 1));
            if (skRows > 1)
            {
                macros["MV_FMA_CHAIN"] = "1";
            }

            var tx = x;
            Func<Bindings, long> rows = b => Rows(b, tx) * rowsPerLead;
            Emit($"matmul {matmul.Output.Name} [{K}→{N}] {w.Type} +bias", [matmul, add],
                [new KernelVariant(b => rows(b) == 1, Lib("matmul.slang", "matvec_sk", macros), _ => new Grid(Grid.Div(N, skRows)))],
                [Pointer(x), Pointer(w), Pointer(y), Pointer(bias), TokenArg(x, rowsPerLead)],
                [x, w, bias], [y]);
            _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) != 1;
        }

        LowerMatMul(matmul, DiagnosticSwitches.NoMatMulEpilogue ? null : new MatMulEpilogue([add], add.Output));
        if (DiagnosticSwitches.NoMatMulEpilogue)
        {
            LowerSeparateEpilogue(matmul, new MatMulEpilogue([add], add.Output), _launchCondition);
        }

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
    /// Token rows of the gemm_cm2 tile for a batch of <paramref name="rows"/>: 32, 48 or 64 for small batches (decode of many
    /// sequences), where a 128-token tile would mostly multiply padding, else 128.
    /// </summary>
    /// <remarks>
    /// The split-f16 form (<paramref name="splitHalfColumns"/>, its output columns; 0 for other GEMMs) uses 64-token tiles up to
    /// <see cref="SplitHalfTile64MaxRows"/> rows when the 128-token grid has under <see cref="SplitHalfTile64MaxWorkgroups"/>
    /// workgroups: its
    /// 128-token tiles hold two accumulator sets in about 220 registers (one 8-warp workgroup per SM), and a few hundred rows
    /// leave such grids to deep K splits of a few steps each (vit-base at 197 rows: 40.8 -> 38.8 us for [3072 -> 768], 34.7 ->
    /// 28.5 us for [768 -> 2304]; bert-base's 128-row matmuls move from the SIMT GEMM to tensor cores). Wide outputs (a vocabulary
    /// head) already fill the device with 128-token tiles, which read each weight tile half as often.
    /// </remarks>
    private static int Gemm2TileRowsFor(long rows, long splitHalfColumns = 0)
        => !DiagnosticSwitches.Gemm2Skinny ? Gemm2TileRows
            : rows <= 32 ? 32
            : rows <= 48 && splitHalfColumns == 0 && DiagnosticSwitches.Gemm2Tile48 ? 48
            : rows <= 64 || (rows <= SplitHalfTile64MaxRows && splitHalfColumns > 0
                && Grid.Div(splitHalfColumns, Gemm2Columns(Gemm2TileRows)) * Grid.Div(rows, Gemm2TileRows) < SplitHalfTile64MaxWorkgroups) ? 64
            : Gemm2TileRows;

    /// <summary>Largest batch the split-f16 GEMM runs with 64-token tiles.</summary>
    private const long SplitHalfTile64MaxRows = 256;

    /// <summary>128-token split-f16 grids of at least this many workgroups keep 128-token tiles.</summary>
    private const long SplitHalfTile64MaxWorkgroups = 160;

    /// <summary>
    /// gemm_cm2 layout (warps M, warps N, fragments M, fragments N) of a tile with <paramref name="tileRows"/> tokens and
    /// a K block of 64 (<paramref name="wideK"/>) or 32. Every thread stages at least one 8-value activation group per K
    /// block, so the 32-token tile with a 32-value K block has only 4 warps. 128-token tiles use 4 warps of 64x64
    /// (<paramref name="splitHalf"/> false) or 8 warps of 64x32 (the split-f16 form, see <see cref="Gemm2Macros"/>).
    /// </summary>
    /// <remarks>
    /// The 4-warp 128x128 tile runs two workgroups per SM (255 registers, 40 KB of shared memory each). On the RTX PRO 4500 it
    /// beat the 8-warp layout, which used to take grids of 384+ workgroups, at every grid size measured (128-1536 tiles; Q4_K,
    /// Q6_K, Q8_0 and bf16 weights): [4096 → 14336] Q4_K at 512 rows 632 → 487 us, bf16 [2048 → 6144] at 2048 rows 575 → 429 us.
    /// </remarks>
    private static int[] Gemm2Layout(int tileRows, bool splitHalf, bool wideK = true) => tileRows switch
    {
        32 => DiagnosticSwitches.Gemm2Layout32 ?? (wideK ? [2, 4, 1, 2] : [1, 4, 2, 2]),
        48 => wideK ? [1, 4, 3, 2] : [1, 2, 3, 4],
        64 => DiagnosticSwitches.Gemm2Layout64 ?? [2, 4, 2, 2],
        _ => DiagnosticSwitches.Gemm2Layout ?? (splitHalf ? [2, 4, 4, 2] : [2, 2, 4, 4]),
    };

    /// <summary>Weight rows (output columns) of the gemm_cm2 tile with <paramref name="tileRows"/> tokens.</summary>
    private static int Gemm2Columns(int tileRows)
    {
        var layout = Gemm2Layout(tileRows, splitHalf: false);
        return layout[1] * layout[3] * 16;
    }

    /// <summary>Macros of gemm_cm2 for a tile of <paramref name="tileRows"/> tokens (layouts: <see cref="Gemm2Layout"/>).</summary>
    private static Dictionary<string, string> Gemm2Macros(Dictionary<string, string> m, int tileRows)
    {
        bool split = m.TryGetValue("GEMM_SPLIT", out var splitMacro) && splitMacro == "1";
        // The single-product form (SPLIT_PARTS 1, fast matrix precision) stages 64-value K blocks like the f16 GEMM (vit-base b1
        // 1.38 -> 1.24 ms, whisper-tiny 1.13 -> 1.00 ms of GPU time against 32-value blocks).
        bool splitLo = split && !(m.TryGetValue("SPLIT_PARTS", out var parts) && parts == "1");
        bool wideK = !splitLo && DiagnosticSwitches.GemmBlockK == 64 && long.Parse(m["K"], CultureInfo.InvariantCulture) % 64 == 0;
        // The split form keeps two accumulator sets live (the step's products, then the f32 sums): the 8-warp layout's
        // 64x32 subgroup tiles leave registers for them, the 4-warp layout's 64x64 tiles spill (vit [768 → 3072]: 151 against 196 us).
        // The single-product form is also faster on the 8-warp layout (whisper-tiny 1.00 -> 0.93 ms).
        int[] layout = Gemm2Layout(tileRows, split, wideK);
        int threads = layout[0] * layout[1] * 32;
        int columns = layout[1] * layout[3] * 16;
        int blockK = wideK ? 64 : 32;
        if (layout[0] * layout[2] * 16 != tileRows || (tileRows == Gemm2TileRows && columns != Gemm2TileColumns)
            || columns != Gemm2Columns(tileRows) || tileRows * blockK / 8 % threads != 0 || columns * blockK / 16 % threads != 0)
        {
            throw new NotSupportedException($"GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles staged evenly by its threads.");
        }

        var result = new Dictionary<string, string>(m)
        {
            ["WARPS_M"] = layout[0].ToString(CultureInfo.InvariantCulture),
            ["WARPS_N"] = layout[1].ToString(CultureInfo.InvariantCulture),
            ["FRAG_M"] = layout[2].ToString(CultureInfo.InvariantCulture),
            ["FRAG_N"] = layout[3].ToString(CultureInfo.InvariantCulture),
            ["GBK"] = blockK.ToString(CultureInfo.InvariantCulture),
        };
        AddTileGroup(result);
        foreach (var (name, value) in DiagnosticSwitches.GemmExtraMacros)
        {
            result[name] = value;
        }

        return result;
    }

    /// <summary>Smallest weight matrix (bytes) whose GEMM launches use the grouped tile order.</summary>
    private const long TileGroupMinWeightBytes = 48L << 20;

    /// <summary>
    /// Token tiles per group of the grouped tile order (<c>grouped_tile</c> in common.slang) of gemm_cm2, the int8 GEMM and the
    /// fp8 GEMM. Launched column tiles first, the workgroups running at once cover one or two token tiles and every weight tile,
    /// so a weight matrix larger than the L2 cache is read from DRAM once per token tile. In groups of 16 token tiles (2048 rows)
    /// they share a few weight tiles instead. RTX PRO 4500, f16 weights, 2048 rows: [4096 → 12288] 104 → 128 TFLOPS, [12288 →
    /// 4096] 99 → 105; 8192 rows: [4096 → 12288] 98 → 125. Each output is computed exactly as before. Launches with K splits keep
    /// the launch order (at 512 rows the grouped order lost 7 % on [12288 → 4096], whose grid is split), and so do matrices under
    /// <see cref="TileGroupMinWeightBytes"/>, which stay in L2 anyway (bf16 [4096 → 4096], Q4_K / Q8_0 of the 8B: 2-4 % slower grouped).
    /// </summary>
    private static void AddTileGroup(Dictionary<string, string> macros)
    {
        if (DiagnosticSwitches.GemmTileGroup <= 0 || !macros.TryGetValue("ROW_BYTES", out var rowBytes) || !macros.TryGetValue("N", out var n))
        {
            return;
        }

        long weightBytes = long.Parse(rowBytes.TrimEnd('u', 'l'), CultureInfo.InvariantCulture) * long.Parse(n, CultureInfo.InvariantCulture);
        if (weightBytes >= TileGroupMinWeightBytes)
        {
            macros["GEMM_GROUP"] = DiagnosticSwitches.GemmTileGroup.ToString(CultureInfo.InvariantCulture);
        }
    }

    /// <summary>
    /// The gemm_cm2 variants (no split-K) for activations with macros <paramref name="m"/>, each also gated by
    /// <paramref name="when"/>: the skinny tiles, then the 128-token tile.
    /// </summary>
    private List<KernelVariant> Gemm2Variants(MatMulPlan p, Dictionary<string, string> m, Func<Bindings, bool> when,
        Func<Dictionary<string, string>, CompiledKernel>? compile = null)
    {
        compile ??= macros => Lib("matmul.slang", "gemm_cm2", macros);
        List<KernelVariant> variants = [];
        foreach (int tileRows in new[] { 32, 48, 64 })
        {
            variants.Add(new KernelVariant(b => when(b) && p.Split(b) == 1 && Gemm2TileRowsFor(p.RowCount(b), SplitHalfColumns(p)) == tileRows,
                compile(Gemm2Macros(m, tileRows)), b => Gemm2Grid(p, b)));
        }

        variants.Add(new KernelVariant(b => when(b) && p.Split(b) == 1,
            compile(Gemm2Macros(m, Gemm2TileRows)), b => Gemm2Grid(p, b)));
        return variants;
    }

    /// <summary>
    /// Lowers weight matmul <paramref name="n"/>. With <paramref name="epilogue"/> (an elementwise group consuming the product), the
    /// split-f16 GEMM and the int-dot matvec apply the group in their store; the other launches write the product and the group runs
    /// as its own launch for the row counts they serve.
    /// </summary>
    private void LowerMatMul(Node n, MatMulEpilogue? epilogue = null)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        if (w.Shape[1] % 8 != 0 && !w.Type.IsQuantized)
        {
            // Weight kernels walk K in groups of 8; odd K goes through the general strided matmul (x · Wᵀ).
            LowerBmm(n, Ref(n.Inputs[0]), w, ta: false, tb: true, alpha: 1f);
            if (epilogue is not null)
            {
                LowerElementwise(epilogue.Group, [epilogue.Output]);
            }

            return;
        }

        var plan = PlanMatMul(n, x, w);
        var outer = _launchCondition;
        // The split-f16 GEMM applies the epilogue in its store; the f16 tensor-core path in its split-K reduction (the bias / residual of
        // projections at 9-256 rows, whose grids split K), the group's own launch covering the other row counts.
        // Epilogues over int-dot weights (folded for the int-dot matvec's store) run as their own launch on the other paths; f16 / bf16
        // weights keep theirs in the split-K reduction (the biases of gpt-oss's projections).
        bool intDot = UseIntDotMatvec(n, plan.W.Type, plan.K);
        bool gemmEpilogue = n.Inputs[1].Type == DataType.F32 || !intDot;
        bool reduceEpilogue = epilogue is not null && gemmEpilogue && !plan.SplitHalf && plan.Coop && !plan.WorkgroupMatrix && DiagnosticSwitches.Gemm2
                              && DiagnosticSwitches.GemmHalfActivations && DiagnosticSwitches.SplitKReduceEpilogue;
        var epilogueCode = epilogue is not null && gemmEpilogue && (plan.SplitHalf || reduceEpilogue) ? MatMulEpilogueSource(n, epilogue) : null;
        Func<Bindings, bool> intDotEpilogue = _ => false;
        bool mmq = QuantizedActivationsAllowed(n) && UseMmq(plan.W.Type, plan.N, plan.K);
        bool fp8Gemm = !mmq && UseFp8Gemm(plan.W.Buffer.Weight, plan.W.Type, plan.K);
        bool fp4Gemm = !mmq && !fp8Gemm && UseNativeFp4Gemm(plan);
        bool nativeBias = !mmq && !fp8Gemm && !fp4Gemm && UseNativeBiasGemm(plan, epilogue);
        bool library = !mmq && !fp8Gemm && !fp4Gemm && (UseLibraryGemm(plan, epilogue is not null) || nativeBias);
        bool skinny = UseSkinnyGemm(plan);
        long skinnyMin = intDot ? IntDotSkinnyMinRows(plan.W, plan.K, plan.N) : SkinnyMinRows;
        bool skinnyHere = skinny && SkinnyMaxRowsFor(plan.N, plan.W) >= skinnyMin;
        // One f16 copy of x for every f16 tensor-core launch of this matmul (skinny GEMM, gemm_cm2), emitted under the outer
        // condition so that the matmuls reading the same rows (Q/K/V, gate/up, ...) share it.
        var half = SharedHalfActivations(plan, intDot, skinny && !intDot, mmq ? MmqMinRows : fp8Gemm ? Fp8GemmMinRows(plan.W.Type, plan.N, plan.K) : fp4Gemm ? NativeFp4MinRows : null);
        if (intDot)
        {
            var outsideIntDot = OutsideIntDotRange(plan.X, plan.RowsPerLead, plan.W.Type, plan.K);
            if (epilogue is not null && plan.Y.IsDense && plan.RowsPerLead == 1)
            {
                // The group in the matvec's store: it writes the group's output for the int-dot rows.
                var code = MatMulEpilogueSource(n, epilogue, "mmvq.slang");
                EmitIntDotMatvec(MatMulLabel(plan) + code.Label, [n, .. epilogue.Group], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], code.Output,
                    plan.N, plan.K, macros => MatMulEpilogueKernel(code, "mmvq", macros, "mmvq_epi.slang"), code.Args);
                intDotEpilogue = b => !outsideIntDot(b);
            }
            else
            {
                EmitIntDotMatvec(MatMulLabel(plan), [n], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], plan.Y, plan.N, plan.K,
                    macros => Lib("mmvq.slang", "mmvq", macros));
            }

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
            _launchCondition = b => (beforeFp8?.Invoke(b) ?? true) && plan.RowCount(b) < Fp8GemmMinRows(plan.W.Type, plan.N, plan.K);
        }
        else if (fp4Gemm)
        {
            // Emitted under the outer condition only, so that matmuls reading the same rows (q / k / v) share the FP4 activations.
            var beforeFp4 = _launchCondition;
            _launchCondition = outer;
            EmitNativeFp4Gemm(plan, epilogue is not null);
            _launchCondition = b => (beforeFp4?.Invoke(b) ?? true) && plan.RowCount(b) < NativeFp4MinRows;
        }
        else if (library)
        {
            var beforeLibrary = _launchCondition;
            _launchCondition = outer;
            EmitLibraryGemm(plan, half, nativeBias ? epilogue : null);
            long libraryMin = nativeBias ? NativeBiasGemmMinRows : LibraryGemmMinRows;
            _launchCondition = b => (beforeLibrary?.Invoke(b) ?? true) && plan.RowCount(b) < libraryMin;
        }

        Func<Bindings, bool> skinnyEpilogue = _ => false;
        if (skinnyHere)
        {
            // Up to 16 rows (above the int-dot range for formats with one): weights read once by the skinny GEMM; the launches below
            // keep single rows (matvec) and larger batches (gemm_cm2). The epilogue (a projection's bias, a residual add) goes into
            // its store where the product is dense rows of N.
            var beforeSkinny = _launchCondition;
            var skinnyCode = epilogue is not null && gemmEpilogue && !plan.SplitHalf && plan.RowsPerLead == 1 && plan.Y.IsDense
                             && plan.Macros["Y_STRIDE"] == plan.N.ToString(CultureInfo.InvariantCulture) && DiagnosticSwitches.SkinnyEpilogue
                ? epilogueCode ?? MatMulEpilogueSource(n, epilogue)
                : null;
            EmitSkinnyGemm(skinnyMin, MatMulLabel(plan), [n], n.Inputs[0], plan.X, plan.XStride, plan.RowsPerLead, [plan.W], plan.Y, plan.N, plan.K, plan.Macros,
                macros => Lib("matmul.slang", "gemm_sk", macros), half, skinnyCode);
            if (skinnyCode is not null)
            {
                skinnyEpilogue = b => (beforeSkinny?.Invoke(b) ?? true) && InSkinnyRange(plan.RowCount(b), plan.N, plan.W, skinnyMin);
            }

            _launchCondition = b => (beforeSkinny?.Invoke(b) ?? true) && !InSkinnyRange(plan.RowCount(b), plan.N, plan.W, skinnyMin);
        }

        // A lone bias add over int-dot weights (its own launch on the GEMM paths): the multi-output GEMM of the q / k / v group adds it
        // in its store for the batches it serves, as for f16 weights.
        var multiBias = epilogue is not null && epilogueCode is null && !gemmEpilogue && !DiagnosticSwitches.NoIntDotMultiBias
                        && _multiGemmOf.ContainsKey(n) && epilogue.Group is [{ Op.Name: "Add" }]
            ? MatMulEpilogueSource(n, epilogue)
            : null;
        if (EpilogueBias(multiBias) is null)
        {
            multiBias = null;
        }

        var reduced = EmitMatMulLaunches(plan, half, epilogueCode, multiBias);
        _launchCondition = outer;
        if (epilogue is not null && epilogueCode is null)
        {
            LowerSeparateEpilogue(n, epilogue, b => (outer?.Invoke(b) ?? true) && !(nativeBias && plan.RowCount(b) >= NativeBiasGemmMinRows) && !intDotEpilogue(b)
                && !skinnyEpilogue(b));
        }
        else if (epilogue is not null && !plan.SplitHalf)
        {
            // Row counts whose split-K reduction did not apply the group (no K split, or another launch served them: e.g. the fp8
            // GEMM, whose multi-output form writes the products without the bias), nor the hand-written GEMM's bias store.
            LowerSeparateEpilogue(n, epilogue, b => (outer?.Invoke(b) ?? true) && !(reduced?.Invoke(b) ?? false)
                && !(nativeBias && plan.RowCount(b) >= NativeBiasGemmMinRows) && !intDotEpilogue(b) && !skinnyEpilogue(b));
        }
    }

    /// <summary>
    /// The shared f16 copy of a matmul's activations for the row counts its f16 tensor-core launches serve: from 2 rows with the
    /// skinny GEMM, else above the int-dot or matvec range, up to the first row count of the int8 / fp8 GEMM
    /// (<paramref name="quantizedGemmMinRows"/>) when one serves larger batches. Null when no tensor-core launch reads f16 activations.
    /// When the skinny GEMM converts f32 rows itself (<c>SHARPLIGHT_SKINNY_X16=0</c>), the copy starts above the rows it serves at
    /// this matmul's width (<see cref="SkinnyMaxRowsFor"/>), where gemm_cm2 takes over.
    /// </summary>
    private TensorRef? SharedHalfActivations(MatMulPlan p, bool intDot, bool skinny, long? quantizedGemmMinRows)
    {
        if (!p.Coop || p.SplitHalf || !DiagnosticSwitches.Gemm2 || !DiagnosticSwitches.GemmHalfActivations)
        {
            return null;
        }

        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long minRows = intDot ? (UseSkinnyGemm(p) ? IntDotSkinnyMinRows(p.W, p.K, p.N) : Math.Max(MatvecMaxRows, IntDotMaxRows(p.W.Type, p.K)) + 1)
            : skinny ? Math.Min(Math.Max(SkinnyMinRows, SkinnyHalfMinRows), Math.Max(MatvecMaxRows, SkinnyMaxRowsFor(p.N, p.W)) + 1)
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
    /// <paramref name="half"/>), split-K. Returns the condition under which the split-K reduction runs (and applies
    /// <paramref name="epilogue"/>), or null when no row count splits K.
    /// </summary>
    /// <param name="plan">The planned matmul.</param>
    /// <param name="half">The shared f16 copy of the activations, if any.</param>
    /// <param name="epilogue">The epilogue the split-K reduction applies, if any.</param>
    /// <param name="multiBias">A bias epilogue the launches here do not apply, which a multi-output group's GEMM adds in its store.</param>
    private Func<Bindings, bool>? EmitMatMulLaunches(MatMulPlan plan, TensorRef? half = null, MatMulEpilogueCode? epilogue = null,
        MatMulEpilogueCode? multiBias = null)
    {
        var activations = plan.X;
        if (plan.Coop && !plan.SplitHalf && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations)
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
        else if (plan.SplitHalf)
        {
            activations = EmitSplitHalfMatMul(plan, epilogue);
        }
        else
        {
            EmitDirectMatMul(plan);
        }

        return EmitSplitKMatMul(plan, activations, epilogue, multiBias);
    }

    /// <summary>
    /// Emits the f32 matvec launch, the split of x into row-scaled f16 hi / lo parts (shared by the matmuls reading the same
    /// rows) and the split-f16 GEMM. Returns the split activations.
    /// </summary>
    private TensorRef EmitSplitHalfMatMul(MatMulPlan p, MatMulEpilogueCode? epilogue = null)
    {
        if (MinRows(p.X) * p.RowsPerLead <= MatvecMaxRows)
        {
            Emit(MatMulLabel(p), [p.Node],
                MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), WithoutSplit(p.Macros), p.N, p.K, p.RowCount),
                [Pointer(p.X), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
                [p.X, p.W], [p.Y]);
            if (epilogue is not null)
            {
                // The matvec rows (up to MatvecMaxRows) apply the group in its own launch.
                var outer = _launchCondition;
                _launchCondition = b => (outer?.Invoke(b) ?? true) && p.RowCount(b) <= MatvecMaxRows;
                LowerElementwise(epilogue.Group, [epilogue.OutputValue]);
                _launchCondition = outer;
            }
        }

        var split = QuantizedActivations(MatMulLabel(p), [p.Node], p.Node.Inputs[0], p.X, p.XStride, p.RowsPerLead, p.K, MatvecMaxRows + 1,
            p.X.Shape[0] * p.RowsPerLead, p.Macros.ContainsKey("SPLIT_PARTS") ? ActivationCode.ScaledF16 : ActivationCode.SplitF16);
        var y = epilogue?.Output ?? p.Y;
        Emit(MatMulLabel(p) + (epilogue?.Label ?? ""), [p.Node, .. epilogue?.Group ?? []],
            Gemm2Variants(p, p.Macros, b => p.RowCount(b) > MatvecMaxRows,
                epilogue is null ? null : macros => MatMulEpilogueKernel(epilogue, "gemm_cm2", macros)),
            [Pointer(split), Pointer(p.WeightParts!), Pointer(y), Pointer(p.WeightScales!), .. MatMulEpilogueArgs(epilogue), TokenArg(p.X, p.RowsPerLead)],
            [split, p.WeightParts!, p.WeightScales!, .. epilogue?.Args ?? []], [y]);
        return split;
    }

    /// <summary>
    /// The fewest rows <paramref name="r"/> can have: its fixed extent, a lone symbol at 1 (0 rows need no launch), 0 for other
    /// expressions. Matmuls whose rows always exceed the matvec range need no matvec launch (nor, for split f16, the f32 weights).
    /// </summary>
    private static long MinRows(TensorRef r)
        => r.Symbol is null ? r.Shape[0] : SymbolicExpression.Parse(r.Symbol).IsSymbol ? 1 : 0;

    /// <summary>The macros without GEMM_SPLIT (the f32 matvecs and the split-K reduction of a split-f16 matmul).</summary>
    private static Dictionary<string, string> WithoutSplit(Dictionary<string, string> m)
    {
        var copy = new Dictionary<string, string>(m);
        copy.Remove("GEMM_SPLIT");
        copy.Remove("SPLIT_PARTS");
        return copy;
    }

    /// <summary>
    /// True when f32 matrix products use one f16 product of row-scaled inputs (<see cref="MatrixPrecision.Fast"/>) instead of the
    /// split form's three; reduced precision, where allowed, takes precedence with its own kernels.
    /// </summary>
    private bool FastMatrixProducts => _opt.Compile.MatrixPrecision == MatrixPrecision.Fast && !HalfActivations;

    /// <summary>
    /// True when reduced precision lets f32 activations enter products as f16 (<see cref="CompileOptions.AllowReducedPrecision"/> with
    /// <see cref="ActivationPrecision.Half"/>). Under <see cref="ActivationPrecision.BFloat16"/> only the kernels with a bf16 form
    /// (conv_cl, conv_cm) compute in 16 bits; every other kernel keeps its strict form, which holds f32's range.
    /// </summary>
    private bool HalfActivations => _opt.Compile.AllowReducedPrecision && _opt.Compile.ActivationPrecision == ActivationPrecision.Half;

    /// <summary>
    /// The split f16 form of the constant f32 weights <paramref name="weight"/> ([N, K] or a conv's [O, CG, KH, KW], read as
    /// <paramref name="w"/>) for the split-f16 GEMM and conv (<see cref="SplitScaleTable"/>): the hi / lo parts and the per-row scales, both computed on the host when
    /// the weights are uploaded. Null when the weights are not a constant of the graph. With <paramref name="taps"/> above 1 (a conv's
    /// KH * KW), each row's values are reordered from [channel][tap] to [tap][channel] first (conv_cm's TAP_MAJOR reduction order).
    /// </summary>
    private (TensorRef Parts, TensorRef Scales)? SplitWeights(Value weight, TensorRef w, long taps = 1)
    {
        if (w.Type != DataType.F32 || w.ByteOffset != 0 || w.Shape.Length < 2 || weight.Kind is not (ValueKind.Weight or ValueKind.Constant))
        {
            return null;
        }

        if (_splitWeights.TryGetValue((weight, taps), out var cached))
        {
            return cached;
        }

        // Rows are the leading dimension (matmul [N, K], conv [O, CG, KH, KW]); the rest is one reduction row.
        long rows = w.Shape[0];
        long k = w.Shape.Skip(1).Aggregate(1L, (a, b) => a * b);
        byte[] Original()
        {
            if (weight.Kind == ValueKind.Constant)
            {
                return weight.ConstantData!;
            }

            var bytes = new byte[rows * k * 4];
            weight.Weight!.Source.Read(0, bytes);
            return bytes;
        }

        byte[] Source() => taps > 1 ? TapMajor(Original(), rows, k / taps, taps) : Original();

        // Fast matrix precision: the hi parts only (one product per multiply-add).
        bool hiOnly = FastMatrixProducts;
        long halves = hiOnly ? 1 : 2;
        (WeightInfo Parts, WeightInfo Scales) Create()
            => (Transient(weight.Name + (hiOnly ? ":scaled-f16" : ":split-f16") + (taps > 1 ? ":tap-major" : ""), DataType.F16, new Shape(rows, halves * k), () => SplitScaleTable.Split(Source(), rows, k, hiOnly)),
                Transient(weight.Name + ":split-scale", DataType.F32, new Shape(rows), () => SplitScaleTable.Scales(Source(), rows, k)));
        static WeightInfo Transient(string name, DataType type, Shape shape, Func<byte[]> produce)
            => new() { Name = name, Type = type, Shape = shape, Source = new TransientWeightSource(type.ByteSize(shape.Elements()), produce) };

        // Model weights: one derived pair per source tensor, so shared-weight registries upload it once for every compilation.
        var (partsInfo, scalesInfo) = weight.Kind == ValueKind.Weight
            ? (DerivedWeightCache.Get(weight.Weight!, ("split-f16", rows, k, hiOnly, taps), () => Create().Parts),
               DerivedWeightCache.Get(weight.Weight!, ("split-scale", rows, k, taps), () => Create().Scales))
            : Create();
        var parts = new TensorRef(NewBuffer(BufferKind.Weight, partsInfo.Name, partsInfo.ByteSize, w: partsInfo), 0, DataType.F16, [rows, halves * k], [halves * k, 1], null);
        var scales = new TensorRef(NewBuffer(BufferKind.Weight, scalesInfo.Name, scalesInfo.ByteSize, w: scalesInfo), 0, DataType.F32, [rows], [1], null);
        _splitWeights[(weight, taps)] = (parts, scales);
        return (parts, scales);
    }

    /// <summary>f32 rows of [<paramref name="channels"/>][<paramref name="taps"/>] values reordered to [taps][channels].</summary>
    private static byte[] TapMajor(byte[] source, long rows, long channels, long taps)
    {
        var src = MemoryMarshal.Cast<byte, float>(source.AsSpan(0, checked((int)(rows * channels * taps * 4))));
        var bytes = new byte[src.Length * 4];
        var dst = MemoryMarshal.Cast<byte, float>(bytes.AsSpan());
        long k = channels * taps;
        for (long r = 0; r < rows; r++)
        {
            for (long c = 0; c < channels; c++)
            {
                for (long t = 0; t < taps; t++)
                {
                    dst[(int)(r * k + t * channels + c)] = src[(int)(r * k + c * taps + t)];
                }
            }
        }

        return bytes;
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
        if (coop && w.Type == DataType.F32 && SmallCoopGrid(N, K, (x.Symbol is null ? x.Shape[0] : _opt.Compile.Bound(x.Symbol)) * rowsPerLead))
        {
            // f32 weights under reduced precision: a tensor-core grid that cannot fill the device (a few hundred rows, short K,
            // so no split either) is slower than the f32 SIMT GEMM, which tiles finer and splits K further.
            coop = false;
        }

        // f32 weights under strict precision: the split-f16 tensor-core GEMM when allowed (CompileOptions.SplitHalfMatMul), on
        // grids large enough for tensor cores to beat the f32 SIMT GEMM.
        long rowsBound = (x.Symbol is null ? x.Shape[0] : _opt.Compile.Bound(x.Symbol)) * rowsPerLead;
        (TensorRef Parts, TensorRef Scales)? splitWeights = null;
        // The single-product form of fast matrix precision beats the SIMT GEMM on small grids too (convnext-tiny b1 1.17 -> 0.97 ms,
        // bert-base b8 2.39 -> 2.27 ms of GPU time).
        if (!coop && SplitHalfGemmUsable(n, w.Type, K) && (FastMatrixProducts || !SmallCoopGrid(N, K, rowsBound, splitHalf: true)))
        {
            splitWeights = SplitWeights(n.Inputs[1], w);
        }

        bool splitHalf = splitWeights is not null;
        if (splitHalf)
        {
            coop = true;
            macros["GEMM_SPLIT"] = "1";
            if (FastMatrixProducts)
            {
                macros["SPLIT_PARTS"] = "1";
            }
        }

        // Workgroup-scope matrix GEMM (device-staged tiles, decode on load) for the formats it decodes.
        bool workgroupMatrix = coop && !splitHalf && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations && PreferWorkgroupMatrixGemm
                               && _opt.Has(DeviceFeatures.MatrixWorkgroup) && K % NvTileK == 0 && format is 1 or 8 or 108;
        // Devices with only 8x8 subgroup matrices (Apple GPUs): the gemm_sg tile for the rows beyond the matvecs, split over K
        // when its grid would leave the GPU idle.
        bool smallMatrix = !coop && SmallMatrixGemmUsable(n, w.Type, K);
        Func<long, int> splitRows = smallMatrix ? rows => ChooseSmallMatrixSplit(N, K, rows) : workgroupMatrix
            ? rows => ChooseSplit(N, K, rows, NvTileTokens, NvTileColumns, NvTileK, GemmWaveTiles)
            : coop
                ? rows => NativeKQuantPlanSplit(w.Type, N, K, rows, splitHalf, ChooseSplit(N, K, rows, Gemm2TileRowsFor(rows, splitHalf ? N : 0),
                    Gemm2Columns(Gemm2TileRowsFor(rows, splitHalf ? N : 0)), (!splitHalf || FastMatrixProducts) && DiagnosticSwitches.GemmBlockK == 64 && K % 64 == 0 ? 64 : 32,
                    GemmWaveTiles))
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
            WeightParts = splitWeights?.Parts,
            WeightScales = splitWeights?.Scales,
            WorkgroupMatrix = workgroupMatrix,
            SmallMatrix = smallMatrix,
            SplitRows = splitRows,
        };
    }

    /// <summary>
    /// True when matmul <paramref name="n"/> may run on the 8x8 subgroup-matrix GEMM (<c>gemm_sg</c>): the device has
    /// <see cref="DeviceFeatures.MatrixF16Small"/> and f16 arithmetic, K in 32-value blocks, and f16 activations acceptable as for the
    /// tensor-core GEMMs (f32 weights only with reduced precision). <c>SHARPLIGHT_GEMM_SG=0</c> disables it.
    /// </summary>
    private bool SmallMatrixGemmUsable(Node n, DataType type, long k)
        => DiagnosticSwitches.SmallMatrixGemm && _opt.Has(DeviceFeatures.MatrixF16Small) && _opt.Has(DeviceFeatures.Float16) && k % 32 == 0
           && (type != DataType.F32 || HalfActivations) && QuantizedActivationsAllowed(n);

    /// <summary>
    /// True when matmul <paramref name="n"/> with f32 weights may run on the split-f16 tensor-core GEMM (gemm_cm2 with GEMM_SPLIT):
    /// allowed by <see cref="CompileOptions.SplitHalfMatMul"/> (the default), not needed because reduced precision is allowed, on
    /// devices with f16 matrix products and 32-wide subgroups, K in 32-value blocks; <c>SHARPLIGHT_GEMM_SPLIT_F16=0</c> disables
    /// it. The weights must also be graph constants (<see cref="SplitWeights"/>).
    /// </summary>
    private bool SplitHalfGemmUsable(Node n, DataType type, long k)
        => type == DataType.F32 && _opt.Compile.SplitHalfMatMul && !HalfActivations && DiagnosticSwitches.GemmSplitHalf
           && DiagnosticSwitches.Gemm2 && _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32
           && k % 32 == 0 && n.Op.Name == "MatMul";

    /// <summary>
    /// True when the gemm_cm2 grid for <paramref name="rows"/> rows (tiles times K splits) stays under
    /// <see cref="SmallCoopGridWorkgroups"/>: too few workgroups to occupy the device.
    /// </summary>
    private static bool SmallCoopGrid(long n, long k, long rows, bool splitHalf = false)
    {
        int tileRows = Gemm2TileRowsFor(rows, splitHalf ? n : 0);
        long columns = Gemm2Columns(tileRows);
        int split = ChooseSplit(n, k, rows, tileRows, columns, DiagnosticSwitches.GemmBlockK == 64 && k % 64 == 0 ? 64 : 32);
        return rows > MatvecMaxRows && (n + columns - 1) / columns * ((rows + tileRows - 1) / tileRows) * split < SmallCoopGridWorkgroups;
    }

    /// <summary>
    /// <see cref="CoopUsable(DataType, long)"/> for matmul node <paramref name="n"/>: a node marked
    /// <see cref="StandardOps.F32ActivationsAttribute"/> (block-quantized ONNX weights, whose ORT reference keeps f32
    /// activations) uses the f16 tensor-core GEMMs only with reduced precision allowed.
    /// </summary>
    private bool CoopUsable(Node n, DataType type, long k) => CoopUsable(type, k) && QuantizedActivationsAllowed(n);

    /// <summary>
    /// True when the f16 tensor-core GEMMs serve weights of <paramref name="type"/> with reduction length <paramref name="k"/>:
    /// f16 matrix MMA (+ f16 arithmetic), the 32-wide subgroups gemm_cm is written for, K in 32-value blocks, and reduced
    /// precision allowed for f32 weights.
    /// </summary>
    private bool CoopUsable(DataType type, long k)
        => _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && k % 32 == 0
           && (type != DataType.F32 || HalfActivations);

    private static string MatMulLabel(MatMulPlan p) => $"matmul {p.Node.Output.Name} [{p.K}→{p.N}] {p.W.Type}";

    /// <summary>The output columns of <paramref name="p"/> when it runs the split-f16 GEMM, else 0 (see <see cref="Gemm2TileRowsFor"/>).</summary>
    private static long SplitHalfColumns(MatMulPlan p) => p.SplitHalf ? p.N : 0;

    private static Grid Gemm2Grid(MatMulPlan p, Bindings b, uint splits = 1)
    {
        int tileRows = Gemm2TileRowsFor(p.RowCount(b), SplitHalfColumns(p));
        return new(Grid.Div(p.N, Gemm2Columns(tileRows)), Grid.Div(p.RowCount(b), tileRows), splits);
    }


    /// <summary>One launch with matvec variants and GEMM variants that read f32 activations.</summary>
    private void EmitDirectMatMul(MatMulPlan p)
    {
        var m = p.Macros;
        if (!p.Coop)
        {
            List<KernelVariant> smallMatrix = p.SmallMatrix
                ? [new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == 1, SmallMatrixKernel(p, 1), b => SmallMatrixGrid(p, b, 1))]
                : [];
            Emit(MatMulLabel(p), [p.Node],
                MemberMatvecVariants(p, m, [.. MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), m, p.N, p.K, p.RowCount), .. smallMatrix,
                    .. SimtGemmVariants(p)]),
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

        Emit(MatMulLabel(p), [p.Node], MemberMatvecVariants(p, m, variants),
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
            MemberMatvecVariants(p, p.Macros, MatvecVariants((entry, macros) => Lib("matmul.slang", entry, macros), p.Macros, p.N, K, p.RowCount)),
            [Pointer(x), Pointer(p.W), Pointer(p.Y), TokenArg(x, rowsPerLead)],
            [x, p.W], [p.Y]);

        var half = shared;
        if (half is null)
        {
            // A row-ranged copy (common.slang, f16_range_scale): x's rows may exceed f16's range.
            long rowsMax = x.Shape[0] * rowsPerLead;
            bool ranged = DiagnosticSwitches.F16RangeGuard;
            half = Temp(DataType.F16, [rowsMax + (ranged ? RangeScaleRows(rowsMax, K) : 0), K], x.Symbol);
            if (ranged)
            {
                _rangeScales[half] = rowsMax * K * 2;
            }

            Emit($"matmul {p.Node.Output.Name} x→{(ranged ? "ranged " : "")}f16", [p.Node],
                [new KernelVariant(b => p.RowCount(b) > MatvecMaxRows,
                    ranged
                        ? Lib("data.slang", "to_f16_ranged", Macros(("K", K), ("X_STRIDE", p.XStride), ("Y_RS", _rangeScales[half] + "ull")))
                        : Lib("data.slang", "to_f16", Macros(("K", K), ("X_STRIDE", p.XStride))),
                    ranged ? b => new Grid((uint)p.RowCount(b)) : b => new Grid(Grid.Div(p.RowCount(b) * (K / 8), 256)))],
                [Pointer(x), Pointer(half), TokenArg(x, rowsPerLead)],
                [x], [half]);
        }

        var halfMacros = WithRangeScales(new Dictionary<string, string>(p.Macros) { ["X_STRIDE"] = K.ToString(CultureInfo.InvariantCulture), ["X_F16"] = "1" }, half);
        if (p.WorkgroupMatrix)
        {
            halfMacros = new Dictionary<string, string>(halfMacros.Concat(new Dictionary<string, string>
            {
                ["NV_BT"] = $"{NvTileTokens}",
                ["NV_BN"] = $"{NvTileColumns}",
                ["NV_BK"] = $"{NvTileK}",
            }.Concat(DiagnosticSwitches.GemmExtraMacros)));
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
        if (_rangeScales.TryGetValue(half, out long scales))
        {
            // The rows of a row-ranged copy that were scaled leave scaled outputs; split-K undoes them in its reduction.
            Emit($"{MatMulLabel(p)} f16 range", [p.Node],
                [new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == 1,
                    Lib("data.slang", "f16_range_unscale", Macros(("N", p.N), ("Y_STRIDE", p.Macros["Y_STRIDE"]), ("X_RS", scales + "ull"))),
                    b => new Grid(Grid.Div(p.RowCount(b), 64)))],
                [Pointer(half), Pointer(p.Y), TokenArg(x, rowsPerLead)],
                [half, p.Y], [p.Y]);
        }

        return (half, halfMacros);
    }

    /// <summary>
    /// Split-K for grids too small to fill the GPU (small N or few token tiles): partial sums into a shared
    /// scratch buffer, then a reduction. Both launches only run when the chosen split is greater than one.
    /// </summary>
    private Func<Bindings, bool>? EmitSplitKMatMul(MatMulPlan p, TensorRef activations, MatMulEpilogueCode? epilogue = null,
        MatMulEpilogueCode? multiBias = null)
    {
        if (!p.Coop)
        {
            EmitSimtSplitKMatMul(p);
            return null;
        }

        if (!DiagnosticSwitches.Gemm2)
        {
            return null;
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
                tiledSplits.Add((p.WorkgroupMatrix ? NvTileTokens : Gemm2TileRowsFor(rows, SplitHalfColumns(p)), split));
                scratchBytes = Math.Max(scratchBytes, split * rows * N * 4);
            }
        }

        if (splits.Count == 0)
        {
            return null;
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var splitMacros = (int split) => new Dictionary<string, string>(WithoutSplit(p.Macros)) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) };
        // A member of a multi-output group (q / k / v) leaves the batches its group's GEMM serves to it.
        Func<Bindings, bool> notMulti = _ => true;
        if (MultiGemmCandidate(p))
        {
            var stored = epilogue ?? multiBias;
            var y0 = stored?.Output ?? p.Y;
            var condition0 = _launchCondition;
            notMulti = NoteMultiGemmPart(p.Node, new MultiGemmPart(false, activations, p.W, y0, EpilogueBias(stored), p.Macros,
                b => (condition0?.Invoke(b) ?? true) && p.RowCount(b) > MatvecMaxRows && p.Split(b) > 1, p.RowCount, rowsMax, TokenArg(p.X, rowsPerLead)));
            if (epilogue is not null && EpilogueBias(epilogue) is null && _multiGemmOf.TryGetValue(p.Node, out var group))
            {
                group.Usable = false;
            }
        }

        // Decode batches of f16 projections on the hand-written f16 GEMM's decode form: one launch for the split GEMM and its reduction.
        var notMultiAlone = notMulti;
        var decode = EmitNativeF16DecodeGemm(p, activations, epilogue, b => notMultiAlone(b) && p.RowCount(b) > MatvecMaxRows && p.Split(b) > 1);
        // K-quant projections likewise on the hand-written K-quant decode GEMM.
        decode ??= EmitNativeKQuantDecodeGemm(p, activations, epilogue, b => notMultiAlone(b) && p.RowCount(b) > MatvecMaxRows && p.Split(b) > 1);
        if (decode is not null)
        {
            notMulti = b => notMultiAlone(b) && !decode(b);
        }

        if (FusedSplitKUsable(p))
        {
            EmitFusedSplitKMatMul(p, activations, epilogue, tiledSplits, scratch, partial, notMulti);
            var fusedCondition = _launchCondition;
            return b => (fusedCondition?.Invoke(b) ?? true) && p.RowCount(b) > MatvecMaxRows && p.Split(b) > 1;
        }

        var partialVariants = tiledSplits.Select(ts => p.WorkgroupMatrix
            ? new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == ts.Split,
                Lib("matmul.slang", "gemm_nv2", splitMacros(ts.Split)),
                b => new Grid(Grid.Div(N, NvTileColumns), Grid.Div(p.RowCount(b), NvTileTokens), (uint)ts.Split))
            : new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == ts.Split && Gemm2TileRowsFor(p.RowCount(b), SplitHalfColumns(p)) == ts.TileRows
                                     && notMulti(b),
                Lib("matmul.slang", "gemm_cm2", new Dictionary<string, string>(Gemm2Macros(p.Macros, ts.TileRows))
                {
                    ["SPLITK"] = ts.Split.ToString(CultureInfo.InvariantCulture),
                }),
                b => Gemm2Grid(p, b, (uint)ts.Split)));
        KernelArg[] scaleArgs = p.WeightScales is { } weightScales ? [Pointer(weightScales)] : [];
        TensorRef[] scaleReads = p.WeightScales is { } scaleRead ? [scaleRead] : [];
        Emit($"{MatMulLabel(p)} split-k", [p.Node],
            [.. partialVariants],
            [Pointer(activations), Pointer(p.WeightParts ?? p.W), new PointerArg(scratch, 0), .. scaleArgs, TokenArg(p.X, rowsPerLead)],
            [activations, p.WeightParts ?? p.W, .. scaleReads], [partial]);

        var reduceVariants = splits.Select(split => new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == split && notMulti(b),
            epilogue is null ? Lib("matmul.slang", "splitk_reduce", splitMacros(split)) : MatMulEpilogueKernel(epilogue, "splitk_reduce", splitMacros(split)),
            b => new Grid(Grid.Div(p.RowCount(b) * N, 256))));
        var y = epilogue?.Output ?? p.Y;
        // w: unused by the reduction, except that a row-ranged f16 copy of the activations (X_RS) goes there for its row scales;
        // the lookup grid of an IQ weight still leads the arguments (the kernel is built for the weight's format).
        bool ranged = p.Macros.ContainsKey("X_RS");
        KernelArg weightArg = Pointer(p.WeightParts ?? p.W);
        KernelArg[] reduceArgs = [new PointerArg(scratch, 0), ranged ? Pointer(activations) : weightArg, Pointer(y), .. MatMulEpilogueArgs(epilogue), TokenArg(p.X, rowsPerLead)];
        if (ranged && UsesGrid((p.WeightParts ?? p.W).Type))
        {
            reduceArgs = [WithGridArgument([weightArg])[0], .. reduceArgs];
        }

        Emit($"matmul {p.Node.Output.Name} split-k reduce{epilogue?.Label}", [p.Node, .. epilogue?.Group ?? []],
            [.. reduceVariants], reduceArgs,
            ranged ? [partial, activations, .. epilogue?.Args ?? []] : [partial, .. epilogue?.Args ?? []], [y]);
        var condition = _launchCondition;
        return b => (condition?.Invoke(b) ?? true) && p.RowCount(b) > MatvecMaxRows && p.Split(b) > 1;
    }

    /// <summary>
    /// The decode form of the hand-written f16 GEMM for the batches of <see cref="NativeF16DecodeMinRows"/> to 64 rows that
    /// <paramref name="serves"/> (the split-K batches its multi-output group leaves to it), when the matmul reads the f16 copy of x
    /// with f16 weights (<see cref="UseNativeF16Decode"/>) into dense rows and its epilogue is none or a lone bias row, which the store adds (and undoes
    /// a row-ranged copy's row scales). Returns the batches it serves, or null.
    /// </summary>
    private Func<Bindings, bool>? EmitNativeF16DecodeGemm(MatMulPlan p, TensorRef activations, MatMulEpilogueCode? epilogue, Func<Bindings, bool> serves)
    {
        var bias = EpilogueBias(epilogue);
        if (!MultiGemmCandidate(p) || p.RowsPerLead != 1 || p.WeightParts is not null || (epilogue is not null && bias is null)
            || !UseNativeF16Decode(p.W.Type, p.N, p.K, p.Macros))
        {
            return null;
        }

        var y = epilogue?.Output ?? p.Y;
        if (!DenseRowsOf(y, p.N))
        {
            return null;
        }

        Func<Bindings, bool> decode = b => serves(b) && p.RowCount(b) >= NativeF16DecodeMinRows && p.RowCount(b) <= NativeF16DecodeTileRows;
        long? rangeScales = _rangeScales.TryGetValue(activations, out long xrs) ? xrs : null;
        KernelArg[] biasArgs = bias is null ? [] : [Pointer(bias)];
        TensorRef[] biasReads = bias is null ? [] : [bias];
        Emit($"{MatMulLabel(p)} native decode{epilogue?.Label}", [p.Node, .. epilogue?.Group ?? []],
            [NativeF16DecodeVariant(p.N, p.K, decode, p.RowCount, bias is not null, rangeScales)],
            [Pointer(activations), Pointer(p.W), Pointer(y), .. biasArgs, TokenArg(p.X, p.RowsPerLead)],
            [activations, p.W, .. biasReads], [y]);
        return decode;
    }

    /// <summary>
    /// True when the split-K launches of <paramref name="p"/> reduce in the GEMM's last workgroup per tile (splitk.slang,
    /// SPLITK_FUSED) instead of a <c>splitk_reduce</c> launch: gemm_cm2 on f16 activations, not the split-f16 or expert forms.
    /// </summary>
    private static bool FusedSplitKUsable(MatMulPlan p) => DiagnosticSwitches.SplitKFused && MultiGemmCandidate(p);

    /// <summary>True for the split-K GEMMs whose kernel has the fused and multi-output forms: gemm_cm2 on f16 activations, not split-f16 or expert forms.</summary>
    private static bool MultiGemmCandidate(MatMulPlan p)
        => !p.WorkgroupMatrix && !p.SplitHalf && !p.Macros.ContainsKey("MOE") && !p.Macros.ContainsKey("DUAL");

    /// <summary>
    /// The split-K GEMM of <paramref name="p"/> with the reduction (and <paramref name="epilogue"/>) in the last workgroup of each
    /// output tile: one launch instead of the partial GEMM and <c>splitk_reduce</c>, bit-identical to them (same slice order).
    /// </summary>
    private void EmitFusedSplitKMatMul(MatMulPlan p, TensorRef activations, MatMulEpilogueCode? epilogue, SortedSet<(int TileRows, int Split)> tiledSplits,
        BufferDecl scratch, TensorRef partial, Func<Bindings, bool> notMulti)
    {
        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long tiles = tiledSplits.Max(ts => Grid.Div(p.N, Gemm2Columns(ts.TileRows)) * Grid.Div(rowsMax, ts.TileRows));
        var arrivals = SplitArrivals(tiles);
        var y = epilogue?.Output ?? p.Y;
        var variants = tiledSplits.Select(ts =>
        {
            var macros = new Dictionary<string, string>(Gemm2Macros(p.Macros, ts.TileRows))
            {
                ["SPLITK"] = ts.Split.ToString(CultureInfo.InvariantCulture),
                ["SPLITK_FUSED"] = "1",
            };
            return new KernelVariant(b => p.RowCount(b) > MatvecMaxRows && p.Split(b) == ts.Split && Gemm2TileRowsFor(p.RowCount(b), SplitHalfColumns(p)) == ts.TileRows
                                          && notMulti(b),
                epilogue is null ? Lib("matmul.slang", "gemm_cm2", macros) : MatMulEpilogueKernel(epilogue, "gemm_cm2", macros),
                b => Gemm2Grid(p, b, (uint)ts.Split));
        });
        Emit($"{MatMulLabel(p)} split-k{epilogue?.Label}", [p.Node, .. epilogue?.Group ?? []], [.. variants],
            [Pointer(activations), Pointer(p.W), new PointerArg(scratch, 0), .. MatMulEpilogueArgs(epilogue), Pointer(y), Pointer(arrivals), TokenArg(p.X, p.RowsPerLead)],
            [activations, p.W, .. epilogue?.Args ?? []], [partial, y, arrivals]);
    }

    /// <summary>
    /// Arrival counters ([output tiles] u32) of the fused split-K reduction (splitk.slang) of the next split matmul: zero when created,
    /// and every launch leaves them zero. One buffer per split-K scratch slot (<see cref="SplitScratch"/>), so sibling matmuls
    /// count in different buffers; matmuls on the same slot are ordered by their scratch.
    /// </summary>
    private TensorRef SplitArrivals(long tiles)
    {
        int slot = (_splitScratchNext - 1 + _splitScratch.Length) % _splitScratch.Length;
        long bytes = Math.Max(16, tiles * 4);
        if (_splitArrivals[slot] is not { } buffer || buffer.Bytes < bytes)
        {
            buffer = _splitArrivals[slot] = NewBuffer(BufferKind.Weight, "const:splitk-arrivals" + slot, bytes, init: new byte[bytes]);
        }

        return new TensorRef(buffer, 0, DataType.U32, [bytes / 4], [1], null);
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
    /// Next row count to probe when searching the split factors: the largest row count of each token tile size (32, 48, 64,
    /// 128) first, then 128-row steps, finishing at <paramref name="rowsMax"/>.
    /// </summary>
    private static long NextSplitProbe(long rows, long rowsMax)
    {
        foreach (long tileEnd in new long[] { 32, 48, 64, 128 })
        {
            if (rows < tileEnd && rowsMax > tileEnd)
            {
                return tileEnd;
            }
        }

        return rows == rowsMax ? rowsMax + 1 : Math.Min(rowsMax, rows + 128);
    }

    /// <summary>
    /// Tensor-core GEMM tiles assumed to run at once (<paramref name="wave"/> of <see cref="ChooseSplit"/>): one per compute unit
    /// the device reports (<see cref="LoweringOptions.ComputeUnits"/>; a 128x128 tile takes most of a unit's shared memory, so
    /// units ≈ concurrent tiles), else <see cref="DefaultGemmWaveTiles"/>. <c>SHARPLIGHT_GEMM_WAVE=n</c> forces n,
    /// <c>fixed</c> the earlier 192 (other splits round the partial sums differently, so greedy output differs). RTX PRO 4500
    /// (82 SMs) against 192: pp2048 Gemma-3-1B +17 %, Llama-3.2-1B +9 %, Qwen3-0.6B +7 %, Mistral-7B +2.5 %, Qwen3-8B +2 %.
    /// </summary>
    private long GemmWaveTiles => DiagnosticSwitches.GemmWaveTiles ?? (_opt.ComputeUnits > 0 ? _opt.ComputeUnits : DefaultGemmWaveTiles);

    /// <summary>Concurrent GEMM tiles assumed when the device does not report its compute units (tuned on a 188-SM GPU).</summary>
    private const long DefaultGemmWaveTiles = 192;

    /// <summary>
    /// K split of a grid of one token tile row (decode-sized batches, whose GEMMs stream their weights): the split with the fewest
    /// estimated K steps per workgroup times waves of <paramref name="wave"/> workgroups, so that the last wave is nearly full (e.g. 32
    /// column tiles of 32 K steps: 4 splits run 128 workgroups in two waves of 82 with 8 steps each, 5 splits 160 workgroups in two
    /// waves with 7 steps each). The smallest such split, keeping at least 2 K steps per workgroup.
    /// </summary>
    private static int WaveFillSplit(long tiles, long steps, long wave)
    {
        int best = 1;
        long bestCost = long.MaxValue;
        for (int split = 1; split <= 32 && steps / split >= 2; split++)
        {
            long cost = (tiles * split + wave - 1) / wave * ((steps + split - 1) / split);
            if (cost < bestCost)
            {
                bestCost = cost;
                best = split;
            }
        }

        return best;
    }

    /// <summary>
    /// K split for the tensor-core GEMM tiles: aim for about 4/3 waves of <paramref name="wave"/> concurrent tiles, keep at least
    /// 4 K blocks per split and no empty splits.
    /// </summary>
    private static int ChooseSplit(long N, long K, long rows, long blockRows = Gemm2TileRows, long blockColumns = Gemm2TileColumns, long blockK = 32,
        long wave = DefaultGemmWaveTiles)
    {
        if (rows <= MatvecMaxRows)
        {
            return 1;
        }

        long tiles = (N + blockColumns - 1) / blockColumns * ((rows + blockRows - 1) / blockRows);
        long steps = K / blockK;
        if (tiles >= wave)
        {
            // Many tiles: split only when two K halves take clearly less time in whole waves (each half wave costs half a tile's
            // steps): e.g. 234 tiles of a 192-tile wave, 2 waves → 3 half waves; Gemma 4 31B's [21504 → 5376] at 512 rows, 168 tiles
            // of an 82-tile wave, 3 waves → 5 half waves (1.72 → 1.47 ms, benchmarks.md section 81). SHARPLIGHT_GEMM_SPLIT_FILL=1: the
            // earlier rule (two halves fill the last wave 15 points better), which kept 168 tiles unsplit (68 → 82 %).
            if (steps < 8 || tiles >= 4 * wave)
            {
                return 1;
            }

            long Waves(long count) => (count + wave - 1) / wave;
            if (DiagnosticSwitches.GemmSplitFillRule)
            {
                double Fill(long count) => (double)count / (Waves(count) * wave);
                return Fill(2 * tiles) > Fill(tiles) + 0.15 ? 2 : 1;
            }

            return Waves(2 * tiles) * 10 < Waves(tiles) * 2 * 9 ? 2 : 1;
        }

        if (DiagnosticSwitches.SplitWaveFill && rows <= blockRows)
        {
            return WaveFillSplit(tiles, steps, wave);
        }

        int split = 1;
        while (split < 32 && tiles * split < wave * 4 / 3 && steps / (split * 2) >= 4)
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
