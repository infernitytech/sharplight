using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// True when the fp8 tensor-core GEMM (<c>gemm_fp8.slang</c>, <see cref="DeviceFeatures.MatrixFp8"/>) serves
    /// prefill-sized batches of a matmul with block-scaled E4M3 weights: activations quantized to E4M3 with one scale per
    /// 128 values, f8 x f8 MMAs with f32 accumulation, one scaling per weight block. Without the feature (or with
    /// <c>SHARPLIGHT_FP8_GEMM=0</c>) the f16 tensor-core GEMM decodes the weights instead.
    /// </summary>
    private bool UseFp8Gemm(DataType type, long k)
        => DiagnosticSwitches.Fp8Gemm && _opt.Has(DeviceFeatures.MatrixFp8) && _opt.Has(DeviceFeatures.SubgroupOps) && _opt.SubgroupSize == 32
           && (type == DataType.F8E4M3_B128 || type == DataType.F8E4M3_B32 || type == ShaderTypes.F8E4M3_B128R || type == ShaderTypes.F8E4M3_B32R
               || type == ShaderTypes.F8E4M3_B128G)
           && k % 128 == 0;

    /// <summary>Emits the E4M3 activation quantization and the fp8 GEMM launches of a planned matmul (see <see cref="UseFp8Gemm"/>).</summary>
    private void EmitFp8Gemm(MatMulPlan p) => EmitQuantizedGemm(p, "fp8", ActivationCode.Fp8, 128,
        (tileRows, split) => Lib("gemm_fp8.slang", "gemm_fp8", Fp8GemmMacros(p.W.Type, p.N, p.K, tileRows, split)), tokenTilesFirst: true,
        native: UseNativeFp8Gemm(p.W.Type, p.N, p.K) ? (quantized, min, max) => DenseScaleCapacityOf(quantized) is > 0 and var xcap
            ? NativeFp8MemberVariants(p, quantized, xcap, min, max) : null : null,
        nativeMinRows: NativeFp8MinRows(!_multiGemmOf.ContainsKey(p.Node)),
        fusedKernel: !Fp8FusedSplitK ? null : (tileRows, split) => Lib("gemm_fp8.slang", "gemm_fp8",
            new Dictionary<string, string>(Fp8GemmMacros(p.W.Type, p.N, p.K, tileRows, split)) { ["SPLITK_FUSED"] = "1" }));

    /// <summary>
    /// The hand-written FP8 GEMM's launch variants of <paramref name="p"/> on its E4M3 rows <paramref name="quantized"/>; a member of a
    /// q / k / v group whose 128-row tiles run on tensor copies leaves the batches its multi-output form serves to it
    /// (<see cref="EmitNativeFp8TallMultiGemm"/>).
    /// </summary>
    private KernelVariant[] NativeFp8MemberVariants(MatMulPlan p, TensorRef quantized, long xcap, long min, long max)
    {
        bool uniform = UniformFp8ScaleGroups(p.Node.Inputs[1], p.W);
        var variants = NativeFp8Variants(p.N, p.K, null, xcap, min, max, p.RowCount, uniform);
        if (!NativeFp8TensorCopies || p.RowsPerLead != 1 || !p.Y.IsDense || p.Macros.GetValueOrDefault("Y_STRIDE") != p.N.ToString(CultureInfo.InvariantCulture))
        {
            return variants;
        }

        var notMulti = NoteMultiGemmTallPart(p.Node, new MultiGemmTallPart(quantized, p.W, p.Y, null, null, p.RowCount, min, p.X.Shape[0],
            TallCondition(min, p.RowCount), TokenArg(p.X), Fp8: true, UniformScales: uniform));
        return Unless(variants, notMulti);
    }

    /// <summary>
    /// Largest batch for which the dual fp8 GEMM serves every row count (one 64-row tile). Larger batches use it only where the
    /// separate GEMMs would not split K (see <see cref="DualFp8Serves"/>).
    /// </summary>
    private const long DualFp8AlwaysMaxRows = 64;

    /// <summary>
    /// Rows above which the dual fp8 GEMM may use 128-row tiles. Below, the 64-row tiles' larger grid wins (Blackwell, 27B
    /// gate / up: 128 rows 44.7 vs 49.9 ms per mixed step, 171 rows 71.0 vs 80.0 ms; at 384 / 512 rows the tall tiles save 3-4 %).
    /// </summary>
    private const long DualFp8TallMinRows = 256;

    /// <summary>
    /// True when the dual fp8 GEMM serves <paramref name="rows"/> activation rows of an [<paramref name="k"/>→<paramref name="n"/>]
    /// pair of <paramref name="type"/> weights: from <see cref="Fp8GemmMinRows"/> to 64 rows always; above, only where the separate fp8 GEMMs run without split-K.
    /// There each output element sums its K blocks in the same order as the separate launch (the accumulation is an explicit
    /// fused multiply-add in both), and the epilogue evaluates the same expression as the elementwise kernel, so prompt
    /// prefill stays bit-identical to the separate launches. Where they split K, their partial sums round differently.
    /// <c>SHARPLIGHT_FP8_DUAL_MAX_ROWS=64</c> keeps larger batches on the separate launches.
    /// </summary>
    private bool DualFp8Serves(long rows, DataType type, long n, long k)
        => rows >= Fp8GemmMinRows(type, n, k) && (rows <= DualFp8AlwaysMaxRows
                                      || (rows <= (DiagnosticSwitches.Fp8DualGemmMaxRows ?? long.MaxValue) && QuantizedGemmSplit(n, k, rows, 128, fp8: true) == 1));

    /// <summary>
    /// Emits the dual fp8 GEMM of a matmul pair sharing x (gate / up) for the batches <see cref="DualFp8Serves"/> accepts: one
    /// launch reads the E4M3 activations once, multiplies both weight matrices (their 16-row fragments alternate in the B tile)
    /// and stores the fused elementwise group <paramref name="epilogueSource"/> defines as <c>epi2</c>, instead of two GEMMs
    /// (with split-K partials and reductions at small batches) and an elementwise launch. No split-K: a workgroup covers half as
    /// many outputs, so the grid has twice the column tiles of a single matmul. Above <see cref="DualFp8TallMinRows"/> rows it
    /// uses 128-row tiles where they pad no more rows than 64-row tiles (half the weight passes; each output still sums its K
    /// blocks in the same order, so the result does not depend on the tile height; <c>SHARPLIGHT_FP8_DUAL_TALL=0</c> keeps
    /// 64-row tiles). Returns false when it does not apply. With <c>cudaEpilogue</c> (the CUDA C++ form of the epilogue,
    /// <see cref="DualEpilogueCudaSource"/>) the hand-written dual GEMM serves batches from <see cref="NativeGemmMinRows"/> rows;
    /// <c>serves</c> tells the row counts the emitted launches cover.
    /// </summary>
    /// <summary>
    /// True (<c>SHARPLIGHT_FP8_DUAL_SPLIT=1</c>) when the dual fp8 GEMM splits K at decode sizes (one token tile row) by wave fill
    /// (<see cref="WaveFillSplit"/>), reducing in the last workgroup of each tile (splitk.slang).
    /// </summary>
    private bool DualFp8Split => Fp8FusedSplitK && DiagnosticSwitches.Fp8DualSplit;

    /// <summary>
    /// True when the fp8 GEMM's split-K launches reduce in their last workgroups (splitk.slang): on targets that compile CUDA C++ only.
    /// On the NVIDIA Vulkan driver the fused form of gemm_fp8 returned wrong outputs (a few elements, or NaN with cached SPIR-V) in
    /// <c>FusionTests.SiblingProjections</c> while gemm_cm2's fused form passed; Vulkan keeps the reduction launch there (benchmarks.md
    /// section 61).
    /// </summary>
    private bool Fp8FusedSplitK => DiagnosticSwitches.SplitKFused && _compiler.CompilesCuda(_opt.Target);

    private bool EmitDualFp8Gemm(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef weightA,
        TensorRef weightB, TensorRef y, long n, long k, string epilogueSource, string? cudaEpilogue, out Func<long, bool> serves, Value? output = null,
        bool uniformScales = false)
    {
        serves = _ => false;
        long minRows = Fp8GemmMinRows(weightA.Type, n, k);
        long rowsMax = x.Shape[0] * rowsPerLead;
        if (!DiagnosticSwitches.Fp8DualGemm || !UseFp8Gemm(weightA.Type, k) || UseMmq(weightA.Type, n, k) || rowsMax < minRows
            || MmqLayout(64)[3] % 2 != 0)
        {
            return false;
        }

        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.Fp8);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        // From NativeGemmMinRows rows the hand-written dual GEMM serves every batch (no K split needed at that size).
        long xcap = DenseScaleCapacityOf(quantized);
        long nativeMin = cudaEpilogue is not null && UseNativeFp8Gemm(weightA.Type, n, k) && weightB.Type == weightA.Type && xcap > 0
            ? Math.Max(minRows, NativeFp8MinRows(decode: true)) : long.MaxValue;
        bool Generated(long r) => r < nativeMin && DualFp8Serves(r, weightA.Type, n, k);
        serves = r => r >= nativeMin || Generated(r);
        bool tall = DiagnosticSwitches.Fp8DualGemmTall && rowsMax > DualFp8AlwaysMaxRows && MmqLayout(128)[3] % 2 == 0;
        int TileRows(long r) => tall && r > DualFp8TallMinRows && (r + 127) / 128 * 2 == (r + 63) / 64 ? 128 : (int)DualFp8AlwaysMaxRows;
        var variants = new List<KernelVariant>();
        // From storeMin rows the pair stores the E4M3 rows of its readers instead (its own launch below).
        long storeMin = rowsMax >= nativeMin && NativeFp8TensorCopies && UseNativeFp8Gemm(weightA.Type, n, k)
            ? Fp8StoreMinRows(output, y, n, cudaEpilogue) : long.MaxValue;
        if (storeMin > rowsMax)
        {
            storeMin = long.MaxValue;
        }

        if (rowsMax >= nativeMin)
        {
            variants.AddRange(NativeFp8Variants(n, k, cudaEpilogue, xcap, nativeMin, rowsMax, rows, uniformScales)
                .Select(v => v with { When = b => v.When(b) && rows(b) < storeMin }));
        }

        // One token tile row (decode): optionally K split with the reduction in the last workgroups (DualFp8Split).
        int outputs64 = MmqColumns((int)DualFp8AlwaysMaxRows) / 2;
        int DualSplit(long r) => DualFp8Split && r <= DualFp8AlwaysMaxRows && Generated(r)
            ? WaveFillSplit(Grid.Div(n, outputs64), k / 128, GemmWaveTiles) : 1;
        foreach (int tileRows in tall ? new[] { 64, 128 } : [(int)DualFp8AlwaysMaxRows])
        {
            int outputs = MmqColumns(tileRows) / 2;
            var macros = Fp8GemmMacros(weightA.Type, n, k, tileRows, 1);
            macros["DUAL"] = "1";
            macros["KERNEL_GEMM_FP8"] = "1";
            var kernel = _compiler.Compile("gemm_fp8_dual.slang", epilogueSource, "gemm_fp8", _opt.Target, macros, _opt.Cache);
            variants.Add(new KernelVariant(b => Generated(rows(b)) && TileRows(rows(b)) == tileRows && DualSplit(rows(b)) == 1, kernel,
                b => new Grid(Grid.Div(rows(b), tileRows), Grid.Div(n, outputs))));
        }

        Emit($"{label} fp8", nodes, variants, [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [quantized, weightA, weightB], [y]);
        if (storeMin != long.MaxValue)
        {
            // The readers' E4M3 rows of the output (as AddFp8CopyOutput allocates them for a norm), written by this launch from storeMin rows;
            // their quantize_fp8 launches run for smaller steps only (NormFp8Copy).
            long outRows = y.Shape.Aggregate(1L, (a, d) => a * d) / n;
            long allocatedRows = Fp8ActivationRows(outRows, n + n / 32);
            var allocation = new object();
            long offset = _alloc.Allocate(allocation, allocatedRows * (n + n / 32));
            if (!_rootScratch.TryGetValue(output!, out var list))
            {
                _rootScratch[output!] = list = [];
            }

            list.Add(allocation);
            long[] shape = [allocatedRows, n + n / 32];
            var copy = new TensorRef(_arena, offset, DataType.U8, shape, TensorRef.DenseStrides(shape), y.Symbol);
            NoteDenseScales(copy, outRows);
            _fp8Copies[output!] = (copy, storeMin);
            long qcap = DenseScaleCapacityOf(copy);
            Emit($"{label} fp8 e4m3 out", nodes, [NativeFp8TmaVariant(n, k, cudaEpilogue, xcap, b => rows(b) >= storeMin, rows, qcap, uniformScales)],
                [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(copy), TokenArg(x, rowsPerLead)],
                [quantized, weightA, weightB], [copy]);
        }
        var splits = new SortedSet<int>();
        long scratchBytes = 0;
        for (long r = minRows; r <= Math.Min(rowsMax, DualFp8AlwaysMaxRows); r++)
        {
            if (DualSplit(r) is > 1 and var s)
            {
                splits.Add(s);
                scratchBytes = Math.Max(scratchBytes, s * r * 2 * n * 4);
            }
        }

        if (splits.Count > 0)
        {
            var scratch = SplitScratch(scratchBytes);
            var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
            var arrivals = SplitArrivals(Grid.Div(n, outputs64));
            var splitVariants = splits.Select(s =>
            {
                var macros = Fp8GemmMacros(weightA.Type, n, k, (int)DualFp8AlwaysMaxRows, s);
                macros["DUAL"] = "1";
                macros["KERNEL_GEMM_FP8"] = "1";
                macros["SPLITK_FUSED"] = "1";
                var kernel = _compiler.Compile("gemm_fp8_dual.slang", epilogueSource, "gemm_fp8", _opt.Target, macros, _opt.Cache);
                return new KernelVariant(b => DualSplit(rows(b)) == s, kernel, b => new Grid(Grid.Div(rows(b), DualFp8AlwaysMaxRows), Grid.Div(n, outputs64), (uint)s));
            });
            Emit($"{label} fp8 split-k", nodes, [.. splitVariants],
                [Pointer(quantized), Pointer(weightA), Pointer(weightB), new PointerArg(scratch, 0), Pointer(y), Pointer(arrivals), TokenArg(x, rowsPerLead)],
                [quantized, weightA, weightB], [partial, y, arrivals]);
        }

        return true;
    }

    private Dictionary<string, string> Fp8GemmMacros(DataType type, long n, long k, int tileRows, int split)
    {
        var layout = MmqLayout(tileRows);
        if (layout[0] * layout[2] * 16 != tileRows)
        {
            throw new NotSupportedException($"fp8 GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles.");
        }

        var macros = Macros(("KERNEL_GEMM_FP8", 1), ("WBLOCK", type.BlockSize), ("K", k), ("N", n), ("Y_STRIDE", n), ("ROW_BYTES", type.ByteSize(k) + "ull"),
            ("F8_WM", layout[0]), ("F8_WN", layout[1]), ("F8_FM", layout[2]), ("F8_FN", layout[3]), ("SPLITK", split),
            ("F8_GENERIC_LAYOUT", DiagnosticSwitches.MmqGenericLayout),
            ("F8_SPLIT", type == ShaderTypes.F8E4M3_B128R || type == ShaderTypes.F8E4M3_B32R || type == ShaderTypes.F8E4M3_B128G ? 1 : 0));
        if (type == ShaderTypes.F8E4M3_B128G)
        {
            macros["F8_GROUPED"] = "1";
        }

        foreach (var (name, value) in DiagnosticSwitches.GemmExtraMacros)
        {
            macros[name] = value;
        }

        return macros;
    }
}
