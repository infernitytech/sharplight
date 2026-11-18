using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Largest activation row count served by the integer-dot matvec for weights of <paramref name="type"/> and depth
    /// <paramref name="k"/>: 8, above which the skinny GEMM or gemm_cm2 win (K-quants: Mistral-7B Q4_K_M batch 16: 12 vs 18 ms
    /// per step; Q8_0 [17408→5120] at 12 rows: 100 µs int-dot vs 72 µs skinny), and 16 for Q8_0 where the f16 tensor-core
    /// path is not usable. With the direct skinny GEMM, Q8_0 at 5-8 rows would be faster there (Blackwell, Qwen3.8-27B Q8_0
    /// decode step at 5 / 6 / 8 rows: 27.7 / 28.1 / 28.7 ms int-dot, 24.0 / 25.6 / 25.3 ms with <c>SHARPLIGHT_MMVQ_MAX_ROWS=4</c>),
    /// but that changes the rounding of those rows and with it greedy MTP output, so it stays a diagnostic setting.
    /// </summary>
    private long IntDotMaxRows(DataType type, long k) => DiagnosticSwitches.IntDotMaxRows
        ?? ((type == ShaderTypes.Q8_0R || type == DataType.Q8_0) && !(DiagnosticSwitches.SkinnyGemm && CoopUsable(type, k)) ? 16 : 8);

    /// <summary>
    /// First row count the skinny GEMM serves for a matmul whose small batches take the int-dot matvec: right above the int-dot
    /// range when that ends below <see cref="MatvecMaxRows"/> (<c>SHARPLIGHT_MMVQ_MAX_ROWS</c>) and the skinny GEMM covers the
    /// rows up to it for <paramref name="n"/> output rows, else above both the int-dot and the f32 matvec ranges.
    /// </summary>
    private long IntDotSkinnyMinRows(DataType type, long k, long n)
    {
        long cap = IntDotMaxRows(type, k);
        return Math.Max((cap < MatvecMaxRows && SkinnyMaxRowsFor(n) >= MatvecMaxRows ? cap : Math.Max(MatvecMaxRows, cap)) + 1, InvariantRows + 1);
    }

    /// <summary>Threads an int-dot matvec launch aims for (splitting K inside workgroups when there are few weight rows).</summary>
    private const long IntDotTargetThreads = 128 * 1024;

    /// <summary>Activation-row buckets the integer-dot matvec is specialized for (each serves rows up to its size).</summary>
    private static readonly int[] s_intDotColumnBuckets = [1, 2, 4, 8, 16, 32];

    /// <summary>
    /// q8 copies of activations, shared by every int-dot matvec that reads the same rows (e.g. the Q, K and V projections),
    /// keyed by the activations' root value and layout, the row range the quantization runs for and the launch condition
    /// it was emitted under.
    /// </summary>
    private readonly Dictionary<(Value Root, long ByteOffset, long XStride, long K, long MinRows, long MaxRows, ActivationCode Code, object? Condition), TensorRef> _quantizedActivations = [];

    /// <summary>Arena allocations that live as long as a root value (released by <see cref="ReleaseRootScratch"/>).</summary>
    private readonly Dictionary<Value, List<object>> _rootScratch = [];

    /// <summary>
    /// True when the integer-dot matvec (<c>mmvq.slang</c>, <see cref="DeviceFeatures.IntDot"/>) handles weights of
    /// <paramref name="type"/> with reduction length <paramref name="k"/>: q8-quantized activations, int8 dot products.
    /// </summary>
    private bool UseIntDotMatvec(Node n, DataType type, long k)
        => QuantizedActivationsAllowed(n) && _opt.Has(DeviceFeatures.IntDot) && _opt.Has(DeviceFeatures.SubgroupOps)
           && (k % 256 == 0 && (type == ShaderTypes.Q8_0R || type == DataType.Q4_K || type == DataType.Q5_K || type == DataType.Q6_K)
               || k % 32 == 0 && DiagnosticSwitches.IntDotBlockFormats && s_intDotBlockFormats.Contains(type));

    /// <summary>
    /// False for matmuls marked <see cref="StandardOps.F32ActivationsAttribute"/> (unless the compile allows reduced
    /// precision): their activations must not be quantized to int8 (int-dot matvec, int8 GEMM).
    /// </summary>
    private bool QuantizedActivationsAllowed(Node n)
        => _opt.Compile.AllowReducedPrecision || !n.Attributes.Get(StandardOps.F32ActivationsAttribute, false);

    /// <summary>
    /// Block formats of the integer-dot matvec read 32 values per chunk: the 32-value GGUF blocks, and the group-scaled formats of
    /// quantized Hugging Face checkpoints (integer codes with a zero point, NVFP4 with and without a global scale).
    /// </summary>
    private static readonly HashSet<DataType> s_intDotBlockFormats =
    [
        DataType.Q8_0, DataType.Q4_0, DataType.Q5_0, DataType.IQ4_NL, DataType.MXFP4, ShaderTypes.MXFP4R,
        DataType.Q4Z_G32, DataType.Q4Z_G128, DataType.Q8Z_G32, DataType.Q8Z_G128, DataType.NVFP4, DataType.NVFP4_GS,
    ];

    /// <summary>Rows (inclusive) from which the int-dot path serves a matmul; single rows may stay on the f32 matvec.</summary>
    private static long IntDotMinRows(DataType type) => (type == ShaderTypes.Q8_0R || type == DataType.Q8_0) && !DiagnosticSwitches.IntDotDecodeQ8 ? 2 : 1;

    /// <summary>
    /// Emits the activation quantization and the integer-dot matvec launches of a matmul (or of a dual matmul when
    /// <paramref name="weights"/> holds two matrices), both conditioned on the row count being in the int-dot range.
    /// <paramref name="compile"/> builds the matvec entry from its macros (library or generated dual source).
    /// </summary>
    private void EmitIntDotMatvec(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef[] weights, TensorRef y,
        long n, long k, Func<Dictionary<string, string>, CompiledKernel> compile)
    {
        var w = weights[0];
        long minRows = IntDotFirstRow(w.Type);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        long maxRows = IntDotMaxRows(w.Type, k);
        bool InRange(Bindings b) => rows(b) >= minRows && rows(b) <= maxRows;
        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, maxRows);

        int format = WeightFormat(w.Type)!.Value;
        long chunks = k / IntDotChunkValues(w.Type);
        var variants = new List<KernelVariant>();
        foreach (int columns in s_intDotColumnBuckets)
        {
            if (columns < minRows || columns / 2 >= maxRows)
            {
                continue;
            }

            // Buckets with row-invariant rows keep the single-row layout (same K split per row).
            var layout = IntDotLayout(InvariantBucket(columns) ? 1 : columns, n, chunks);
            var macros = Macros(("WFMT", format), ("K", k), ("N", n), ("Y_STRIDE", n), ("ROW_BYTES", w.Type.ByteSize(k) + "ull"),
                ("MQ_COLS", columns), ("MQ_ROWS", layout.Rows), ("MQ_SG", layout.Subgroups), ("MQ_LPR", layout.LanesPerRow),
                ("DUAL", weights.Length - 1), ("KERNEL_MMVQ", 1));
            int bucket = columns;
            long rowsPerWorkgroup = layout.LanesPerRow < _opt.SubgroupSize
                ? layout.Subgroups * (_opt.SubgroupSize / layout.LanesPerRow) * layout.Rows
                : layout.Rows;
            variants.Add(new KernelVariant(b => InRange(b) && rows(b) <= bucket, compile(macros), _ => new Grid(Grid.Div(n, rowsPerWorkgroup))));
        }

        Emit($"{label} int-dot", nodes, variants,
            [Pointer(quantized), .. weights.Select(Pointer), Pointer(y), TokenArg(x, rowsPerLead)],
            [quantized, .. weights], [y]);
    }

    /// <summary>
    /// The quantized copy of <paramref name="x"/> (q8_1 for the int8 kernels, E4M3 for the fp8 GEMM, f16 rows for the f16
    /// tensor-core GEMMs), made by a quantization launch that runs while x has <paramref name="minRows"/> to <paramref name="maxRows"/> rows; reused when
    /// an earlier matmul over the same rows (same root value, layout, row range, encoding and launch condition) already made one.
    /// </summary>
    private TensorRef QuantizedActivations(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, long k, long minRows,
        long maxRows, ActivationCode code = ActivationCode.Q8_1)
    {
        var result = QuantizedActivationsCore(label, nodes, source, x, xStride, rowsPerLead, k, minRows, maxRows, code);
        NoteCopyReader(RootOf(source), result, nodes, minRows, maxRows, x.Shape[0] * rowsPerLead);
        return result;
    }

    private TensorRef QuantizedActivationsCore(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, long k, long minRows,
        long maxRows, ActivationCode code)
    {
        var root = RootOf(source);
        // Shared only while x is the root's own arena memory (not a concatenation computed in slices: its slices and views
        // share one root): copies made for this matmul are temporaries.
        bool shareable = x.Buffer == _arena && ReferenceEquals(x, Ref(source)) && !_concatSliceRoots.Contains(root)
                         && _refs.TryGetValue(root, out var rootRef) && rootRef.Buffer == _arena;
        var cacheKey = (root, x.ByteOffset, xStride, k, minRows, maxRows, code, (object?)_launchCondition);
        if (shareable && _quantizedActivations.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        if (code == ActivationCode.F16 && shareable && NormHalfCopy(root, x, xStride, k, minRows) is { } normCopy)
        {
            if (normCopy.RangeScales is { } scales)
            {
                _rangeScales[normCopy.Copy] = scales;
            }

            // The producer writes the copy for some steps only; the others convert into the same buffer.
            if (normCopy.Written is { } written)
            {
                var conversionKey = (root, x.ByteOffset, xStride, k, minRows, maxRows, code, (object?)_launchCondition);
                if (_quantizedActivations.TryAdd(conversionKey, normCopy.Copy))
                {
                    EmitActivationConversion(label, nodes, x, xStride, rowsPerLead, k, minRows, maxRows, code, normCopy.Copy, written);
                }
            }
            else if (maxRows > normCopy.MaxRows)
            {
                long from = Math.Max(minRows, normCopy.MaxRows + 1);
                var extensionKey = (root, x.ByteOffset, xStride, k, from, maxRows, code, (object?)_launchCondition);
                if (_quantizedActivations.TryAdd(extensionKey, normCopy.Copy))
                {
                    EmitActivationConversion(label, nodes, x, xStride, rowsPerLead, k, from, maxRows, code, normCopy.Copy);
                }
            }

            return normCopy.Copy;
        }

        if (code == ActivationCode.Fp8 && shareable && NormFp8Copy(root, x, xStride, k) is { } fp8Copy)
        {
            // The norm writes the rows from fp8Copy.MinRows rows on; smaller steps quantize into the same rows.
            if (_quantizedActivations.TryAdd(cacheKey, fp8Copy.Copy))
            {
                EmitActivationConversion(label, nodes, x, xStride, rowsPerLead, k, minRows, maxRows, code, fp8Copy.Copy, rows => rows >= fp8Copy.MinRows);
            }

            return fp8Copy.Copy;
        }

        long rowsMax = x.Shape[0] * rowsPerLead;
        long[] shape = [rowsMax, code switch { ActivationCode.Fp8 => k + k / 32, ActivationCode.F16 => 2 * k, ActivationCode.SplitF16 => 4 * k + 16, ActivationCode.ScaledF16 => 2 * k + 16, _ => k + k / 4 }];
        if (code == ActivationCode.F16 && DiagnosticSwitches.F16RangeGuard)
        {
            // A row-ranged copy (x is no norm output, so its rows may exceed f16's range): the scale words follow the rows.
            shape[0] += RangeScaleRows(rowsMax, k);
        }

        TensorRef quantized;
        if (shareable)
        {
            var allocation = new object();
            long offset = _alloc.Allocate(allocation, DataType.U8.ByteSize(shape[0] * shape[1]));
            quantized = new TensorRef(_arena, offset, DataType.U8, shape, TensorRef.DenseStrides(shape), x.Symbol);
            if (!_rootScratch.TryGetValue(root, out var list))
            {
                _rootScratch[root] = list = [];
            }

            list.Add(allocation);
            _quantizedActivations[cacheKey] = quantized;
        }
        else
        {
            quantized = Temp(DataType.U8, shape, x.Symbol);
        }

        if (code == ActivationCode.F16 && DiagnosticSwitches.F16RangeGuard)
        {
            _rangeScales[quantized] = rowsMax * k * 2;
        }

        EmitActivationConversion(label, nodes, x, xStride, rowsPerLead, k, minRows, maxRows, code, quantized);
        return quantized;
    }

    /// <summary>
    /// The launch converting x into <paramref name="quantized"/> while x has <paramref name="minRows"/> to <paramref name="maxRows"/> rows,
    /// except for the row counts where <paramref name="written"/> holds (the producer of x wrote the copy).
    /// </summary>
    private void EmitActivationConversion(string label, Node[] nodes, TensorRef x, long xStride, long rowsPerLead, long k, long minRows, long maxRows,
        ActivationCode code, TensorRef quantized, Func<long, bool>? written = null)
    {
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        bool ranged = code == ActivationCode.F16 && _rangeScales.ContainsKey(quantized);
        var kernel = code switch
        {
            ActivationCode.Fp8 => Lib("gemm_fp8.slang", "quantize_fp8", Macros(("K", k), ("X_STRIDE", xStride), ("KERNEL_QUANTIZE_FP8", 1))),
            ActivationCode.F16 when ranged => Lib("data.slang", "to_f16_ranged", Macros(("K", k), ("X_STRIDE", xStride), ("Y_RS", _rangeScales[quantized] + "ull"))),
            ActivationCode.F16 => Lib("data.slang", "to_f16", Macros(("K", k), ("X_STRIDE", xStride))),
            ActivationCode.SplitF16 => Lib("data.slang", "split_rows", Macros(("K", k), ("X_STRIDE", xStride))),
            ActivationCode.ScaledF16 => Lib("data.slang", "split_rows", Macros(("K", k), ("X_STRIDE", xStride), ("SPLIT_PARTS", 1))),
            _ => Lib("mmvq.slang", "quantize_q8_1", Macros(("K", k), ("X_STRIDE", xStride), ("KERNEL_QUANTIZE_Q8_1", 1))),
        };
        long valuesPerThread = code == ActivationCode.F16 ? 8 : 4;
        Func<Bindings, Grid> grid = code is ActivationCode.SplitF16 or ActivationCode.ScaledF16 || ranged
            ? b => new Grid((uint)rows(b))
            : b => new Grid(Grid.Div(rows(b) * (k / valuesPerThread), 256));
        string what = code switch { ActivationCode.Fp8 => "fp8", ActivationCode.F16 => ranged ? "ranged f16" : "f16", ActivationCode.SplitF16 => "split f16", ActivationCode.ScaledF16 => "scaled f16", _ => "q8" };
        Emit($"{label} x→{what}", nodes,
            [new KernelVariant(b => rows(b) >= minRows && rows(b) <= maxRows && !(written?.Invoke(rows(b)) ?? false), kernel, grid)],
            [Pointer(x), Pointer(quantized), TokenArg(x, rowsPerLead)],
            [x], [quantized]);
    }

    /// <summary>
    /// Emits the int-dot form of the per-item expert matvec of a MatMulId (decode): quantizes the activation rows and runs
    /// <c>moe_mmvq</c> while there are at most <paramref name="maxItems"/> (token, slot) items.
    /// </summary>
    private void EmitIntDotExpertMatvec(string label, Node n, TensorRef x, long xStride, long rowsPerLead, TensorRef w, TensorRef ids, TensorRef y,
        Dictionary<string, string> macros, long maxItems, long slots)
        => EmitIntDotExpertMatvec(label, [n], n, x, xStride, rowsPerLead, [w], ids, y, macros, maxItems, slots, [], m => Lib("mmvq.slang", "moe_mmvq", m));

    /// <summary>
    /// The int-dot expert matvec over one or two expert matrices (<paramref name="weights"/>; with two, the kernel source that
    /// <paramref name="compile"/> builds defines the epilogue <c>MOE_EPI</c>, which may read the per-expert <paramref name="tables"/>).
    /// </summary>
    private void EmitIntDotExpertMatvec(string label, Node[] nodes, Node n, TensorRef x, long xStride, long rowsPerLead, TensorRef[] weights, TensorRef ids,
        TensorRef y, Dictionary<string, string> macros, long maxItems, long slots, TensorRef[] tables, Func<Dictionary<string, string>, CompiledKernel> compile,
        bool slotSum = false)
    {
        var w = weights[0];
        long N = w.Shape[1];
        long K = w.Shape[2];
        // Activation rows: one per token, or one per (token, slot) item when x is per slot (rowsPerLead = slots).
        long maxRows = Math.Max(1, rowsPerLead == 1 ? maxItems / slots : maxItems);
        var quantized = QuantizedActivations(label, nodes, n.Inputs[0], x, xStride, rowsPerLead, K, 1, maxRows);
        Func<Bindings, long> items = b => Rows(b, ids) * slots;
        int lanesPerRow = Math.Min(_opt.SubgroupSize, DiagnosticSwitches.IntDotLayout is { Length: > 2 } forced ? forced[2] : _opt.SubgroupSize);
        int rowsPerGroup = DiagnosticSwitches.IntDotLayout is { } layout ? layout[0] : 1;
        int subgroups = DiagnosticSwitches.IntDotLayout is { } layout2 ? layout2[1] : 4;
        if (slotSum)
        {
            subgroups = (int)slots;   // one subgroup per slot
        }

        long rowsPerWorkgroup = (slotSum ? 1 : subgroups) * (_opt.SubgroupSize / lanesPerRow) * rowsPerGroup;
        var m = new Dictionary<string, string>(macros)
        {
            ["MQ_COLS"] = "1",
            ["MQ_ROWS"] = rowsPerGroup.ToString(CultureInfo.InvariantCulture),
            ["MQ_SG"] = subgroups.ToString(CultureInfo.InvariantCulture),
            ["MQ_LPR"] = lanesPerRow.ToString(CultureInfo.InvariantCulture),
            ["KERNEL_MOE_MMVQ"] = "1",
            ["DUAL"] = (weights.Length - 1).ToString(CultureInfo.InvariantCulture),
            ["SLOT_SUM"] = slotSum ? "1" : "0",
        };
        Emit(label + " int-dot", nodes,
            [new KernelVariant(b => items(b) <= maxItems && Rows(b, x) * rowsPerLead <= maxRows, compile(m),
                b => new Grid(Grid.Div(N, rowsPerWorkgroup), (uint)Math.Max(1, slotSum ? Rows(b, ids) : items(b))))],
            [Pointer(quantized), .. weights.Select(Pointer), Pointer(ids), Pointer(y), .. tables.Select(Pointer), TokenArg(ids)],
            [quantized, .. weights, ids, .. tables], [y]);
    }

    /// <summary>Frees the arena allocations tied to <paramref name="root"/> and forgets the q8 copies made of it.</summary>
    private void ReleaseRootScratch(Value root)
    {
        if (!_rootScratch.Remove(root, out var allocations))
        {
            return;
        }

        foreach (var allocation in allocations)
        {
            _alloc.Free(allocation);
        }

        _halfCopies.Remove(root);
        foreach (var key in _quantizedActivations.Keys.Where(key => key.Root == root).ToList())
        {
            _quantizedActivations.Remove(key);
        }
    }

    /// <summary>Values per thread work item of the int-dot matvec for a weight format (see <c>CHUNK_VALUES</c>).</summary>
    private static long IntDotChunkValues(DataType type) => type == ShaderTypes.Q8_0R ? 16 : type == DataType.Q6_K ? 64 : 32;

    /// <summary>
    /// Weight rows per workgroup and subgroups per workgroup of the int-dot matvec: enough threads along K to keep
    /// several chunks per thread in flight, and more rows per workgroup for more activation rows (amortizes x loads).
    /// </summary>
    private (int Rows, int Subgroups, int LanesPerRow) IntDotLayout(int columns, long n, long chunks)
    {
        if (DiagnosticSwitches.IntDotLayout is { } forced)
        {
            return (forced[0], forced[1], forced.Length > 2 ? Math.Min(forced[2], _opt.SubgroupSize) : _opt.SubgroupSize);
        }

        // Several weight rows per workgroup share each activation load (2-4x fewer), when there are rows enough to fill the GPU.
        int rowsPerGroup = n < 4096 ? 1 : columns >= 4 && n % 4 == 0 ? 4 : columns >= 2 && n % 2 == 0 ? 2 : 1;
        long workgroups = (n + rowsPerGroup - 1) / rowsPerGroup;
        int subgroups = 1;
        while (subgroups < 8 && workgroups * subgroups * _opt.SubgroupSize < IntDotTargetThreads && chunks >= subgroups * 2 * _opt.SubgroupSize)
        {
            subgroups *= 2;
        }

        return (rowsPerGroup, subgroups, _opt.SubgroupSize);
    }

    /// <summary>The launch-condition guard that keeps the f32 matvec/GEMM launches off the int-dot row range.</summary>
    private Func<Bindings, bool> OutsideIntDotRange(TensorRef x, long rowsPerLead, DataType weightType, long k)
    {
        long minRows = IntDotFirstRow(weightType);
        long maxRows = IntDotMaxRows(weightType, k);
        return b => Rows(b, x) * rowsPerLead is var r && (r < minRows || r > maxRows);
    }
}
