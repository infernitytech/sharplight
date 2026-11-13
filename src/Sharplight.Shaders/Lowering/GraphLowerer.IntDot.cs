using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Largest activation row count served by the integer-dot matvec.</summary>
    private static readonly long IntDotMaxRows = DiagnosticSwitches.IntDotMaxRows;

    /// <summary>Threads an int-dot matvec launch aims for (splitting K inside workgroups when there are few weight rows).</summary>
    private const long IntDotTargetThreads = 128 * 1024;

    /// <summary>Activation-row buckets the integer-dot matvec is specialized for (each serves rows up to its size).</summary>
    private static readonly int[] s_intDotColumnBuckets = [.. new[] { 1, 2, 4, 8, 16, 32 }.Where(c => c / 2 < IntDotMaxRows)];

    /// <summary>
    /// q8 copies of activations, shared by every int-dot matvec that reads the same rows (e.g. the Q, K and V projections),
    /// keyed by the activations' root value and layout, the row range the quantization runs for and the launch condition
    /// it was emitted under.
    /// </summary>
    private readonly Dictionary<(Value Root, long ByteOffset, long XStride, long K, long MinRows, long MaxRows, object? Condition), TensorRef> _quantizedActivations = [];

    /// <summary>Arena allocations that live as long as a root value (released by <see cref="ReleaseRootScratch"/>).</summary>
    private readonly Dictionary<Value, List<object>> _rootScratch = [];

    /// <summary>
    /// True when the integer-dot matvec (<c>mmvq.slang</c>, <see cref="DeviceFeatures.IntDot"/>) handles weights of
    /// <paramref name="type"/> with reduction length <paramref name="k"/>: q8-quantized activations, int8 dot products.
    /// </summary>
    private bool UseIntDotMatvec(DataType type, long k)
        => _opt.Has(DeviceFeatures.IntDot) && _opt.Has(DeviceFeatures.SubgroupOps) && k % 256 == 0
           && (type == ShaderTypes.Q8_0R || type == DataType.Q4_K || type == DataType.Q5_K || type == DataType.Q6_K);

    /// <summary>Rows (inclusive) from which the int-dot path serves a matmul; single rows may stay on the f32 matvec.</summary>
    private static long IntDotMinRows(DataType type) => type == ShaderTypes.Q8_0R && !DiagnosticSwitches.IntDotDecodeQ8 ? 2 : 1;

    /// <summary>
    /// Emits the activation quantization and the integer-dot matvec launches of a matmul (or of a dual matmul when
    /// <paramref name="weights"/> holds two matrices), both conditioned on the row count being in the int-dot range.
    /// <paramref name="compile"/> builds the matvec entry from its macros (library or generated dual source).
    /// </summary>
    private void EmitIntDotMatvec(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef[] weights, TensorRef y,
        long n, long k, Func<Dictionary<string, string>, CompiledKernel> compile)
    {
        var w = weights[0];
        long minRows = IntDotMinRows(w.Type);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        bool InRange(Bindings b) => rows(b) >= minRows && rows(b) <= IntDotMaxRows;
        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, IntDotMaxRows);

        int format = WeightFormat(w.Type)!.Value;
        long chunks = k / IntDotChunkValues(w.Type);
        var variants = new List<KernelVariant>();
        foreach (int columns in s_intDotColumnBuckets)
        {
            if (columns < minRows)
            {
                continue;
            }

            var layout = IntDotLayout(columns, n, chunks);
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
    /// The q8 copy of <paramref name="x"/> for the int-dot matvecs, made by a quantization launch that runs while x has
    /// <paramref name="minRows"/> to <paramref name="maxRows"/> rows; reused when an earlier matvec over the same rows
    /// (same root value, layout, row range and launch condition) already made one.
    /// </summary>
    private TensorRef QuantizedActivations(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, long k, long minRows,
        long maxRows)
    {
        var root = RootOf(source);
        // Shared only while x is the root's own arena memory: copies made for this matmul are temporaries.
        bool shareable = x.Buffer == _arena && ReferenceEquals(x, Ref(source)) && _refs.TryGetValue(root, out var rootRef) && rootRef.Buffer == _arena;
        var cacheKey = (root, x.ByteOffset, xStride, k, minRows, maxRows, (object?)_launchCondition);
        if (shareable && _quantizedActivations.TryGetValue(cacheKey, out var cached))
        {
            return cached;
        }

        long rowsMax = x.Shape[0] * rowsPerLead;
        long[] shape = [rowsMax, k + k / 4];
        TensorRef quantized;
        if (shareable)
        {
            var allocation = new object();
            long offset = _alloc.Allocate(allocation, DataType.U8.ByteSize(rowsMax * shape[1]));
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

        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        Emit($"{label} x→q8", nodes,
            [new KernelVariant(b => rows(b) >= minRows && rows(b) <= maxRows, Lib("mmvq.slang", "quantize_q8_1", Macros(("K", k), ("X_STRIDE", xStride), ("KERNEL_QUANTIZE_Q8_1", 1))),
                b => new Grid(Grid.Div(rows(b) * (k / 4), 256)))],
            [Pointer(x), Pointer(quantized), TokenArg(x, rowsPerLead)],
            [x], [quantized]);
        return quantized;
    }

    /// <summary>
    /// Emits the int-dot form of the per-item expert matvec of a MatMulId (decode): quantizes the activation rows and runs
    /// <c>moe_mmvq</c> while there are at most <paramref name="maxItems"/> (token, slot) items.
    /// </summary>
    private void EmitIntDotExpertMatvec(string label, Node n, TensorRef x, long xStride, long rowsPerLead, TensorRef w, TensorRef ids, TensorRef y,
        Dictionary<string, string> macros, long maxItems, long slots)
    {
        long N = w.Shape[1];
        long K = w.Shape[2];
        // Activation rows: one per token, or one per (token, slot) item when x is per slot (rowsPerLead = slots).
        long maxRows = Math.Max(1, rowsPerLead == 1 ? maxItems / slots : maxItems);
        var quantized = QuantizedActivations(label, [n], n.Inputs[0], x, xStride, rowsPerLead, K, 1, maxRows);
        Func<Bindings, long> items = b => Rows(b, ids) * slots;
        int lanesPerRow = Math.Min(_opt.SubgroupSize, DiagnosticSwitches.IntDotLayout is { Length: > 2 } forced ? forced[2] : _opt.SubgroupSize);
        int rowsPerGroup = DiagnosticSwitches.IntDotLayout is { } layout ? layout[0] : 1;
        int subgroups = DiagnosticSwitches.IntDotLayout is { } layout2 ? layout2[1] : 4;
        long rowsPerWorkgroup = subgroups * (_opt.SubgroupSize / lanesPerRow) * rowsPerGroup;
        var m = new Dictionary<string, string>(macros)
        {
            ["MQ_COLS"] = "1",
            ["MQ_ROWS"] = rowsPerGroup.ToString(CultureInfo.InvariantCulture),
            ["MQ_SG"] = subgroups.ToString(CultureInfo.InvariantCulture),
            ["MQ_LPR"] = lanesPerRow.ToString(CultureInfo.InvariantCulture),
            ["KERNEL_MOE_MMVQ"] = "1",
        };
        Emit(label + " int-dot", [n],
            [new KernelVariant(b => items(b) <= maxItems && Rows(b, x) * rowsPerLead <= maxRows, Lib("mmvq.slang", "moe_mmvq", m),
                b => new Grid(Grid.Div(N, rowsPerWorkgroup), (uint)Math.Max(1, items(b))))],
            [Pointer(quantized), Pointer(w), Pointer(ids), Pointer(y), TokenArg(ids)],
            [quantized, w, ids], [y]);
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
    private Func<Bindings, bool> OutsideIntDotRange(TensorRef x, long rowsPerLead, DataType weightType)
    {
        long minRows = IntDotMinRows(weightType);
        return b => Rows(b, x) * rowsPerLead is var r && (r < minRows || r > IntDotMaxRows);
    }
}
