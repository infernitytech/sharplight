using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Parallel top-k lowering (kernels in topk.slang).</summary>
public sealed partial class GraphLowerer
{
    /// <summary>Threads per workgroup of the per-row top-k kernel.</summary>
    private const int TopKWorkgroup = 256;

    /// <summary>Largest k handled by the subgroup kernel (each lane keeps k candidates in registers).</summary>
    private const long TopKSubgroupMaxK = 16;

    /// <summary>
    /// Top-k along one axis. Small k with subgroup operations: one subgroup per row. Otherwise one workgroup per row
    /// (radix select + bitonic sort in shared memory, portable), and for k beyond its shared-memory sort the serial kernel.
    /// </summary>
    /// <summary>
    /// Router top-k units with the softmax they read (and the normalization of the selected weights, when the values only feed it): the
    /// softmax, the ReduceSum, the Scale and the Div computed in the top-k launch (<see cref="FuseRouterTopK"/>). Selected: the softmax
    /// reads the top-k values instead (top-k of the logits first).
    /// </summary>
    private readonly Dictionary<Node, (Node Softmax, bool Selected, (Node Reduce, Node Scale, Node Div)? Norm)> _routerTopK = [];

    /// <summary>
    /// Finds mixture-of-experts routers: a TopK (k up to <see cref="TopKSubgroupMaxK"/>, largest, last axis of [rows, experts]) of a Softmax
    /// it alone reads, optionally followed by the renormalization of the selected values (ReduceSum, Scale and Div by the scaled sum, the
    /// values read by nothing else). The top-k unit absorbs them: one launch instead of up to five per layer (decode: ~13 µs to ~4 µs).
    /// Also the other order (gpt-oss): a TopK of the logits whose values only a Softmax over the k selected reads.
    /// </summary>
    private void FuseRouterTopK(List<LoweringUnit> units, Dictionary<Value, List<Node>> consumers)
    {
        if (!_opt.Has(DeviceFeatures.SubgroupOps))
        {
            return;
        }

        var unitOf = new Dictionary<Node, LoweringUnit>();
        foreach (var unit in units)
        {
            foreach (var node in unit.Fused ?? (unit.Node is { } single ? [single] : []))
            {
                unitOf[node] = unit;
            }
        }

        bool Alone(Node node) => unitOf.TryGetValue(node, out var unit) && (unit.Node == node || unit.Fused is [var only] && only == node)
            && unit.Absorbed is null && unit.NormHead is null && unit.NormTail is null && unit.Gathers is null && unit.MatMulHead is null && unit.ConvHead is null;
        List<Node> Readers(Value v) => consumers.TryGetValue(v, out var readers) ? readers : [];
        bool Internal(Value v) => !_segmentOutputs.Contains(v);

        var removed = new HashSet<LoweringUnit>();
        foreach (var unit in units)
        {
            if (unit.Node is not { Op.Name: "TopK" } topk || !Alone(topk) || topk.Attributes.Has("k_symbol") || !topk.Attributes.Get("largest", true)
                || topk.Inputs[0] is not { Shape.Rank: 2 } logits || NormAxis(topk.Attributes.Get("axis", -1L), 2) != 1 || logits.Shape[1].IsSymbolic
                || topk.Attributes.Get<long>("k") is var k && (k > TopKSubgroupMaxK || k > _opt.SubgroupSize || k > logits.Shape[1].Fixed))
            {
                continue;
            }

            bool PlainSoftmax(Node softmax) => Alone(softmax) && softmax.Attributes.Get("scale", 1f) == 1f && !softmax.Attributes.Get("log", false)
                                               && softmax.Inputs[0].Type == DataType.F32;
            if (logits.Type == DataType.F32 && Readers(topk.Outputs[0]) is [{ Op.Name: "Softmax" } selected] && selected.Inputs[0] == topk.Outputs[0]
                && PlainSoftmax(selected) && Internal(topk.Outputs[0]))
            {
                removed.Add(unitOf[selected]);
                unit.Absorbed = [selected];
                _routerTopK[topk] = (selected, true, null);
                continue;
            }

            if (logits.Producer is not { Op.Name: "Softmax" } softmax || !PlainSoftmax(softmax) || Readers(logits) is not [_] || !Internal(logits))
            {
                continue;
            }

            (Node, Node, Node)? norm = null;
            var values = topk.Outputs[0];
            if (Readers(values) is [var first, var second] && Internal(values)
                && (first.Op.Name == "ReduceSum" ? (first, second) : (second, first)) is ({ Op.Name: "ReduceSum" } reduce, { Op.Name: "Div" } div)
                && reduce.Inputs[0] == values && div.Inputs[0] == values && Alone(reduce) && Alone(div)
                && Readers(reduce.Output) is [{ Op.Name: "Scale" } scale] && Internal(reduce.Output) && Alone(scale)
                && Readers(scale.Output) is [var divReader] && divReader == div && div.Inputs[1] == scale.Output && Internal(scale.Output)
                && div.Output.Shape.Equals(values.Shape))
            {
                norm = (reduce, scale, div);
                removed.Add(unitOf[reduce]);
                removed.Add(unitOf[scale]);
                removed.Add(unitOf[div]);
            }

            removed.Add(unitOf[softmax]);
            unit.Absorbed = norm is var (r, sc, d) ? [softmax, r, sc, d] : [softmax];
            _routerTopK[topk] = (softmax, false, norm);
        }

        units.RemoveAll(removed.Contains);
    }

    /// <summary>Lowers a router top-k with its absorbed softmax and normalization (<see cref="FuseRouterTopK"/>) as one launch.</summary>
    private void LowerRouterTopK(Node n, Node softmax, bool selected, (Node Reduce, Node Scale, Node Div)? norm)
    {
        var x = AsDenseF32(Ref(selected ? n.Inputs[0] : softmax.Inputs[0]), "topk");
        long K = n.Attributes.Get<long>("k");
        var vals = Produce(norm is { } nd ? nd.Div.Output : selected ? softmax.Output : n.Outputs[0]);
        var idxs = Produce(n.Outputs[1]);
        var macros = Macros(("LEN", x.Shape[1]), ("K", K), ("INNER", 1), ("LARGEST", true), ("WG", TopKWorkgroup), ("P2", TopKSortSlots(K)), ("NORM", norm is not null),
            ("NORM_SCALE", norm?.Scale.Attributes.Get("scale", 1f) ?? 1f), ("NORM_BIAS", norm?.Scale.Attributes.Get("bias", 0f) ?? 0f), ("SELECTED_SOFTMAX", selected),
            ("ROW_SLOTS", TopKRowSlots(x.Shape[1])));
        var kernel = Lib("topk.slang", "topk_softmax", macros);
        var tx = x;
        Emit($"router topk {n.Outputs[1].Name}", [softmax, n, .. norm is var (r, s, d) ? new[] { r, s, d } : []],
            [Always(kernel, b => TopKGrid(Rows(b, tx)))],
            [Pointer(x), Pointer(vals), Pointer(idxs), TopKRowCount(x, 1, 1)], [x], [vals, idxs]);
    }

    /// <summary>
    /// Values per lane when a router row of <paramref name="length"/> logits stays in registers in <c>topk_softmax</c> (a multiple of the
    /// subgroup size, up to 32 per lane), else 0 (per-lane sorted lists). Qwen3-Coder-Next (512 experts, top 10) on Blackwell: 15 → ~3 µs.
    /// </summary>
    private long TopKRowSlots(long length)
        => DiagnosticSwitches.NoTopKRegisterRow || length % _opt.SubgroupSize != 0 || length / _opt.SubgroupSize > 32 ? 0 : length / _opt.SubgroupSize;

    private void LowerTopK(Node n)
    {
        if (_routerTopK.TryGetValue(n, out var router))
        {
            LowerRouterTopK(n, router.Softmax, router.Selected, router.Norm);
            return;
        }

        if (n.Attributes.Get("k_symbol", "") is { Length: > 0 } kSymbol)
        {
            LowerTopKDynamic(n, kSymbol);
            return;
        }

        long K = n.Attributes.Get<long>("k");
        long sortSlots = TopKSortSlots(K);
        bool subgroup = K <= TopKSubgroupMaxK && _opt.Has(DeviceFeatures.SubgroupOps);
        if (!subgroup && sortSlots * 8 + 4 * 1024 > _opt.MaxSharedMemory)
        {
            LowerTopKSerial(n);
            return;
        }

        var x = AsDenseF32(Ref(n.Inputs[0]), "topk");
        var vals = Produce(n.Outputs[0]);
        var idxs = Produce(n.Outputs[1]);
        int axis = NormAxis(n.Attributes.Get("axis", -1L), x.Rank);
        long len = x.Shape[axis];
        long inner = 1;
        for (int k = axis + 1; k < x.Rank; k++)
        {
            inner *= x.Shape[k];
        }

        long rowsPerBatch = 1;
        for (int k = 1; k < x.Rank; k++)
        {
            rowsPerBatch *= k == axis ? 1 : x.Shape[k];
        }

        var macros = Macros(("LEN", len), ("K", K), ("INNER", inner), ("LARGEST", n.Attributes.Get("largest", true)),
            ("WG", TopKWorkgroup), ("P2", sortSlots));
        var kernel = Lib("topk.slang", subgroup ? "topk_subgroup" : "topk_rows", macros);
        var tx = x;
        Emit($"topk {n.Outputs[0].Name}", [n],
            [Always(kernel, b => TopKGrid(TopKRows(b, tx, axis, rowsPerBatch)))],
            [Pointer(x), Pointer(vals), Pointer(idxs), TopKRowCount(x, axis, rowsPerBatch)], [x], [vals, idxs]);
    }

    /// <summary>
    /// Top-k with a symbolic k (e.g. <c>min(200, detections)</c>) along the outermost axis, whose length may be symbolic too: the
    /// workgroup kernel (or, for k bounds beyond its shared-memory sort, the serial one) reads k and the length as push constants.
    /// </summary>
    private void LowerTopKDynamic(Node n, string kSymbol)
    {
        var x = AsDenseF32(Ref(n.Inputs[0]), "topk");
        var vals = Produce(n.Outputs[0]);
        var idxs = Produce(n.Outputs[1]);
        long kBound = Math.Max(1, _opt.Compile.Bound(kSymbol));
        long sortSlots = TopKSortSlots(kBound);
        bool serial = sortSlots * 8 + 4 * 1024 > _opt.MaxSharedMemory;
        long inner = x.MaxElements / x.Shape[0];
        var macros = Macros(("LEN", x.Shape[0]), ("K", kBound), ("INNER", inner), ("LARGEST", n.Attributes.Get("largest", true)),
            ("WG", TopKWorkgroup), ("P2", sortSlots), ("DYNAMIC", 1));
        var kernel = Lib("topk.slang", serial ? "topk_serial" : "topk_rows", macros);
        KernelArg length = x.Symbol is null ? new UIntArg((uint)x.Shape[0]) : new SymbolArg(x.Symbol, 1);
        var grid = serial ? new Grid(Grid.Div(inner, 64)) : TopKGrid(inner);
        Emit($"topk {n.Outputs[0].Name}", [n], [Always(kernel, _ => grid)],
            [Pointer(x), Pointer(vals), Pointer(idxs), new UIntArg((uint)inner), length, new SymbolArg(kSymbol, 1)], [x], [vals, idxs]);
    }

    /// <summary>Power-of-two slot count of the bitonic sort for <paramref name="k"/> results.</summary>
    private static long TopKSortSlots(long k)
    {
        long slots = 2;
        while (slots < k)
        {
            slots <<= 1;
        }

        return slots;
    }

    /// <summary>Rows (outer x inner positions) for the bindings; dim 0 may be the symbolic batch.</summary>
    private static long TopKRows(Bindings b, TensorRef x, int axis, long rowsPerBatch)
        => axis == 0 ? rowsPerBatch : Rows(b, x) * rowsPerBatch;

    private static KernelArg TopKRowCount(TensorRef x, int axis, long rowsPerBatch)
    {
        if (axis == 0 || x.Symbol is null)
        {
            return new UIntArg((uint)(axis == 0 ? rowsPerBatch : x.Shape[0] * rowsPerBatch));
        }

        return new SymbolArg(x.Symbol, rowsPerBatch);
    }

    /// <summary>One workgroup per row; rows beyond 65535 continue along y (row = y * 65535 + x).</summary>
    private static Grid TopKGrid(long rows) => new((uint)Math.Min(rows, 65535), Grid.Div(rows, 65535));
}
