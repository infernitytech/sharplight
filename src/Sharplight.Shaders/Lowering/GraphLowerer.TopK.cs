using Sharplight.Graph;
using Sharplight.Runtime;

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
    private void LowerTopK(Node n)
    {
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
