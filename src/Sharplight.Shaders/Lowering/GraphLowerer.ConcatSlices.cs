using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Concatenation input → (concatenation output, byte offset of its slice): the producer writes into the slice directly.</summary>
    private readonly Dictionary<Value, (Value Concat, long ByteOffset)> _concatSlices = [];

    /// <summary>Concatenation outputs with at least one input computed in place.</summary>
    private readonly HashSet<Value> _concatSliceRoots = [];

    /// <summary>Ops whose lowering writes its rank-4 output through the output's outermost stride, so it can fill a batch-strided slice.</summary>
    private static readonly HashSet<string> s_batchStridedWriters = ["Resize"];

    /// <summary>
    /// Plans concatenations whose inputs are computed in place, in their slices of the output, so the concatenation needs no
    /// launch for them (the copy of a feature-pyramid head, e.g. SegFormer's 4 x 256 channels, costs a full read and write of
    /// the output). A slice is either contiguous (every axis before the concatenation axis has extent 1, see
    /// <see cref="HasSlices"/>: any producer) or batch-strided (only the outermost axis before it is longer: producers in
    /// <see cref="s_batchStridedWriters"/> whose value has no other consumer). An input qualifies when it is produced by a
    /// launch of this segment (not a view, another concatenation, a segment input or output), appears once, and its slice
    /// starts 256-byte aligned and spans a multiple of 16 bytes per outer index, the alignment kernels may assume for vector
    /// loads and stores. Off with <c>SHARPLIGHT_NO_CONCAT_IN_PLACE=1</c>. The slice values get the concatenation output as
    /// their liveness root, so the output stays allocated from the first producer to the last reader of any slice.
    /// </summary>
    private void PlanConcatSlices(List<LoweringUnit> units)
    {
        if (!DiagnosticSwitches.ConcatInPlace)
        {
            return;
        }

        var producers = new Dictionary<Value, Node>();
        var consumers = new Dictionary<Value, int>();
        foreach (var unit in units)
        {
            foreach (var node in unit.Nodes)
            {
                foreach (var output in node.Outputs)
                {
                    producers[output] = node;
                }

                foreach (var input in node.Inputs)
                {
                    consumers[input] = consumers.GetValueOrDefault(input) + 1;
                }
            }
        }

        foreach (var unit in units)
        {
            foreach (var node in unit.Nodes)
            {
                if (node.Op.Name == "Concat" && !_channelsLastConcats.ContainsKey(node.Output))
                {
                    PlanConcatSlices(node, producers, consumers);
                }
            }
        }
    }

    private void PlanConcatSlices(Node concat, Dictionary<Value, Node> producers, Dictionary<Value, int> consumers)
    {
        var y = concat.Output;
        int axis = concat.Attributes.Get<int>("axis");
        if (axis < 0)
        {
            axis += y.Rank;
        }

        bool contiguous = HasSlices(y, axis, batchStrided: false);
        if (!(contiguous || HasSlices(y, axis, batchStrided: true)) || concat.Inputs.Any(i => i.Rank != y.Rank || i.Shape.Dims[axis].IsSymbolic))
        {
            return;
        }

        long inner = 1;
        for (int k = axis + 1; k < y.Rank; k++)
        {
            inner *= y.Shape.Dims[k].Fixed;
        }

        long rowBytes = inner * y.Type.ElementSize;
        long start = 0;
        foreach (var input in concat.Inputs)
        {
            long extent = input.Shape.Dims[axis].Fixed;
            long offset = start * rowBytes;
            long bytes = extent * rowBytes;
            start += extent;
            bool produced = producers.TryGetValue(input, out var producer) && producer.Op.Category != OpCategory.View
                            && producer.Op.Name != "Concat";
            bool fits = contiguous ? HasSlices(input, axis, batchStrided: false)
                : produced && s_batchStridedWriters.Contains(producer!.Op.Name) && input.Rank == 4 && consumers.GetValueOrDefault(input) == 1
                  && HasSlices(input, axis, batchStrided: true);
            if (!produced || !fits || input.Type != y.Type || _segmentOutputs.Contains(input) || _concatSlices.ContainsKey(input)
                || concat.Inputs.Count(i => i == input) != 1 || offset % 256 != 0 || bytes % 16 != 0 || bytes == 0)
            {
                continue;
            }

            _concatSlices[input] = (y, offset);
            _concatSliceRoots.Add(y);
        }
    }

    /// <summary>
    /// True when slices of <paramref name="v"/> along <paramref name="axis"/> are contiguous: plain elements, every axis before it
    /// of extent 1 (a symbolic outermost axis counts when its bound is 1, e.g. a dynamic batch compiled for batch 1), the axis
    /// itself and every later one static. With <paramref name="batchStrided"/> the outermost axis (before the concatenation
    /// axis) may have any extent: each outer index then holds one contiguous run of the slice.
    /// </summary>
    private bool HasSlices(Value v, int axis, bool batchStrided)
    {
        if (v.Type.BlockSize != 1 || v.Type.IsQuantized || axis >= v.Rank || (batchStrided && axis == 0))
        {
            return false;
        }

        for (int k = 0; k < v.Rank; k++)
        {
            var d = v.Shape.Dims[k];
            bool ok = k == 0 && batchStrided ? true
                : k < axis ? Bound(d) == 1 && (!d.IsSymbolic || k == 0)
                : !d.IsSymbolic;
            if (!ok)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The slice of the concatenation output that <paramref name="v"/> is computed in (allocating the output on first use): the
    /// output's strides, so dense when the axes before the concatenation axis have extent 1.
    /// </summary>
    private TensorRef ConcatSlice(Value v, (Value Concat, long ByteOffset) slice)
    {
        var y = Produce(slice.Concat);
        return y with { ByteOffset = y.ByteOffset + slice.ByteOffset, Type = v.Type, Shape = BoundShape(v.Shape), Symbol = v.Shape.Symbol };
    }

    /// <summary>True when <paramref name="src"/> already is the slice of the concatenation output <paramref name="y"/> at <paramref name="byteOffset"/>.</summary>
    private static bool IsConcatSlice(TensorRef src, TensorRef y, long byteOffset)
    {
        if (src.Buffer != y.Buffer || src.ByteOffset != y.ByteOffset + byteOffset || src.Rank != y.Rank)
        {
            return false;
        }

        for (int k = 0; k < src.Rank; k++)
        {
            if (src.Shape[k] != 1 && src.Strides[k] != y.Strides[k])
            {
                return false;
            }
        }

        return true;
    }
}
