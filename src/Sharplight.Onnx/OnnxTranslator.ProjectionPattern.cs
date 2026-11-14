using Sharplight.Graph;

namespace Sharplight.Onnx;

// Fusion of sibling projections: several MatMuls of one activation by constant weights (an encoder's Q, K and V).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Translates <paramref name="n"/> = <c>MatMul(x, W)</c> with a constant 2-D <c>W</c> together with every later MatMul of the
    /// same <c>x</c> by a constant 2-D weight of the same input size and element type, as one MatMul over the weights
    /// concatenated along the output axis, followed by one slice per original product. When each product's only reader is an
    /// <c>Add</c> of a constant bias vector of its width, the biases are concatenated as well and the slices replace the Adds.
    /// One wider product keeps more of the machine busy than several narrow ones (an encoder's Q, K and V projections) and
    /// reads <c>x</c> once; the result is the same as the separate products. Returns false when <paramref name="n"/> has no
    /// such sibling or fusion is off (<see cref="OnnxImportOptions.FuseProjections"/>).
    /// </summary>
    private bool TryFusedProjections(OnnxNode n, Value x, OnnxConst weight)
    {
        if (!_opts.FuseProjections || x.Rank < 2 || weight.Rank != 2 || !weight.IsFloat || !_consumers.TryGetValue(n.Inputs[0], out var readers))
        {
            return false;
        }

        long k = weight.Dims[0];
        var group = new List<(OnnxNode Node, OnnxConst Weight)>();
        foreach (var r in readers)
        {
            if (r.OpType == "MatMul" && r.Inputs.Count == 2 && r.Inputs[0] == n.Inputs[0] && r.Inputs[1] != n.Inputs[0] && !_skip.Contains(r)
                && !_graphOutputs.Contains(r.Outputs[0])
                && _env.TryGetValue(r.Inputs[1], out var t) && t.C is { Rank: 2, HasSymbols: false } c && c.Dims[0] == k && c.ElemType == weight.ElemType
                && !group.Any(g => g.Node == r))
            {
                group.Add((r, c));
            }
        }

        if (group.Count < 2 || group[0].Node != n)
        {
            return false;
        }

        // Biases: each product's only reader is an Add of a constant vector of its width (all of them, or the Adds stay separate).
        var biases = new List<(OnnxNode Add, OnnxConst Bias)>();
        foreach (var (node, w) in group)
        {
            if (SoleConsumer(node.Outputs[0], "Add") is { } add && ConstOperand(add, out _) is { } b && b.Count == w.Dims[1] && b.Dims.Length >= 1
                && b.Dims[..^1].All(d => d == 1) && b.Dims.Length <= x.Rank && b.ElemType == OnnxType.Float && !_graphOutputs.Contains(add.Outputs[0]))
            {
                biases.Add((add, b));
            }
        }

        bool biased = biases.Count == group.Count;
        string name = Nm(n) + ".fused";
        long total = group.Sum(g => g.Weight.Dims[1]);
        var parts = group.Select(g => g.Weight.Transposed([1, 0])).ToList();
        var fusedWeight = OnnxConst.Lazy(weight.ElemType, [total, k], () => Concatenated(parts), name + ".weight");
        var y = Emit("MatMul", [x, WeightValue(fusedWeight, keepHalf: true, name + ".weight")], null, name);
        Heavy();
        if (biased)
        {
            var biasParts = biases.Select(b => b.Bias).ToList();
            var fusedBias = OnnxConst.Lazy(OnnxType.Float, [total], () => Concatenated(biasParts), name + ".bias");
            y = Emit("Add", [y, ConstValue(fusedBias, name + ".bias")], null, name + ".biased");
        }

        long start = 0;
        for (int i = 0; i < group.Count; i++)
        {
            long width = group[i].Weight.Dims[1];
            var target = biased ? biases[i].Add : group[i].Node;
            SetDyn(target, 0, _b.Slice(y, y.Rank - 1, start, width, Nm(target)));
            start += width;
            _skip.Add(group[i].Node);
            if (biased)
            {
                _skip.Add(biases[i].Add);
            }
        }

        return true;
    }

    /// <summary>The bytes of <paramref name="parts"/> one after another (row-major tensors concatenated along their first axis).</summary>
    private static byte[] Concatenated(List<OnnxConst> parts)
    {
        var bytes = new byte[parts.Sum(p => p.ByteSize)];
        long offset = 0;
        foreach (var p in parts)
        {
            var b = p.Bytes();
            b.CopyTo(bytes, offset);
            offset += b.Length;
        }

        return bytes;
    }
}
