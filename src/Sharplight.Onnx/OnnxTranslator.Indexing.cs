using Sharplight.Graph;

namespace Sharplight.Onnx;

// Gather and scatter style operators.
internal sealed partial class OnnxTranslator
{
    private void Gather(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        if (!ins[1]!.IsConst)
        {
            var indices = Idx(ins[1]!);
            int gatherAxis = NormalizeAxis(n.Int("axis", 0), x.Rank);
            if (Sym(indices) && gatherAxis > 0 && !Sym(x) && x.Shape.Dims.Take(gatherAxis).All(d => d.Fixed == 1))
            {
                // Data-dependent indices behind unit dims ([1, S, 4] gathered on axis 1): the unit dims stay in front of the count.
                var inner = Rs(x, x.Shape.Dims.Skip(gatherAxis).Select(d => d.Fixed).ToArray());
                SetDyn(n, 0, Emit("GatherAxis", [inner, indices], Attributes.From(("axis", 0L)), Nm(n)));
                _env[n.Outputs[0]].LeadingUnits = gatherAxis + ins[1]!.LeadingUnits;
                return;
            }

            SetDyn(n, 0, Emit("GatherAxis", [x, indices], Attributes.From(("axis", n.Int("axis", 0))), Nm(n)));
            return;
        }
        var idxC = ConstIn(ins, 1, "Gather indices");
        int axis = NormalizeAxis(n.Int("axis", 0), x.Rank);
        bool sym = Sym(x);
        if (axis == 0 && sym)
        {
            throw new NotSupportedException("Gather along the symbolic batch axis");
        }

        long dimA = x.Shape[axis].Fixed;
        var idx = idxC.Ints().Select(v => v < 0 ? v + dimA : v).ToArray();
        if (idx.Any(v => v < 0 || v >= dimA))
        {
            throw new InvalidDataException("Gather index out of range.");
        }
        // output shape: x.dims[:axis] + idx.dims + x.dims[axis+1:]
        var outDims = x.Shape.Dims.Take(axis).Select(d => d.IsSymbolic ? 0L : d.Fixed)
            .Concat(idxC.Dims).Concat(x.Shape.Dims.Skip(axis + 1).Select(d => d.Fixed)).ToArray();
        Value y;
        if (idx.Length == 0)
        {
            throw new NotSupportedException("Gather with empty indices");
        }

        bool contiguous = true;
        for (int i = 1; i < idx.Length; i++)
        {
            if (idx[i] != idx[i - 1] + 1)
            {
                contiguous = false;
            }
        }

        if (contiguous)
        {
            y = SliceAxis(x, axis, idx[0], idx.Length);
        }
        else
        {
            if (idx.Length > 4096)
            {
                throw new NotSupportedException($"Gather with {idx.Length} indices");
            }

            var pieces = idx.Select(i => SliceAxis(x, axis, i, 1)).ToList();
            while (pieces.Count > 64)
            {
                pieces = pieces.Chunk(64).Select(c => c.Length == 1 ? c[0] : _b.Concat(axis, c)).ToList();
            }

            y = pieces.Count == 1 ? pieces[0] : _b.Concat(axis, pieces.ToArray());
        }
        SetDyn(n, 0, Emit("Reshape", [y], Attributes.From(("shape", outDims)), Nm(n)));
    }

    private void GatherElements(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("axis", n.Int("axis", 0)));
        SetDyn(n, 0, Emit("GatherElements", [X(ins, 0), Idx(ins[1]!)], attributes, Nm(n)));
    }

    private void GatherNd(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("batch_dims", n.Int("batch_dims", 0)));
        SetDyn(n, 0, Emit("GatherND", [X(ins, 0), Idx(ins[1]!)], attributes, Nm(n)));
    }

    /// <summary>ScatterElements, and its deprecated alias Scatter.</summary>
    private void ScatterElements(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("axis", n.Int("axis", 0)), ("reduction", n.Str("reduction", "none")));
        SetDyn(n, 0, Emit("ScatterElements", [X(ins, 0), Idx(ins[1]!), X(ins, 2)], attributes, Nm(n)));
    }

    private void ScatterNd(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("reduction", n.Str("reduction", "none")));
        SetDyn(n, 0, Emit("ScatterND", [X(ins, 0), Idx(ins[1]!), X(ins, 2)], attributes, Nm(n)));
    }

    private void OneHot(OnnxNode n, TranslatedTensor?[] ins)
    {
        long depth = ConstIn(ins, 1, "OneHot depth").ScalarInt();
        var values = ConstIn(ins, 2, "OneHot values").Floats();
        var attributes = Attributes.From(("depth", depth), ("axis", n.Int("axis", -1)), ("on", (float)values[1]), ("off", (float)values[0]));
        SetDyn(n, 0, Emit("OneHot", [Idx(ins[0]!)], attributes, Nm(n)));
    }
}
