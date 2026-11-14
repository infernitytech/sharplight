using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Recognition of LayerNormalization written out with elementary ops (exports below opset 17).
internal sealed partial class OnnxTranslator
{
    /// <summary>
    /// Translates <paramref name="div"/> as one <c>LayerNorm</c> when it ends the decomposed form PyTorch exports below opset 17:
    /// <c>d = x - ReduceMean(x); Div(d, Sqrt(ReduceMean(d²) + eps))</c> over the last axis (d² as <c>Pow(d, 2)</c> or
    /// <c>Mul(d, d)</c>), together with a following <c>Mul</c> by a constant scale and <c>Add</c> of a constant bias of the
    /// normalized size. The elementary nodes already emitted for the statistics are left to dead-node elimination.
    /// Returns false when the pattern does not match (the Div is then translated on its own).
    /// </summary>
    private bool TryDecomposedLayerNorm(OnnxNode div)
    {
        if (Producer(div.Inputs[1], "Sqrt") is not { } sqrt
            || Producer(sqrt.Inputs[0], "Add") is not { } addEps
            || ConstOperand(addEps, out int varianceIndex) is not { Count: 1 } epsilon
            || Producer(addEps.Inputs[varianceIndex], "ReduceMean") is not { } varianceMean
            || Producer(varianceMean.Inputs[0], "Pow", "Mul") is not { } square
            || Producer(div.Inputs[0], "Sub") is not { } sub
            || sub.Outputs[0] != div.Inputs[0]
            || !Squares(square, sub.Outputs[0])
            || Producer(sub.Inputs[1], "ReduceMean") is not { } mean
            || mean.Inputs[0] != sub.Inputs[0]
            || !_env.TryGetValue(sub.Inputs[0], out var xt)
            || xt is not { LeadingUnits: 0, V: { Rank: > 0 } x }
            || x.Type != DataType.F32
            || x.Shape[-1].IsSymbolic
            || !OverLastAxis(mean, xt) || !OverLastAxis(varianceMean, xt))
        {
            return false;
        }

        long d = x.Shape[-1].Fixed;
        var last = div;
        var absorbed = new List<OnnxNode>();
        OnnxConst? scale = null;
        OnnxConst? bias = null;
        if (SoleConsumer(div.Outputs[0], "Mul") is { } mul && ConstOperand(mul, out _) is { } s && IsVectorOf(s, d, x.Rank))
        {
            scale = s;
            absorbed.Add(last = mul);
            if (SoleConsumer(mul.Outputs[0], "Add") is { } add && ConstOperand(add, out _) is { } b && IsVectorOf(b, d, x.Rank))
            {
                bias = b;
                absorbed.Add(last = add);
            }
        }

        string name = Nm(last);
        scale ??= OnnxConst.FromFloats(OnnxType.Float, [d], Enumerable.Repeat(1.0, (int)d).ToArray(), name + ".scale");
        var inputs = new List<Value> { x, ConstValue(scale.Reshaped([d]), name + ".scale") };
        if (bias is not null)
        {
            inputs.Add(ConstValue(bias.Reshaped([d]), name + ".bias"));
        }

        SetDyn(last, 0, Emit("LayerNorm", [.. inputs], Attributes.From(("eps", epsilon.ScalarF32())), name));
        _skip.UnionWith(absorbed);
        return true;
    }

    /// <summary>The live node producing <paramref name="name"/> when it is one of <paramref name="opTypes"/>.</summary>
    private OnnxNode? Producer(string name, params string[] opTypes)
        => _liveProducers.TryGetValue(name, out var node) && opTypes.Contains(node.OpType) ? node : null;

    /// <summary>The only node reading <paramref name="name"/> when it is a <paramref name="opType"/> and the value is not a graph output.</summary>
    private OnnxNode? SoleConsumer(string name, string opType)
        => !_graphOutputs.Contains(name) && _consumers.TryGetValue(name, out var list) && list.Count == 1 && list[0].OpType == opType
            ? list[0]
            : null;

    /// <summary>
    /// The constant operand of a binary node whose other operand is not constant (or not translated yet); <paramref name="other"/>
    /// is that operand's index.
    /// </summary>
    private OnnxConst? ConstOperand(OnnxNode n, out int other)
    {
        other = -1;
        if (n.Inputs.Count != 2)
        {
            return null;
        }

        for (int i = 0; i < 2; i++)
        {
            if (_env.TryGetValue(n.Inputs[i], out var t) && t.C is { HasSymbols: false } c
                && !(_env.TryGetValue(n.Inputs[1 - i], out var o) && o.IsConst) && OnnxType.IsFloat(c.ElemType))
            {
                other = 1 - i;
                return c;
            }
        }

        return null;
    }

    /// <summary>Whether <paramref name="square"/> computes the square of <paramref name="d"/> (<c>Pow(d, 2)</c> or <c>Mul(d, d)</c>).</summary>
    private bool Squares(OnnxNode square, string d)
    {
        if (square.OpType == "Mul")
        {
            return square.Inputs.Count == 2 && square.Inputs[0] == d && square.Inputs[1] == d;
        }

        return square.Inputs.Count == 2 && square.Inputs[0] == d && _env.TryGetValue(square.Inputs[1], out var e)
            && e.C is { Count: 1, HasSymbols: false } c && c.ScalarF32() == 2f;
    }

    /// <summary>Whether a ReduceMean averages exactly the last axis of <paramref name="x"/> and keeps the reduced dim.</summary>
    private bool OverLastAxis(OnnxNode reduce, TranslatedTensor x)
    {
        if (reduce.Int("keepdims", 1) == 0)
        {
            return false;
        }

        long[]? axes = reduce.Has("axes") ? reduce.Ints("axes")
            : reduce.Inputs.Count > 1 && reduce.Inputs[1] != "" && _env.TryGetValue(reduce.Inputs[1], out var a) && a.C is { HasSymbols: false } ac ? ac.Ints()
            : null;
        return axes is [var axis] && (axis == -1 || x.LeadingUnits == 0 && axis == x.V!.Rank - 1);
    }

    /// <summary>
    /// Whether a constant holds <paramref name="d"/> values along the last axis (all other dims 1) and broadcasting it against a
    /// tensor of rank <paramref name="rank"/> keeps that rank.
    /// </summary>
    private static bool IsVectorOf(OnnxConst c, long d, int rank)
        => c.Count == d && c.Rank <= rank && (c.Rank == 0 ? d == 1 : c.Dims[^1] == d);
}
