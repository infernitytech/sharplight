namespace Sharplight.Onnx;

// Recognition of the exact (erf) GELU written out with elementary ops (exports below opset 20).
internal sealed partial class OnnxTranslator
{
    private const double Sqrt2 = 1.4142135623730951;

    /// <summary>
    /// Translates <paramref name="mul"/> as one <c>Gelu</c> when it ends <c>x · 0.5 · (1 + erf(x / √2))</c> (the three factors
    /// multiplied in any order and grouping, the division also written as a product with 1/√2). The elementary nodes
    /// already emitted are left to dead-node elimination. Returns false when the pattern does not match.
    /// </summary>
    private bool TryDecomposedGelu(OnnxNode mul)
    {
        var factors = new List<string>();
        CollectFactors(mul, factors, depth: 0);
        if (factors.Count != 3)
        {
            return false;
        }

        string? half = factors.FirstOrDefault(f => IsScalarConst(f, 0.5));
        string? onePlusErf = factors.FirstOrDefault(f => f != half && Producer(f, "Add") is { } add
            && ConstOperand(add, out int e) is { Count: 1 } one && Math.Abs(one.ScalarF32() - 1f) < 1e-7f
            && Producer(add.Inputs[e], "Erf") is not null);
        if (half is null || onePlusErf is null)
        {
            return false;
        }

        string x = factors.First(f => f != half && f != onePlusErf);
        var add1 = _liveProducers[onePlusErf];
        var erf = _liveProducers[add1.Inputs[ConstOperandIndex(add1)]];
        if (Producer(erf.Inputs[0], "Div", "Mul") is not { } scaled || ConstOperand(scaled, out int xi) is not { Count: 1 } c
            || scaled.Inputs[xi] != x || (scaled.OpType == "Div" && xi != 0))
        {
            return false;
        }

        double expected = scaled.OpType == "Div" ? Sqrt2 : 1 / Sqrt2;
        if (Math.Abs(c.ScalarF32() - expected) > 1e-5 * expected
            || !_env.TryGetValue(x, out var xt) || xt is not { LeadingUnits: 0, V: { } xv } || xv.Type != Tensors.DataType.F32)
        {
            return false;
        }

        SetDyn(mul, 0, Emit("Gelu", [xv], null, Nm(mul)));
        return true;
    }

    /// <summary>Operands of a tree of Mul nodes rooted at <paramref name="mul"/> (inner products only when read by the tree alone).</summary>
    private void CollectFactors(OnnxNode mul, List<string> factors, int depth)
    {
        foreach (var input in mul.Inputs)
        {
            if (depth < 2 && Producer(input, "Mul") is { } inner && SoleConsumer(input, "Mul") == mul)
            {
                CollectFactors(inner, factors, depth + 1);
            }
            else
            {
                factors.Add(input);
            }
        }
    }

    private bool IsScalarConst(string name, double value)
        => _env.TryGetValue(name, out var t) && t.C is { Count: 1, HasSymbols: false } c && OnnxType.IsFloat(c.ElemType) && Math.Abs(c.ScalarF32() - value) < 1e-7;

    private int ConstOperandIndex(OnnxNode n)
    {
        _ = ConstOperand(n, out int other);
        return other;
    }
}
